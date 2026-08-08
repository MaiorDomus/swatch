"""Detects sustained mechanical noise (e.g. a kitchen hood fan) from a
camera's audio stream.

This is a heuristic, not a trained sound classifier: a window of audio is
treated as "hood-like" when it is both loud enough (``threshold_db``) and
spectrally steady over time (``max_spectral_flux``). A running fan produces a
fairly constant hum/whoosh, whereas speech and music have much higher
spectral flux -- their frequency content keeps changing (phonemes, notes) --
which is what lets this tell them apart from a loud appliance without needing
a model. It won't be perfect (e.g. a sustained drone in a song could still
fool it), but it's deliberately lightweight and dependency-free (just numpy),
consistent with the rest of swatch's detection code.
"""

import datetime
import logging
import multiprocessing
import subprocess
import threading

import numpy as np

from swatch.config import AudioMonitorConfig
from swatch.debounce import SustainedStateTracker
from swatch.models import Detection
from swatch.util import get_random_suffix

logger = logging.getLogger(__name__)

SILENT_DBFS = -120.0


def compute_rms_dbfs(samples: np.ndarray) -> float:
    """Compute the RMS loudness of int16 PCM samples in dBFS (0 = full scale,
    quieter audio is more negative)."""
    if samples.size == 0:
        return SILENT_DBFS

    normalized = samples.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(np.square(normalized))))

    if rms <= 0:
        return SILENT_DBFS

    return 20 * float(np.log10(rms))


SPECTRUM_BANDS = 32


def compute_normalized_spectrum(
    samples: np.ndarray, bands: int = SPECTRUM_BANDS
) -> np.ndarray:
    """Compute a unit-norm, coarsely-banded magnitude spectrum, so spectral
    flux measures changes in broad spectral *shape* rather than bin-by-bin
    jitter. Broadband noise (e.g. a fan) has essentially random magnitude in
    any single high-resolution FFT bin from one window to the next even
    though its overall shape is stable, so comparing raw per-bin magnitudes
    would make steady noise look "unsteady". Averaging into a handful of
    bands smooths that jitter out while still catching real shape changes
    (like speech moving between phonemes, or music changing notes)."""
    if samples.size == 0:
        return np.zeros(1)

    windowed = samples.astype(np.float64) * np.hanning(len(samples))
    magnitude = np.abs(np.fft.rfft(windowed))
    banded = np.array([band.mean() for band in np.array_split(magnitude, bands)])
    norm = float(np.linalg.norm(banded))

    if norm == 0:
        return banded

    return banded / norm


def compute_spectral_flux(
    prev_spectrum: np.ndarray, curr_spectrum: np.ndarray
) -> float:
    """Euclidean distance between two unit-norm magnitude spectra: ~0 for an
    unchanging spectral shape (a fan's steady hum), up to ~1.4 (both are
    non-negative unit vectors) for a completely different one (e.g. speech
    moving between phonemes)."""
    if prev_spectrum.shape != curr_spectrum.shape:
        # window sizes only differ at stream start/end; treat as "no info yet"
        return 0.0

    return float(np.linalg.norm(curr_spectrum - prev_spectrum))


# SoundStateClassifier is the same debounce primitive AutoDetector now uses
# for noisy per-frame object detections (see swatch/detection.py) -- kept as
# an alias here since it's the established public name/import path for the
# audio side.
SoundStateClassifier = SustainedStateTracker


class AudioMonitor(threading.Thread):
    """Background thread that pulls audio from a stream via ffmpeg and
    classifies whether sustained hood-like noise is currently audible."""

    def __init__(
        self,
        config: AudioMonitorConfig,
        stop_event: multiprocessing.Event,
        input_source: str | None = None,
    ) -> None:
        """Initialize the audio monitor.

        input_source overrides config.rtsp_url with an arbitrary ffmpeg input
        (e.g. a local file path), which is what makes this testable without a
        real camera.
        """
        threading.Thread.__init__(self)
        # AudioMonitor is only ever constructed with monitors sourced from
        # SwatchConfig.runtime_config, which always stamps config.name.
        assert config.name is not None
        self.name = f"audio_monitor_{config.name}"
        self.monitor_name: str = config.name
        self.config = config
        self.stop_event = stop_event
        self._input_source = input_source or config.rtsp_url
        self.is_on = False
        self._classifier = SoundStateClassifier(
            config.window_seconds, config.min_on_seconds, config.min_off_seconds
        )
        # Tracks the currently-open Detection row (if any), so on/off
        # history for audio monitors shows up in /api/detections the same
        # way object detections' does -- lets the dashboard build one
        # combined "last N on/off" table from a single endpoint.
        self._was_on = False
        self._open_detection_id: str | None = None

    def _build_ffmpeg_cmd(self) -> list[str]:
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        if self._input_source.startswith("rtsp"):
            cmd += [
                "-rtsp_transport",
                "tcp",
                "-allowed_media_types",
                "audio",
                "-timeout",
                "15000000",  # microseconds; reconnect if the stream stalls
            ]
        else:
            # A local file (only used by tests, in place of a real camera) is
            # otherwise decoded as fast as possible rather than arriving over
            # time like a live stream does; -re paces it at its native rate
            # so file-based tests actually exercise the "still running" case.
            cmd += ["-re"]

        cmd += [
            "-i",
            self._input_source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(self.config.sample_rate),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "pipe:1",
        ]
        return cmd

    def __record_transition__(self) -> None:
        """Create/end a Detection row whenever is_on flips, mirroring
        AutoDetector's new/end bookkeeping for object detections. camera/
        zone/color_variant/top_area don't have an audio equivalent, so
        they're left blank/zero -- label (the monitor name) is what
        /api/detections?label=... filters on."""
        if self.is_on and not self._was_on:
            self._open_detection_id = f"{self.monitor_name}.{get_random_suffix()}"
            Detection.replace(
                id=self._open_detection_id,
                label=self.monitor_name,
                camera="",
                zone="",
                color_variant="audio",
                start_time=datetime.datetime.now().timestamp(),
                top_area=0,
            ).execute()
        elif not self.is_on and self._was_on and self._open_detection_id:
            Detection.update(
                end_time=datetime.datetime.now().timestamp()
            ).where(Detection.id == self._open_detection_id).execute()
            self._open_detection_id = None

        self._was_on = self.is_on

    def __close_stale_detection__(self) -> None:
        """If the stream dropped mid-"on", end the open detection and reset
        state so a reconnect starts clean -- otherwise is_on (and any open
        Detection row) would stay stuck "on" through a real disconnect until
        the next stream happens to report enough consecutive quiet windows,
        which never dependably happens if the stream just isn't there."""
        if self.is_on:
            self.is_on = False
            self.__record_transition__()

        self._classifier = SoundStateClassifier(
            self.config.window_seconds,
            self.config.min_on_seconds,
            self.config.min_off_seconds,
        )

    def _process_stream(self) -> None:
        """Run one ffmpeg session end-to-end, classifying audio until it
        stops (stream ended, crashed, or stop_event was set)."""
        window_bytes = int(self.config.sample_rate * self.config.window_seconds) * 2

        process = subprocess.Popen(
            self._build_ffmpeg_cmd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

        try:
            prev_spectrum: np.ndarray | None = None
            assert process.stdout is not None

            while not self.stop_event.is_set():
                raw = process.stdout.read(window_bytes)

                if not raw:
                    break  # ffmpeg exited / stream ended

                samples = np.frombuffer(raw, dtype=np.int16)
                loudness_db = compute_rms_dbfs(samples)
                curr_spectrum = compute_normalized_spectrum(samples)

                flux = (
                    compute_spectral_flux(prev_spectrum, curr_spectrum)
                    if prev_spectrum is not None
                    else 0.0
                )
                prev_spectrum = curr_spectrum

                is_candidate = (
                    loudness_db >= self.config.threshold_db
                    and flux <= self.config.max_spectral_flux
                )
                self.is_on = self._classifier.update(is_candidate)
                self.__record_transition__()
        finally:
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            if process.stdout is not None:
                process.stdout.close()

            # 0/None: exited cleanly on its own (e.g. end of a local file).
            # -15: killed by our SIGTERM directly (rare -- usually caught, see below).
            # 255: ffmpeg's own exit code when it catches SIGTERM/SIGINT and shuts
            # down cleanly, which is what happens every time we call terminate()
            # above and ffmpeg was still running -- this is the common case, not
            # an error.
            if process.returncode not in (0, None, -15, 255):
                logger.warning(
                    "ffmpeg for audio monitor %s exited with code %s",
                    self.monitor_name,
                    process.returncode,
                )

    def run(self) -> None:
        logger.info("Starting audio monitor for %s", self.monitor_name)
        reconnect_backoff_seconds = 5

        while not self.stop_event.is_set():
            try:
                self._process_stream()
            except Exception:
                logger.exception(
                    "Audio monitor for %s crashed, retrying", self.monitor_name
                )

            self.__close_stale_detection__()

            if self.stop_event.wait(reconnect_backoff_seconds):
                break

        logger.info("Stopping audio monitor for %s", self.monitor_name)
