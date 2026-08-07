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

import logging
import multiprocessing
import subprocess
import threading

import numpy as np

from swatch.config import AudioMonitorConfig

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


class SoundStateClassifier:
    """Debounces per-window loud+steady booleans into an on/off state.

    A state flip only takes effect once the *opposite* classification has
    been seen for min_on_seconds (to turn on) or min_off_seconds (to turn
    off) worth of consecutive windows, so a single stray loud noise or a
    brief lull doesn't flicker the reported state.
    """

    def __init__(
        self,
        window_seconds: float,
        min_on_seconds: float,
        min_off_seconds: float,
    ) -> None:
        self.min_on_windows = max(1, round(min_on_seconds / window_seconds))
        self.min_off_windows = max(1, round(min_off_seconds / window_seconds))
        self.is_on = False
        self._pending_windows = 0

    def update(self, is_candidate: bool) -> bool:
        """Feed in whether the latest window qualifies as loud+steady;
        returns the (possibly still-debouncing) current on/off state."""
        if is_candidate == self.is_on:
            self._pending_windows = 0
            return self.is_on

        self._pending_windows += 1
        required_windows = self.min_on_windows if is_candidate else self.min_off_windows

        if self._pending_windows >= required_windows:
            self.is_on = is_candidate
            self._pending_windows = 0

        return self.is_on


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
        finally:
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

            if process.stdout is not None:
                process.stdout.close()

            if process.returncode not in (0, None, -15):  # -15 == terminated by us
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

            if self.stop_event.wait(reconnect_backoff_seconds):
                break

        logger.info("Stopping audio monitor for %s", self.monitor_name)
