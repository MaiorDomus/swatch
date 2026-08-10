"""Tests for swatch.audio"""

import multiprocessing
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from peewee import SqliteDatabase

from swatch.audio import (
    SILENT_DBFS,
    AudioMonitor,
    SoundStateClassifier,
    compute_low_band_energy_ratio,
    compute_normalized_spectrum,
    compute_rms_dbfs,
    compute_spectral_flux,
)
from swatch.config import AudioMonitorConfig
from swatch.models import Detection

HAS_FFMPEG = shutil.which("ffmpeg") is not None


class TestComputeRmsDbfs(unittest.TestCase):
    """Testing RMS loudness computation."""

    def test_silence_is_very_negative(self) -> None:
        samples = np.zeros(1000, dtype=np.int16)
        assert compute_rms_dbfs(samples) == SILENT_DBFS

    def test_empty_array_is_very_negative(self) -> None:
        assert compute_rms_dbfs(np.array([], dtype=np.int16)) == SILENT_DBFS

    def test_full_scale_square_wave_is_near_zero_dbfs(self) -> None:
        samples = np.array([32767, -32768] * 500, dtype=np.int16)
        assert compute_rms_dbfs(samples) > -0.5

    def test_quieter_signal_has_lower_dbfs(self) -> None:
        loud = np.array([20000, -20000] * 500, dtype=np.int16)
        quiet = np.array([2000, -2000] * 500, dtype=np.int16)
        assert compute_rms_dbfs(quiet) < compute_rms_dbfs(loud)


class TestComputeNormalizedSpectrum(unittest.TestCase):
    """Testing spectrum computation."""

    def test_empty_array_returns_zeros(self) -> None:
        spectrum = compute_normalized_spectrum(np.array([], dtype=np.int16))
        assert (spectrum == 0).all()

    def test_spectrum_is_unit_norm(self) -> None:
        rng = np.random.default_rng(seed=1)
        samples = (rng.standard_normal(4000) * 5000).astype(np.int16)
        spectrum = compute_normalized_spectrum(samples)
        assert abs(np.linalg.norm(spectrum) - 1.0) < 1e-9

    def test_silence_returns_unnormalized_zeros(self) -> None:
        spectrum = compute_normalized_spectrum(np.zeros(1000, dtype=np.int16))
        assert (spectrum == 0).all()

    def test_cutoff_ignores_high_frequency_differences(self) -> None:
        """Two windows sharing the same low tone but differing only above the
        cutoff should look identical once the high frequency is excluded --
        this is what keeps podcast/speech content (high flux, high frequency)
        from swamping the shape comparison for a fan's low-frequency hum."""
        t = np.arange(1600) / 16000
        low_tone = np.sin(2 * np.pi * 120 * t)

        window_a = ((low_tone + np.sin(2 * np.pi * 3000 * t)) * 10000).astype(
            np.int16
        )
        window_b = ((low_tone + np.sin(2 * np.pi * 5000 * t)) * 10000).astype(
            np.int16
        )

        full_a = compute_normalized_spectrum(window_a)
        full_b = compute_normalized_spectrum(window_b)
        assert compute_spectral_flux(full_a, full_b) > 0.1

        low_a = compute_normalized_spectrum(
            window_a, sample_rate=16000, cutoff_hz=500.0
        )
        low_b = compute_normalized_spectrum(
            window_b, sample_rate=16000, cutoff_hz=500.0
        )
        assert compute_spectral_flux(low_a, low_b) < 0.05

    def test_extreme_cutoff_does_not_produce_nan(self) -> None:
        """A cutoff so low it zeroes out virtually every bin must not poison
        the normalized vector with nan (e.g. from a stray empty band)."""
        samples = (np.sin(2 * np.pi * 120 * np.arange(1600) / 16000) * 10000).astype(
            np.int16
        )
        spectrum = compute_normalized_spectrum(
            samples, sample_rate=16000, cutoff_hz=0.5
        )
        assert not np.isnan(spectrum).any()

    def test_cutoff_preserves_band_bin_counts(self) -> None:
        """Zeroing bins above cutoff (rather than truncating the array before
        banding) must not shrink how many bins each remaining band averages
        over -- otherwise steady broadband noise loses the jitter-smoothing
        that makes it look "steady" in the first place. Two independent
        draws of the same broadband noise should still look nearly identical
        under a low cutoff, the same way they do with no cutoff at all."""
        rng = np.random.default_rng(seed=2)
        window_a = (rng.standard_normal(16000) * 5000).astype(np.int16)
        window_b = (rng.standard_normal(16000) * 5000).astype(np.int16)

        full_a = compute_normalized_spectrum(window_a)
        full_b = compute_normalized_spectrum(window_b)
        full_flux = compute_spectral_flux(full_a, full_b)

        low_a = compute_normalized_spectrum(window_a, sample_rate=16000, cutoff_hz=500.0)
        low_b = compute_normalized_spectrum(window_b, sample_rate=16000, cutoff_hz=500.0)
        low_flux = compute_spectral_flux(low_a, low_b)

        assert low_flux < full_flux + 0.05


class TestComputeSpectralFlux(unittest.TestCase):
    """Testing spectral flux (spectral shape distance)."""

    def test_identical_spectra_have_zero_flux(self) -> None:
        spectrum = compute_normalized_spectrum(
            np.array([1000, -1000] * 500, dtype=np.int16)
        )
        assert compute_spectral_flux(spectrum, spectrum) == 0.0

    def test_mismatched_shapes_return_zero(self) -> None:
        assert compute_spectral_flux(np.zeros(4), np.zeros(8)) == 0.0

    def test_different_spectra_have_nonzero_flux(self) -> None:
        low_tone = compute_normalized_spectrum(
            (np.sin(2 * np.pi * 100 * np.arange(1600) / 16000) * 20000).astype(np.int16)
        )
        high_tone = compute_normalized_spectrum(
            (np.sin(2 * np.pi * 4000 * np.arange(1600) / 16000) * 20000).astype(
                np.int16
            )
        )
        assert compute_spectral_flux(low_tone, high_tone) > 0.5


class TestComputeLowBandEnergyRatio(unittest.TestCase):
    """Testing the raw (pre-normalization) low-band energy share used to
    guard against FFT leakage being mistaken for a real low-frequency hum."""

    def test_empty_array_is_zero(self) -> None:
        assert compute_low_band_energy_ratio(np.array([], dtype=np.int16), 16000, 500) == 0.0

    def test_silence_is_zero(self) -> None:
        samples = np.zeros(1600, dtype=np.int16)
        assert compute_low_band_energy_ratio(samples, 16000, 500) == 0.0

    def test_low_tone_has_high_ratio(self) -> None:
        samples = (np.sin(2 * np.pi * 120 * np.arange(1600) / 16000) * 20000).astype(
            np.int16
        )
        assert compute_low_band_energy_ratio(samples, 16000, 500) > 0.8

    def test_high_tone_has_low_ratio(self) -> None:
        """A tone well above the cutoff should have (almost) none of its
        energy below it -- only negligible FFT windowing leakage, not a
        genuine low-frequency component."""
        samples = (np.sin(2 * np.pi * 4000 * np.arange(1600) / 16000) * 20000).astype(
            np.int16
        )
        assert compute_low_band_energy_ratio(samples, 16000, 500) < 0.02


class TestSoundStateClassifier(unittest.TestCase):
    """Testing the on/off debounce state machine."""

    def test_starts_off(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=3.0, min_off_seconds=3.0
        )
        assert classifier.is_on is False

    def test_single_candidate_window_does_not_flip_on(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=3.0, min_off_seconds=3.0
        )
        assert classifier.update(True) is False

    def test_sustained_candidate_windows_flip_on(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=3.0, min_off_seconds=3.0
        )
        assert classifier.update(True) is False
        assert classifier.update(True) is False
        assert classifier.update(True) is True

    def test_a_gap_resets_the_sustained_count(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=3.0, min_off_seconds=3.0
        )
        classifier.update(True)
        classifier.update(True)
        classifier.update(False)  # resets progress toward "on"
        assert classifier.update(True) is False
        assert classifier.update(True) is False
        assert classifier.update(True) is True

    def test_sustained_quiet_flips_back_off(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=2.0, min_off_seconds=2.0
        )
        classifier.update(True)
        assert classifier.update(True) is True

        assert classifier.update(False) is True
        assert classifier.update(False) is False

    def test_on_and_off_durations_are_independent(self) -> None:
        """A fast min_on_seconds with a slow min_off_seconds shouldn't flip
        off after only one quiet window."""
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=1.0, min_off_seconds=5.0
        )
        assert classifier.update(True) is True

        assert classifier.update(False) is True
        assert classifier.update(False) is True
        assert classifier.update(False) is True
        assert classifier.update(False) is True
        assert classifier.update(False) is False

    def test_sub_second_windows_still_require_at_least_one(self) -> None:
        """round(min_on_seconds / window_seconds) could come out to 0 for a
        very short duration; must not allow an instant flip."""
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=0.1, min_off_seconds=0.1
        )
        assert classifier.min_on_windows >= 1
        assert classifier.update(True) is True


class TestAudioMonitorDetectionHistory(unittest.TestCase):
    """Testing that on/off transitions get recorded in the Detection table,
    the same way object detections' history does, so a dashboard table can
    show both from the one /api/detections endpoint."""

    def setUp(self) -> None:
        self.db = SqliteDatabase(":memory:")
        Detection.bind(self.db)
        # Detection.create_table() would make end_time NOT NULL (the field
        # has no null=True), rejecting an in-progress detection -- the real
        # schema (migrations/001_init_detection_table.py) leaves it
        # nullable, so build the table that way here too.
        self.db.execute_sql(
            'CREATE TABLE IF NOT EXISTS "detection" ('
            '"id" VARCHAR(30) NOT NULL PRIMARY KEY, "label" VARCHAR(20) NOT NULL, '
            '"camera" VARCHAR(20) NOT NULL, "zone" VARCHAR(20) NOT NULL, '
            '"color_variant" VARCHAR(20) NOT NULL, "start_time" DATETIME NOT NULL, '
            '"end_time" DATETIME, "top_area" INTEGER NOT NULL)'
        )

    def tearDown(self) -> None:
        self.db.drop_tables([Detection])
        self.db.close()

    def _make_monitor(self) -> AudioMonitor:
        config = AudioMonitorConfig(name="kitchen_hood", rtsp_url="unused")
        return AudioMonitor(config, multiprocessing.Event(), input_source="unused")

    def test_turning_on_creates_an_open_detection_row(self) -> None:
        monitor = self._make_monitor()
        monitor.is_on = True
        monitor.__record_transition__()

        rows = list(Detection.select().where(Detection.label == "kitchen_hood"))
        assert len(rows) == 1
        assert rows[0].end_time is None
        assert rows[0].camera == ""
        assert rows[0].color_variant == "audio"

    def test_turning_off_closes_the_open_detection_row(self) -> None:
        monitor = self._make_monitor()
        monitor.is_on = True
        monitor.__record_transition__()

        monitor.is_on = False
        monitor.__record_transition__()

        rows = list(Detection.select().where(Detection.label == "kitchen_hood"))
        assert len(rows) == 1
        assert rows[0].end_time is not None

    def test_repeated_on_readings_do_not_create_duplicate_rows(self) -> None:
        monitor = self._make_monitor()
        monitor.is_on = True
        monitor.__record_transition__()
        monitor.__record_transition__()
        monitor.__record_transition__()

        rows = list(Detection.select().where(Detection.label == "kitchen_hood"))
        assert len(rows) == 1

    def test_close_stale_detection_ends_an_open_row_and_resets_is_on(self) -> None:
        """A stream that drops mid-"on" shouldn't leave is_on stuck True or
        a Detection row with no end_time forever."""
        monitor = self._make_monitor()
        monitor.is_on = True
        monitor.__record_transition__()

        monitor.__close_stale_detection__()

        assert monitor.is_on is False
        rows = list(Detection.select().where(Detection.label == "kitchen_hood"))
        assert len(rows) == 1
        assert rows[0].end_time is not None

    def test_close_stale_detection_is_a_noop_when_already_off(self) -> None:
        monitor = self._make_monitor()
        monitor.__close_stale_detection__()

        rows = list(Detection.select().where(Detection.label == "kitchen_hood"))
        assert len(rows) == 0


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg is not installed")
class TestAudioMonitorEndToEnd(unittest.TestCase):
    """End-to-end tests that actually invoke ffmpeg against local WAV fixtures,
    exercising the same subprocess/PCM-parsing code path used against a real
    RTSP stream in production."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp_dir = tempfile.mkdtemp()
        cls.fan_noise_wav = cls._generate_fan_noise()
        cls.varying_tone_wav = cls._generate_varying_tone()
        cls.fan_with_podcast_wav = cls._generate_fan_noise_with_podcast()
        cls.silence_wav = cls._generate_silence()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def setUp(self) -> None:
        # _process_stream() records on/off transitions to Detection -- a
        # test run reaching is_on=True needs a bound database for that
        # write to succeed. See TestAudioMonitorDetectionHistory for why
        # the table is built via raw SQL instead of create_table().
        self.db = SqliteDatabase(":memory:")
        Detection.bind(self.db)
        self.db.execute_sql(
            'CREATE TABLE IF NOT EXISTS "detection" ('
            '"id" VARCHAR(30) NOT NULL PRIMARY KEY, "label" VARCHAR(20) NOT NULL, '
            '"camera" VARCHAR(20) NOT NULL, "zone" VARCHAR(20) NOT NULL, '
            '"color_variant" VARCHAR(20) NOT NULL, "start_time" DATETIME NOT NULL, '
            '"end_time" DATETIME, "top_area" INTEGER NOT NULL)'
        )

    def tearDown(self) -> None:
        self.db.drop_tables([Detection])
        self.db.close()

    @classmethod
    def _generate_fan_noise(cls) -> str:
        """Steady pink noise + a low hum, standing in for a running hood fan."""
        path = str(Path(cls.tmp_dir) / "fan_noise.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=pink:amplitude=0.3:duration=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=120:duration=6",
                "-filter_complex",
                "amix=inputs=2:duration=shortest",
                "-ar",
                "16000",
                "-ac",
                "1",
                path,
            ],
            check=True,
        )
        return path

    @classmethod
    def _generate_varying_tone(cls) -> str:
        """A frequency sweep: loud, but its spectral shape keeps changing --
        standing in for speech/music rather than a steady mechanical drone."""
        path = str(Path(cls.tmp_dir) / "varying.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "aevalsrc='0.5*sin(2*PI*(200+1800*(t/6))*t)':d=6",
                "-ar",
                "16000",
                "-ac",
                "1",
                path,
            ],
            check=True,
        )
        return path

    @classmethod
    def _generate_fan_noise_with_podcast(cls) -> str:
        """Fan noise (pink noise + low hum) mixed with a speech-like varying
        tone, standing in for a podcast playing near the camera while the
        hood is running -- what should still register as "on" with
        flux_band_cutoff_hz excluding the podcast's higher-frequency content
        from the shape comparison."""
        path = str(Path(cls.tmp_dir) / "fan_with_podcast.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=color=pink:amplitude=0.3:duration=6",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=120:duration=6",
                "-f",
                "lavfi",
                "-i",
                "aevalsrc='0.5*sin(2*PI*(200+1800*(t/6))*t)':d=6",
                "-filter_complex",
                "amix=inputs=3:duration=shortest",
                "-ar",
                "16000",
                "-ac",
                "1",
                path,
            ],
            check=True,
        )
        return path

    @classmethod
    def _generate_silence(cls) -> str:
        path = str(Path(cls.tmp_dir) / "silence.wav")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=16000:cl=mono:d=4",
                path,
            ],
            check=True,
        )
        return path

    def _run_to_completion(
        self, wav_path: str, flux_band_cutoff_hz: float | None = 500.0
    ) -> bool:
        config = AudioMonitorConfig(
            name="test",
            rtsp_url="unused",
            min_on_seconds=2,
            min_off_seconds=2,
            flux_band_cutoff_hz=flux_band_cutoff_hz,
        )
        monitor = AudioMonitor(config, multiprocessing.Event(), input_source=wav_path)
        monitor._process_stream()
        return monitor.is_on

    def test_steady_fan_noise_switches_on(self) -> None:
        assert self._run_to_completion(self.fan_noise_wav) is True

    def test_varying_tone_stays_off(self) -> None:
        """Loud but spectrally unsteady audio should not be mistaken for the
        hood, even though it clears the loudness threshold."""
        assert self._run_to_completion(self.varying_tone_wav) is False

    def test_fan_with_podcast_switches_on_with_band_cutoff(self) -> None:
        """A podcast (speech-like varying tone) playing alongside the fan
        raises the full-spectrum flux enough to hide the fan -- but with
        flux_band_cutoff_hz restricting the comparison to the fan's
        low-frequency hum, it should still register as on."""
        assert self._run_to_completion(self.fan_with_podcast_wav) is True

    def test_fan_with_podcast_stays_off_without_band_cutoff(self) -> None:
        """Sanity check for the above: without flux_band_cutoff_hz, the
        podcast's higher-frequency content dominates the shape comparison
        and the fan underneath it is missed."""
        assert (
            self._run_to_completion(self.fan_with_podcast_wav, flux_band_cutoff_hz=None)
            is False
        )

    def test_silence_stays_off(self) -> None:
        assert self._run_to_completion(self.silence_wav) is False

    def test_missing_input_does_not_raise(self) -> None:
        assert self._run_to_completion("/nonexistent/path/does-not-exist.wav") is False

    def test_stopping_mid_stream_does_not_log_a_warning(self) -> None:
        """Regression: ffmpeg exits with code 255 when it catches the
        SIGTERM from our own terminate() call -- a normal, clean shutdown,
        not an error worth warning about. This is what happens every time an
        AudioMonitor is stopped while its source is still live (e.g. a real
        RTSP stream, or here, a WAV file that hasn't reached EOF yet)."""
        config = AudioMonitorConfig(name="test", rtsp_url="unused")
        stop_event = multiprocessing.Event()
        monitor = AudioMonitor(config, stop_event, input_source=self.fan_noise_wav)

        thread = threading.Thread(target=monitor._process_stream)
        with self.assertNoLogs("swatch.audio", level="WARNING"):
            thread.start()
            time.sleep(2)
            stop_event.set()
            thread.join(timeout=10)

        assert not thread.is_alive()


if __name__ == "__main__":
    unittest.main()
