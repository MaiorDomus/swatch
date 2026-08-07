"""Tests for swatch.audio"""

import multiprocessing
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from swatch.audio import (
    SILENT_DBFS,
    AudioMonitor,
    SoundStateClassifier,
    compute_normalized_spectrum,
    compute_rms_dbfs,
    compute_spectral_flux,
)
from swatch.config import AudioMonitorConfig

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
        cls.silence_wav = cls._generate_silence()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

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

    def _run_to_completion(self, wav_path: str) -> bool:
        config = AudioMonitorConfig(
            name="test", rtsp_url="unused", min_on_seconds=2, min_off_seconds=2
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

    def test_silence_stays_off(self) -> None:
        assert self._run_to_completion(self.silence_wav) is False

    def test_missing_input_does_not_raise(self) -> None:
        assert self._run_to_completion("/nonexistent/path/does-not-exist.wav") is False


if __name__ == "__main__":
    unittest.main()
