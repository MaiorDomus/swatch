"""Tests for swatch.audio"""

import multiprocessing
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from peewee import SqliteDatabase

from swatch.audio import (
    SILENT_DBFS,
    AudioMonitor,
    AudioSource,
    SoundStateClassifier,
    compute_band_rms_dbfs,
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


class TestComputeBandRmsDbfs(unittest.TestCase):
    """Testing the absolute loudness of just the low band, used so that
    other sound playing on top of a fan doesn't hide it."""

    @staticmethod
    def _tone(freq: float, amplitude: float = 20000) -> np.ndarray:
        return np.sin(2 * np.pi * freq * np.arange(16000) / 16000) * amplitude

    def test_empty_array_is_very_negative(self) -> None:
        assert (
            compute_band_rms_dbfs(np.array([], dtype=np.int16), 16000, 500)
            == SILENT_DBFS
        )

    def test_silence_is_very_negative(self) -> None:
        samples = np.zeros(16000, dtype=np.int16)
        assert compute_band_rms_dbfs(samples, 16000, 500) == SILENT_DBFS

    def test_low_tone_matches_full_rms(self) -> None:
        """With nothing above the cutoff, the band level is the whole level."""
        samples = self._tone(120).astype(np.int16)
        band_db = compute_band_rms_dbfs(samples, 16000, 500)
        assert abs(band_db - compute_rms_dbfs(samples)) < 0.1

    def test_high_tone_is_far_below_full_rms(self) -> None:
        samples = self._tone(4000).astype(np.int16)
        band_db = compute_band_rms_dbfs(samples, 16000, 500)
        assert band_db < compute_rms_dbfs(samples) - 60

    def test_loud_content_above_cutoff_does_not_change_band_level(self) -> None:
        """The whole point versus compute_low_band_energy_ratio: adding a much
        louder sound above the cutoff (a podcast over a running fan) leaves
        the low band's own level where it was, even though its share of the
        total collapses."""
        hum = self._tone(120, amplitude=500)
        podcast = self._tone(2000, amplitude=15000)
        alone = hum.astype(np.int16)
        mixed = (hum + podcast).astype(np.int16)

        mixed_db = compute_band_rms_dbfs(mixed, 16000, 500)
        alone_db = compute_band_rms_dbfs(alone, 16000, 500)
        assert abs(mixed_db - alone_db) < 0.5
        assert compute_low_band_energy_ratio(mixed, 16000, 500) < 0.1


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

    def test_force_sets_state_and_drops_pending_progress(self) -> None:
        classifier = SoundStateClassifier(
            window_seconds=1.0, min_on_seconds=1.0, min_off_seconds=3.0
        )
        classifier.update(True)
        classifier.update(False)
        classifier.update(False)

        assert classifier.force(False) is False
        # the two off windows already seen don't count toward the next flip
        assert classifier.update(True) is True
        assert classifier.update(False) is True
        assert classifier.update(False) is True
        assert classifier.update(False) is False


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
        cls.low_fan_wav = cls._generate_low_fan()
        cls.low_fan_with_loud_podcast_wav = cls._generate_low_fan_with_loud_podcast()
        cls.hiss_over_faint_hum_wav = cls._generate_hiss_over_faint_hum()
        cls.fan_then_silence_wav = cls._generate_fan_then("anullsrc=r=16000:d=4")
        cls.fan_then_speech_wav = cls._generate_fan_then(
            "aevalsrc='0.5*sin(2*PI*(200+1800*(t/4))*t)':d=4:s=16000"
        )

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
    def _ffmpeg_mix(cls, name: str, inputs: list[str], filter_complex: str) -> str:
        """Render lavfi sources through a filter graph to a 16kHz mono WAV."""
        path = str(Path(cls.tmp_dir) / name)
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        for source in inputs:
            cmd += ["-f", "lavfi", "-i", source]
        cmd += ["-filter_complex", filter_complex, "-ar", "16000", "-ac", "1", path]
        subprocess.run(cmd, check=True)
        return path

    @classmethod
    def _generate_low_fan(cls) -> str:
        """Low-passed noise + a 120Hz hum at roughly a real hood's measured
        level (~-49 dBFS, almost all of it below 500Hz)."""
        return cls._ffmpeg_mix(
            "low_fan.wav",
            [
                "anoisesrc=color=brown:amplitude=0.02:duration=8",
                "sine=frequency=120:duration=8",
            ],
            "[0]lowpass=f=400[n];[1]volume=0.01[h];"
            "[n][h]amix=inputs=2:duration=shortest:normalize=0",
        )

    @classmethod
    def _generate_low_fan_with_loud_podcast(cls) -> str:
        """The low fan above with a much louder speech-like sweep on top (a
        video playing near the camera): the fan's share of the total drops
        to ~0.05, below min_band_energy_ratio's default, while its own
        low-band level is unchanged."""
        return cls._ffmpeg_mix(
            "low_fan_with_loud_podcast.wav",
            [
                "anoisesrc=color=brown:amplitude=0.02:duration=8",
                "sine=frequency=120:duration=8",
                "aevalsrc='0.04*sin(2*PI*(600+2400*mod(t,1.5)/1.5)*t)':d=8:s=16000",
            ],
            "[0]lowpass=f=400[n];[1]volume=0.01[h];"
            "[n][h][2]amix=inputs=3:duration=shortest:normalize=0",
        )

    @classmethod
    def _generate_hiss_over_faint_hum(cls) -> str:
        """Loud, steady high-frequency hiss (a running tap, say) over a hum
        far too faint to be the hood: loud and steady overall (~-51 dBFS),
        but only ~-65 dBFS below 500Hz."""
        return cls._ffmpeg_mix(
            "hiss_over_faint_hum.wav",
            [
                "anoisesrc=color=pink:amplitude=0.03:duration=8",
                "sine=frequency=120:duration=8",
            ],
            "[0]highpass=f=1000[n];[1]volume=0.006[h];"
            "[n][h]amix=inputs=2:duration=shortest:normalize=0",
        )

    @classmethod
    def _generate_fan_then(cls, after: str) -> str:
        """6s of the fan noise above, then 4s of another lavfi source."""
        name = "fan_then_" + after.split("=")[0] + ".wav"
        return cls._ffmpeg_mix(
            name,
            [
                "anoisesrc=color=pink:amplitude=0.3:duration=6",
                "sine=frequency=120:duration=6",
                after,
            ],
            "[0][1]amix=inputs=2:duration=shortest,aresample=16000,"
            "aformat=channel_layouts=mono[f];"
            "[2]aresample=16000,aformat=channel_layouts=mono[a];"
            "[f][a]concat=n=2:v=0:a=1",
        )

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
        self,
        wav_path: str,
        flux_band_cutoff_hz: float | None = 500.0,
        **overrides: float | None,
    ) -> bool:
        config = AudioMonitorConfig(
            **{
                "name": "test",
                "rtsp_url": "unused",
                "min_on_seconds": 2,
                "min_off_seconds": 2,
                "flux_band_cutoff_hz": flux_band_cutoff_hz,
                **overrides,
            }
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

    def test_quiet_off_seconds_switches_off_when_the_hum_stops(self) -> None:
        """4s of silence after the fan is well short of a long
        min_off_seconds, but long enough for quiet_off_seconds."""
        assert (
            self._run_to_completion(self.fan_then_silence_wav, min_off_seconds=30)
            is True
        )
        assert (
            self._run_to_completion(
                self.fan_then_silence_wav, min_off_seconds=30, quiet_off_seconds=2
            )
            is False
        )

    def test_quiet_off_seconds_ignores_loud_unsteady_audio(self) -> None:
        """Speech-like audio after the fan fails the flux check but isn't
        quiet, so it still has to wait out min_off_seconds."""
        assert (
            self._run_to_completion(
                self.fan_then_speech_wav, min_off_seconds=30, quiet_off_seconds=2
            )
            is True
        )
        assert (
            self._run_to_completion(
                self.fan_then_speech_wav, min_off_seconds=2, quiet_off_seconds=2
            )
            is False
        )

    def test_loud_podcast_hides_fan_from_band_energy_ratio(self) -> None:
        """Regression baseline for min_band_level_db: with only the ratio
        guard, a video playing loudly over a running fan drags the fan's
        share of the total below min_band_energy_ratio and it's missed."""
        assert self._run_to_completion(self.low_fan_with_loud_podcast_wav) is False

    def test_band_level_catches_fan_under_loud_podcast(self) -> None:
        assert (
            self._run_to_completion(
                self.low_fan_with_loud_podcast_wav,
                min_band_energy_ratio=0.0,
                min_band_level_db=-56.0,
            )
            is True
        )

    def test_band_level_rejects_loud_hiss_over_faint_hum(self) -> None:
        """Loud, steady sound that's mostly above the cutoff is what the
        ratio guard was for; the level check rejects it by itself too, so
        the ratio can be switched off when using it."""
        assert (
            self._run_to_completion(
                self.hiss_over_faint_hum_wav, min_band_energy_ratio=0.0
            )
            is True
        )
        assert (
            self._run_to_completion(
                self.hiss_over_faint_hum_wav,
                min_band_energy_ratio=0.0,
                min_band_level_db=-56.0,
            )
            is False
        )

    def test_band_level_threshold_is_respected(self) -> None:
        """The low fan measures ~-49 dBFS in its band: on just below that
        threshold, off just above it."""
        assert (
            self._run_to_completion(self.low_fan_wav, min_band_level_db=-52.0) is True
        )
        assert (
            self._run_to_completion(self.low_fan_wav, min_band_level_db=-46.0) is False
        )

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


class RecordingConsumer:
    def __init__(self) -> None:
        self.audio = bytearray()
        self.events: list[str] = []

    def on_audio(self, raw: bytes) -> None:
        self.audio += raw

    def on_stream_end(self) -> None:
        self.events.append("end")

    def on_disconnect(self) -> None:
        self.events.append("disconnect")


class FailingConsumer(RecordingConsumer):
    def on_audio(self, raw: bytes) -> None:
        raise RuntimeError("boom")


class TestAudioSource(unittest.TestCase):
    """Testing that one stream fans out to everything listening to it."""

    def _cmd(self, url: str, has_tls_verify: bool) -> list[str]:
        with mock.patch(
            "swatch.audio.rtsp_has_tls_verify", return_value=has_tls_verify
        ):
            return AudioSource(url, 16000, multiprocessing.Event())._build_ffmpeg_cmd()

    def test_rtsps_skips_certificate_verification(self) -> None:
        cmd = self._cmd("rtsps://cam/abc", True)
        assert cmd[cmd.index("-tls_verify") + 1] == "0"

    def test_tls_verify_is_left_out_where_ffmpeg_lacks_it(self) -> None:
        """Debian bookworm's ffmpeg 5.1 rejects the option outright."""
        assert "-tls_verify" not in self._cmd("rtsps://cam/abc", False)

    def test_plain_rtsp_has_no_tls_options(self) -> None:
        assert "-tls_verify" not in self._cmd("rtsp://cam/abc", True)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg is not installed")
    def test_every_consumer_gets_the_same_audio(self) -> None:
        tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        wav = str(Path(tmp_dir) / "tone.wav")
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
                "sine=frequency=440:duration=1",
                "-ar",
                "16000",
                wav,
            ],
            check=True,
        )

        source = AudioSource(wav, 16000, multiprocessing.Event())
        first, second = RecordingConsumer(), RecordingConsumer()
        source.subscribe(first)
        source.subscribe(FailingConsumer())  # mustn't starve the others
        source.subscribe(second)
        source.process_stream()

        assert len(first.audio) == 16000 * 2
        assert first.audio == second.audio
        assert first.events == second.events == ["end"]

    def test_disconnect_is_signalled_before_a_retry(self) -> None:
        stop_event = multiprocessing.Event()
        source = AudioSource("/nonexistent/input.wav", 16000, stop_event)
        consumer = RecordingConsumer()
        source.subscribe(consumer)
        source.start()

        deadline = time.monotonic() + 10
        while "disconnect" not in consumer.events and time.monotonic() < deadline:
            time.sleep(0.05)

        stop_event.set()
        source.join(timeout=10)
        assert consumer.events[:2] == ["end", "disconnect"]


if __name__ == "__main__":
    unittest.main()
