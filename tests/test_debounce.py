"""Tests for swatch.debounce"""

import unittest

from swatch.debounce import TimeBasedSustainedStateTracker


class TestTimeBasedSustainedStateTracker(unittest.TestCase):
    """Testing TimeBasedSustainedStateTracker."""

    def test_starts_off(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=1.0, min_off_seconds=1.0
        )
        assert tracker.is_on is False

    def test_single_candidate_reading_does_not_flip_on_with_a_real_delay(
        self,
    ) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=3.0, min_off_seconds=3.0
        )
        assert tracker.update(True, now=0.0) is False

    def test_sustained_candidate_flips_on_once_min_on_seconds_elapses(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=3.0, min_off_seconds=3.0
        )
        assert tracker.update(True, now=0.0) is False
        assert tracker.update(True, now=1.0) is False
        assert tracker.update(True, now=3.1) is True

    def test_sustained_quiet_flips_back_off(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=0.0, min_off_seconds=3.0
        )
        assert tracker.update(True, now=0.0) is True
        # off-streak starts here, at t=1.0 -- 3.0s of *that* must elapse
        assert tracker.update(False, now=1.0) is True
        assert tracker.update(False, now=3.9) is True  # only 2.9s into the streak
        assert tracker.update(False, now=4.1) is False  # 3.1s into the streak

    def test_a_gap_resets_the_sustained_timer(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=0.0, min_off_seconds=3.0
        )
        assert tracker.update(True, now=0.0) is True
        assert tracker.update(False, now=1.0) is True
        # a hit before min_off_seconds elapses should reset the off-streak
        assert tracker.update(True, now=1.5) is True
        assert tracker.update(False, now=4.0) is True  # only 2.5s into the new streak

    def test_slow_ticks_do_not_stretch_the_configured_duration(self) -> None:
        """The whole reason this exists instead of the tick-counted
        SustainedStateTracker: ticks that arrive slower than assumed
        shouldn't make min_off_seconds take any longer in real time."""
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=0.0, min_off_seconds=5.0
        )
        assert tracker.update(True, now=0.0) is True

        # ticks arriving every 1.3s -- the off-streak starts at the first
        # miss (t=1.3), so it flips 5.0s after *that* (t=6.3), not 5.0s
        # after the tracker started and not after "5 ticks" (~t=7.8)
        assert tracker.update(False, now=1.3) is True
        assert tracker.update(False, now=2.6) is True
        assert tracker.update(False, now=3.9) is True
        assert tracker.update(False, now=5.2) is True  # only 3.9s into the streak
        assert tracker.update(False, now=6.5) is False  # 5.2s into the streak

    def test_on_and_off_durations_are_independent(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=0.5, min_off_seconds=5.0
        )
        assert tracker.update(True, now=0.0) is False
        assert tracker.update(True, now=0.6) is True

    def test_defaults_to_real_clock_when_now_omitted(self) -> None:
        tracker = TimeBasedSustainedStateTracker(
            min_on_seconds=0.0, min_off_seconds=0.0
        )
        # just confirms it doesn't crash without an explicit `now`
        assert tracker.update(True) is True


if __name__ == "__main__":
    unittest.main()
