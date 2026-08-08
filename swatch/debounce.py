"""Debouncing noisy per-tick booleans into a stable on/off state.

Any detector that samples the world periodically (an audio loudness/flux
check, a single-frame color/shape match) can have its per-tick boolean flip
on transient noise even while the underlying real-world state hasn't
changed. The trackers below require the *opposite* classification to be
seen for a configured duration before reporting a state flip, so a stray
miss or false positive doesn't flicker the reported state.
"""

import time


class SustainedStateTracker:
    """Debounces per-tick booleans into an on/off state.

    A state flip only takes effect once the *opposite* classification has
    been seen for min_on_seconds (to turn on) or min_off_seconds (to turn
    off) worth of consecutive ticks, so a single stray reading doesn't
    flicker the reported state.
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
        """Feed in whether the latest tick qualifies as "on"; returns the
        (possibly still-debouncing) current on/off state."""
        if is_candidate == self.is_on:
            self._pending_windows = 0
            return self.is_on

        self._pending_windows += 1
        required_windows = (
            self.min_on_windows if is_candidate else self.min_off_windows
        )

        if self._pending_windows >= required_windows:
            self.is_on = is_candidate
            self._pending_windows = 0

        return self.is_on


class TimeBasedSustainedStateTracker:
    """Debounces booleans into an on/off state based on real elapsed time,
    for callers whose tick interval isn't reliably uniform.

    SustainedStateTracker counts ticks and assumes each one takes
    window_seconds of real time -- fine for audio, whose windows come from a
    continuous stream at a fixed sample rate, but wrong for something like
    object detection's auto_detect loop, where each tick also does an HTTP
    fetch of a camera snapshot: if that fetch routinely takes longer than
    auto_detect's configured interval (as measured against a real camera,
    ~1.3s actual vs. 1s configured), min_off_seconds worth of "ticks" ends up
    being a noticeably longer real debounce than configured. This tracks
    actual elapsed time instead, so the configured seconds mean what they
    say regardless of how long each tick actually takes.
    """

    def __init__(
        self,
        min_on_seconds: float,
        min_off_seconds: float,
    ) -> None:
        self.min_on_seconds = min_on_seconds
        self.min_off_seconds = min_off_seconds
        self.is_on = False
        self._divergence_started_at: float | None = None

    def update(self, is_candidate: bool, now: float | None = None) -> bool:
        """Feed in the latest reading (and optionally the time it was taken
        at, for testing -- defaults to time.monotonic()); returns the
        (possibly still-debouncing) current on/off state."""
        if now is None:
            now = time.monotonic()

        if is_candidate == self.is_on:
            self._divergence_started_at = None
            return self.is_on

        if self._divergence_started_at is None:
            self._divergence_started_at = now

        required_seconds = (
            self.min_on_seconds if is_candidate else self.min_off_seconds
        )

        if now - self._divergence_started_at >= required_seconds:
            self.is_on = is_candidate
            self._divergence_started_at = None

        return self.is_on
