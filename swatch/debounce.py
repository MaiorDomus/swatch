"""Debouncing noisy per-tick booleans into a stable on/off state.

Any detector that samples the world periodically (an audio loudness/flux
check, a single-frame color/shape match) can have its per-tick boolean flip
on transient noise even while the underlying real-world state hasn't
changed. SustainedStateTracker requires the *opposite* classification to be
seen for a configured number of consecutive ticks before it'll report a
state flip, so a stray miss or false positive doesn't flicker the reported
state.
"""


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
