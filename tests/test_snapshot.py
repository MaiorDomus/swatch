"""Tests for swatch.snapshot cleanup / retention logic."""

import datetime
import multiprocessing
import os
import shutil
import tempfile
import unittest
from unittest import mock

from swatch.config import CameraConfig
from swatch.snapshot import SnapshotCleanup

# Fixed, mid-month "now" so tests never flake around month boundaries.
FIXED_NOW = datetime.datetime(2026, 8, 15, 12, 0, 0)


class TestSnapshotCleanup(unittest.TestCase):
    """Testing that cleanup_snapshots respects retain_days."""

    def setUp(self) -> None:
        self.media_dir = tempfile.mkdtemp()
        os.makedirs(f"{self.media_dir}/snapshots")
        self._orig_media_dir = os.environ.get("MEDIA_DIR")
        os.environ["MEDIA_DIR"] = self.media_dir

    def tearDown(self) -> None:
        if self._orig_media_dir is None:
            os.environ.pop("MEDIA_DIR", None)
        else:
            os.environ["MEDIA_DIR"] = self._orig_media_dir
        shutil.rmtree(self.media_dir, ignore_errors=True)

    def _make_snapshot_dir(self, days_ago: int, camera_name: str = "front") -> str:
        date = FIXED_NOW - datetime.timedelta(days=days_ago)
        path = f"{self.media_dir}/snapshots/{date.strftime('%m-%d')}/{camera_name}"
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/snapshot.jpg", "w") as f:
            f.write("fake")
        return path

    def _existing_dirs(self) -> set:
        return set(os.listdir(f"{self.media_dir}/snapshots"))

    @mock.patch("swatch.snapshot.datetime")
    def test_default_retain_days_keeps_only_today(self, mock_datetime) -> None:
        """retain_days=1 (the new default) should only keep today's snapshots."""
        mock_datetime.datetime.now.return_value = FIXED_NOW
        mock_datetime.timedelta = datetime.timedelta

        self._make_snapshot_dir(days_ago=0)
        self._make_snapshot_dir(days_ago=1)
        self._make_snapshot_dir(days_ago=2)
        self._make_snapshot_dir(days_ago=5)

        camera_config = CameraConfig(name="front")
        cleanup = SnapshotCleanup(config=None, stop_event=multiprocessing.Event())
        cleanup.cleanup_snapshots(camera_config)

        assert self._existing_dirs() == {"08-15"}

    @mock.patch("swatch.snapshot.datetime")
    def test_larger_retain_days_keeps_more_history(self, mock_datetime) -> None:
        mock_datetime.datetime.now.return_value = FIXED_NOW
        mock_datetime.timedelta = datetime.timedelta

        self._make_snapshot_dir(days_ago=0)
        self._make_snapshot_dir(days_ago=1)
        self._make_snapshot_dir(days_ago=2)
        self._make_snapshot_dir(days_ago=10)

        camera_config = CameraConfig(name="front", snapshot_config={"retain_days": 3})
        cleanup = SnapshotCleanup(config=None, stop_event=multiprocessing.Event())
        cleanup.cleanup_snapshots(camera_config)

        assert self._existing_dirs() == {"08-15", "08-14", "08-13"}

    def test_cleanup_snapshots_no_snapshots_dir_yet(self) -> None:
        """Regression: a fresh install with no snapshots saved yet must not
        crash cleanup_snapshots with a FileNotFoundError. This matters more
        since SnapshotCleanup now runs its first pass immediately on startup
        instead of 24h later."""
        shutil.rmtree(f"{self.media_dir}/snapshots")

        camera_config = CameraConfig(name="front")
        cleanup = SnapshotCleanup(config=None, stop_event=multiprocessing.Event())

        cleanup.cleanup_snapshots(camera_config)  # should not raise

    @mock.patch("swatch.snapshot.datetime")
    def test_cleanup_only_removes_matching_camera(self, mock_datetime) -> None:
        """A second camera's snapshots in the same dated dir must survive."""
        mock_datetime.datetime.now.return_value = FIXED_NOW
        mock_datetime.timedelta = datetime.timedelta

        self._make_snapshot_dir(days_ago=5, camera_name="front")
        self._make_snapshot_dir(days_ago=5, camera_name="back")

        camera_config = CameraConfig(name="front")
        cleanup = SnapshotCleanup(config=None, stop_event=multiprocessing.Event())
        cleanup.cleanup_snapshots(camera_config)

        remaining = os.listdir(f"{self.media_dir}/snapshots/08-10")
        assert remaining == ["back"]


if __name__ == "__main__":
    unittest.main()
