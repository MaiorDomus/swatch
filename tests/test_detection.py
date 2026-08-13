"""Tests for AutoDetector's result debouncing and DetectionCleanup."""

import datetime
import multiprocessing
import unittest
from typing import Any

from peewee import SqliteDatabase

from swatch.config import SwatchConfig
from swatch.detection import AutoDetector, DetectionCleanup
from swatch.image import ImageProcessor
from swatch.models import Detection
from swatch.snapshot import SnapshotProcessor


class TestAutoDetectorDebounce(unittest.TestCase):
    """Testing AutoDetector.__debounce_results__.

    Real auto_detect ticks aren't reliably spaced exactly auto_detect
    seconds apart (each cycle also does an HTTP snapshot fetch), so
    debouncing is time-based, not tick-counted -- tests drive it with
    explicit `now` timestamps rather than relying on real elapsed time
    between calls.
    """

    def setUp(self) -> None:
        self.config_dict: dict[str, Any] = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "1, 1, 1",
                            "color_upper": "2, 2, 2",
                        },
                    },
                    "min_area": 0,
                    "max_area": 100000,
                },
            },
            "cameras": {
                "test_cam": {
                    "auto_detect": 1,
                    "snapshot_config": {"url": "http://localhost/snap.jpg"},
                    "zones": {
                        "test_zone": {
                            "coordinates": "1, 2, 3, 4",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
        }

    def _make_detector(self) -> AutoDetector:
        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        camera_config = swatch_config.cameras["test_cam"]
        return AutoDetector(
            image_processor,
            snapshot_processor,
            camera_config,
            multiprocessing.Event(),
        )

    def _result(self, is_on: bool) -> dict[str, Any]:
        return {
            "test_zone": {
                "test_obj": {
                    "result": is_on,
                    "area": 100 if is_on else -1,
                    "camera_name": "test_cam",
                },
            },
        }

    def test_no_debounce_config_passes_through_raw_result(self) -> None:
        # default min_on_seconds/min_off_seconds are both 0 -- any single
        # tick flips state, matching pre-debounce behavior exactly.
        detector = self._make_detector()

        result = self._result(True)
        detector.__debounce_results__(result, now=0.0)
        assert result["test_zone"]["test_obj"]["result"] is True

        result = self._result(False)
        detector.__debounce_results__(result, now=0.1)
        assert result["test_zone"]["test_obj"]["result"] is False

    def test_sustained_off_holds_state_through_a_quick_miss(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_off_seconds"] = 3
        detector = self._make_detector()

        detector.__debounce_results__(self._result(True), now=0.0)

        # only 1s elapsed since the miss started, short of min_off_seconds=3
        miss_result = self._result(False)
        detector.__debounce_results__(miss_result, now=1.0)
        assert miss_result["test_zone"]["test_obj"]["result"] is True

    def test_sustained_off_flips_once_enough_real_time_elapses(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_off_seconds"] = 3
        detector = self._make_detector()

        detector.__debounce_results__(self._result(True), now=0.0)
        detector.__debounce_results__(self._result(False), now=1.0)

        # 3.5s since the miss streak started at t=1.0 -- past min_off_seconds
        final_result = self._result(False)
        detector.__debounce_results__(final_result, now=4.5)
        assert final_result["test_zone"]["test_obj"]["result"] is False

    def test_a_recovering_hit_resets_the_off_debounce(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_off_seconds"] = 3
        detector = self._make_detector()

        detector.__debounce_results__(self._result(True), now=0.0)
        detector.__debounce_results__(self._result(False), now=1.0)
        # a real hit before min_off_seconds elapses resets the miss streak
        detector.__debounce_results__(self._result(True), now=2.0)

        # even though 3.5s have passed since the *original* miss at t=1.0,
        # the streak restarted at t=2.0 so this is well short of 3s off
        final_result = self._result(False)
        detector.__debounce_results__(final_result, now=4.5)
        assert final_result["test_zone"]["test_obj"]["result"] is True

    def test_sustained_on_requires_real_elapsed_time(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_on_seconds"] = 2
        detector = self._make_detector()

        first_result = self._result(True)
        detector.__debounce_results__(first_result, now=0.0)
        assert first_result["test_zone"]["test_obj"]["result"] is False

        # only 1s later -- still short of min_on_seconds=2
        second_result = self._result(True)
        detector.__debounce_results__(second_result, now=1.0)
        assert second_result["test_zone"]["test_obj"]["result"] is False

        # 2.5s after the candidate streak started -- past min_on_seconds
        third_result = self._result(True)
        detector.__debounce_results__(third_result, now=2.5)
        assert third_result["test_zone"]["test_obj"]["result"] is True

    def test_trackers_are_scoped_per_camera_zone_object(self) -> None:
        detector = self._make_detector()
        detector.__debounce_results__(self._result(True))
        assert "test_cam.test_zone.test_obj" in detector.trackers


class TestDetectionCleanup(unittest.TestCase):
    """Testing DetectionCleanup.__cleanup_db__.

    Regression coverage: Detection.delete().where(...) previously never
    called .execute(), so retention silently deleted nothing for object
    detections either -- there was no prior test to catch it.
    """

    def setUp(self) -> None:
        self.db = SqliteDatabase(":memory:")
        Detection.bind(self.db)
        # Matches migrations/001_init_detection_table.py; end_time is
        # nullable there, unlike Detection.create_table() from the bare
        # model (see TestAudioMonitorDetectionHistory in test_audio.py).
        self.db.execute_sql(
            'CREATE TABLE IF NOT EXISTS "detection" ('
            '"id" VARCHAR(30) NOT NULL PRIMARY KEY, "label" VARCHAR(20) NOT NULL, '
            '"camera" VARCHAR(20) NOT NULL, "zone" VARCHAR(20) NOT NULL, '
            '"color_variant" VARCHAR(20) NOT NULL, "start_time" DATETIME NOT NULL, '
            '"end_time" DATETIME, "top_area" INTEGER NOT NULL)'
        )

        self.config_dict: dict[str, Any] = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "1, 1, 1",
                            "color_upper": "2, 2, 2",
                        },
                    },
                },
            },
            "cameras": {
                "test_cam": {
                    "snapshot_config": {
                        "url": "http://localhost/snap.jpg",
                        "retain_days": 1,
                    },
                    "zones": {
                        "test_zone": {
                            "coordinates": "1, 2, 3, 4",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
            "audio_monitors": {
                "kitchen_hood": {
                    "rtsp_url": "unused",
                    "retain_days": 1,
                },
            },
        }

    def tearDown(self) -> None:
        self.db.drop_tables([Detection])
        self.db.close()

    def _create_detection(
        self, detection_id: str, label: str, camera: str, days_ago: float
    ) -> None:
        start_time = (
            datetime.datetime.now() - datetime.timedelta(days=days_ago)
        ).timestamp()
        Detection.create(
            id=detection_id,
            label=label,
            camera=camera,
            zone="",
            color_variant="default",
            start_time=start_time,
            end_time=start_time + 1,
            top_area=0,
        )

    def test_expired_camera_detections_are_deleted(self) -> None:
        self._create_detection("old", "test_obj", "test_cam", days_ago=5)
        self._create_detection("recent", "test_obj", "test_cam", days_ago=0)

        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        cleanup = DetectionCleanup(swatch_config, multiprocessing.Event())
        cleanup.__cleanup_db__()

        remaining = {d.id for d in Detection.select()}
        assert remaining == {"recent"}

    def test_expired_audio_monitor_detections_are_deleted(self) -> None:
        """audio_monitors' rows have no camera (see
        AudioMonitor.__record_transition__), so they need their own
        label-based cleanup pass rather than the camera-based one."""
        self._create_detection("old_audio", "kitchen_hood", "", days_ago=5)
        self._create_detection("recent_audio", "kitchen_hood", "", days_ago=0)

        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        cleanup = DetectionCleanup(swatch_config, multiprocessing.Event())
        cleanup.__cleanup_db__()

        remaining = {d.id for d in Detection.select()}
        assert remaining == {"recent_audio"}


class TestAutoDetectorRunLoopResilience(unittest.TestCase):
    """AutoDetector.run()'s loop must survive an unexpected exception from a
    single detection cycle (e.g. a snapshot fetch failure) rather than
    silently ending the thread -- see swatch/image.py's fetch_snapshot_bytes,
    which is what used to let a bare network exception propagate all the way
    out of this loop."""

    def setUp(self) -> None:
        self.db = SqliteDatabase(":memory:")
        Detection.bind(self.db)
        self.db.execute_sql(
            'CREATE TABLE IF NOT EXISTS "detection" ('
            '"id" VARCHAR(30) NOT NULL PRIMARY KEY, "label" VARCHAR(20) NOT NULL, '
            '"camera" VARCHAR(20) NOT NULL, "zone" VARCHAR(20) NOT NULL, '
            '"color_variant" VARCHAR(20) NOT NULL, "start_time" DATETIME NOT NULL, '
            '"end_time" DATETIME, "top_area" INTEGER NOT NULL)'
        )

        self.config_dict: dict[str, Any] = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "1, 1, 1",
                            "color_upper": "2, 2, 2",
                        },
                    },
                },
            },
            "cameras": {
                "test_cam": {
                    # 0 makes stop_event.wait(0) return immediately each
                    # iteration instead of blocking, so the test doesn't
                    # depend on real wall-clock time.
                    "auto_detect": 0,
                    "snapshot_config": {"url": "http://localhost/snap.jpg"},
                    "zones": {
                        "test_zone": {
                            "coordinates": "1, 2, 3, 4",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
        }

    def tearDown(self) -> None:
        self.db.drop_tables([Detection])
        self.db.close()

    def test_run_survives_an_exception_and_keeps_polling(self) -> None:
        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        camera_config = swatch_config.cameras["test_cam"]
        stop_event = multiprocessing.Event()
        detector = AutoDetector(
            image_processor, snapshot_processor, camera_config, stop_event
        )

        call_count = 0

        def fake_detect(camera_name: str, image_url: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                raise RuntimeError("simulated snapshot fetch failure")

            # stop after the second (successful) cycle so run() returns
            stop_event.set()
            return {}

        image_processor.detect = fake_detect  # type: ignore[method-assign]

        detector.run()

        assert call_count == 2


if __name__ == "__main__":
    unittest.main()
