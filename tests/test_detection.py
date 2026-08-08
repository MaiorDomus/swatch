"""Tests for AutoDetector's result debouncing."""

import multiprocessing
import unittest
from typing import Any

from swatch.config import SwatchConfig
from swatch.detection import AutoDetector
from swatch.image import ImageProcessor
from swatch.snapshot import SnapshotProcessor


class TestAutoDetectorDebounce(unittest.TestCase):
    """Testing AutoDetector.__debounce_results__."""

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
        detector.__debounce_results__(result)
        assert result["test_zone"]["test_obj"]["result"] is True

        result = self._result(False)
        detector.__debounce_results__(result)
        assert result["test_zone"]["test_obj"]["result"] is False

    def test_sustained_off_holds_state_through_single_miss(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_off_seconds"] = 3
        detector = self._make_detector()

        detector.__debounce_results__(self._result(True))

        miss_result = self._result(False)
        detector.__debounce_results__(miss_result)
        assert miss_result["test_zone"]["test_obj"]["result"] is True

    def test_sustained_off_eventually_flips_after_enough_misses(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_off_seconds"] = 2
        detector = self._make_detector()

        detector.__debounce_results__(self._result(True))
        detector.__debounce_results__(self._result(False))

        final_result = self._result(False)
        detector.__debounce_results__(final_result)
        assert final_result["test_zone"]["test_obj"]["result"] is False

    def test_sustained_on_requires_consecutive_hits(self) -> None:
        self.config_dict["objects"]["test_obj"]["min_on_seconds"] = 2
        detector = self._make_detector()

        first_result = self._result(True)
        detector.__debounce_results__(first_result)
        assert first_result["test_zone"]["test_obj"]["result"] is False

        second_result = self._result(True)
        detector.__debounce_results__(second_result)
        assert second_result["test_zone"]["test_obj"]["result"] is True

    def test_trackers_are_scoped_per_camera_zone_object(self) -> None:
        detector = self._make_detector()
        detector.__debounce_results__(self._result(True))
        assert "test_cam.test_zone.test_obj" in detector.trackers


if __name__ == "__main__":
    unittest.main()
