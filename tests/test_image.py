"""Tests for SwatchImage"""

import unittest

import numpy as np

from swatch.config import SwatchConfig
from swatch.image import ImageProcessor
from swatch.snapshot import SnapshotProcessor


class TestImage(unittest.TestCase):
    """Testing the configuration is parsed correctly."""

    def setUp(self) -> None:
        """setup simple"""
        self.config = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "1, 1, 1",
                            "color_upper": "2, 2, 2",
                            "time_range": {
                                "after": "08:00",
                                "before": "18:00",
                            },
                        },
                    },
                    "min_area": 0,
                    "max_area": 100000,
                },
            },
            "cameras": {
                "test_cam": {
                    "auto_detect": 300,
                    "snapshot_config": {
                        "url": "http://localhost/snap.jpg",
                        "save_detections": True,
                        "save_misses": True,
                        "mode": "mask",
                        "retain_days": 100,
                    },
                    "zones": {
                        "test_zone": {
                            "coordinates": "1, 2, 3, 4",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
        }

    def test_valid_time_range(self) -> None:
        # Mirrors the skip condition in ImageProcessor.__check_image__: a
        # variant is valid (not skipped) when after <= now_time <= before.
        swatch_config = SwatchConfig(**self.config)
        now_time = "12:00"
        color_variant = swatch_config.objects["test_obj"].color_variants["default"]
        assert not (
            now_time < color_variant.time_range.after
            or now_time > color_variant.time_range.before
        )

    def test_invalid_time_range(self) -> None:
        swatch_config = SwatchConfig(**self.config)
        now_time = "04:00"
        color_variant = swatch_config.objects["test_obj"].color_variants["default"]
        assert (
            now_time < color_variant.time_range.after
            or now_time > color_variant.time_range.before
        )


class TestCheckImage(unittest.TestCase):
    """Testing ImageProcessor.__check_image__ returns a well-formed result
    even on a miss, not the bare {} it used to return when a variant
    matched zero pixels."""

    def setUp(self) -> None:
        self.config_dict = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                        },
                    },
                    "min_area": 10,
                    "max_area": 1000,
                },
            },
            "cameras": {
                "test_cam": {
                    "snapshot_config": {
                        "url": "http://localhost/snap.jpg",
                        "save_detections": False,
                        "save_misses": False,
                    },
                    "zones": {
                        "test_zone": {
                            "coordinates": "0, 0, 20, 20",
                            "objects": ["test_obj"],
                        },
                    },
                },
            },
        }
        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        self.image_processor = ImageProcessor(swatch_config, snapshot_processor)
        self.obj_config = swatch_config.objects["test_obj"]

    def test_no_color_match_returns_well_formed_miss(self) -> None:
        """A crop with zero matching pixels (the common "definitely off"
        case) must not return a bare {} -- that's indistinguishable from
        "no data" downstream, in the HTTP API and then the HA integration."""
        crop = np.zeros((20, 20, 3), dtype="uint8")  # solid black, no white

        result = self.image_processor.__check_image__(
            crop, "test_cam", "test_file", self.obj_config
        )

        assert result == {
            "result": False,
            "area": 0,
            "variant": "default",
            "camera_name": "test_cam",
        }

    def test_color_match_above_min_area_returns_result_true(self) -> None:
        crop = np.zeros((20, 20, 3), dtype="uint8")
        crop[2:8, 2:8] = (255, 255, 255)  # 6x6 white block, area 36 >= min_area 10

        result = self.image_processor.__check_image__(
            crop, "test_cam", "test_file", self.obj_config
        )

        assert result["result"] is True
        assert result["variant"] == "default"

    def test_all_variants_time_gated_out_still_returns_well_formed(self) -> None:
        """If every color variant's time_range excludes right now, the loop
        body never runs at all -- must still return "result": False rather
        than a bare {}."""
        config_dict = {
            **self.config_dict,
            "objects": {
                "test_obj": {
                    **self.config_dict["objects"]["test_obj"],
                    "color_variants": {
                        "default": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                            # after > before makes this window never valid
                            "time_range": {"after": "23:59", "before": "00:00"},
                        },
                    },
                },
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        crop = np.zeros((20, 20, 3), dtype="uint8")

        result = image_processor.__check_image__(
            crop, "test_cam", "test_file", swatch_config.objects["test_obj"]
        )

        assert result["result"] is False
        assert result != {}
