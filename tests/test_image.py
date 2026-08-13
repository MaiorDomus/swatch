"""Tests for SwatchImage"""

import datetime
import unittest
import unittest.mock

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
        """If an object has no color variants at all, the loop body never
        runs -- must still return "result": False rather than a bare {}."""
        config_dict = {
            **self.config_dict,
            "objects": {
                "test_obj": {
                    **self.config_dict["objects"]["test_obj"],
                    "color_variants": {},
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

    def test_wraparound_time_range_valid_late_at_night(self) -> None:
        """after > before (e.g. 22:00-06:00) should be treated as a window
        spanning midnight, valid outside [before, after) rather than never
        valid."""
        config_dict = {
            **self.config_dict,
            "objects": {
                "test_obj": {
                    **self.config_dict["objects"]["test_obj"],
                    "color_variants": {
                        "night": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                            "time_range": {"after": "22:00", "before": "06:00"},
                        },
                    },
                },
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        crop = np.zeros((20, 20, 3), dtype="uint8")
        crop[2:8, 2:8] = (255, 255, 255)  # 6x6 white block, area 36 >= min_area 10

        with unittest.mock.patch("swatch.image.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 1, 1, 23, 30
            )
            result = image_processor.__check_image__(
                crop, "test_cam", "test_file", swatch_config.objects["test_obj"]
            )

        assert result["result"] is True
        assert result["variant"] == "night"

    def test_wraparound_time_range_invalid_during_day(self) -> None:
        """The same overnight window must still gate out a daytime tick."""
        config_dict = {
            **self.config_dict,
            "objects": {
                "test_obj": {
                    **self.config_dict["objects"]["test_obj"],
                    "color_variants": {
                        "night": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                            "time_range": {"after": "22:00", "before": "06:00"},
                        },
                    },
                },
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        crop = np.zeros((20, 20, 3), dtype="uint8")
        crop[2:8, 2:8] = (255, 255, 255)

        with unittest.mock.patch("swatch.image.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(
                2024, 1, 1, 12, 0
            )
            result = image_processor.__check_image__(
                crop, "test_cam", "test_file", swatch_config.objects["test_obj"]
            )

        assert result["result"] is False


class TestVariantGeometryOverride(unittest.TestCase):
    """A color_variant's own min_area/max_area/etc, when set, should override
    the object's default for that variant's match -- e.g. a night variant
    needing a larger max_area than the day variant for the same light."""

    def setUp(self) -> None:
        self.config_dict = {
            "objects": {
                "test_obj": {
                    "color_variants": {
                        "default": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                            "max_area": 20,
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

    def test_variant_max_area_override_rejects_match_object_default_would_allow(
        self,
    ) -> None:
        swatch_config = SwatchConfig(**self.config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        obj_config = swatch_config.objects["test_obj"]

        # 6x6 white block, area 36: within the object's max_area (1000) but
        # outside the variant's override (20).
        crop = np.zeros((20, 20, 3), dtype="uint8")
        crop[2:8, 2:8] = (255, 255, 255)

        result = image_processor.__check_image__(
            crop, "test_cam", "test_file", obj_config
        )

        assert result["result"] is False

    def test_no_variant_override_falls_back_to_object_default(self) -> None:
        config_dict = {
            **self.config_dict,
            "objects": {
                "test_obj": {
                    **self.config_dict["objects"]["test_obj"],
                    "color_variants": {
                        "default": {
                            "color_lower": "250, 250, 250",
                            "color_upper": "255, 255, 255",
                        },
                    },
                },
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        snapshot_processor = SnapshotProcessor(swatch_config)
        image_processor = ImageProcessor(swatch_config, snapshot_processor)
        obj_config = swatch_config.objects["test_obj"]

        crop = np.zeros((20, 20, 3), dtype="uint8")
        crop[2:8, 2:8] = (255, 255, 255)  # area 36, within object's max_area 1000

        result = image_processor.__check_image__(
            crop, "test_cam", "test_file", obj_config
        )

        assert result["result"] is True
