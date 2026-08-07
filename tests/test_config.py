"""Tests for SwatchConfig"""

import unittest

from pydantic import ValidationError

from swatch.config import CameraConfig, SnapshotConfig, SwatchConfig


class TestConfig(unittest.TestCase):
    """Testing the configuration is parsed correctly."""

    def setUp(self) -> None:
        """setup simple"""
        self.minimal = {
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
                    "snapshot_config": {
                        "url": "http://localhost/snap.jpg",
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
        self.full = {
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

    def test_minimal_config_class(self) -> None:
        swatch_config = SwatchConfig(**self.minimal)
        assert self.minimal == swatch_config.model_dump(exclude_unset=True)

    def test_full_config_class(self) -> None:
        swatch_config = SwatchConfig(**self.full)
        assert self.full == swatch_config.model_dump(exclude_unset=True)

    def test_extra_fields_are_forbidden(self) -> None:
        """SwatchBaseModel sets extra="forbid", unknown keys should raise."""
        invalid = {**self.minimal, "not_a_real_field": True}
        with self.assertRaises(ValidationError):
            SwatchConfig(**invalid)

    def test_snapshot_config_default_retain_days_is_one(self) -> None:
        """Regression test: default retention should be 1 day, not 7."""
        assert SnapshotConfig().retain_days == 1

    def test_camera_name_defaults_to_none(self) -> None:
        """A bare CameraConfig has no name until runtime_config assigns one."""
        camera = CameraConfig()
        assert camera.name is None

    def test_runtime_config_assigns_camera_name_from_key(self) -> None:
        """runtime_config should stamp each camera with its dict key as name."""
        swatch_config = SwatchConfig(**self.minimal).runtime_config
        assert swatch_config.cameras["test_cam"].name == "test_cam"

    def test_runtime_config_preserves_camera_settings(self) -> None:
        """runtime_config shouldn't lose any explicitly set camera fields."""
        swatch_config = SwatchConfig(**self.full).runtime_config
        camera = swatch_config.cameras["test_cam"]
        assert camera.auto_detect == 300
        assert camera.snapshot_config.retain_days == 100
        assert camera.snapshot_config.mode == "mask"

    def test_runtime_config_does_not_mutate_original(self) -> None:
        """runtime_config should be a deep copy, not mutate the source config."""
        swatch_config = SwatchConfig(**self.minimal)
        swatch_config.runtime_config
        assert swatch_config.cameras["test_cam"].name is None
