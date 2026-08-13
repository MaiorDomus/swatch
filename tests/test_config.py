"""Tests for SwatchConfig"""

import unittest

from pydantic import ValidationError

from swatch.config import (
    AudioMonitorConfig,
    CameraConfig,
    ColorVariantConfig,
    SnapshotConfig,
    SwatchConfig,
)


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

    def test_audio_monitors_default_to_empty(self) -> None:
        """Existing configs without audio_monitors should still parse fine."""
        swatch_config = SwatchConfig(**self.minimal)
        assert swatch_config.audio_monitors == {}

    def test_audio_monitor_requires_rtsp_url(self) -> None:
        with self.assertRaises(ValidationError):
            AudioMonitorConfig()

    def test_runtime_config_assigns_audio_monitor_name_from_key(self) -> None:
        config_dict = {
            **self.minimal,
            "audio_monitors": {
                "kitchen_hood": {"rtsp_url": "rtsps://192.168.1.1:7441/abc"},
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        assert swatch_config.audio_monitors["kitchen_hood"].name == "kitchen_hood"

    def test_runtime_config_preserves_audio_monitor_settings(self) -> None:
        config_dict = {
            **self.minimal,
            "audio_monitors": {
                "kitchen_hood": {
                    "rtsp_url": "rtsps://192.168.1.1:7441/abc",
                    "threshold_db": -20.0,
                    "min_on_seconds": 3.0,
                },
            },
        }
        swatch_config = SwatchConfig(**config_dict).runtime_config
        monitor = swatch_config.audio_monitors["kitchen_hood"]
        assert monitor.rtsp_url == "rtsps://192.168.1.1:7441/abc"
        assert monitor.threshold_db == -20.0
        assert monitor.min_on_seconds == 3.0


class TestColorVariantGeometryOverrides(unittest.TestCase):
    """A color_variant can optionally override its object's geometry
    thresholds (e.g. a night variant needing a larger max_area than the day
    variant for the same physical light, since camera exposure/gain changes
    how large it blooms in-frame)."""

    def test_overrides_default_to_none(self) -> None:
        """By default a variant has no geometry overrides -- the object's
        thresholds apply unchanged."""
        variant = ColorVariantConfig(color_lower="1, 1, 1", color_upper="2, 2, 2")
        assert variant.min_area is None
        assert variant.max_area is None
        assert variant.min_ratio is None
        assert variant.max_ratio is None
        assert variant.min_solidity is None
        assert variant.max_solidity is None

    def test_explicit_overrides_are_preserved(self) -> None:
        variant = ColorVariantConfig(
            color_lower="1, 1, 1",
            color_upper="2, 2, 2",
            min_area=45,
            max_area=3000,
            min_ratio=0.5,
            max_ratio=5.0,
            min_solidity=0.6,
            max_solidity=1.1,
        )
        assert variant.min_area == 45
        assert variant.max_area == 3000
        assert variant.min_ratio == 0.5
        assert variant.max_ratio == 5.0
        assert variant.min_solidity == 0.6
        assert variant.max_solidity == 1.1
