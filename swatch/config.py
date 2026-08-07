"""Configuration for SwatchApp."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, ConfigDict, Field
import yaml


class SwatchBaseModel(BaseModel):
    """Base config that sets rules."""

    model_config = ConfigDict(extra="forbid")


class SnapshotModeEnum(str, Enum):
    """Types of snapshots to retain."""

    ALL = "all"
    CROP = "crop"
    MASK = "mask"
    NONE = "none"


class SnapshotConfig(SwatchBaseModel):
    """Configuration for saving snapshots."""

    url: str | None = Field(title="Camera Snapshot Url.", default=None)
    mode: SnapshotModeEnum = Field(title="Snapshot mode.", default=SnapshotModeEnum.ALL)
    clean_snapshot: bool = Field(title="Save clean snapshot.", default=True)
    bounding_box: bool = Field(
        title="Write bounding boxes for detected objects on the snapshot.",
        default=True,
    )
    save_detections: bool = Field(
        title="Save snapshots of detections that are found.", default=True
    )
    save_misses: bool = Field(
        title="Save snapshots of missed detections.", default=False
    )
    retain_days: int = Field(title="Number of days to retain snapshots.", default=1)


class TimeRangeConfig(SwatchBaseModel):
    """Configuration of time range for color variants."""

    after: str = Field(
        title="Color variant is valid if current time is > this 24H time.",
        default="00:00",
    )
    before: str = Field(
        title="Color variant is valid if current time is < this 24H time.",
        default="24:00",
    )


class ColorVariantConfig(SwatchBaseModel):
    """Configuration of color values."""

    color_lower: str = Field(title="Lower R, G, B color values")
    color_upper: str = Field(title="Higher R, G, B color values")
    time_range: TimeRangeConfig = Field(
        title="Valid time range for this config.", default_factory=TimeRangeConfig
    )


class ObjectConfig(SwatchBaseModel):
    """Configuration of the object detection."""

    color_variants: dict[str, ColorVariantConfig] = Field(
        title="Color variants for this object", default_factory=dict
    )
    min_area: int = Field(title="Min Area", default=0)
    max_area: int = Field(title="Max Area", default=240000)
    min_ratio: float = Field(
        title="Min ratio of width/height for valid detection.", default=0
    )
    max_ratio: float = Field(
        title="Max ratio of width/height for valid detection.", default=24000000
    )


class ZoneConfig(SwatchBaseModel):
    """Configuration for cropped parts of camera frame."""

    coordinates: str = Field(title="Coordinates polygon for the defined zone.")
    objects: list[str] = Field(title="Included Objects.")


class CameraConfig(SwatchBaseModel):
    """Configuration for camera."""

    auto_detect: int = Field(
        title="Frequency to automatically run detection.", default=0
    )
    name: str | None = Field(
        title="Camera name.", pattern="^[a-zA-Z0-9_-]+$", default=None
    )
    snapshot_config: SnapshotConfig = Field(
        title="Snapshot config for this zone.", default_factory=SnapshotConfig
    )
    zones: dict[str, ZoneConfig] = Field(
        default_factory=dict, title="Zones for this camera."
    )


class AudioMonitorConfig(SwatchBaseModel):
    """Configuration for detecting a sustained mechanical noise (e.g. a kitchen
    hood fan) from a camera's audio stream."""

    name: str | None = Field(
        title="Audio monitor name.", pattern="^[a-zA-Z0-9_-]+$", default=None
    )
    rtsp_url: str = Field(title="RTSP url to pull audio from.")
    sample_rate: int = Field(title="Sample rate to decode audio at, in Hz.", default=16000)
    window_seconds: float = Field(
        title="Length of each analysis window, in seconds.", default=1.0
    )
    threshold_db: float = Field(
        title=(
            "Minimum RMS loudness (in dBFS, where 0 is full digital scale and "
            "quieter sounds are more negative) for a window to be considered loud."
        ),
        default=-35.0,
    )
    max_spectral_flux: float = Field(
        title=(
            "Maximum spectral flux (0-1ish, how much the frequency shape changes "
            "between windows) for a window to be considered steady, mechanical "
            "noise rather than speech or music."
        ),
        default=0.15,
    )
    min_on_seconds: float = Field(
        title="How long loud + steady audio must be sustained before switching on.",
        default=5.0,
    )
    min_off_seconds: float = Field(
        title="How long quiet or unsteady audio must be sustained before switching off.",
        default=10.0,
    )


class SwatchConfig(SwatchBaseModel):
    """Main configuration for SwatchApp."""

    objects: dict[str, ObjectConfig] = Field(title="Object configuration.")
    cameras: dict[str, CameraConfig] = Field(title="Camera configuration.")
    audio_monitors: dict[str, AudioMonitorConfig] = Field(
        title="Audio monitors.", default_factory=dict
    )

    @property
    def runtime_config(self) -> SwatchConfig:
        """Merge camera config with globals."""
        config = self.model_copy(deep=True)

        for name, camera in config.cameras.items():
            camera_dict = camera.model_dump(exclude_unset=True)
            camera_config: CameraConfig = CameraConfig.model_validate(
                {"name": name, **camera_dict}
            )

            config.cameras[name] = camera_config

        for name, monitor in config.audio_monitors.items():
            monitor_dict = monitor.model_dump(exclude_unset=True)
            monitor_config: AudioMonitorConfig = AudioMonitorConfig.model_validate(
                {"name": name, **monitor_dict}
            )

            config.audio_monitors[name] = monitor_config

        return config

    @classmethod
    def parse_yaml_file(cls, path: str) -> SwatchConfig:
        """Parses a raw YAML file to return config."""
        with open(path) as f:
            raw_config = f.read()

        config = yaml.safe_load(raw_config)
        return cls.model_validate(config)
