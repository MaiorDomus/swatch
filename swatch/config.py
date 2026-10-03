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
    min_area: int | None = Field(
        title="Overrides the object's min_area when this variant matches.",
        default=None,
    )
    max_area: int | None = Field(
        title="Overrides the object's max_area when this variant matches.",
        default=None,
    )
    min_ratio: float | None = Field(
        title="Overrides the object's min_ratio when this variant matches.",
        default=None,
    )
    max_ratio: float | None = Field(
        title="Overrides the object's max_ratio when this variant matches.",
        default=None,
    )
    min_solidity: float | None = Field(
        title="Overrides the object's min_solidity when this variant matches.",
        default=None,
    )
    max_solidity: float | None = Field(
        title="Overrides the object's max_solidity when this variant matches.",
        default=None,
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
    min_solidity: float = Field(
        title=(
            "Min solidity (matched area / convex hull area, 0-1) for valid "
            "detection. A smooth, convex shape like an oval light sits close "
            "to 1.0; an irregular/jagged shape (e.g. a diffuse reflection) is "
            "lower. Useful for telling a real fixture apart from a "
            "similarly-sized-and-shaped false positive elsewhere in the zone."
        ),
        default=0,
    )
    max_solidity: float = Field(
        title="Max solidity (matched area / convex hull area, 0-1) for valid detection.",
        default=1.1,
    )
    min_on_seconds: float = Field(
        title=(
            "How long a match must be sustained across consecutive auto_detect "
            "ticks before switching on (default: shown below, i.e. any single "
            "match). A single video frame can be too noisy for area/ratio/"
            "solidity thresholds to hold every tick even while genuinely "
            "matching (JPEG artifacts, exposure flicker), so this and "
            "min_off_seconds debounce the reported state the same way "
            "audio_monitors already do."
        ),
        default=0.0,
    )
    min_off_seconds: float = Field(
        title=(
            "How long a match must be absent across consecutive auto_detect "
            "ticks before switching off (default: shown below, i.e. any "
            "single miss)."
        ),
        default=0.0,
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
    sample_rate: int = Field(
        title="Sample rate to decode audio at, in Hz.", default=16000
    )
    window_seconds: float = Field(
        title="Length of each analysis window, in seconds.", default=1.0
    )
    threshold_db: float = Field(
        title=(
            "Minimum RMS loudness (in dBFS, where 0 is full digital scale and "
            "quieter sounds are more negative) for a window to be considered loud. "
            "A camera mic picking up an appliance from across a room tends to run "
            "quieter than you'd expect -- tune this against your own footage."
        ),
        default=-60.0,
    )
    max_spectral_flux: float = Field(
        title=(
            "Maximum spectral flux (0-1ish, how much the frequency shape changes "
            "between windows) for a window to be considered steady, mechanical "
            "noise rather than speech or music."
        ),
        default=0.15,
    )
    flux_band_cutoff_hz: float | None = Field(
        title=(
            "Only compare spectral shape at or below this frequency (Hz) when "
            "computing flux, instead of the full spectrum. A hood fan's hum "
            "concentrates in low frequencies, while speech/music -- e.g. a podcast "
            "playing near the camera -- carries much more energy above it; without "
            "this, that higher-frequency content can dominate the shape comparison "
            "and mask a real fan running underneath it. Set to None to compare the "
            "full spectrum instead."
        ),
        default=500.0,
    )
    min_band_energy_ratio: float = Field(
        title=(
            "Minimum fraction (0-1) of a window's total energy that must fall at "
            "or below flux_band_cutoff_hz before its spectral shape is trusted at "
            "all. Only applies when flux_band_cutoff_hz is set: a loud sound with "
            "almost none of its real energy down there can still leak a tiny "
            "amount in (FFT windowing sidelobes), and since the shape comparison "
            "force-normalizes whatever survives the cutoff to unit norm, that "
            "negligible leakage can otherwise look like a perfectly steady hum. "
            "Tested live against a real UniFi camera with flux_band_cutoff_hz=500: "
            "0.02 was too permissive and let a podcast playing near the camera "
            "(with the hood off) sustain enough incidental low-band energy to "
            "read as the hood running -- 0.05-0.10 correctly rejected it while "
            "still catching the hood."
        ),
        default=0.08,
    )
    min_band_level_db: float | None = Field(
        title=(
            "Minimum RMS loudness (dBFS) of just the audio at or below "
            "flux_band_cutoff_hz (or of the whole window, if no cutoff is set) "
            "for a window to count as the hum. An absolute level, unlike "
            "min_band_energy_ratio: other sound playing on top of the fan adds "
            "energy above the cutoff and drags that ratio down while the fan's "
            "own hum stays just as loud. Tested live against a real UniFi camera "
            "with flux_band_cutoff_hz=500: the hood measured -53 to -48 dBFS in "
            "that band, with or without a video playing loudly on top, while the "
            "video alone, a running tap and a kettle mostly sat around -80 to "
            "-64 dBFS, with only brief spikes above -56 (rejected anyway by the "
            "flux check and min_on_seconds). "
            "When set, min_band_energy_ratio can usually be set to 0 -- FFT "
            "leakage below the cutoff is far too quiet to pass this level. Unset "
            "(None) to skip this check."
        ),
        default=None,
    )
    min_on_seconds: float = Field(
        title="How long loud + steady audio must be sustained before switching on.",
        default=5.0,
    )
    min_off_seconds: float = Field(
        title="How long quiet or unsteady audio must be sustained before switching off.",
        default=10.0,
    )
    retain_days: int = Field(
        title="Number of days of on/off history to keep (default: shown below).",
        default=1,
    )


class ProtectConfig(SwatchBaseModel):
    """UniFi Protect connection used to play replies through a camera's
    speaker (talkback)."""

    host: str = Field(title="UniFi Protect / UniFi OS console host or IP.")
    port: int = Field(title="UniFi OS console HTTPS port.", default=443)
    username: str = Field(
        title=(
            "Local UniFi OS account. Needs permission to use the camera's "
            "talkback (Full Management on the camera)."
        )
    )
    password: str = Field(title="Password for the local UniFi OS account.")
    api_key: str | None = Field(
        title=(
            "Optional UniFi Protect integration API key. When set, talkback "
            "sessions are requested through Protect's public API; otherwise "
            "audio is sent straight to the camera's talkback port over UDP."
        ),
        default=None,
    )
    verify_ssl: bool = Field(
        title="Verify the console's TLS certificate (self-signed by default).",
        default=False,
    )
    camera: str = Field(
        title="Name or id of the Protect camera whose speaker plays the replies."
    )
    speaker_volume: int | None = Field(
        title=(
            "Set the camera's speaker volume (0-100) in UniFi Protect when "
            "connecting. Unset leaves Protect's own setting alone."
        ),
        default=None,
        ge=0,
        le=100,
    )


class VoiceSatelliteConfig(SwatchBaseModel):
    """Turns a camera's microphone (and speaker, via UniFi Protect talkback)
    into a Home Assistant Assist satellite, served over the ESPHome native
    API so Home Assistant's ESPHome integration can add it."""

    name: str | None = Field(
        title="Voice satellite name.", pattern="^[a-zA-Z0-9_-]+$", default=None
    )
    friendly_name: str | None = Field(
        title="Name shown in Home Assistant (default: the satellite name).",
        default=None,
    )
    rtsp_url: str = Field(
        title=(
            "RTSP url to pull microphone audio from. Using the same url as an "
            "audio monitor shares one stream between them."
        )
    )
    port: int = Field(
        title="TCP port for the ESPHome native API (one per satellite).",
        default=6053,
    )
    wake_words: list[str] = Field(
        title=(
            "Wake word ids active at startup (at most 2). Built in: okay_nabu, "
            "hey_jarvis, hey_mycroft, alexa; plus any microWakeWord model in "
            "wake_word_dir. Home Assistant can change these at runtime."
        ),
        default_factory=lambda: ["okay_nabu"],
        max_length=2,
    )
    wake_word_dir: str | None = Field(
        title=(
            "Directory of extra microWakeWord models (each a <id>.json config "
            "next to its .tflite file), e.g. a custom trained wake word."
        ),
        default=None,
    )
    stop_word: bool = Field(
        title=(
            'Listen for "stop" while a reply is playing, to cut it off. Off by '
            "default: a camera has no echo cancellation, so its mic hears the "
            "reply itself and the stop model fires on it, cutting replies off."
        ),
        default=False,
    )
    refractory_seconds: float = Field(
        title="Ignore further wake words for this long after one triggers.",
        default=2.0,
    )
    echo_guard_seconds: float = Field(
        title=(
            "Keep the mic closed for this long after a reply finishes playing, "
            "so the tail of the camera's own speaker audio isn't heard as the "
            "start of a follow-up question or a wake word."
        ),
        default=0.5,
    )
    protect: ProtectConfig | None = Field(
        title=(
            "UniFi Protect connection for playing replies on the camera's "
            "speaker. Without it the satellite still listens and runs "
            "commands, but replies aren't spoken."
        ),
        default=None,
    )


class SwatchConfig(SwatchBaseModel):
    """Main configuration for SwatchApp."""

    objects: dict[str, ObjectConfig] = Field(title="Object configuration.")
    cameras: dict[str, CameraConfig] = Field(title="Camera configuration.")
    audio_monitors: dict[str, AudioMonitorConfig] = Field(
        title="Audio monitors.", default_factory=dict
    )
    voice_satellites: dict[str, VoiceSatelliteConfig] = Field(
        title="Voice satellites.", default_factory=dict
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

        for name, satellite in config.voice_satellites.items():
            satellite_dict = satellite.model_dump(exclude_unset=True)
            satellite_config: VoiceSatelliteConfig = (
                VoiceSatelliteConfig.model_validate({"name": name, **satellite_dict})
            )

            config.voice_satellites[name] = satellite_config

        return config

    @classmethod
    def parse_yaml_file(cls, path: str) -> SwatchConfig:
        """Parses a raw YAML file to return config."""
        with open(path) as f:
            raw_config = f.read()

        config = yaml.safe_load(raw_config)
        return cls.model_validate(config)
