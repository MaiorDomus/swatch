# Config

Setting up the config requires two main sections. Objects are used to define the different objects that swatch can detect, and cameras are used to define the common image producers that will be used.

## `objects`

```yaml
# REQUIRED: Define a list of objects that are expected to be seen. These can be specific
# to one camera or common between many / all cameras
objects:
  # REQUIRED: Name of the object
  trash_can:
    # REQUIRED: the list of color variants that this object can be detected as. Useful for
    # different lighting conditions
    color_variants:
      # REQUIRED: the name of the color variant
      default:
        # REQUIRED: the lower R, G, B values that are considered a potential match for the
        # color variant of the object.
        color_lower: 70, 70, 0
        # REQUIRED: the upper R, G, B values that are considered a potential match for the
        # color variant of the object.
        color_upper: 110, 100, 50
        # OPTIONAL: the time range for when this color variant is allowed
        # NOTE: make sure that /etc/localtime is passed to the container so it has valid time
        time_range:
          # OPTIONAL: Color variant is valid if current time is > this 24H time (Default: shown below).
          after: "00:00"
          # OPTIONAL: Color variant is valid if current time is < this 24H time (Default: shown below).
          before: "24:00"
    # OPTIONAL: the min area of the bounding box around groups of matching R, G, B pixels
    # considered a true positive. This is not recommended to be set as a super small amount
    # could be a false positive. (Default: shown below)
    min_area: 1000
    # OPTIONAL: the max area of the bounding box around groups of pixels with R, G, B
    # values within the bounds to be considered a true positive (Default: shown below).
    max_area: 100000
    # OPTIONAL: the min ratio of width/height of bounding box for valid object detection (default: shown below).
    min_ratio: 0
     # OPTIONAL: the max ratio of width/height of bounding box for valid object detection (default: shown below).
    max_ratio: 24000000
```

### `cameras`

```yaml
# REQUIRED: Define list of cameras that will be used for color detection.
cameras:
  # REQUIRED: Name of the camera
  front_doorbell_cam:
    # OPTIONAL: Frequency in seconds to run detection on the camera.
    # a value of 0 disables auto detection (Default: shown below).
    auto_detect: 0
    # OPTIONAL: Configure the url and retention of snapshots. (Default: Shown Below)
    snapshot_config:
        # OPTIONAL: but highly recommended, setting the default url for a snapshot to be
        # processed by this camera. This is required for auto detection (Default: none).
        url: "http://ip.ad.dr.ess/jpg"
        # OPTIONAL: Whether or not to draw bounding boxes for confirmed objects in the snapshots (Default: shown below).
        bounding_box: true
        # OPTIONAL: Whether or not to save a clean png of the snapshot along with the annotated jpg (Default: shown below).
        clean_snapshot: true
        # OPTIONAL: Whether or not to save the snapshots of confirmed detections (Default: shown below).
        save_detections: true
        # OPTIONAL: Whether or not to save the snapshots of missed detections (Default: shown below).
        save_misses: false
        # OPTIONAL: Variations of snapshots to keep. Options are all, mask, crop (Default: shown below).
        mode: "all"
        # OPTIONAL: Number of days of snapshots to keep (Default: shown below).
        retain_days: 1
    # REQUIRED: Zones are cropped areas where the object can be expected to be.
    # This makes searching / matches for efficient and more predictable than searching
    # the entire image.
    zones:
      # REQUIRED: Name of the zone.
      street:
        # REQUIRED: Coordinates to crop the zone by.
        # NOTE: The order of the coordinates are: x, y, x+w, y+h starting in the top left corner as 0, 0.
        coordinates: 225, 540, 350, 620
        # REQUIRED: List of objects that may be in this zone. These correspond to
        # the objects list defined previously and are matched by name.
        objects:
          - trash_can
```

### `audio_monitors`

Detects a sustained mechanical noise (e.g. a kitchen hood fan) from a camera's audio
stream, pulled via `ffmpeg` over RTSP. This is a heuristic based on the audio being both
loud enough and spectrally *steady* over a sustained period -- it is not a trained sound
classifier, and it looks for "loud, steady noise", not specifically a hood fan. A running
fan produces a fairly constant hum, whereas speech and music have much more varied
frequency content moment to moment (changing phonemes/notes), which is what lets it tell
them apart. It won't be perfect -- a sustained drone in music could still trigger it -- so
tune `threshold_db`/`max_spectral_flux` for your environment.

Tested live against a real UniFi G6 Instant with its RTSP audio alias enabled, pointed at
a kitchen hood fan, with both the fan on and off:

| Condition                       | RMS loudness (dBFS) | Spectral flux |
| -------------------------------- | -------------------- | ------------- |
| Hood on (steady running)         | -53 to -50           | 0.02-0.08     |
| Hood off (quiet room)            | -90 to -78 typical   | 0.02-0.07     |
| Hood off (brief ambient sounds)  | -90 to -70 (still quiet) | 0.3-0.46 (unsteady, correctly ignored) |

The spectral-steadiness heuristic worked well out of the box, and there's a clean ~17dB
gap between the loudest "off" moment and the quietest "on" measurement, which is what
`threshold_db` now defaults to the middle of (-60.0). Camera-off ambient noise (talking,
footsteps, cabinets) shows up as loud, brief, high-flux spikes -- correctly rejected by
the steadiness check even when momentarily as loud as the running hood. Still, a mic's
distance, sensitivity, and any automatic gain control varies enough between cameras and
kitchens that you should expect to tune this against your own setup rather than trust the
default blindly.

The resulting on/off state shows up alongside object detection results at
`GET /<name>/latest` and `GET /all/latest`, so it works with the Home Assistant
integration's existing polling with no extra setup.

```yaml
# OPTIONAL: Define audio monitors that listen to a camera's RTSP audio stream for
# sustained, steady loud noise (e.g. a kitchen hood fan running).
audio_monitors:
  # REQUIRED: Name of the audio monitor.
  kitchen_hood:
    # REQUIRED: RTSP url to pull audio from (rtsp:// or rtsps://). This needs to have
    # been enabled for the camera in your NVR (e.g. UniFi Protect's per-camera RTSP
    # alias).
    rtsp_url: "rtsps://192.168.1.1:7441/abcdefghijk"
    # OPTIONAL: Sample rate to decode audio at, in Hz (Default: shown below).
    sample_rate: 16000
    # OPTIONAL: Length of each analysis window, in seconds (Default: shown below).
    window_seconds: 1.0
    # OPTIONAL: Minimum RMS loudness, in dBFS, for a window to be considered loud
    # (0 is full digital scale, quieter is more negative) (Default: shown below).
    threshold_db: -60.0
    # OPTIONAL: Maximum spectral flux (how much the frequency shape changes between
    # windows, roughly 0-1) for a window to be considered steady/mechanical rather than
    # speech or music (Default: shown below).
    max_spectral_flux: 0.15
    # OPTIONAL: How long loud + steady audio must be sustained before switching on,
    # in seconds (Default: shown below).
    min_on_seconds: 5.0
    # OPTIONAL: How long quiet or unsteady audio must be sustained before switching
    # off, in seconds (Default: shown below).
    min_off_seconds: 10.0
```
