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
        # NOTE: if `after` is later than `before` (e.g. after: "22:00", before: "06:00"),
        # the window is treated as spanning midnight -- valid outside [before, after)
        # rather than inside [after, before].
        time_range:
          # OPTIONAL: Color variant is valid if current time is > this 24H time (Default: shown below).
          after: "00:00"
          # OPTIONAL: Color variant is valid if current time is < this 24H time (Default: shown below).
          before: "24:00"
        # OPTIONAL: any of min_area, max_area, min_ratio, max_ratio, min_solidity,
        # max_solidity may also be set here to override the object's default for this
        # variant specifically (default: unset, i.e. use the object's value). Useful
        # when the same physical object looks a different size/shape by variant --
        # e.g. a light that blooms larger in-frame at night than during the day due
        # to camera exposure/gain, needing a bigger max_area for a night-time variant
        # than for a daytime one.
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
    # OPTIONAL: the min solidity (matched area / convex hull area, 0-1) for valid object
    # detection (default: shown below). A smooth, filled shape like an oval light sits
    # close to 1.0; an irregular/jagged shape (e.g. a diffuse reflection) is lower. Useful
    # for telling a real fixture apart from a similarly-sized-and-shaped false positive
    # elsewhere in the zone that area/ratio alone can't distinguish.
    min_solidity: 0
    # OPTIONAL: the max solidity (matched area / convex hull area, 0-1) for valid object
    # detection (default: shown below).
    max_solidity: 1.1
    # OPTIONAL: how long (in seconds) a match must be sustained across consecutive
    # auto_detect ticks before switching on (default: shown below, i.e. any single
    # match). A small object can occupy few enough pixels that JPEG artifacts or
    # exposure flicker cause area/ratio/solidity to fail on some frames even while
    # genuinely matching -- raise this (and min_off_seconds) if you see the reported
    # state flicker despite a real, unchanging object. Debounces the same way
    # audio_monitors' min_on_seconds/min_off_seconds do.
    min_on_seconds: 0
    # OPTIONAL: how long (in seconds) a match must be absent across consecutive
    # auto_detect ticks before switching off (default: shown below, i.e. any single
    # miss).
    min_off_seconds: 0
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

The shape comparison only looks at frequencies at or below `flux_band_cutoff_hz`
(default 500Hz). A hood fan's hum concentrates down there, while speech/music carries
much more energy above it -- e.g. a podcast playing near the camera while the hood is
running. Without this restriction, that higher-frequency content can dominate the
shape comparison each window and mask the fan running underneath it. Set it to `null`
to compare the full spectrum instead.

`min_band_energy_ratio` (default 0.08) guards against a subtler failure of that same
cutoff: a loud sound with almost none of its real energy below the cutoff can still
leak a tiny amount in there (FFT windowing sidelobes), and since the shape comparison
force-normalizes whatever survives the cutoff to unit norm, that negligible leakage can
otherwise look like a perfectly steady hum. This requires a real fraction of a window's
total energy to actually sit below the cutoff before its shape is trusted at all --
only relevant when `flux_band_cutoff_hz` is set. Tested live with `flux_band_cutoff_hz`
at its default: 0.02 was too permissive and let a podcast playing near the camera, with
the hood off, sustain enough incidental low-band energy to read as the hood running;
0.05-0.10 correctly rejected it while still catching the hood.

`min_band_level_db` (default: unset) is an alternative to that ratio: an absolute
minimum loudness, in dBFS, of just the audio at or below `flux_band_cutoff_hz`. The
ratio has a blind spot in both directions, found by recording a real hood alongside
common kitchen sounds:

- Something loud playing *on top of* the running fan (a video, a podcast) adds energy
  above the cutoff and drags the fan's share of the total down -- with a video 6-12 dB
  louder than normal, the hood's share fell from 0.60-0.77 to as low as 0.03, so no
  ratio threshold both kept catching the hood and rejected other sounds.
- A coffee grinder, or the same video alone, can briefly carry a 0.1-0.36 share,
  enough to pass a permissive ratio.

The fan's own low-band level doesn't care what's playing above it: the same hood
measured -53 to -48 dBFS below 500Hz whether or not the video was playing, while the
video alone, a running tap and a kettle mostly sat around -80 to -64 dBFS. A threshold
of about -56 separated them cleanly. When you set `min_band_level_db`, you can set
`min_band_energy_ratio` to 0: FFT leakage below the cutoff is far too quiet to pass an
absolute level anyway.

Short bursts of genuinely steady, bass-heavy noise -- the same grinder measured 4-5
seconds at -45 dBFS with very low flux -- look exactly like the hood for their
duration. What gives them away is how short they are, so set `min_on_seconds` longer
than those bursts (8-10 seconds worked for that grinder).

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
    # OPTIONAL: Only compare spectral shape at or below this frequency (Hz) when
    # computing flux, instead of the full spectrum -- keeps higher-frequency content
    # (e.g. a podcast playing near the camera) from masking the fan's low-frequency
    # hum. Set to null to compare the full spectrum instead (Default: shown below).
    flux_band_cutoff_hz: 500.0
    # OPTIONAL: Minimum fraction (0-1) of a window's total energy that must fall at or
    # below flux_band_cutoff_hz before its spectral shape is trusted -- guards against
    # a loud sound whose FFT windowing leakage below the cutoff can otherwise look like
    # a steady hum once normalized. Only applies when flux_band_cutoff_hz is set
    # (Default: shown below).
    min_band_energy_ratio: 0.08
    # OPTIONAL: Minimum RMS loudness, in dBFS, of just the audio at or below
    # flux_band_cutoff_hz (the whole window if no cutoff is set) for a window to count
    # as the hum -- unlike min_band_energy_ratio, not dragged down by other sound
    # playing on top of the fan. When set, min_band_energy_ratio can be set to 0
    # (Default: unset, no check).
    # min_band_level_db: -56.0
    # OPTIONAL: How long loud + steady audio must be sustained before switching on,
    # in seconds (Default: shown below).
    min_on_seconds: 5.0
    # OPTIONAL: How long quiet or unsteady audio must be sustained before switching
    # off, in seconds (Default: shown below).
    min_off_seconds: 10.0
    # OPTIONAL: Number of days of on/off history to keep, shown on the dashboard's
    # activity table and via /api/detections (Default: shown below).
    retain_days: 1
```

### `voice_satellites`

Turns a camera's microphone into a Home Assistant [Assist](https://www.home-assistant.io/voice_control/)
satellite, with replies played through the camera's own speaker via UniFi Protect
talkback. Swatch serves each satellite over the ESPHome native API, the same protocol
ESPHome voice devices (and the Open Home Foundation's Linux Voice Assistant) use, so
Home Assistant adds it through its regular **ESPHome** integration: Settings → Devices &
services → Add integration → ESPHome → host = the machine running swatch, port = the
satellite's `port`. Home Assistant then creates an `assist_satellite` entity for it,
and the device page lets you pick the Assist pipeline (language, speech-to-text,
conversation agent, text-to-speech) and the wake words.

How it works:

- The wake word is detected locally in swatch, on the camera's RTSP audio, using
  [microWakeWord](https://github.com/kahrendt/microWakeWord) models. Built in:
  `okay_nabu` (also trained on Dutch, French, German, Italian, Spanish and Swedish
  speakers), `hey_jarvis`, `hey_mycroft`, `alexa`. Up to two can be active at once.
- After the wake word, swatch streams the audio to Home Assistant, which runs the
  pipeline. Your commands can be in any language your pipeline supports -- the wake
  word only starts the listening.
- The reply (and any `assist_satellite.announce` / `start_conversation`) is played on
  the camera's speaker. Wake words are ignored while a pipeline runs or a reply plays,
  so the camera can't wake itself up.
- A satellite on the same `rtsp_url` as an audio monitor shares its stream, so the
  camera only serves one RTSP connection for both.

Without a `protect` block the satellite still listens and runs commands, it just
doesn't speak the replies (or play the listening tones).

A camera turns its mic down for about a second after it has played something (measured
on a G6 Instant: roughly 10 dB), and the start-of-listening tone plays right as you
start talking. If commands are recognised worse with it, turn `wake_sound` off.

The connection is plaintext (no ESPHome API encryption), like Linux Voice Assistant's;
keep the port on your LAN. Expose the port from the container (the add-on publishes
6053 by default).

```yaml
# OPTIONAL: Define voice satellites that turn a camera's microphone (and speaker) into
# a Home Assistant Assist satellite.
voice_satellites:
  # REQUIRED: Name of the voice satellite.
  living_room:
    # OPTIONAL: Name shown in Home Assistant (Default: the satellite name, title-cased).
    friendly_name: "Living Room Camera"
    # REQUIRED: RTSP url to pull microphone audio from. Using the same url as an audio
    # monitor shares one stream between them.
    rtsp_url: "rtsps://192.168.1.1:7441/abcdefghijk"
    # OPTIONAL: TCP port Home Assistant connects to (ESPHome native API). Each
    # satellite needs its own (Default: shown below).
    port: 6053
    # OPTIONAL: Wake words active at startup, at most 2. Home Assistant can change them
    # at runtime from the device page; they go back to these on a swatch restart
    # (Default: shown below).
    wake_words:
      - okay_nabu
    # OPTIONAL: Probability (0-1) a wake word must reach to trigger. Lower is more
    # sensitive but more prone to false triggers. A camera across the room hears you
    # quieter and with more echo than a satellite on a table: on a G6 Instant a clear
    # "Okay Nabu" from the far side of the room scored 0.75, while ordinary room noise,
    # TV and speech never went above 0.07 (Default: each model's own, 0.85 for
    # okay_nabu).
    # wake_word_threshold: 0.7
    # OPTIONAL: Directory of extra microWakeWord models, each a <id>.json config next to
    # its .tflite file (the format ESPHome and Linux Voice Assistant use), e.g. a
    # custom-trained wake word. They're offered to Home Assistant alongside the built-in
    # ones (Default: none).
    # wake_word_dir: /config/wake_words
    # OPTIONAL: Listen for "stop" while a reply is playing, to cut it off. Off by
    # default: a camera has no echo cancellation, so its mic hears the reply itself and
    # the "stop" model fires on it -- tested on a G6 Instant, every reply got cut off
    # about 1.5 seconds in. Only worth trying with the speaker volume turned well down
    # (Default: shown below).
    stop_word: false
    # OPTIONAL: Play a short, soft rising tone on the camera speaker when the satellite
    # starts listening, after the wake word. The mic keeps listening while it plays, so
    # you can talk straight away. Needs protect (Default: shown below).
    wake_sound: true
    # OPTIONAL: Play a short, soft falling tone when the satellite stops listening,
    # before the reply. Needs protect (Default: shown below).
    done_sound: true
    # OPTIONAL: Ignore further wake words for this many seconds after one triggers
    # (Default: shown below).
    refractory_seconds: 2.0
    # OPTIONAL: Keep the mic closed for this many seconds after a reply finishes, so the
    # tail of the speaker's own audio isn't heard as a follow-up question
    # (Default: shown below).
    echo_guard_seconds: 0.5
    # OPTIONAL: UniFi Protect connection for playing replies on the camera's speaker.
    protect:
      # REQUIRED: UniFi OS console host or IP.
      host: 192.168.1.1
      # OPTIONAL: Console HTTPS port (Default: shown below).
      port: 443
      # REQUIRED: A local UniFi OS account allowed to use the camera's talkback.
      username: swatch
      password: "your-password"
      # OPTIONAL: Send replies through a talkback session requested from Protect's
      # public API (needs api_key) instead of straight to the camera's talkback port
      # over UDP. Off by default: on a G6 Instant those sessions left the speaker
      # amplifier humming after every reply, until the camera's audio settings were
      # changed (Default: shown below).
      talkback_via_api: false
      # OPTIONAL: UniFi Protect integration API key, only for talkback_via_api
      # (Default: none).
      # api_key: ...
      # OPTIONAL: Verify the console's TLS certificate (Default: shown below).
      verify_ssl: false
      # REQUIRED: Name or id of the Protect camera whose speaker plays the replies.
      camera: "G6 Instant"
      # OPTIONAL: Set the camera's speaker volume (0-100) in UniFi Protect when
      # swatch connects. Protect defaults cameras to 100, which is loud for replies
      # (Default: unset, leaves Protect's setting alone).
      speaker_volume: 60
```
