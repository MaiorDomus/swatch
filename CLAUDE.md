# Swatch workspace

This repo is one of three (checked out as siblings under a common parent) that
together make up the "Swatch" project: a service that detects known objects by
color in camera images (and sustained mechanical sounds, e.g. a kitchen hood fan,
from camera audio), plus its Home Assistant integration and add-on packaging.
They're versioned and released independently but are meant to be used together.

## `swatch/` (this repo) — the core service

The actual detection engine and HTTP API. Everyone else in this workspace is a thin
client of this service.

- **Object detection**: watches configured camera snapshot URLs, crops out
  configured `zones`, and color-matches configured `objects` against R/G/B bounds
  (`color_variants`) to decide if something is present/on. Geometry filters
  (`min_area`/`max_area`/`min_ratio`/`max_ratio`/`min_solidity`/`max_solidity`) reject
  matches that are the wrong size or shape. `color_variants` can be time-gated
  (`time_range`) for different lighting conditions across the day, and can now also
  override the object's geometry thresholds per-variant (e.g. a light that blooms
  larger in-frame at night than during the day needs a bigger `max_area` for its
  night variant).
- **Audio monitors**: pulls RTSP audio from a camera and flags sustained,
  spectrally-steady loud noise (e.g. a running range hood fan), distinguishing it
  from speech/music by how much the frequency shape changes moment to moment.
- Exposes both via a REST API (`docs/api.md`) — `/api/<label>/latest`,
  `/api/detections` (on/off history), snapshot endpoints, and a `/api/colortest/*`
  pair for tuning color bounds against a real image.
- Has its own web UI (Flutter app in `web/`) including a "Color Playground" for
  tuning `color_lower`/`color_upper` against real snapshots.
- Config lives in a single `config.yaml`/`swatch.yml` (`docs/config.md` documents
  every field); schema is defined with pydantic in `swatch/config.py`.

Key source files:
- `swatch/config.py` — pydantic config schema (`SwatchConfig`, `ObjectConfig`,
  `ColorVariantConfig`, `CameraConfig`, `AudioMonitorConfig`, ...).
- `swatch/image.py` — `ImageProcessor`: fetches a snapshot, crops zones, evaluates
  each object's color variants against `time_range` and geometry thresholds.
- `swatch/util.py` — color masking (`mask_image`), contour/bounding-box detection
  (`detect_objects`), solidity computation, and the shared hardened snapshot-fetch
  helper (`fetch_snapshot_bytes`).
- `swatch/detection.py` — `AutoDetector` (per-camera polling thread that debounces
  raw per-frame results into sustained on/off state and records it to the DB) and
  `DetectionCleanup`.
- `swatch/audio.py` — the audio monitor equivalent of the above for RTSP streams.
- `swatch/snapshot.py` — saving/serving snapshot images (clean, masked, bounding-box
  annotated) to disk and via the API.
- `swatch/http.py` — Flask routes (see `docs/api.md`).
- `migrations/` — `peewee_migrate` DB migrations for the detection-history table.

Run via `docker-compose.yml` / `docker/Dockerfile`, or `make local` (builds the
Flutter web UI, then the Docker image).

## `swatch-hass-integration/` — Home Assistant custom integration (sibling repo)

A `custom_components/swatch` HACS integration that polls a running `swatch/`
instance's HTTP API and exposes results as native Home Assistant entities:

- Binary sensor entities for each detected object/zone.
- Binary sensor entities for each audio monitor.
- Supports multiple Swatch instances configured through the HA UI (`config_flow.py`).

This is how a Swatch detection (e.g. "kitchen hood light is on") actually becomes
something you can use in HA automations/dashboards. It talks to `swatch/` purely
over HTTP (`custom_components/swatch/api.py`) — no shared code with the core service.

## `swatch-hass-addon/` — Home Assistant OS add-on (sibling repo)

Packages `swatch/` itself as a Home Assistant OS/Supervisor add-on, so it can run
directly on a Home Assistant box instead of needing separately-managed Docker/NAS
hosting. Two variants: `swatch/` (stable) and `swatch-beta/` (beta channel), each
with its own `config.yaml` (Supervisor add-on manifest — ports, env vars like
`CONFIG_FILE`/`MEDIA_DIR`/`DB_FILE`, HA `/config` mount) and `Dockerfile`.

Typical setup: install this add-on (or run `swatch/` via your own Docker), point
`swatch-hass-integration` at its URL, then define `objects`/`cameras`/
`audio_monitors` in `swatch`'s own config to start getting entities in HA.

**Important**: this add-on's `Dockerfile` builds by `git clone`-ing this (`swatch/`)
repo at build time (`ARG SWATCH_REPO`/`SWATCH_REF`, defaulting to `main`) rather than
vendoring the source — see the comment above the `RUN git clone` in
`swatch/Dockerfile` and `swatch-beta/Dockerfile` in that repo. Home Assistant
Supervisor only busts that clone's Docker layer cache when `config.yaml`'s `version`
field changes; otherwise a "Rebuild" silently reuses whatever was cloned on the very
first build, forever. **So whenever this repo's backend code changes,
`swatch-hass-addon` needs a matching version bump** (in both `swatch/config.yaml`
and `swatch-beta/config.yaml`, plus a matching `CHANGELOG.md` entry in each) to
actually pick up the change — purely bumping the version, no `Dockerfile` edits
needed. See recent commits in that repo (e.g. "Bump to 3.2.17-local to pick
up...") for the exact pattern to follow.

## Inspecting a live instance

The workspace's primary live `swatch` instance runs at `http://192.168.10.10:4500`.
Its web UI has a read-only Settings page that renders the raw `config.yaml`, but for
scripted/agent access, hit the API directly instead of scraping that page:
- `GET /api/config/raw` — the config file's exact contents (comments and formatting
  included), e.g. `curl http://192.168.10.10:4500/api/config/raw`.
- `GET /api/config` — the parsed `SwatchConfig` as JSON (normalized, no comments).

## Tuning color-matched object thresholds

When a color-matched `object` seems to need much looser `min_area`/`max_area`/
`min_ratio`/`max_ratio` (e.g. "it works during the day but not at night"), don't
just widen the bounds to fit whatever blob shows up in a failing frame. A brighter
scene (more camera exposure/gain at night) can push a *nearby* surface — a wall,
tile, reflection — into the same broad color range as the real object, producing a
much bigger/differently-shaped blob that has nothing to do with the object actually
changing size. Loosening geometry to accommodate that blob just starts matching the
false positive instead.

Before changing thresholds, visually confirm what's actually matching: run the
object's `color_lower`/`color_upper` through `swatch.util.mask_image()` (or
`/api/colortest/mask`) against a real snapshot/crop and look at the resulting mask,
or diff it against `swatch.util.detect_objects()`'s per-contour `area`/`ratio`/
`solidity` output. If there are multiple separate blobs, only widen bounds by the
amount justified by the real object's own contour drift, keeping enough margin to
still exclude any other blob in the mask.
