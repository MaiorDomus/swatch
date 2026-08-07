# 2026-08 Dependency Upgrade

Upgraded the backend to the newest available libraries (validated against Python 3.14 /
Home Assistant 2026.8.0-era dependency versions).

## Dependency bumps

| Package                  | Before        | After   |
| ------------------------ | ------------- | ------- |
| Flask                    | 2.1.*         | 3.1.*   |
| numpy                    | 1.22.*        | 2.5.*   |
| opencv-python-headless   | unpinned (4.x)| 5.0.*   |
| peewee                   | 3.14.*        | 4.3.*   |
| peewee_migrate           | 1.4.*         | 1.15.*  |
| pydantic                 | 1.9.*         | 2.13.*  |
| PyYAML                   | 6.0.*         | 6.0.*   |
| requests                 | 2.27.*        | 2.34.*  |
| mypy                     | 0.991         | 2.3.0   |
| matplotlib               | 3.5.* (unused)| removed |

## Code changes required by the bumps

- **`swatch/config.py`** — migrated from pydantic v1 to v2: `Extra.forbid` →
  `model_config = ConfigDict(extra="forbid")`, `Field(regex=...)` → `Field(pattern=...)`,
  `.copy()`/`.dict()`/`.parse_obj()` → `.model_copy()`/`.model_dump()`/`.model_validate()`.
  Also had to add `default=None` to `CameraConfig.name` — pydantic v2 no longer treats an
  `Optional[str]` field as implicitly defaulting to `None`; this was a real behavioral
  break caught by the test suite.
- **`swatch/app.py`** — peewee 4 removed `playhouse.sqlite_ext.SqliteExtDatabase`
  (replaced by a Cython-based `CySqliteDatabase` extra requiring a compiled dependency).
  Swapped in plain `peewee.SqliteDatabase`, since none of the FTS/JSON extension features
  were actually used.
- **`swatch/http.py`** — `.dict()` → `.model_dump()`; `.schema_json()` →
  `json.dumps(.model_json_schema())` (pydantic v2 returns a dict, not a JSON string);
  `np.fromstring(...)` (deprecated for binary data) → `np.frombuffer(...)`.
- **`swatch/mypy.ini`** — `python_version` was pinned to 3.9, which current mypy no longer
  accepts as a target; bumped to 3.13.
- Removed now-dead `typing.Dict`/`Optional`/`Set`/`Tuple` imports left behind after
  `pyupgrade --py313-plus` rewrote annotations to builtin generics (`dict`, `X | None`, etc).
- `.pre-commit-config.yaml` — bumped pyupgrade, black, codespell, pre-commit-hooks and
  mirrors-prettier to current releases.

## Validated

- All requirements installed cleanly into a clean venv (Python 3.14).
- Full test suite run (`python -m unittest discover -s tests`).
- End-to-end smoke test: config parsing → peewee migrations → DB read/write →
  Flask app → `/config` and `/config/schema` endpoints.
- Image pipeline smoke test: opencv/numpy/colorthief/Pillow round trip.
- `pre-commit run --all-files` clean.

## Test suite

The suite went from 5 tests (2 of them pre-existing broken) to 41, all passing:

- **Fixed the 3 pre-existing broken tests.** `test_valid_time_range`/`test_invalid_time_range`
  had their `assert`/`assert not` swapped relative to each other — verified against the real
  skip condition in `ImageProcessor.__check_image__` before fixing, so this locks in actual
  behavior rather than an assumption. `test_db_created` called `SwatchApp()` before setting
  `DB_FILE`/`CONFIG_FILE`, so it always hit the missing-config `sys.exit(1)` path instead of
  exercising `__init_db__`; rewrote it to point at an isolated temp dir and added a
  `tearDown` that calls `app.stop()` so the background cleanup threads don't leak.
- **`tests/test_util.py`** (new) — `mask_image`/`detect_objects`, the core color-detection
  algorithm, had zero prior coverage.
- **`tests/test_snapshot.py`** (new) — `SnapshotCleanup.cleanup_snapshots` retention logic,
  covering the `retain_days` default change and the "no snapshots dir yet" crash fix below.
- **`tests/test_http.py`** (new) — Flask route smoke tests, including regression coverage
  for the `jsonify` status-code bug below.
- **`tests/test_config.py`** (extended) — `extra="forbid"` rejection, `runtime_config`
  camera-name merging, and a regression test locking in `retain_days=1`.

## Bugs found and fixed along the way

Re-enabled mypy for the whole `swatch` package — it had been globally disabled via
`[mypy-swatch.*] ignore_errors = true` in `mypy.ini`, which is why some of these went
unnoticed. Fixing everything mypy surfaced turned up genuine bugs, not just annotation gaps:

- **`http.py`** — `jsonify({...}, status_code)` returns a corrupted `[dict, status]` JSON
  array with a **200** status instead of the intended body + status. This was the pattern in
  ~15 error-response routes (`/detections/<id>`, `/<camera>/snapshot.jpg`,
  `/<camera>/detect`, etc.); fixed to `make_response(jsonify({...}), status)`. Also fixed
  `/colortest/values` returning a *success* body with a hardcoded 404 status.
- **`snapshot.py`** — `cv2.imdecode()`/`requests.get(url)` results were used without
  None-checks in several methods; a malformed camera response or an unset snapshot URL would
  crash instead of failing cleanly. Also a `crop` variable that was only assigned inside an
  `if img.size > 0:` block but read unconditionally afterward (`UnboundLocalError` risk).
  `cleanup_snapshots` crashed with `FileNotFoundError` on a fresh install with no snapshots
  saved yet — more exposed now that cleanup runs immediately on startup rather than 24h
  later (see below).
- **`app.py`** — `stop()` crashed with `KeyError` joining `camera_processes` for any camera
  that never got an `AutoDetector` thread (i.e. `auto_detect=0`); it iterated over all
  configured cameras instead of the ones actually running. Also removed a dead `return`
  after `sys.exit(1)`, and renamed `SwatchConfig.parse_file` → `parse_yaml_file` since it
  collided with pydantic's own deprecated `BaseModel.parse_file` (different signature).
- **`util.py`/`config.py`** — `detect_objects` and `parse_colors_from_image` were
  type-hinted with the wrong container types (`set` where the code returns a `list`, `str`
  where it returns a `tuple`); `SnapshotConfig.url` was declared `str` but defaulted to
  `None`.
- **`detection.py`** — `AutoDetector` now asserts and caches `camera_config.name` /
  `snapshot_config.url` once in `__init__` (both are guaranteed non-`None` by how
  `SwatchApp` constructs it) instead of re-reading `Optional` fields throughout.

## Not touched

The Flutter/Dart web frontend (`web/`, SDK constraint `>=2.17.0 <3.0.0`) was left alone —
there's no Flutter/Dart toolchain available to validate an upgrade with, and it has no
bearing on Home Assistant compatibility.

## Snapshot retention

`snapshot_config.retain_days` now defaults to **1 day** instead of 7, so a fresh install
doesn't silently accumulate a week of snapshots under `/media/swatch/snapshots` before
anyone notices. It's still fully configurable per-camera via `retain_days` in
`config.yaml` (see `docs/config.md`).

`SnapshotCleanup` also now runs an initial pass immediately on startup instead of waiting
a full 24 hours for its first run — so upgrading an existing install with a large
snapshot backlog cleans it up right away instead of a day later.
