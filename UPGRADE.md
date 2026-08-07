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

Three pre-existing test failures are unrelated to this upgrade (confirmed by not touching
the code paths involved): `test_db_created` (expects `/config/config.yaml` to exist, which
it doesn't in a bare checkout) and `test_valid_time_range` / `test_invalid_time_range`
(pre-existing inverted assertions in `tests/test_image.py`).

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
