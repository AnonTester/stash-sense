# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Import AGENTS.md

@AGENTS.md

## Commands

All commands run from `api/` with the venv active (or via `make` which handles this automatically):

```bash
# Start sidecar (dev, hot-reload)
cd api && source ../.venv/bin/activate && make sidecar

# Run tests
cd api && make test            # all tests
cd api && make test-fast       # fail-fast (-x)
cd api && make test-ci         # skip @pytest.mark.heavy (no ML/GPU required)
cd api && make test-heavy      # only heavy/GPU tests

# Run a single test file
cd api && ../.venv/bin/python -m pytest tests/test_upstream_field_mapper.py -v

# Linting
cd api && make lint            # check only
cd api && make lint-fix        # auto-fix

# Deploy plugin to Unraid
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/

# Build and push Docker image
docker build -t carrotwaxr/stash-sense:latest . && docker push carrotwaxr/stash-sense:latest
```

Dev API at `http://localhost:5000`, docs at `http://localhost:5000/docs`. Requires `api/.env` with `STASH_API_KEY` and stash-box API keys.

**Hot-reload caveat:** Background analysis tasks block uvicorn `--reload` on file changes — kill and restart the process.

## Architecture

Two components talking to one Stash instance:

- **`api/`** — FastAPI sidecar (Python). Face recognition, recommendations engine, upstream sync. Runs as Docker container on unRAID at port 6960.
- **`plugin/`** — JS/CSS/Python injected into Stash web UI. All sidecar calls go through `stash_sense_backend.py` to bypass browser CSP.

**Two databases:**
- `performers.db` — Read-only, distributed via GitHub Releases. Face metadata, stash-box IDs, Voyager ANN indices.
- `stash_sense.db` — Read-write, user-local. Recommendations, watermarks, upstream snapshots, scene fingerprints. Schema version 9 (`recommendations_db.py`). Survives face DB updates.

**Startup sequence (`main.py`):** Hardware detection → model manager init → ResourceManager registration (face recognition registered as *lazy*, not loaded) → recommendations DB init → settings system → StashBox connection manager → queue manager start. Face recognition loads on first `/identify` request.

## Key Systems

### Face Recognition (lazy-loaded)
3-phase batch pipeline in `recognizer.py`: extract frames (ffmpeg, 8 workers) → detect faces (RetinaFace ONNX) → batch embed + match (FaceNet512 + ArcFace ONNX, Voyager ANN index). `resource_manager.py` manages lazy load/unload with 30-min idle timeout.

Multi-signal late fusion in `multi_signal_matcher.py`: `final_score = face_score × body_multiplier × tattoo_multiplier`. Body via MediaPipe pose (`body_proportions.py`), tattoo via YOLO detection (`tattoo_detector.py`).

### Recommendations Engine
`BaseAnalyzer` (`analyzers/base.py`) + incremental watermarking pattern. Each analyzer type has a `logic_version` class attribute — bumping it auto-clears stale snapshots/watermarks for full re-analysis on next run. Analyzers: duplicate scenes, duplicate performers, upstream performer/scene/studio/tag changes, scene fingerprint matching, missing stash-box links.

Jobs run via `QueueManager` (`queue_manager.py`) with `JOB_REGISTRY` in `job_models.py`. `BaseJob` (`base_job.py`) provides `JobContext` with stop signaling, cursor-based checkpointing, and yield-to-higher-priority support.

### Upstream Performer Sync
3-way diff in `upstream_field_mapper.py`: upstream (current stash-box) vs local (current Stash) vs snapshot (last-seen upstream, stored in `upstream_snapshots` table). Distinguishes intentional local changes from actual upstream drift. Translation to Stash mutation format in `recommendations_router.py:update_performer_fields()`.

### Duplicate Scene Detection
Candidate generation via SQL joins + inverted indices (O(n) pairs, not O(n²)). Scored with signal hierarchy: stash-box ID match = 100%, face fingerprint ≤ 85%, metadata ≤ 60%. Diminishing returns: `primary + secondary × 0.3`.

## Conventions

- **Logging:** Default level is WARNING. `logger.warning()` is user-visible; `logger.info()` is not.
- **Rate limiting:** Shared 5 req/s for Stash and StashBox APIs. StashBox calls use `Priority.LOW`.
- **Plugin logging:** Use Stash log protocol with level-prefix bytes (`\x01` + level_char + `\x02`), not plain JSON to stderr. See `stash_sense_backend.py:_log_prefix()`.
- **Local-only fields:** `favorite`, `rating`, `o_count` are Stash-local metadata — never compare against upstream StashBox values.
- **Test marking:** ML/GPU tests are marked `@pytest.mark.heavy`. CI runs `make test-ci` which excludes them. `conftest.py` mocks ML modules so heavy-marked files can be collected without GPU.
- **Background tasks:** Don't inherit shell activation. Use explicit venv python path for background processes.

## Field Name Mapping (Upstream Sync)

Stash-box uses separate fields that Stash combines into compound strings:

| Diff Engine | Stash Mutation | Notes |
|---|---|---|
| `aliases` | `alias_list` | |
| `height` | `height_cm` | Integer |
| `breast_type` | `fake_tits` | |
| `career_start_year` + `career_end_year` | `career_length` | Combined "YYYY-YYYY" |
| `cup_size` + `band_size` + `waist_size` + `hip_size` | `measurements` | Combined "38F-24-35" |

Translation: `recommendations_router.py:update_performer_fields()`

## Key Files

- `api/main.py` — App entry point, lifespan, router wiring, lazy-load setup
- `api/recommendations_router.py` — All recommendation API endpoints
- `api/recommendations_db.py` — SQLite layer (schema v9), migrations
- `api/queue_router.py` / `api/queue_manager.py` — Job queue API and execution engine
- `api/job_models.py` — `JOB_REGISTRY` and all job type definitions
- `api/base_job.py` — `BaseJob` ABC and `JobContext`
- `api/settings_router.py` — Settings and system info API
- `api/upstream_field_mapper.py` — Field mapping, parsing, 3-way diff engine
- `api/analyzers/base_upstream.py` — Base class with logic versioning
- `api/resource_manager.py` — Lazy load / idle-unload for face recognition
- `api/stash_client_unified.py` — Stash GraphQL client
- `api/stashbox_client.py` — StashBox GraphQL client
- `plugin/stash-sense-recommendations.js` — Recommendations dashboard UI
- `plugin/stash-sense-settings.js` — Settings and model management UI
- `plugin/stash-sense-operations.js` — Operation queue UI
- `plugin/stash-sense.css` — All styles
- `plugin/stash_sense_backend.py` — Plugin backend proxy
