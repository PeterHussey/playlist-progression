# AGENTS.md — playlist-progression

## Project Overview

Weekend-scale **Python** prototype for music similarity and playlist generation from local audio files. Ingests audio → extracts Essentia DSP features + optional CLAP embeddings via subprocess → stores in SQLite → generates JSON playlist using a branching recommender (near/mid/far distance bands + axis-controlled jumps).

**Key fact:** The implementation is Python (not Java as originally planned). The `src/recommender/` package contains the core pipeline.

---

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # essentia-tensorflow, tinytag, tensorflow
pip install laion-clap            # optional (for CLAP embeddings)

# Run pipeline (recursive scan of music directory)
make run MUSIC_DIR=/path/to/music
# or directly:
python run.py /path/to/music database/playlist.db [--clap]

# Database init (idempotent)
make init-db

# Clean temp files
make clean
```

---

## Project Structure

```
playlist-progression/
├── src/recommender/          # Core pipeline
│   ├── track.py              # Track dataclass
│   ├── feature_extractor.py  # Subprocess wrapper for extraction scripts
│   ├── feature_converter.py  # Axis layout (AXIS_NAMES) + JSON→vector convert()
│   ├── ingest_pipeline.py    # Main entry: scan → extract → store
│   ├── branch_sampler.py     # Distance bands + directed jumps
│   └── playlist_writer.py    # JSON output
├── scripts/
│   ├── extract_essentia.py   # Essentia CLI wrapper (20 axes via feature_converter.py)
│   └── extract_clap.py       # CLAP embedding wrapper (512-dim)
├── database/
│   ├── init.db               # SQLite schema
│   └── playlist.db           # Runtime DB (gitignored)
├── docs/                     # ARCHITECTURE.md, SCHEMA.md, BRANCHING.md, INTEGRATION.md
├── run.py                    # CLI entry point
├── Makefile                  # run, init-db, clean targets
└── requirements.txt          # essentia-tensorflow, tinytag, tensorflow (laion-clap optional, installed separately)
```

---

## Critical Gotchas

| Issue | Details |
|-------|---------|
| **Essentia install** | `pip install essentia-tensorflow` — not in requirements.txt. Must be installed in venv before running. |
| **CLAP optional** | `--clap` flag enables 512-dim embeddings. Requires `pip install laion-clap`. Pipeline works without it. |
| **Subprocess timeout** | 30 seconds per track (hardcoded in `feature_extractor.py:TIMEOUT_SECONDS`). |
| **Idempotent ingestion** | Tracks tracked by absolute `file_path` (UNIQUE constraint). Re-running skips existing files. |
| **No real audio in tests** | `tests/sample_audio/` contains placeholder `.mp3` files — Essentia will fail on them. Use real audio for actual runs. |
| **Python version** | Requires Python 3.10+ (uses `list[Type]` syntax). |

---

## Branching Algorithm (Key Design)

**Distance:** Standardised weighted Euclidean across the feature axes (z-scored per axis using population mean/stddev). Axis layout is owned by `src/recommender/feature_converter.py` (`AXIS_NAMES` + `convert()`) — read it, don't copy axis lists from here.

**Bands:** Near (`d ≤ 0.3σ`) → Mid (`0.3σ < d ≤ 0.7σ`) → Far-but-directed (`d > 0.7σ` except `hold_axis ≤ 0.3σ`). Default schedule: Near → Mid → Far → Mid → Near. Full design (weights, pseudocode, fallback order): `docs/BRANCHING.md`.

---

## Database Schema

Single `tracks` table: one row per audio file, keyed by absolute `file_path` (UNIQUE — this is what makes ingestion idempotent). Feature data lives in `feature_json` TEXT; optional CLAP embedding in `clap_embedding`. Exact DDL: `docs/SCHEMA.md` (runtime source of truth: `init_database()` in `src/recommender/ingest_pipeline.py`).

---

## Feature Axes

Count, order, and names are owned by `src/recommender/feature_converter.py` (`AXIS_NAMES` + `convert()`) — read it, don't copy axis lists from here. Human-readable descriptor definitions: `README.md` ("Essentia Descriptor Definitions") and `docs/SCHEMA.md`.

Non-obvious semantics worth knowing inline: `key.fifths_x`/`key.fifths_y` are cos/sin coordinates on a 24-slot circle of fifths where relative major/minor are adjacent (C–Am = 1 step, C–G = 2 steps, C–F# = 12 steps); unknown key/mode → (0, 0). `key.confidence` is retained as a separate reliability axis.

**CLAP embeddings** (optional): 512-dim, stored in `clap_embedding` BLOB. **Not used for distance** in this prototype.

---

## Testing / QA

```bash
# Run tests (network tests skipped by default)
pytest

# Run QA script (verifies structure, DB init, imports, sampler, JSON output)
bash tests/run_qa.sh

# Outputs:
# - tests/QA_REPORT.md (human-readable)
# - .omo/evidence/task-6-playlist-progression.md (machine-readable counts)
```

**QA script checks:** project structure, DB init, core imports, BranchSampler methods, pipeline components, mood extraction, JSON output format, git status.

**Tests:** Markers defined in `pytest.ini` — `network` tests are skipped by default (see `tests/conftest.py`).

---

## Documentation References

- `docs/ARCHITECTURE.md` — Pipeline stages, integration approach, schema overview, branching design, scope boundaries
- `docs/SCHEMA.md` — Exact SQL schema, Essentia descriptor definitions with branching roles
- `docs/BRANCHING.md` — Distance band thresholds, axis weights, directed jump pseudocode, sampling strategy
- `docs/INTEGRATION.md` — Subprocess calling convention, JSON sidecar formats, error handling

---

## Scope Boundaries (What This Is NOT)

- ❌ No Spotify/streaming API integration
- ❌ No JNI or native bindings (subprocess only)
- ❌ No production auth, multi-user, or server deployment
- ❌ No MiMo-Audio natural-language layer
- ❌ No real-time playback or streaming server
- ❌ No GPU dependency for core similarity

---

## Common Tasks for Agents

| Task | Command / File |
|------|----------------|
| Add new Essentia descriptor | Edit `scripts/extract_essentia.py` + `src/recommender/feature_converter.py` (`AXIS_NAMES` + `convert()`), add tests in `tests/test_feature_converter.py`, update `docs/SCHEMA.md` |
| Modify distance bands | Edit `src/recommender/branch_sampler.py` thresholds |
| Change hold axis for directed jumps | Pass `hold_axis` to `select_directed_jump()` (e.g., `"tempo.bpm"` — must match an `AXIS_NAMES` entry) |
| Debug extraction failure | Check stderr from `extract_essentia.py` / `extract_clap.py` (30s timeout) |
| Inspect database | `sqlite3 database/playlist.db "SELECT * FROM tracks;"` |
| Verify JSON output | `cat branch_playlist.json \| python -m json.tool` |

---

**`.gitignore` excludes:** `.venv/`, `__pycache__/`, `*.db-journal`, `*.db-wal`, `*.db-shm`, `essentia_*.json`, `clap_*.json`, `branch_playlist.json`