# AGENTS.md — playlist-progression

## Project Overview

Weekend-scale **Python** prototype for music similarity and playlist generation from local audio files. Ingests audio → extracts Essentia DSP features + optional CLAP embeddings via subprocess → stores in SQLite → generates JSON playlist using a branching recommender (near/mid/far distance bands + axis-controlled jumps).

**Key fact:** The implementation is Python (not Java as originally planned). The `src/recommender/` package contains the core pipeline.

---

## Quick Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install essentia              # required
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
│   ├── ingest_pipeline.py    # Main entry: scan → extract → store
│   ├── branch_sampler.py     # Distance bands + directed jumps
│   └── playlist_writer.py    # JSON output
├── scripts/
│   ├── extract_essentia.py   # Essentia CLI wrapper (18 descriptors)
│   └── extract_clap.py       # CLAP embedding wrapper (512-dim)
├── database/
│   ├── init.db               # SQLite schema
│   └── playlist.db           # Runtime DB (gitignored)
├── docs/                     # ARCHITECTURE.md, SCHEMA.md, BRANCHING.md, INTEGRATION.md
├── run.py                    # CLI entry point
├── Makefile                  # run, init-db, clean targets
└── requirements.txt          # stdlib only (Essentia/CLAP installed separately)
```

---

## Critical Gotchas

| Issue | Details |
|-------|---------|
| **Essentia install** | `pip install essentia` — not in requirements.txt. Must be installed in venv before running. |
| **CLAP optional** | `--clap` flag enables 512-dim embeddings. Requires `pip install laion-clap`. Pipeline works without it. |
| **Subprocess timeout** | 30 seconds per track (hardcoded in `feature_extractor.py:TIMEOUT_SECONDS`). |
| **Idempotent ingestion** | Tracks tracked by absolute `file_path` (UNIQUE constraint). Re-running skips existing files. |
| **No real audio in tests** | `tests/sample_audio/` contains placeholder `.mp3` files — Essentia will fail on them. Use real audio for actual runs. |
| **Python version** | Requires Python 3.10+ (uses `list[Type]` syntax). |

---

## Branching Algorithm (Key Design)

**Distance:** Standardised weighted Euclidean across 18 Essentia axes (z-scored per axis using population mean/stddev).

**Three bands:**
- **Near** (`d ≤ 0.3σ`): Close neighbours, minimal perceptual change
- **Mid** (`0.3σ < d ≤ 0.7σ`): Moderate jumps, noticeable but related
- **Far-but-directed** (`d > 0.7σ` on all axes **except** `hold_axis`, where `d ≤ 0.3σ`): Large leaps anchored by one constant quality (e.g., same tempo, different mood)

**Directed jump:** `select_directed_jump(seed, candidates, hold_axis)` — holds one axis constant while jumping far on others.

**Default schedule:** Near → Mid → Far → Mid → Near (repeating cycle).

---

## Database Schema

```sql
CREATE TABLE tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT UNIQUE NOT NULL,
    title TEXT,
    artist TEXT,
    duration_sec REAL,
    feature_json TEXT,          -- full Essentia JSON output
    clap_embedding BLOB,        -- nullable 512-dim float array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## Essentia Descriptors (18 dimensions)

| Group | Descriptors | Dimensions |
|-------|-------------|------------|
| Timbre | spectral_centroid (mean, std), spectral_complexity (mean, std), spectral_rolloff (mean, std) | 6 |
| Tonal | hkey_scale (key + scale), chord | 3 |
| Rhythm | bpm, danceability | 2 |
| Mood | happy, sad, aggressive, relaxed, electronic, party, acoustic | 7 |

**CLAP embeddings** (optional): 512-dim, stored in `clap_embedding` BLOB. **Not used for distance** in this prototype.

---

## Testing / QA

```bash
# Run QA script (verifies structure, DB init, compilation checks)
bash tests/run_qa.sh

# Outputs:
# - tests/QA_REPORT.md (human-readable)
# - .omo/evidence/task-6-playlist-progression.md (machine-readable counts)
```

**QA script checks:** project structure, DB init, Java compilation (legacy check), pipeline run, BranchSampler methods, JSON output format, GitHub init commands.

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
| Add new Essentia descriptor | Edit `scripts/extract_essentia.py` and update `docs/SCHEMA.md` |
| Modify distance bands | Edit `src/recommender/branch_sampler.py` thresholds |
| Change hold axis for directed jumps | Pass `hold_axis` to `select_directed_jump()` (e.g., `"rhythm.bpm"`) |
| Debug extraction failure | Check stderr from `extract_essentia.py` / `extract_clap.py` (30s timeout) |
| Inspect database | `sqlite3 database/playlist.db "SELECT * FROM tracks;"` |
| Verify JSON output | `cat branch_playlist.json \| python -m json.tool` |

---

## Git / Version Control

```bash
# Initialize repo (if not done)
git init
git add .
git commit -m "Initial prototype"
gh repo create playlist-progression --public
git remote add origin <url-from-gh>
git push -u origin main
```

**`.gitignore` excludes:** `.venv/`, `__pycache__/`, `*.db-journal`, `*.db-wal`, `*.db-shm`, `essentia_*.json`, `clap_*.json`, `branch_playlist.json`