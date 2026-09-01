# playlist-progression

A weekend-scale Python prototype for music similarity and playlist generation from local audio files.

## Overview

Ingests local audio files → extracts Essentia DSP features + optional CLAP embeddings via subprocess → stores in SQLite → generates a JSON playlist using a branching recommender (near/mid/far distance bands + axis-controlled jumps).

## Features

- **Subprocess integration**: Java-free — calls Essentia CLI and CLAP Python scripts via `subprocess.run()`
- **Full Essentia descriptor set**: timbre (spectral centroid/complexity/rolloff), tonal (key/scale/chord), rhythm (tempo/BPM, danceability), mood (happy/sad/aggressive/relaxed/electronic/party/acoustic)
- **SQLite storage**: `tracks` table with feature JSON and optional CLAP embeddings
- **Branching recommender**: 3 distance bands — near (≤0.3σ), mid (0.3–0.7σ), far-but-directed (≥0.7σ along specific axis, holding one descriptor constant)
- **JSON output**: `branch_playlist.json` with seed info, distance band per track, reason string

## Quick Start

```bash
# 1. Create venv and install Essentia
python -m venv .venv
source .venv/bin/activate
pip install essentia

# 2. Run the pipeline
make run MUSIC_DIR=/path/to/your/music
# or directly:
python run.py /path/to/your/music database/playlist.db
```

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
│   ├── extract_essentia.py   # Essentia CLI wrapper
│   └── extract_clap.py       # CLAP embedding wrapper
├── docs/                     # Architecture, schema, integration, branching design
├── database/init.db          # SQLite schema
├── run.py                    # CLI entry point
├── Makefile                  # run, init-db, clean targets
└── requirements.txt          # No external deps (stdlib only)
```

## Scope Boundaries

- ✅ Local music library only (owned files)
- ✅ Subprocess integration (no JNI, no HTTP service)
- ❌ No Spotify/streaming API integration
- ❌ No production auth, multi-user, or server deployment
- ❌ No MiMo-Audio natural-language layer
- ❌ No real-time playback or streaming server
- ❌ No GPU dependency for core similarity

## License

Personal exploration tool — not a commercial product.