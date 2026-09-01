# playlist-progression

A weekend-scale Python prototype for music similarity and playlist generation from local audio files.

## Overview

Ingests local audio files → extracts Essentia DSP features + optional CLAP embeddings via subprocess → stores in SQLite → generates a JSON playlist using a branching recommender (near/mid/far distance bands + axis-controlled jumps).

## Branching Methodology

### Feature Extraction Pipeline

**Essentia (DSP descriptors)** — `scripts/extract_essentia.py` calls the Essentia CLI to compute a fixed descriptor set per track:

| Axis Group | Descriptors | Dimensions |
|------------|-------------|------------|
| **Timbre** | `lowlevel.spectral_centroid` (mean, std), `lowlevel.spectral_complexity` (mean, std), `lowlevel.spectral_rolloff` (mean, std) | 6 |
| **Tonal** | `tonal.hkey_scale` (key + scale), `tonal.chord` (chord) | 3 |
| **Rhythm** | `rhythm.bpm` (tempo), `rhythm.danceability` | 2 |
| **Mood** | `highlevel.mood_happy`, `mood_sad`, `mood_aggressive`, `mood_relaxed`, `mood_electronic`, `mood_party`, `mood_acoustic` | 7 |

Total: **18 dimensions** per track, stored as a flat `double[]` in `Track.features` and as JSON in `tracks.feature_json`.

### Essentia Descriptor Definitions

| Descriptor | Essentia Name | Range | What It Measures |
|------------|---------------|-------|------------------|
| **Spectral Centroid (mean)** | `lowlevel.spectral_centroid.mean` | 0–Nyquist Hz | "Brightness" — centre of mass of the spectrum. Higher = brighter, more high-frequency energy. |
| **Spectral Centroid (std)** | `lowlevel.spectral_centroid.std` | ≥ 0 | Variability of brightness over time. High = timbral fluctuation (e.g., filter sweeps, evolving textures). |
| **Spectral Complexity (mean)** | `lowlevel.spectral_complexity.mean` | ≥ 0 | Number of spectral peaks — proxy for harmonic density. High = rich harmonics (piano, strings); low = pure tones (sine, flute). |
| **Spectral Complexity (std)** | `lowlevel.spectral_complexity.std` | ≥ 0 | Harmonic density variation. High = evolving harmonic content (arpeggios, modulation). |
| **Spectral Rolloff (mean)** | `lowlevel.spectral_rolloff.mean` | 0–Nyquist Hz | Frequency below which 85% of energy lies. Higher = more high-frequency content (cymbals, noise). |
| **Spectral Rolloff (std)** | `lowlevel.spectral_rolloff.std` | ≥ 0 | High-frequency energy variation. |
| **Key + Scale** | `tonal.hkey_scale` | Key: 0–11 (C–B), Scale: 0=major, 1=minor | Estimated musical key and mode. Used for tonal distance (circle-of-fifths aware). |
| **Chord** | `tonal.chord` | Chord label string | Estimated chord at each frame; aggregated to dominant chord. |
| **BPM (Tempo)** | `rhythm.bpm` | ~40–200 | Beats per minute. Primary rhythmic anchor for directed jumps. |
| **Danceability** | `rhythm.danceability` | 0–1 | How suitable for dancing — based on beat strength, regularity, tempo. High = steady 4/4 groove. |
| **Mood: Happy** | `highlevel.mood_happy` | 0–1 | Positive, upbeat, major-key feel. |
| **Mood: Sad** | `highlevel.mood_sad` | 0–1 | Melancholic, minor-key, slow tempo feel. |
| **Mood: Aggressive** | `highlevel.mood_aggressive` | 0–1 | High energy, distortion, fast tempo, heavy rhythm. |
| **Mood: Relaxed** | `highlevel.mood_relaxed` | 0–1 | Low energy, slow tempo, soft dynamics. |
| **Mood: Electronic** | `highlevel.mood_electronic` | 0–1 | Synthetic timbres, drum machines, quantised rhythm. |
| **Mood: Party** | `highlevel.mood_party` | 0–1 | High energy, danceable, celebratory. |
| **Mood: Acoustic** | `highlevel.mood_acoustic` | 0–1 | Organic timbres, minimal electronic processing. |

**Notes:**
- All `lowlevel.*` descriptors are computed frame-wise (typically 44100/2048 ≈ 21.5 ms frames) then aggregated to mean/std across the track.
- `tonal.*` descriptors use the HPCP (Harmonic Pitch Class Profile) chroma representation.
- `highlevel.mood_*` are classifier outputs from pre-trained models (Essentia's Gaia/MTG-Jamendo models).
- Mood scores are **not mutually exclusive** — a track can score high on both `mood_happy` and `mood_party`.

**CLAP (semantic embeddings)** — `scripts/extract_clap.py` runs LAION-CLAP (via `laion-clap` Python package) to produce a 512-dim embedding vector per track. Stored as JSON array in `tracks.clap_embedding`. Optional — pipeline works without it.

### Distance Computation

`BranchSampler.compute_distance()` uses **standardised weighted Euclidean distance** across the 18 Essentia axes:

```
distance = sqrt( Σ weight[i] * (z_a[i] - z_b[i])² )
```

Where `z = (value - population_mean) / population_stddev` per axis. Axis weights are configurable (default 1.0). Population statistics are computed across the full library at sampler initialisation.

CLAP embeddings are **not used for distance** in this prototype — they're stored for future semantic similarity experiments.

### Three Distance Bands

Given a seed track, candidates are partitioned by standardised distance:

| Band | Threshold | Purpose |
|------|-----------|---------|
| **Near** | `d ≤ 0.3σ` | Close neighbours — minimal perceptual change. Establishes mood/groove. |
| **Mid** | `0.3σ < d ≤ 0.7σ` | Moderate jumps — noticeable shift but related. Drives transitions. |
| **Far-but-directed** | `d > 0.7σ` on all axes **except** `hold_axis`, where `d ≤ 0.3σ` | Large leaps anchored by one constant quality (e.g., same tempo, different mood). Creates contrast/surprise. |

### Directed Jump Algorithm

`select_directed_jump(seed, candidates, hold_axis)`:

1. Compute full distance on all 18 axes
2. Compute distance on all axes **except** `hold_axis` (zero out that dimension)
3. Compute distance on `hold_axis` only
4. Candidate qualifies if: `non_hold_distance > 0.7σ` **AND** `hold_distance ≤ 0.3σ`

Example: `hold_axis="rhythm.bpm"` → playlist jumps far in timbre/mood/key while keeping tempo constant.

### Playlist Generation Flow

```
seed track
    │
    ├─► select_near()  ──► pick 1 ──► next seed
    │
    ├─► select_mid()   ──► pick 1 ──► next seed
    │
    └─► select_directed_jump(hold_axis="rhythm.bpm") ──► pick 1 ──► next seed
    │
    └─► repeat until playlist length reached or candidates exhausted
```

Fallback: if a band is empty, widen to next band. If all empty, pick globally nearest unvisited track.

### Output

`branch_playlist.json`:
```json
{
  "seed": { "id": 1, "title": "...", "artist": "..." },
  "playlist": [
    { "position": 1, "id": 5, "title": "...", "artist": "...", "band": "Near", "distance": 0.12, "reason": "Close timbral neighbour, minimal shift" },
    { "position": 2, "id": 12, "title": "...", "artist": "...", "band": "Mid", "distance": 0.45, "reason": "Moderate shift in mood, related rhythm" },
    { "position": 3, "id": 27, "title": "...", "artist": "...", "band": "Far", "distance": 0.89, "reason": "Directed jump: same tempo, far mood/timbre" }
  ]
}
```

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

## Input Processing

### Directory Scanning

The pipeline **recursively scans** the supplied music directory — no flat structure required. It walks all subdirectories and finds any file with a supported extension:

```python
# From src/recommender/ingest_pipeline.py
AUDIO_EXTENSIONS = {".mp3", ".flac", ".ogg", ".wav"}
```

**No playlist file is needed** — just point at your music root:

```bash
python run.py /path/to/music database/playlist.db
```

This will find:
```
/path/to/music/
├── Artist A/
│   ├── Album 1/
│   │   ├── 01 Track.mp3
│   │   └── 02 Track.flac
│   └── Album 2/
│       └── 01 Track.ogg
└── Artist B/
    └── Single.wav
```

All 5 files above would be discovered and processed.

### Re-ingestion Guard

The pipeline is **idempotent** — it tracks processed files by absolute path in SQLite. Re-running on the same directory:

- Skips files already in the database (by `file_path` UNIQUE constraint)
- Only processes new/changed files
- Safe to run incrementally as you add music

### Seed Selection

The first track in the generated playlist is the **seed**. Currently the seed is the first track returned by the database query (lowest `id`). Future enhancement: allow specifying a seed track by ID or path.

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