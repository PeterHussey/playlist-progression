# Architecture Design — Playlist Progression Prototype

> **Status:** Weekend-scale prototype, not a production system.

## Overview

This document describes the architecture for a local music similarity and discovery
prototype. The system reads a user's local music library, extracts acoustic and
neural features, stores them in a lightweight database, and generates a playlist
that follows a deliberate progression through similarity space. The entire pipeline
runs on a single machine with no cloud dependencies and no external API calls.

---

## Pipeline

The pipeline has five sequential stages: **Ingest → Extract → Store → Sample → Output**.

1. **Ingest.** The Java orchestrator scans a configured directory for audio files
   (MP3, FLAC, OGG, WAV). It normalises file paths, reads any embedded ID3/Vorbis
   tags (title, artist, album, duration), and writes an initial metadata row to
   SQLite. Files that are corrupt, unreadable, or duplicated by content hash are
   skipped with a logged warning. No audio conversion happens at this stage — the
   raw files are left untouched.

2. **Extract.** For each ingested track the orchestrator shells out to Python
   scripts via `ProcessBuilder`. Essentia runs first and produces a JSON sidecar
   containing loudness, tempo, key, mode, spectral descriptors, and rhythm
   statistics. If CLAP extraction is enabled, a second subprocess call runs
   `python3 extract_clap.py` and returns a 512-dimensional embedding vector.
   CLAP extraction is optional — the system is fully functional with Essentia
   features alone. Each subprocess writes its JSON output to a temporary file, which
   the Java side reads back and closes.

Worker notes: the batch worker loads TensorFlow models once and reuses them across
tracks, amortising the initialisation cost. Single-track extraction remains retained
for debugging and interactive use.

3. **Store.** The Java layer persists all extracted features into SQLite. Essentia
   output is stored as a `TEXT` column holding the raw JSON. CLAP embeddings are
   stored as a `TEXT` column holding a JSON array that is populated only when extraction was enabled.
   Metadata rows are updated with file hashes and feature-extraction timestamps so
   re-runs can skip already-processed tracks.

4. **Sample.** A similarity matrix is computed on demand by loading all feature
   vectors into memory, running pairwise distance calculations, and partitioning
   neighbours into **Near**, **Mid**, and **Far** distance bands. The band
   boundaries are configurable thresholds expressed in normalised Euclidean distance.
   This stage also handles axis-controlled directed jumps — the user can bias
   sampling toward a particular feature axis (e.g. tempo increase, energy
   progression) by weighting that dimension more heavily in the distance function.

5. **Output.** The sampler produces an ordered sequence of track IDs and writes
   them to a JSON playlist file. The JSON includes the full track metadata, the
   selected distance band for each transition, and the feature axes that drove each
   decision. This file can be loaded by any local player or inspected manually.

---

## Integration Approach

The system uses a **Java + Python subprocess boundary** orchestrated through
`ProcessBuilder`. Java owns the main loop, file I/O, database access, and
playlist generation logic. Python is invoked only for feature extraction, which
benefits from the mature Essentia and CLAP libraries without requiring JNI or
native bindings.

Concrete subprocess calls:

```
ProcessBuilder pb = new ProcessBuilder("python3", "extract_essentia.py", audioPath, outputPath);
ProcessBuilder pb = new ProcessBuilder("python3", "extract_clap.py", audioPath, outputPath);
```

Each Python script is a standalone CLI tool that reads one audio file and writes
one JSON sidecar. The scripts are intentionally stateless — no shared memory,
no daemon process. This makes them easy to test, debug, and replace independently.

CLAP extraction is gated behind a configuration flag. When disabled, the pipeline
runs Essentia-only and the CLAP TEXT column remains `NULL`. This keeps the core
similarity engine functional on machines where the neural embedding model is too
large or where GPU acceleration is unavailable.

---

## Schema Overview

The prototype uses a single SQLite database file (`playlist.db`). SQLite is
chosen for zero-configuration deployment and straightforward file-based backup.

### Tables

| Table | Purpose |
|---|---|
| `tracks` | One row per audio file. Columns: `id` (INTEGER PK), `path` (TEXT UNIQUE), `title`, `artist`, `album`, `duration_sec`, `sha256`, `ingested_at`. |
| `features` | One row per extracted track. Columns: `track_id` (FK → tracks), `essentia_json` (TEXT — full Essentia output), `clap_embedding` (TEXT — nullable 512-dim float array as JSON), `extracted_at`. |
| `runs` | Audit log for pipeline executions. Columns: `id`, `started_at`, `completed_at`, `tracks_processed`, `config_snapshot` (TEXT — JSON of thresholds and flags). |

Feature data is stored in its native JSON shape rather than being decomposed into
individual columns. This avoids schema migrations every time Essentia adds a new
descriptor and keeps the extract scripts decoupled from database schema changes.
CLAP embeddings use `TEXT` (JSON array) storage for consistency with `feature_json`
— human-readable and directly matching what `ingest_pipeline.py` writes via `json.dumps`.

---

## Branching Design

The playlist progression engine moves through similarity space using **distance
bands** — discrete zones that control how far each successive track can deviate
from the current one.

### Distance Bands

| Band | Normalised Distance Range | Behaviour |
|---|---|---|
| **Near** | 0.0 – 0.15 | Close neighbours; minimal perceptual change. Good for establishing a mood. |
| **Mid** | 0.15 – 0.40 | Moderate jumps; noticeable shift but still related. Good for transition. |
| **Far** | 0.40 – 1.0 | Large leaps; contrast and surprise. Good for discovery moments. |

Band boundaries are stored as configuration parameters and can be tuned per run.

### Axis-Controlled Directed Jumps

Beyond distance thresholds, the system supports **axis bias** — weighting
specific feature dimensions more heavily when computing pairwise distance. For
example, setting a high weight on the `tempo` axis forces the playlist to
progressively accelerate (or decelerate), even while other musical qualities
stay within the Near band. Axis weights are passed as a JSON config object and
modify the Euclidean distance function at sampling time.

### Sampling Strategy

The sampler maintains a candidate list from the nearest unvisited tracks in the
target band. When a band runs out of candidates, it falls back to the next wider
band with a logged notice. The progression is: start in Near, transition through
Mid, and introduce Far for discovery peaks. The exact schedule is configurable
but defaults to a repeating Near → Mid → Far cycle.

---

## Scope Boundaries

This project is deliberately scoped as a **weekend-scale prototype**, not a
production system. The following boundaries apply:

- **Local library only.** The system reads files from a local directory tree. There
  is no Spotify API integration, no streaming-service authentication, and no
  remote file access. The user's own audio files are the sole data source.
- **No production auth.** There is no user authentication, no multi-tenancy, and
  no access control. The SQLite file is open to whoever has filesystem access.
- **No JNI or native bindings.** The Java-to-Python interface is strictly
  subprocess-based via `ProcessBuilder`. There are no JNI wrappers, no
  shared-memory mechanisms, and no in-process Python embedding.
- **No neural audio generation.** While the architecture notes mention CLAP
  embeddings and MiMo-Audio was evaluated during research, this prototype does not
  implement any natural-language or generative audio layer. CLAP is used solely as
  a similarity feature, not for generation.
- **Single-machine deployment.** There are no Docker containers, no CI/CD pipelines,
  no cloud hosting. The system runs as a local Java application invoking local
  Python scripts. The SQLite database file and JSON playlist output live alongside
  the source tree.

---

## Decisions Log

| Decision | Rationale |
|---|---|
| Essentia as primary feature extractor | Mature, well-documented, extensive DSP descriptor set, no GPU required. |
| CLAP as optional secondary extractor | Rich neural embeddings improve similarity when available, but the system must work without them. |
| SQLite over PostgreSQL or H2 | Zero-install, single-file database ideal for a local prototype. No server process. |
| JSON feature storage over relational decomposition | Avoids schema churn; keeps extract scripts decoupled from DB migrations. |
| Subprocess (ProcessBuilder) over JNI | Simpler debugging, no native compilation, Python scripts are independently testable. |
| Distance bands over continuous similarity scoring | Discrete bands give the user predictable playlist behaviour and make axis-controlled jumps intuitive. |
