# Subprocess Integration — Playlist Progression Prototype

> **Status:** Design document. Implements the Integration Approach described in
> [ARCHITECTURE.md](ARCHITECTURE.md#integration-approach).

## Overview

The Java orchestrator calls Python feature-extraction scripts as subprocesses via
`ProcessBuilder`. Each script is a stateless, single-file CLI tool: it reads one
audio file, writes one JSON sidecar, and exits. There is no daemon, no shared
state, and no IPC beyond the filesystem.

Two scripts exist:

| Script | Purpose | Required? |
|---|---|---|
| `extract_essentia.py` | DSP features (loudness, tempo, key, spectral descriptors) | Yes |
| `extract_clap.py` | 512-dimensional neural embedding | No — gated by config flag |

---

## Calling Convention

### ProcessBuilder Example

```java
import java.io.File;
import java.util.concurrent.TimeUnit;

public class FeatureExtractor {

    // Timeout overridden by Python pipeline: 180s default, phase-specific (DSP 60, mood 180)
    private static final long TIMEOUT_SECONDS = 180;

    /**
     * Run a Python extraction script via ProcessBuilder.
     *
     * @param script   script filename (e.g. "extract_essentia.py")
     * @param audioPath absolute path to the input audio file
     * @param outputPath absolute path for the JSON sidecar output
     * @throws RuntimeException if the process times out or returns non-zero
     */
    public void extract(String script, String audioPath, String outputPath)
            throws RuntimeException {

        ProcessBuilder pb = new ProcessBuilder(
            "python3", script, audioPath, outputPath
        );
        pb.redirectErrorStream(false); // keep stderr separate

        try {
            Process process = pb.start();

            // Capture stderr for diagnostics
            String stderr = new String(process.getErrorStream().readAllBytes());

            boolean finished = process.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS);
            if (!finished) {
                process.destroyForcibly();
                throw new RuntimeException(
                    "Script " + script + " timed out after " + TIMEOUT_SECONDS + "s"
                );
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                throw new RuntimeException(
                    "Script " + script + " exited with code " + exitCode
                    + "\nstderr: " + stderr
                );
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted while waiting for " + script, e);
        }
    }
}
```

### Invocation Sequence

For each ingested track, the orchestrator calls extraction in a two-phase order:

```
Phase 1 (DSP — always runs):
  python3 extract_essentia.py <audio_path> <essentia_output.json> --no-mood

Phase 2 (Mood — unless --no-mood):
  python3 extract_essentia.py <audio_path> <essentia_output.json> --mood-only

CLAP (if enabled):
  python3 extract_clap.py     <audio_path> <clap_output.json>
```

The two-phase split lets DSP features complete fast (60 s default) while mood
retrieval runs on a longer timeout (180 s default).  Mood failure is non-fatal:
the mood key is omitted from the sidecar (`NULL`) and retried on the next run
via `--mood-only`.

**Batch variant** (`--batch MANIFEST`):

```
python3 extract_essentia.py --batch <manifest.json> [--no-mood] [--models-dir DIR]
```

The batch worker loads TF graphs once, processes all tracks, and writes a
summary JSON with `ok`/`failed` per entry.  Manifest entries use unique
output paths (`batch_NNNN_stem.json`) to avoid collisions across directories.

---

## Python Script Interface

### General Contract

Both scripts follow the same interface pattern:

```
python3 <script>.py <audio_path> <output_path>
```

| Argument | Description |
|---|---|
| `audio_path` | Absolute path to the input audio file (MP3, FLAC, OGG, WAV) |
| `output_path` | Absolute path where the JSON sidecar will be written |

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | Success — output file written at `output_path` |
| `1` | Runtime error — message printed to stderr |
| `2` | Bad arguments — missing or invalid paths |

The scripts write to `stdout` only for debugging traces in development builds.
Production output goes exclusively to the JSON sidecar file.

### `extract_essentia.py`

Reads one audio file and writes a JSON sidecar containing DSP descriptors.

**Output format:**

```json
{
  "version": "1.0",
  "duration_sec": 234.5,
  "loudness": {
    "integrated": -12.3,
    "range": 8.7
  },
  "tempo": {
    "bpm": 128.0,
    "confidence": 2.27
  },
  "key": {
    "key": "C",
    "mode": "minor",
    "scale": "C minor",
    "confidence": 0.85
  },
  "spectral": {
    "centroid": 2340.5,
    "bandwidth": 1820.3,
    "rolloff": 4500.0,
    "flatness": 0.15
  },
  "rhythm": {
    "danceability": 1.04,
    "onset_rate": 3.4
  }
}
```

**Value ranges (stored raw, never clamped).** `rhythm.danceability` is the
Essentia `Danceability` output, 0–~3 by design (higher = more danceable), so
values > 1 are normal. `tempo.confidence` is the `RhythmExtractor2013`
(multifeature) beat-tracking confidence on a 0–5.32 scale ([0,1) very low,
[1,1.5] low, (1.5,3.5] good, (3.5,5.32] excellent) — **not** 0–1.
`key.confidence` is the `KeyExtractor` strength value (Essentia publishes no
bounded range). The example above uses realistic values (`confidence: 2.27`
= good beat confidence; `danceability: 1.04` = moderately danceable).
Downstream distance computation z-scores every axis, so raw scales need no
pre-normalisation; clamping at extract time would discard real signal and is
deliberately not done.

### `extract_clap.py`

Reads one audio file and writes a JSON sidecar containing a CLAP embedding
vector.

**Output format:**

```json
{
  "version": "1.0",
  "model": "clap-v1",
  "embedding": [0.012, -0.034, 0.056, "...", -0.021]
}
```

The `embedding` array contains exactly 512 floating-point values representing
the audio's position in CLAP's learned latent space. The Java side deserialises
this into a `float[512]` and stores it as a `BLOB` in the `features` table.

---

## Error Handling Rules

### Timeout

Each subprocess has a **default 180-second timeout** measured from `Process.start()` 
to `Process.waitFor()`, with phase-specific overrides. The effective timeout is 
determined by precedence: explicit flag > environment variable > phase default.

### Phase defaults
- **DSP phase**: 60 seconds (environment variable `EXTRACT_DSP_TIMEOUT_SEC`, default 60)
- **Mood phase**: 180 seconds (environment variable `EXTRACT_MOOD_TIMEOUT_SEC`, default 180)
- **Overall extraction** (single-track and batch): 180 seconds (environment variable 
  `EXTRACT_TIMEOUT_SEC`, default 180)

### CLI flags
- `--timeout SECONDS`: override overall extraction timeout
- `--dsp-timeout SECONDS`: override DSP-only timeout
- `--mood-timeout SECONDS`: override mood-only timeout
- `--no-mood`: skip mood extraction entirely
- `--batch MANIFEST`: run batch worker (single TF graph load across all tracks)
- `--prefetch`: download mood classification models

If the process does not complete within the effective timeout:
1. `process.destroyForcibly()` is called to kill the process tree.
2. A `RuntimeError` is thrown with a descriptive message.
3. The track is logged as failed and the orchestrator **continues** to the next 
   track. The pipeline is not halted.

Per-file timeout is applied via `subprocess.run(timeout=eff)`. The batch worker 
shares a single overall timeout for the entire manifest.

After the process completes, the orchestrator checks the exit code:

- **`0`** — Success. The sidecar file is read and stored.
- **`1`** — Runtime error (corrupt file, missing codec, model load failure).
  The stderr output is captured and logged. The track is marked as failed.
- **`2`** — Bad arguments. Indicates a bug in the orchestrator, not a data
  problem. Logged at ERROR level with a stack trace.

In all non-zero cases the orchestrator **logs and continues**. A single failed
extraction does not stop the pipeline.

### Stderr Capture

Each subprocess writes diagnostic messages to `stderr`. The orchestrator reads
`getErrorStream()` after the process exits (or is killed) and includes the text
in the error log entry. This captures Python tracebacks, Essentia warnings, and
model-loading diagnostics without mixing them into the JSON output.

### Sidecar File Validation

After a successful exit code, the orchestrator validates the sidecar file before
storing it:

1. **File exists** — the script must have created `output_path`.
2. **File is non-empty** — a zero-byte file indicates a silent failure.
3. **Valid JSON** — the file must parse without error.
4. **Required fields present** — the `version` key must exist.

If any check fails, the orchestrator logs a warning and skips the track. The
partial sidecar file is deleted to avoid confusion on subsequent runs.

---

## Script Design Rules

1. **Single-file, no daemon.** Each script is a standalone Python file with no
   shared state between invocations. No background processes, no persistent
   connections, no in-memory caches across calls.

2. **Read one file, write one file.** The input is an audio path; the output is
   a JSON sidecar path. No database access, no network calls.

3. **No external services.** Scripts do not call Spotify, MusicBrainz, or any
   network API. All model weights are loaded from local paths.

4. **Idempotent.** Running the same script on the same audio file produces
   identical output. The orchestrator handles deduplication via content hashes
   before calling extraction.

5. **Independently testable.** Each script can be run from the command line:
   ```
   python3 extract_essentia.py /path/to/song.mp3 /tmp/song_essentia.json
   ```
   This makes debugging extraction issues straightforward — no Java required.

---

## Configuration Flags

The orchestrator reads a JSON configuration file that controls extraction
behaviour:

```json
{
  "clap": {
    "enabled": true,
    "model_path": "/path/to/clap-model.pt"
  },
  "extraction": {
    "timeout_sec": 180,
    "python_bin": "python3"
  }
}
```

| Key | Default | Description |
|---|---|---|
| `clap.enabled` | `false` | When `false`, `extract_clap.py` is never called and the CLAP BLOB column stays `NULL`. |
| `clap.model_path` | (required if enabled) | Local filesystem path to the CLAP model weights. |
| `extraction.timeout_sec` | `180` | Per-file subprocess timeout in seconds. (Overridden by `--timeout` flag or `EXTRACT_TIMEOUT_SEC` env var.) |
| `extraction.python_bin` | `"python3"` | Path to the Python interpreter. Allows override for virtual environments. |

---



## Batch Protocol

### Manifest Schema
Each entry in the manifest JSON must conform to:

```json
[
  {"audio_path": "<absolute-path-to-audio>", "output_path": "<absolute-path-for-sidecar>"},
]
```

### Summary Schema
The batch worker writes a summary JSON at `<manifest_path>.summary.json`:

```json
{
  "ok": ["<absolute-path-to-sidecar>", ...],
  "failed": [{"output": "<absolute-path-to-sidecar>", "error": "<error-message>"}, ...]
}
```

The pipeline stores sidecars from `output_path` into the `features` table. Failed entries are logged in the `runs` table `config_snapshot` and retried on subsequent runs via the `--mood-only` flag.

### Error Isolation
Batch processing isolates failures per-track. A non-zero exit or timeout for one track does not halt processing of remaining tracks; the summary records `ok`/`failed` per entry, and failed tracks fall back to single-track extraction on the next run.

## Error Recovery

The orchestrator tracks which tracks have been successfully extracted in the
`features` table via the `extracted_at` timestamp. On re-run:

- Tracks with a non-null `extracted_at` are skipped.
- Tracks that previously failed (no row in `features`) are retried.
  - If the failure was a DSP error, the full extraction re-runs.
  - If the failure was a mood error, the mood key is omitted (NULL) in the stored
    `feature_json`, and the track is flagged for retry via `--mood-only` on the next run.
- This makes the extraction step resumable after crashes or interruptions.

### NULL+retry Rule
When mood extraction fails, the `mood` key is intentionally omitted from the stored
`feature_json` (i.e. `mood` is `NULL`). This is not a data corruption — it signals
the pipeline to retry mood extraction on the next run via `--mood-only`. The DSP
features (loudness, tempo, key, spectral, rhythm) are always retained from the first
successful extraction, so only the mood component needs re-computation.

The `runs` table logs each pipeline execution with a `config_snapshot` so that
changes in configuration between runs are visible in the audit trail.

---

## Relationship to ARCHITECTURE.md

This document expands the **Integration Approach** section of
[ARCHITECTURE.md](ARCHITECTURE.md#integration-approach). Key references:

- **Pipeline stage 2 (Extract)** — the subprocess calls described here
- **Schema (features table)** — where the sidecar JSON is persisted
- **Decisions Log** — rationale for subprocess over JNI, JSON storage over
  relational decomposition, and CLAP as optional
