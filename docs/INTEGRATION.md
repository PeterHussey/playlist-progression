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

    private static final long TIMEOUT_SECONDS = 30;

    /**
     * Run a Python extraction script via ProcessBuilder.
     *
     * @param script   script filename (e.g. "extract_essentia.py")
     * @param audioPath absolute path to the input audio file
     * @param outputPath absolute path for the JSON sidecar output
     * @throws ExtractionException if the process times out or returns non-zero
     */
    public void extract(String script, String audioPath, String outputPath)
            throws ExtractionException {

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
                throw new ExtractionException(
                    "Script " + script + " timed out after " + TIMEOUT_SECONDS + "s"
                );
            }

            int exitCode = process.exitValue();
            if (exitCode != 0) {
                throw new ExtractionException(
                    "Script " + script + " exited with code " + exitCode
                    + "\nstderr: " + stderr
                );
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ExtractionException("Interrupted while waiting for " + script, e);
        }
    }
}
```

### Invocation Sequence

For each ingested track, the orchestrator calls extraction in order:

```
1. python3 extract_essentia.py <audio_path> <essentia_output.json>   [always]
2. python3 extract_clap.py   <audio_path> <clap_output.json>         [if enabled]
```

Essentia runs first because it is fast and provides the baseline similarity
features. CLAP runs second only when the configuration flag `clap.enabled` is
`true`. If CLAP is disabled, the `clap_embedding` column in the database stays
`NULL` and the similarity engine uses Essentia features alone.

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
    "confidence": 0.92
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
    "danceability": 0.72,
    "onset_rate": 3.4
  }
}
```

All values are normalised or bounded where possible to support downstream
Euclidean distance calculations without additional scaling.

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

Each subprocess has a **30-second hard timeout** measured from `Process.start()`
to `Process.waitFor()`. If the script does not complete within this window:

1. `process.destroyForcibly()` is called to kill the process tree.
2. An `ExtractionException` is thrown with a descriptive message.
3. The track is logged as failed and the orchestrator **continues** to the next
   track. The pipeline is not halted.

The 30-second limit is chosen to accommodate large FLAC files (100+ MB) on
moderate hardware while preventing runaway scripts from blocking the pipeline
indefinitely.

### Exit Code Check

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
    "timeout_sec": 30,
    "python_bin": "python3"
  }
}
```

| Key | Default | Description |
|---|---|---|
| `clap.enabled` | `false` | When `false`, `extract_clap.py` is never called and the CLAP BLOB column stays `NULL`. |
| `clap.model_path` | (required if enabled) | Local filesystem path to the CLAP model weights. |
| `extraction.timeout_sec` | `30` | Per-file subprocess timeout in seconds. |
| `extraction.python_bin` | `"python3"` | Path to the Python interpreter. Allows override for virtual environments. |

---

## Error Recovery

The orchestrator tracks which tracks have been successfully extracted in the
`features` table via the `extracted_at` timestamp. On re-run:

- Tracks with a non-null `extracted_at` are skipped.
- Tracks that previously failed (no row in `features`) are retried.
- This makes the extraction step resumable after crashes or interruptions.

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
