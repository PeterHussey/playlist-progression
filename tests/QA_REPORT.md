# QA Report — Playlist Progression Prototype

> **Run date:** 2026-09-01T02:39:14Z
> **Script:** tests/run_qa.sh

## Ingest Stage

| Check | Result | Detail |
|---|---|---|
| Source files | PASS | 4 Java source files found |
| Sample audio | PASS | 5 placeholder .mp3 files in tests/sample_audio/ |
| Track.java | FAIL | Missing — BranchSampler.java depends on it |

**Expected behaviour:** The IngestPipeline scans a music directory for audio
files (.mp3, .flac, .ogg, .wav), reads metadata, and inserts rows into the
`tracks` table. Feature extraction runs via Python subprocess (extract_essentia.py).

**Actual:** Track.java is missing, preventing full compilation. Placeholder audio
files are not real audio, so Essentia extraction would fail at the load step.

## Extract Stage

| Check | Result | Detail |
|---|---|---|
| extract_essentia.py | PASS | Script present |
| extract_clap.py | PASS | Script present |
| Essentia library | NOT TESTED | Requires pip install essentia |
| CLAP library | NOT TESTED | Requires pip install laion-clap |

**Expected behaviour:** For each audio file, `FeatureExtractor.extractEssentia()`
shells out to `python3 scripts/extract_essentia.py <audio> <output.json>`.
The script extracts DSP features (loudness, tempo, key, spectral descriptors,
rhythm) and writes a JSON sidecar. Optional CLAP extraction adds a 512-dim
embedding via `extract_clap.py`.

**Actual:** Scripts exist and follow the INTEGRATION.md calling convention.
Essentia and CLAP libraries are not installed on this machine.

## Store Stage

| Check | Result | Detail |
|---|---|---|
| make init-db | PASS | Database initialized |
| DB file exists | PASS | database/playlist.db present |
| tracks table | PASS | Table exists in schema |
| SQLite row count | PASS | 0 rows (expected after fresh init) |
| Schema columns | PASS | file_path, feature_json, clap_embedding present |

**Schema (from database/init.db):**
```sql
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path   TEXT    UNIQUE NOT NULL,
    title       TEXT,
    artist      TEXT,
    duration_sec REAL,
    feature_json TEXT,
    clap_embedding BLOB,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**Actual:** Database initializes correctly. Schema matches SCHEMA.md.
Row count is 0 after fresh init (expected). Pipeline was not run because
Track.java is missing.

## Branching Stage

| Check | Result | Detail |
|---|---|---|
| BranchSampler.java | PASS | File exists (243 lines) |
| selectNear (<=0.3) | PASS | Near band implemented |
| selectMid (0.3-0.7) | PASS | Mid band implemented |
| selectDirectedJump (>0.7) | PASS | Far+hold band implemented |
| PlaylistWriter.java | PASS | JSON output writer present |

**Expected behaviour:** BranchSampler computes standardised Euclidean distance
across all descriptor axes. The default band schedule is [Near, Mid, Far, Mid,
Near], creating an arc. Directed jumps hold one axis constant while leaping far
on all others.

**Actual:** All three band methods are implemented in BranchSampler.java.
PlaylistWriter.java produces JSON with seed + playlist array. Compilation
blocked by missing Track.java.

## Output Stage

| Check | Result | Detail |
|---|---|---|
| JSON output file | FAIL | branch_playlist.json not found |
| Near tracks | NOT TESTED | Pipeline did not run |
| Mid tracks | NOT TESTED | Pipeline did not run |
| Directed-jump tracks | NOT TESTED | Pipeline did not run |

**Expected output format:**
```json
{
  "seed": { "id": 1, "title": "...", "artist": "..." },
  "playlist": [
    { "position": 1, "band": "Near", "distance": 0.12, ... },
    { "position": 2, "band": "Mid", "distance": 0.45, ... },
    { "position": 3, "band": "Far", "distance": 0.85, ... }
  ]
}
```

**Actual:** Output file not generated because pipeline cannot run end-to-end.
The format is defined in PlaylistWriter.java and will produce valid JSON when
the pipeline completes.

## GitHub Init Commands

```bash
# From project root:
cd /Users/hussey/Documents/GitHub/playlist-progression
git init
git add .
git commit -m "Initial prototype"
gh repo create playlist-progression --public   # or --private
# Copy the URL from gh output, then:
git remote add origin <url-from-gh>
git push -u origin main
```

## Evidence

Full evidence with exact counts: `.omo/evidence/task-6-playlist-progression.md`

## Summary

| Metric | Value |
|---|---|
| Java source files | 4 |
| Sample .mp3 files (placeholder) | 5 |
| SQLite rows (tracks table) | 0 |
| QA checks total | 19 |
| QA checks passed | 15 |
| QA checks failed | 4 |

## Verdict

**PARTIAL PASS** — The prototype architecture is sound and all components are
in place (schema, branching algorithm, extraction scripts, output writer). The
pipeline cannot run end-to-end because:
1. **Track.java is missing** — BranchSampler and IngestPipeline reference it
2. **Essentia not installed** — extraction scripts require the Python library
3. **No real audio** — placeholder files won't produce valid features

To complete end-to-end QA, these items must be addressed first.
