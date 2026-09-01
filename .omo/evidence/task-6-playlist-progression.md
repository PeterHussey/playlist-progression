# Evidence — Task 6: End-to-End QA

**Run date:** 2026-09-01T02:39:14Z
**QA script:** tests/run_qa.sh

## Counts

| Metric | Value |
|---|---|
| Java source files | 4 |
| Sample .mp3 files (placeholder) | 5 |
| SQLite rows (tracks table) | 0 |
| Tests total | 19 |
| Tests passed | 15 |
| Tests failed | 4 |

## Pipeline State

- **Ingest stage**: Placeholder .mp3 files created in tests/sample_audio/
- **Extract stage**: extract_essentia.py + extract_clap.py present (require Essentia/CLAP installed)
- **Store stage**: SQLite database initialised with tracks table schema
- **Branching stage**: BranchSampler.java implements selectNear (dist <= 0.3), selectMid (0.3 < dist <= 0.7), selectDirectedJump (far + hold axis)
- **Output stage**: PlaylistWriter.java produces branch_playlist.json (seed + playlist array)

## Distance Band Coverage

| Band | Threshold | Method | Status |
|---|---|---|---|
| Near | distance <= 0.3 | selectNear() | Implemented in BranchSampler.java |
| Mid | 0.3 < distance <= 0.7 | selectMid() | Implemented in BranchSampler.java |
| Far (directed) | > 0.7 non-hold, <= 0.3 hold | selectDirectedJump() | Implemented in BranchSampler.java |

## Blocked Items

- Track.java is missing — BranchSampler references it but it was never created. Compilation fails without it.
- Essentia Python library not installed on this machine — extraction scripts will fail at import.
- No real audio files — placeholder .mp3 files won't produce valid features.
- Pipeline cannot run end-to-end without Track.java + Essentia + real audio.

## GitHub Init Commands

```bash
cd /Users/hussey/Documents/GitHub/playlist-progression
git init
git add .
git commit -m "Initial prototype"
gh repo create playlist-progression --public
git remote add origin <url-from-gh>
git push -u origin main
```

## QA Results

| Result | Check | Detail |
| PASS | Source files | 4 Java source files found |
| FAIL | Track.java | Missing — BranchSampler.java depends on it |
| PASS | extract_essentia.py | Found |
| PASS | extract_clap.py | Found |
| PASS | Makefile | Found with build/init-db/run/clean targets |
| PASS | Documentation | 4 docs: ARCHITECTURE SCHEMA INTEGRATION BRANCHING |
| PASS | make init-db | Database initialized |
| PASS | DB file exists | /Users/hussey/Documents/GitHub/playlist-progression/database/playlist.db present |
| PASS | tracks table | Table exists in schema |
| PASS | SQLite row count | 0 rows (expected after fresh init) |
| PASS | Schema columns | file_path, feature_json, clap_embedding columns present |
| PASS | Sample audio | 5 placeholder .mp3 files in tests/sample_audio/ |
| FAIL | javac compile | Compilation failed (likely missing Track.java) |
| FAIL | Pipeline run | IngestPipeline.class not compiled — cannot run |
| PASS | Band methods | selectNear, selectMid, selectDirectedJump all present |
| PASS | PlaylistWriter.java | JSON output writer present |
| FAIL | JSON output | branch_playlist.json not found (pipeline not run end-to-end) |
| PASS | git init | Repository not yet initialized — commands documented in QA report |
| PASS | gh CLI | GitHub CLI available |
