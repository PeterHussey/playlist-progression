# Evidence — End-to-End QA (Python Pipeline)

**Run date:** 2026-09-03T17:25:43Z
**QA script:** tests/run_qa.sh

## Counts

| Metric | Value |
|---|---|
| Python source files | 8 |
| SQLite rows (tracks table) | 0 |
| Tests total | 25 |
| Tests passed | 25 |
| Tests failed | 0 |

## Pipeline State

- **Ingest stage**: scan_directory + read_metadata (tinytag) + process_file
- **Extract stage**: extract_essentia.py + extract_clap.py via subprocess
- **Store stage**: SQLite database with tracks table
- **Branching stage**: BranchSampler with RMS-normalized distance, select_near/select_mid/select_directed_jump
- **Output stage**: PlaylistWriter produces branch_playlist.json (seed + playlist)

## QA Results

| Result | Check | Detail |
| PASS | Python source files | 8 files found |
| PASS | track.py | Found |
| PASS | branch_sampler.py | Found |
| PASS | ingest_pipeline.py | Found |
| PASS | playlist_writer.py | Found |
| PASS | feature_extractor.py | Found |
| PASS | extract_essentia.py | Found |
| PASS | extract_clap.py | Found |
| PASS | Makefile | Found |
| PASS | Documentation | 9 docs found |
| PASS | make init-db | Database initialized |
| PASS | DB file exists | /Users/my188/Documents/GitHub/playlist-progression/database/playlist.db present |
| PASS | tracks table | Table exists |
| PASS | SQLite row count | 0 rows (expected) |
| PASS | Schema columns | file_path, title, artist columns present |
| PASS | Core imports | track, branch_sampler, playlist_writer importable |
| PASS | Band methods | select_near, select_mid, select_directed_jump all present |
| PASS | RMS normalization | Distance is RMS-normalized |
| PASS | ingest_pipeline.py | Found |
| PASS | feature_extractor.py | Found |
| PASS | playlist_writer.py | Found |
| PASS | tinytag import | TinyTag importable |
| PASS | mood extraction | extract_mood function available |
| PASS | JSON output | Valid JSON with seed + playlist keys |
| PASS | git init | Repository initialized |
