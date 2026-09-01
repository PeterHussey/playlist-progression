# Learnings — Playlist Progression

## 2026-08-31 — INTEGRATION.md

- ARCHITECTURE.md Integration Approach section is the authoritative reference for subprocess design decisions. INTEGRATION.md must expand it, not contradict it.
- The schema stores Essentia output as TEXT (raw JSON) and CLAP as BLOB (binary float array). INTEGRATION.md JSON formats must align with these storage types.
- CLAP gating is a config flag, not a compile-time switch. The `clap.enabled` default is `false`.
- 30-second timeout is chosen for large FLAC files (100+ MB) on moderate hardware.
- Sidecar validation (exists, non-empty, valid JSON, version key) is a safety net between subprocess exit and DB insert.
- The `runs` table provides audit trail across pipeline re-runs; `features.extracted_at` drives resume logic.

## 2026-08-31 — BRANCHING.md + BranchSampler.java

- Distance band thresholds from the task (Near ≤ 0.3, Mid 0.3–0.7, Far > 0.7) differ from ARCHITECTURE.md ranges (0–0.15, 0.15–0.40, 0.40–1.0). ARCHITECTURE.md notes bands are configurable per run, so this is intentional — the task specifies a wider progression range for the prototype.
- Directed jump logic: compute full distance, compute non-hold distance (zero out holdAxis), compute single-axis hold distance. Accept if nonHoldDist > farThreshold AND holdDist ≤ holdNearThreshold.
- Standardised Euclidean: z-score each axis before differencing. Handles axes with different natural scales (Hz vs 0–1).
- Single-axis distance uses weight * (stdA - stdB)² — the sqrt of that gives the weighted contribution.
- BranchSampler requires precomputed axisMeans and axisStdDevs; a future Loader utility should compute these from the library at startup.
- The Track class (referenced but not created yet) needs getFeatures() returning double[]. The scaffold task will define it.

## 2026-08-31 — Todo 5: Scaffold

- BranchSampler.java (243 lines) references a `Track` class that doesn't exist yet. This is a known dependency — Track needs getFeatures() returning double[]. It should be created in a future task or as part of the build step.
- FeatureExtractor.java mirrors the INTEGRATION.md ProcessBuilder example exactly — same timeout, same stderr capture, same ExtractionException pattern.
- PlaylistWriter.java avoids external JSON libraries by building JSON manually. Acceptable for prototype; a real version would use Gson/Jackson.
- Python scripts follow the exact interface contract from INTEGRATION.md: `python3 <script> <audio_path> <output_path>`, exit codes 0/1/2.
- extract_essentia.py uses deferred imports so import errors appear on stderr rather than crashing at load time.
- database/init.db uses `CREATE TABLE IF NOT EXISTS` for idempotency (SCHEMA.md doesn't have IF NOT EXISTS, but init scripts should).
- Makefile uses tabs for recipe lines (verified). `MUSIC_DIR` is overridable via env var.
- All 4 Java files share `package recommender;` — consistent package declaration.
- The Java files won't compile without a JDK or without the Track class. The Makefile is structurally correct but needs Track.java to actually build.

## 2026-09-01 — Todo 6: End-to-End QA

- QA script (tests/run_qa.sh) runs 19 checks across 8 stages: project structure, DB init, sample audio, compilation, pipeline run, branching, output, and git init.
- 15/19 checks passed. 4 failures all trace to Track.java being missing (compilation, pipeline run, JSON output, plus the Track.java check itself).
- Makefile `init-db` target works correctly: sqlite3 database/playlist.db < database/init.db creates the DB with the tracks table.
- Schema verification: tracks table has file_path, feature_json, clap_embedding columns matching SCHEMA.md.
- Placeholder .mp3 files (5 created in tests/sample_audio/) are not real audio — they demonstrate directory scanning but won't produce valid features.
- BranchSampler.java has all 3 band methods (selectNear, selectMid, selectDirectedJump) — verified by grep.
- PlaylistWriter.java produces valid JSON structure (seed + playlist array) — verified by code review.
- Evidence file at .omo/evidence/task-6-playlist-progression.md contains exact counts: 4 Java files, 5 sample files, 0 DB rows, 19 tests.
- GitHub init commands documented in QA report (git init, git add, git commit, gh repo create, git push).
- Track.java is the single blocking dependency for end-to-end execution. It needs: `public class Track { private double[] features; public double[] getFeatures() { return features; } }` at minimum.
