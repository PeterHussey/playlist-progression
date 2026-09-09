# playlist-progression

**Brief written:** 2026-09-09 (retrofit — reconstructed after the fact; criteria post-date the code).
**Tier:** Tool

## Objective

Personal music discovery tool: turn a local library of thousands of owned audio
files into deliberate Near / Mid / Far playlist progressions.

- Primary similarity: LAION-CLAP 512-dim embeddings, L2-normalized cosine
  (`src/recommender/clap_similarity.py`), quantile bands
  (`partition_quantile_bands`, default `near_quantile=0.10`, `mid_quantile=0.40`).
- Essentia DSP + mood features are constraint gates only (tempo window, key
  compatibility, mood box + relax in `src/recommender/constraint_filter.py`) —
  not a distance measure.
- Path: validate on a mid-size subset (~150–250 tracks) before the full-library
  run.

## Success criteria

- [ ] **1. Subset scale test (ROADMAP M3).** `python3 run.py <subset_dir> <db> --clap --batch` on ~150–250 tracks yields 100% CLAP coverage per `PlaylistSampler.load_library`; 3 seeds × clap sampler produce playlists exercising Near / Mid / Far with minimal fallbacks; forced-choice ear-test votes recorded.
- [ ] **2. Full-library run (ROADMAP M4).** `python3 run.py <full_dir> <db> --clap --batch` completes on the thousands-track library; re-running skips clean (UNIQUE `file_path`); runbook notes (timeouts, retries) recorded under `docs/`.
- [ ] **3. Deprecation cleanup (ROADMAP M5, after M3 passes).** `--sampler essentia` flag, `src/recommender/branch_sampler.py`, and matching README / AGENTS / SCHEMA / BRANCHING references removed; `pytest` green; `git grep -i 'essentia.*sampler\|branch_sampler'` returns no live hits.
- [ ] **4. Repeatability / partial-failure recovery.** Re-running `run.py` skips rows whose stored `EXTRACTOR_VERSION` matches (status `skipped:`); stale rows auto-refresh (status `re-extracted:`); mood-extraction failures recover via `--mood-only` / `--prefetch` / `--no-mood`.
- [ ] **5. Playlist summary parity.** `generate_playlist.py --sampler clap` writes `playlist_summary.txt` matching the JSON entries; verified by `diff <(jq -r '.playlist[].file' branch_playlist.json) <(awk '/track:/{print $2}' playlist_summary.txt)` returning empty.
- [ ] **6. Doc consistency.** `git grep -in 'Java\b\|javac\|ProcessBuilder\|three-table'` returns no live hits across README, AGENTS.md, ARCHITECTURE.md, SCHEMA.md, BRANCHING.md, PROJECT.md, ROADMAP.md.
- [ ] **7. M3U / PLS export.** `python3 run.py … --generate-playlist --export m3u` (or `--export pls` / `--export all`) produces a valid M3U / PLS referencing files present in the playlist JSON; verified by `grep -F "$(jq -r '.playlist[0].file' branch_playlist.json)" playlist.m3u`.

## Out of scope

- **Not building:** Spotify / streaming-API integration — local owned library is the input.
- **Not building:** production auth, multi-user, or server deployment — single-user prototype.
- **Not building:** MiMo-Audio natural-language layer — not part of this codebase's target.
- **Not building:** real-time playback or streaming server — output is JSON / M3U / PLS, not audio.
- **Not building:** GPU dependency for core similarity — GPU helps the CLAP extraction pass only; cosine over 512-dim stays CPU-side.

## Constraints

- Solo author, weekend-scale prototype.
- Python 3.10+ (code uses `list[Type]` syntax; verified on macOS Python 3.12 in `.venv`).
- Pre-existing dependency commitments: `essentia-tensorflow` 2.1b6.dev1389, `tensorflow` 2.21.0, `tinytag` 2.3.1, `torchaudio` 2.11.0. `laion-clap` is optional and installed separately (tested 1.1.7).
- Subprocess-only extraction (no JNI, no native bindings).
- Mood classification uses Essentia pre-trained MusiCNN TF models (one binary model per mood, 7 axes: happy, sad, aggressive, relaxed, electronic, party, acoustic); `scripts/extract_essentia.py` downloads models to `models/` on first use.

## Decisions

| Decision | Rationale |
|---|---|
| CLAP cosine + quantile bands as default (`PlaylistSampler`, `--sampler clap`) | 49-track blinded A/B (`docs/eval/blinded-2026-09-05/REPORT.md`): tempo adherence 0.33–0.44 → 0.89–1.00, ear test 3/3 to clap, full Near / Mid / Far schedule with 0 fallbacks vs Essentia Far-only fallback chain. |
| Essentia distance **deprecated** (`branch_sampler.py`, `--sampler essentia` retained only as fallback) | 17-track comparison (`docs/comparison-clap-vs-essentia.md`): Spearman ρ = 0.078, NN match 5.9% — spaces are uncorrelated; Essentia z-space puts every candidate at d > 0.7 from typical seeds. Removal waits until M3 passes (success criterion #3). |
| Subset-before-full-scale rule | Thousands-track target is untested; 17- and 49-track results do not prove quantile stability or batch throughput at scale. |
| Batch worker as the scale path (`--batch`, TF graphs / CLAP model loaded once) | Amortises ~5 s model-init cost; single-track subprocess per file does not scale. |

## The "is done" bar

| | Looks done | Is done |
|---|---|---|
| **Scale coverage** | `--batch --clap` runs end-to-end on the `test-playlist-music/` subset. | Runs on the full thousands-track library; `PlaylistSampler.load_library` reports 100% CLAP coverage; runbook notes (timeouts, retries) recorded under `docs/`. |
| **Repeatability** | A single `--clap --batch` invocation produces a playlist JSON. | Re-running is idempotent (`skipped:` for existing non-stale rows); partial-failure rows auto-recover via `--mood-only` / `--prefetch`; version-checked re-extract works on stale rows. |
| **Playlist quality vs activity** | `branch_playlist.json` has N entries with no repeats. | Near / Mid / Far schedule exercises all three bands with minimal fallbacks; forced-choice ear-test recorded per seed; `playlist_summary.txt` written on the clap path. |

## How to run and verify

- **Run (ingest):** `python3 run.py <music_dir> <db_path> --clap --batch` — or `make run MUSIC_DIR=/path/to/music`.
- **Run (ingest + playlist):** `python3 run.py <music_dir> <db_path> --clap --batch --generate-playlist --seed-title "<substring>" --export m3u` — or `make playlist SEED="<substring>"`.
- **Test:** `pytest` (network tests skipped by default per `pytest.ini` + `tests/conftest.py`); `bash tests/run_qa.sh` (verifies structure, DB init, imports, sampler, JSON output, git status — runs against a temp DB so it doesn't touch the real library).
- **Lint:** `ruff check <touched-paths>` (CI scope — see `.github/workflows/ci.yml`).
- **Fixtures / sample data:** `test-playlist-music/` (17 real-audio files used by E2E checks); `database/qa_temp_playlist.db` is the QA script's ephemeral DB; real runs use `database/playlist.db`.

## Review cadence

- **Status review** (grade against this brief): weekly while active.
- **Expansion review** (should this brief change): at milestones only — after each success criterion (M3 / M4 / M5) is met, or when a divergence between brief and code is found that doesn't fit a status-review row.

## Review log

| Date | Type | Summary | File |
|---|---|---|---|
| _none yet_ | | | |
