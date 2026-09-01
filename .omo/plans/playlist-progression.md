# playlist-progression - Work Plan

## TL;DR (For humans)

**What you'll get:** A weekend-scale Java prototype that ingests your local audio files, extracts interpretable DSP features via Essentia CLI (called from Java via ProcessBuilder) plus optional CLAP embeddings, stores everything in SQLite, and outputs a JSON playlist with a deliberate "branching" structure: near matches, mid-range adjacent tracks, and directed far jumps (same axis varied) — not just nearest-neighbor duplicates.

**Why this approach:** Subprocess keeps the prototype simple (no JNI, no extra service), SQLite is portable, the branching algorithm is the interesting design contribution, and the full descriptor set gives you controllable axes for directed jumps.

**What it will NOT do:** No Spotify integration, no real-time playback server, no production user accounts, no MiMo-Audio natural-language layer. This is a personal exploration tool, not a commercial product.

**Effort:** Medium — architecture + schema design + scaffold prototype with ingest, extract, store, branch, and output stages.
**Risk:** Low — all technology choices are well-understood; the main risk is Python environment setup for Essentia/CLAP.
**Decisions to sanity-check:** Integration via subprocess; full descriptor set; JSON output; full GitHub init included.

Your next move: approve or run a high-accuracy review, then `$start-work playlist-progression`.

---

> TL;DR (machine): Medium effort, low risk — Java prototype with subprocess Essentia/CLAP, SQLite store, branching recommender, JSON playlist output, GitHub init included.

## Scope

### Must have
- Java prototype in `/Users/hussey/Documents/GitHub/playlist-progression/`
- Subprocess integration: Java calls Essentia CLI and CLAP Python script via `ProcessBuilder`
- SQLite database (`music_features.db`) with `tracks` table (path, metadata, feature_json) and optional `clap_embeddings` table
- Feature extraction pipeline: scan folder, run Essentia descriptors, optionally run CLAP embedding script
- Branching recommender algorithm with concrete design and Java-level pseudocode (near/mid/far distance bands + directed axis jumps)
- JSON playlist output (`branch_playlist.json`) with seed info, distance band per track, and reason string
- `.gitignore` excluding Python environments (`.venv/`, `__pycache__/`), build artifacts (`/target/`, `*.class`), and SQLite temporary files
- GitHub repo initialization: `git init`, `gh repo create` with public/private selection, remote push instructions

### Must NOT have (guardrails, anti-slop, scope boundaries)
- No Spotify or streaming service API integration (local library only)
- No JNI bindings or native C++ compilation within the Java build
- No production authentication, multi-user support, or server deployment
- No MiMo-Audio natural-language explanation layer (deferred)
- No real-time audio playback or streaming server component
- No dependency on GPU for core similarity (CLAP embedding optional; prototype should work with CPU-only fallback)

## Verification strategy

> Zero human intervention - all verification is agent-executed.
- Test decision: tests-after + manual QA script — no TDD framework required for prototype scale
- Evidence: `.omo/evidence/` folder with agent-executed output logs from ingest, extraction, and branching runs
- Each todo includes exact QA invocation (command + expected output format)

## Execution strategy

### Parallel execution waves
- Wave 1 (Architecture + Schema): Design concrete pipeline stages, data flow, SQLite schema, descriptor selection, and branching algorithm pseudocode
- Wave 2 (Scaffold): Create `.omo/`, `.gitignore`, Java source structure, Python scripts, SQLite init, ingest pipeline
- Wave 3 (Integration + Test): End-to-end test: ingest sample folder → extract → store → generate JSON playlist from seed

### Dependency matrix
| Todo | Depends on | Blocks | Can parallelize with |
| --- | --- | --- | --- |
| 1. Architecture document | — | — | — |
| 2. SQLite schema + descriptor selection | 1 | 3 | — |
| 3. Subprocess integration design | 1 | 4 | 2 |
| 4. Branching algorithm (pseudocode + Java design) | 1, 2, 3 | 5 | — |
| 5. Scaffold prototype (Java + Python scripts) | 2, 3, 4 | 6 | — |
| 6. End-to-end QA + GitHub init verification | 5 | — | — |

## Todos

> Implementation + Test = ONE todo. Never separate.

- [ ] 1. Write architecture design document (`docs/ARCHITECTURE.md`)
  What to do / Must NOT do: Document concrete pipeline stages (Ingest → Extract → Store → Sample → Output), data flow diagram, where Java touches Python/native (ProcessBuilder), and how CLAP embedding is optional. Must NOT include JNI design or production deployment plans.
  Parallelization: Wave 1 | Blocked by: — | Blocks: 2, 3
  References (executor has NO interview context - be exhaustive): User handoff; `ulw-plan` references; workspace `/Users/hussey/Documents/GitHub/playlist-progression/`
  Acceptance criteria (agent-executable): File `docs/ARCHITECTURE.md` exists; contains sections for Pipeline, Integration Approach, Schema Overview, Branching Design, and Scope Boundaries; all sections have at least 3 sentences each.
  QA scenarios (name the exact tool + invocation): happy — read file and grep for "ProcessBuilder" and "SQLite" and "distance bands"; failure — check file missing or sections empty. Evidence `.omo/evidence/task-1-playlist-progression.md`
  Commit: Y | docs(ARCHITECTURE): add architecture design document

- [ ] 2. Define SQLite schema and descriptor selection (`docs/SCHEMA.md`)
  What to do / Must NOT do: Specify `tracks` table columns (id INTEGER PRIMARY KEY, file_path TEXT UNIQUE, title TEXT, artist TEXT, duration_sec REAL, feature_json TEXT, created_at TIMESTAMP); `clap_embedding` optional BLOB column. List exact Essentia descriptors: `lowlevel.timbre` (mean/std of spectral centroid, spectral complexity, spectral rolloff), `lowlevel.tonal` (hkey_scale, chord), `rhythm.tempo` (BPM), `rhythm.danceability`, `highlevel.mood` (happy, sad, aggressive, relaxed, electronic, party, acoustic). Must NOT include MiMo-Audio descriptors.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 5
  References: User handoff (Essentia descriptor list); workspace empty; `essentia` CLI docs (general knowledge)
  Acceptance criteria: `docs/SCHEMA.md` exists with exact SQL `CREATE TABLE` statements and descriptor list with axis explanations; file contains at least 200 words.
  QA scenarios: happy — grep for `CREATE TABLE` and "danceability"; failure — file missing or SQL invalid. Evidence `.omo/evidence/task-2-playlist-progression.md`
  Commit: Y | docs(SCHEMA): define SQLite schema and descriptor selection

- [ ] 3. Design subprocess integration (`docs/INTEGRATION.md`)
  What to do / Must NOT do: Specify how Java calls `python3 extract_clap.py <audio_path>` and `essentia extract <audio_path>` via `ProcessBuilder`; specify JSON/line-delimited output format; specify error handling (timeout 30s, exit code check). Recommend Python script as single-file wrapper. Must NOT design JNI or HTTP service.
  Parallelization: Wave 1 | Blocked by: 1 | Blocks: 4, 5
  References: User selection (subprocess); workspace; `ProcessBuilder` Java docs
  Acceptance criteria: `docs/INTEGRATION.md` exists; contains `ProcessBuilder` example snippet, Python script interface description, and error handling rules.
  QA scenarios: happy — grep for "ProcessBuilder" and "timeout"; failure — missing error handling section. Evidence `.omo/evidence/task-3-playlist-progression.md`
  Commit: Y | docs(INTEGRATION): design subprocess integration

- [ ] 4. Design branching sampling algorithm (`docs/BRANCHING.md` + `src/main/java/recommender/BranchSampler.java` skeleton)
  What to do / Must NOT do: Concrete pseudocode and Java-level class skeleton for distance bands (near: distance ≤ 0.3 standard deviation from seed mean; mid: 0.3 < d ≤ 0.7; far-but-directed: d > 0.7 along a single selected axis, e.g., same tempo/different mood). Include directed jump logic: select one descriptor axis to hold constant, vary others. Must NOT implement full nearest-neighbor only; must include all 3 bands.
  Parallelization: Wave 2 | Blocked by: 2, 3 | Blocks: 5
  References: User design request (branching recommender); workspace; distance metric theory (standardized Euclidean)
  Acceptance criteria: `docs/BRANCHING.md` exists with pseudocode; `BranchSampler.java` skeleton exists with methods `computeDistance()`, `selectNear()`, `selectMid()`, `selectDirectedJump()`; all 3 distance bands referenced in code comments.
  QA scenarios: happy — read file and verify 3 methods present; failure — method names missing or pseudocode missing far-band. Evidence `.omo/evidence/task-4-playlist-progression.md`
  Commit: Y | design(BRANCHING): design branching sampling algorithm with skeleton

- [ ] 5. Scaffold prototype (`.gitignore`, `src/`, `scripts/`, `music_features.db` init)
  What to do / Must NOT do: Create `.gitignore` (exclude `.venv/`, `__pycache__/`, `/target/`, `*.class`, `*.db-journal`); create `src/main/java/` package structure (`IngestPipeline.java`, `FeatureExtractor.java`, `BranchSampler.java`, `PlaylistWriter.java`); create `scripts/extract_essentia.py` and `scripts/extract_clap.py` (single-file wrappers); create `database/init.db` SQL; initialize `music_features.db`. Must NOT include production build system (just simple `javac` + `java` commands or basic `Makefile`).
  Parallelization: Wave 2 | Blocked by: 2, 3, 4 | Blocks: 6
  References: Workspace; previous docs; Java package conventions
  Acceptance criteria: `.gitignore` exists; 4 Java source skeleton files exist; 2 Python script skeletons exist; `database/init.db` SQL matches SCHEMA.md; `Makefile` or `run.sh` script exists with `javac` + `java` commands.
  QA scenarios: happy — list `src/` and `scripts/` contents; failure — `.gitignore` missing `.venv/` or `Makefile` missing. Evidence `.omo/evidence/task-5-playlist-progression.md`
  Commit: Y | scaffold(PROTOTYPE): scaffold prototype directory structure and scripts

- [ ] 6. End-to-end QA and GitHub init verification (`tests/QA_REPORT.md` + `run_qa.sh`)
  What to do / Must NOT do: Run full pipeline: scan a sample audio folder (create sample folder with at least 3 dummy `.mp3` files or reference paths), run ingest, verify SQLite rows exist, run branching with seed, verify JSON output file exists with near/mid/far entries. Then verify GitHub init commands (`git init`, `git add .`, `git commit -m "Initial prototype"`, `gh repo create` steps). Must NOT claim QA passed without evidence file.
  Parallelization: Wave 3 | Blocked by: 5 | Blocks: —
  References: All previous todos; workspace; `tests/` folder created by this todo
  Acceptance criteria: `tests/QA_REPORT.md` exists with happy/failure results for each stage (ingest count, SQLite row count, JSON track count, distance band counts, GitHub init commands listed); `run_qa.sh` executable and produces `.omo/evidence/task-6-playlist-progression.md` with exact counts.
  QA scenarios: happy — `run_qa.sh` executes without errors and `QA_REPORT.md` reports 3+ tracks in output with all 3 bands; failure — script errors or JSON output missing directed-jump tracks. Evidence `.omo/evidence/task-6-playlist-progression.md`
  Commit: Y | test(QA): end-to-end QA report and GitHub init verification

## Final verification wave
> Runs in parallel after ALL todos. ALL must APPROVE. Surface results and wait for the user's explicit okay before declaring complete.
- [ ] F1. Plan compliance audit
- [ ] F2. Code quality review
- [ ] F3. Real manual QA
- [ ] F4. Scope fidelity

## Commit strategy
- Each todo commits independently (Y/N in todo row) to keep history granular.
- Wave 1 todos (docs) commit first; Wave 2 (scaffold) commits next; Wave 3 (QA) commits last.
- Final verification wave runs after Wave 3 and does not produce code commits, only approval evidence.
- The `.omo/` folder is excluded from `git` if needed, but the user requested full init; recommend adding `.omo/` to `.gitignore` unless user wants plan artifacts in repo.

## Success criteria
- `docs/ARCHITECTURE.md`, `SCHEMA.md`, `INTEGRATION.md`, `BRANCHING.md` all exist with required content.
- `BranchSampler.java` skeleton exists with 4 required methods.
- Prototype runs from `run_qa.sh` and produces `branch_playlist.json` with at least 1 near, 1 mid, and 1 directed-jump track.
- SQLite `tracks` table has data after ingest.
- GitHub init steps are documented and executable.
- No scope creep: no Spotify API, no JNI, no production auth, no MiMo-Audio layer.
