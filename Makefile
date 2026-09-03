# playlist-progression prototype — Python version
#
# Usage:
#   make run — run the ingestion pipeline
#   make clean — remove temp files
#   make init-db — initialise SQLite database

# ── Configuration ──────────────────────────────────────────────

DB_FILE ?= database/playlist.db
DB_INIT = database/init.db

# Default music directory (override with: make run MUSIC_DIR=/path/to/music)
MUSIC_DIR ?= .

# ── Targets ────────────────────────────────────────────────────


.PHONY: run init-db clean

## Initialise the SQLite database (idempotent)
init-db:
	@echo "==> Initialising database..."
	@mkdir -p database
	sqlite3 $(DB_FILE) < $(DB_INIT)
	@echo "==> Database ready: $(DB_FILE)"

## Run the ingestion pipeline
run: init-db
	@echo "==> Running ingestion pipeline..."
	python3 run.py $(MUSIC_DIR) $(DB_FILE)

## Run ingestion + generate playlist (SEED="substring" for seed title match)
playlist: init-db
	@echo "==> Running ingestion + playlist generation..."
	python3 run.py $(MUSIC_DIR) $(DB_FILE) --generate-playlist --seed-title "$(SEED)"

## Remove temp files
clean:
	@echo "==> Cleaning..."
	rm -f essentia_*.json clap_*.json
	@echo "==> Clean"