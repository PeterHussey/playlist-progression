#!/usr/bin/env python3
"""CLI entry point for the playlist-progression pipeline.

Usage:
    python run.py /path/to/music database/music_features.db

Or with CLAP embedding extraction:
    python run.py /path/to/music database/music_features.db --clap

Or forcing re-extraction of tracks already in the database:
    python run.py /path/to/music database/music_features.db --re-extract
"""

import argparse
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.recommender.ingest_pipeline import run_pipeline


def _positive_int(value: str) -> int:
    """argparse type function: reject non-positive integers."""
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return ivalue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Playlist-progression: ingest music, extract features, generate branch playlists."
    )
    parser.add_argument(
        "music_dir",
        type=str,
        help="Root directory containing audio files to process",
    )
    parser.add_argument(
        "db_path",
        type=str,
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--clap",
        action="store_true",
        help="Also extract CLAP embeddings",
    )
    parser.add_argument(
        "--re-extract",
        action="store_true",
        help="Re-extract features even for tracks already in the database "
        "(rows with a stale extractor version are refreshed automatically)",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_int,
        default=None,
        help="Per-file subprocess timeout in seconds (default 180, env EXTRACT_TIMEOUT_SEC)",
    )
    parser.add_argument(
        "--dsp-timeout",
        type=_positive_int,
        default=None,
        help="DSP extraction timeout in seconds (default 60, env EXTRACT_DSP_TIMEOUT_SEC)",
    )
    parser.add_argument(
        "--mood-timeout",
        type=_positive_int,
        default=None,
        help="Mood extraction timeout in seconds (default 180, env EXTRACT_MOOD_TIMEOUT_SEC)",
    )
    parser.add_argument(
        "--no-mood",
        action="store_true",
        help="Skip mood extraction (DSP only)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use batch worker (single TF graph load for all tracks)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory for mood classification models (passed to batch worker)",
    )
    args = parser.parse_args()

    music_dir = Path(args.music_dir)
    db_path = Path(args.db_path)

    if not music_dir.is_dir():
        print(f"Error: Not a directory: {music_dir}", file=sys.stderr)
        sys.exit(1)

    tracks = run_pipeline(
        music_dir, db_path,
        extract_clap=args.clap,
        force=args.re_extract,
        timeout=args.timeout,
        dsp_timeout=args.dsp_timeout,
        mood_timeout=args.mood_timeout,
        no_mood=args.no_mood,
        batch=args.batch,
        models_dir=Path(args.models_dir) if args.models_dir else None,
    )
    print(f"\nIngestion complete: {len(tracks)} tracks processed")


if __name__ == "__main__":
    main()
