#!/usr/bin/env python3
"""Convert recorded playlist JSONs to M3U/PLS without regenerating them.

Faithful export: track order, bands, and distances in the JSON are kept
as-is; file paths and durations are resolved from the database by track
id. Fails loudly on any id missing from the database.

Usage:
    python scripts/export_eval_playlists.py --db database/eval.db \
        docs/eval/blinded-2026-09-05/s1-A.json [...] [--format all]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.recommender.playlist_export import export_playlists
from src.recommender.track import Track


def load_ordered_tracks(db_path: Path, json_path: Path) -> list[Track]:
    """Resolve a recorded playlist JSON to ordered Tracks (seed first).

    Raises:
        KeyError: if any referenced track id is missing from the database.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    ordered_ids = [data["seed"]["id"]] + [
        e["id"] for e in sorted(data["playlist"], key=lambda e: e["position"])
    ]
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        tracks = []
        for track_id in ordered_ids:
            row = conn.execute(
                "SELECT id, file_path, title, artist, duration_sec"
                " FROM tracks WHERE id = ?",
                (track_id,),
            ).fetchone()
            if row is None:
                raise KeyError(
                    f"{json_path}: track id={track_id} not in {db_path}"
                )
            tracks.append(
                Track(
                    id=row["id"],
                    file_path=Path(row["file_path"]) if row["file_path"] else None,
                    title=row["title"],
                    artist=row["artist"],
                    duration_sec=row["duration_sec"] or 0.0,
                )
            )
        return tracks
    finally:
        conn.close()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert recorded playlist JSONs to M3U/PLS (no regeneration)."
    )
    parser.add_argument("--db", type=Path, required=True, help="SQLite database file")
    parser.add_argument("jsons", type=Path, nargs="+", help="Playlist JSON files")
    parser.add_argument(
        "--format",
        choices=["m3u", "pls", "all"],
        default="all",
        help="Export format (default: all)",
    )
    args = parser.parse_args(argv)
    for json_path in args.jsons:
        tracks = load_ordered_tracks(args.db, json_path)
        for export_path in export_playlists(json_path, tracks, fmt=args.format):
            print(f"Export playlist: {export_path}")


if __name__ == "__main__":
    main()
