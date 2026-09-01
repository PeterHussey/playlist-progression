"""Playlist writer — outputs branch_playlist.json.

Produces a JSON file containing the ordered sequence of tracks with seed info,
distance band per track, and a human-readable reason string explaining why
each track was selected.

Output format:
{
  "seed": { "id": ..., "title": "...", "artist": "..." },
  "playlist": [
    {
      "position": 1,
      "id": 5,
      "title": "...",
      "artist": "...",
      "band": "Near",
      "distance": 0.12,
      "reason": "Close timbral neighbour, minimal shift"
    },
    ...
  ]
}
"""

import json
from pathlib import Path

from .track import Track


def _json_value(value: object) -> object:
    """Escape a value for JSON output (handles None, strings, numbers)."""
    if value is None:
        return None
    if isinstance(value, str):
        # Escape backslashes, quotes, newlines, carriage returns, tabs
        v = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        return f'"{v}"'
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value)


def _build_playlist_json(seed: Track, entries: list[dict[str, object]]) -> str:
    """Build the JSON string for the playlist output.

    Args:
        seed: the seed track (first track in the playlist)
        entries: list of dicts with keys: position, track_id, title, artist,
                 band, distance, reason

    Returns:
        formatted JSON string
    """
    lines = []
    lines.append("{")
    # seed
    lines.append('  "seed": {')
    lines.append(f'    "id": {_json_value(seed.get_id())},')
    s_title = seed.get_title() or ""
    s_artist = seed.get_artist() or ""
    lines.append(f'    "title": {_json_value(s_title)},')
    lines.append(f'    "artist": {_json_value(s_artist)}')
    lines.append("  },")
    # playlist
    lines.append('  "playlist": [')
    for i, entry in enumerate(entries):
        comma = "," if i < len(entries) - 1 else ""
        lines.append("    {")
        lines.append(f'      "position": {entry["position"]},')
        lines.append(f'      "id": {entry["track_id"]},')
        t = entry.get("title") or ""
        a = entry.get("artist") or ""
        lines.append(f'      "title": {_json_value(t)},')
        lines.append(f'      "artist": {_json_value(a)},')
        lines.append(f'      "band": {_json_value(entry["band"])},')
        lines.append(f'      "distance": {entry["distance"]:.4f},')
        reason = entry.get("reason") or ""
        lines.append(f'      "reason": {_json_value(reason)}')
        lines.append("    }" + comma)
    lines.append("  ]")
    lines.append("}")
    return "\n".join(lines)


def write_playlist(output_path: Path, seed: Track, entries: list[dict[str, object]]) -> None:
    """Write the playlist to a JSON file.

    Args:
        output_path: path to write the JSON file
        seed: the seed track (first track in the playlist)
        entries: ordered list of playlist entry dicts
    """
    json_str = _build_playlist_json(seed, entries)
    output_path.write_text(json_str, encoding="utf-8")
    # Optionally log
    import sys
    print(f"Playlist written to: {output_path} ({len(entries)} tracks)", file=sys.stderr)


# Convenience helper: build a single entry dict
def make_entry(
    position: int,
    track_id: int,
    title: str | None,
    artist: str | None,
    band: str,
    distance: float,
    reason: str | None,
) -> dict[str, object]:
    """Create a playlist entry dict.

    Args:
        position: 1-based position in the playlist
        track_id: database row ID of the track
        title: track title
        artist: artist name
        band: "Near", "Mid", or "Far"
        distance: distance from previous track
        reason: human-readable selection reason

    Returns:
        dict suitable for inclusion in the playlist JSON
    """
    return {
        "position": position,
        "track_id": track_id,
        "title": title,
        "artist": artist,
        "band": band,
        "distance": distance,
        "reason": reason or "",
    }