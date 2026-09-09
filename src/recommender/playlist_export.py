"""M3U/PLS playlist export — local player handoff.

Converts an ordered list of Tracks (seed first) into extended-M3U and
PLS files with paths relativized against the export file's directory
(falls back to absolute when relativization fails, e.g. different
Windows drives).
"""

import os
from pathlib import Path

from .track import Track


def _display_title(track: Track) -> str:
    """Human-readable "Artist - Title" label with Unknown fallbacks."""
    artist = track.get_artist() or "Unknown"
    title = track.get_title() or "Unknown"
    return f"{artist} - {title}"


def _duration_sec(track: Track) -> int:
    """Whole-second duration, or -1 when unknown (M3U/PLS convention)."""
    duration = track.get_duration_sec() or 0.0
    return int(duration) if duration > 0 else -1


def _export_path(track: Track, base_dir: Path) -> str:
    """Path string for the export file, relative when possible."""
    raw = str(track.get_file_path()) if track.get_file_path() is not None else ""
    if not raw:
        return raw
    try:
        return os.path.relpath(raw, base_dir)
    except ValueError:
        return os.path.abspath(raw)


def write_m3u(output_path: Path, tracks: list[Track]) -> None:
    """Write an extended-M3U playlist (seed first, in order)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output_path.parent
    lines = ["#EXTM3U"]
    for track in tracks:
        lines.append(f"#EXTINF:{_duration_sec(track)},{_display_title(track)}")
        lines.append(_export_path(track, base_dir))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pls(output_path: Path, tracks: list[Track]) -> None:
    """Write a PLS playlist (seed first, in order)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_dir = output_path.parent
    lines = ["[playlist]"]
    for i, track in enumerate(tracks, start=1):
        lines.append(f"File{i}={_export_path(track, base_dir)}")
        lines.append(f"Title{i}={_display_title(track)}")
        lines.append(f"Length{i}={_duration_sec(track)}")
    lines.append(f"NumberOfEntries={len(tracks)}")
    lines.append("Version=2")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_playlists(
    base_output: Path, tracks: list[Track], fmt: str = "all"
) -> list[Path]:
    """Export M3U/PLS siblings next to the JSON playlist output.

    Args:
        base_output: JSON output path; exports share its stem.
        tracks: ordered tracks, seed first.
        fmt: "m3u", "pls", or "all".

    Returns:
        list of written export paths.
    """
    if fmt not in ("m3u", "pls", "all"):
        raise ValueError(f"unknown export format '{fmt}' (expected m3u, pls, or all)")
    base_output = Path(base_output)
    written: list[Path] = []
    if fmt in ("m3u", "all"):
        m3u_path = base_output.with_suffix(".m3u")
        write_m3u(m3u_path, tracks)
        written.append(m3u_path)
    if fmt in ("pls", "all"):
        pls_path = base_output.with_suffix(".pls")
        write_pls(pls_path, tracks)
        written.append(pls_path)
    return written
