"""Tests for M3U/PLS playlist export (local player handoff)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recommender.playlist_export import export_playlists, write_m3u, write_pls
from src.recommender.track import Track


def _mk(i, name, artist="Artist", duration=200.0):
    return Track(
        id=i,
        file_path=Path(f"/music/{name}"),
        title=f"Title {i}",
        artist=artist,
        duration_sec=duration,
    )


def test_write_m3u_uses_relative_paths_and_extinf(tmp_path):
    tracks = [_mk(1, "a.mp3", duration=200.0), _mk(2, "sub/b.mp3", duration=185.6)]
    out = tmp_path / "music" / "play.m3u"
    # point track paths inside tmp_path so they relativize
    for t in tracks:
        t.set_file_path(tmp_path / "music" / Path(t.get_file_path()).name
                        if t.id == 1 else tmp_path / "music" / "sub" / "b.mp3")
    write_m3u(out, tracks)
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[0] == "#EXTM3U"
    assert lines[1] == "#EXTINF:200,Artist - Title 1"
    assert lines[2] == "a.mp3"
    assert lines[3] == "#EXTINF:185,Artist - Title 2"
    assert lines[4] == "sub/b.mp3"


def test_write_m3u_unknown_duration_uses_minus_one(tmp_path):
    tracks = [_mk(1, "a.mp3", duration=0.0)]
    tracks[0].set_file_path(tmp_path / "a.mp3")
    out = tmp_path / "play.m3u"
    write_m3u(out, tracks)
    assert "#EXTINF:-1,Artist - Title 1" in out.read_text(encoding="utf-8")


def test_write_pls_content_and_relative_paths(tmp_path):
    tracks = [_mk(1, "a.mp3", duration=200.0), _mk(2, "b.mp3", duration=0.0)]
    for t in tracks:
        t.set_file_path(tmp_path / Path(t.get_file_path()).name)
    out = tmp_path / "play.pls"
    write_pls(out, tracks)
    text = out.read_text(encoding="utf-8")
    assert "[playlist]" in text
    assert "File1=a.mp3" in text
    assert "Title1=Artist - Title 1" in text
    assert "Length1=200" in text
    assert "Length2=-1" in text
    assert "NumberOfEntries=2" in text
    assert "Version=2" in text


def test_export_playlists_derives_sibling_paths(tmp_path):
    tracks = [_mk(1, "a.mp3"), _mk(2, "b.mp3")]
    for t in tracks:
        t.set_file_path(tmp_path / Path(t.get_file_path()).name)
    base = tmp_path / "branch_playlist.json"
    written = export_playlists(base, tracks, fmt="all")
    assert written == [tmp_path / "branch_playlist.m3u", tmp_path / "branch_playlist.pls"]
    assert (tmp_path / "branch_playlist.m3u").exists()
    assert (tmp_path / "branch_playlist.pls").exists()
    written_m3u = export_playlists(base, tracks, fmt="m3u")
    assert written_m3u == [tmp_path / "branch_playlist.m3u"]


def test_export_playlists_rejects_bad_format(tmp_path):
    import pytest

    with pytest.raises(ValueError):
        export_playlists(tmp_path / "x.json", [_mk(1, "a.mp3")], fmt="xspf")


def test_generate_playlist_accepts_export_flag():
    import generate_playlist

    assert generate_playlist.parse_args([]).export is None
    assert generate_playlist.parse_args(["--export", "all"]).export == "all"
    assert generate_playlist.parse_args(["--export", "m3u"]).export == "m3u"
    assert generate_playlist.parse_args(["--export", "pls"]).export == "pls"


def _load_converter():
    import importlib.util

    path = Path(__file__).parent.parent / "scripts" / "export_eval_playlists.py"
    spec = importlib.util.spec_from_file_location("export_eval_playlists", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_converter_returns_seed_first_in_position_order(tmp_path):
    import sqlite3

    conv = _load_converter()
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE tracks (id INTEGER PRIMARY KEY, file_path TEXT, "
        "title TEXT, artist TEXT, duration_sec REAL)"
    )
    for i in (7, 13, 43):
        (tmp_path / f"t{i}.mp3").touch()
        conn.execute(
            "INSERT INTO tracks VALUES (?,?,?,?,?)",
            (i, str(tmp_path / f"t{i}.mp3"), f"Title {i}", "Artist", 100.0 + i),
        )
    conn.commit()
    conn.close()
    playlist = {
        "seed": {"id": 7, "title": "Title 7", "artist": "Artist"},
        "playlist": [
            {"position": 1, "id": 43},
            {"position": 2, "id": 13},
        ],
    }
    json_path = tmp_path / "s1-A.json"
    json_path.write_text(__import__("json").dumps(playlist))
    tracks = conv.load_ordered_tracks(db, json_path)
    assert [t.get_id() for t in tracks] == [7, 43, 13]
    assert str(tracks[0].get_file_path()) == str(tmp_path / "t7.mp3")
    assert tracks[1].get_duration_sec() == 143.0


def test_converter_rejects_unknown_track_id(tmp_path):
    import sqlite3

    conv = _load_converter()
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE tracks (id INTEGER PRIMARY KEY, file_path TEXT, "
        "title TEXT, artist TEXT, duration_sec REAL)"
    )
    conn.commit()
    conn.close()
    playlist = {"seed": {"id": 99, "title": "Ghost", "artist": "Nobody"}, "playlist": []}
    json_path = tmp_path / "s9-Z.json"
    json_path.write_text(__import__("json").dumps(playlist))
    import pytest

    with pytest.raises(KeyError):
        conv.load_ordered_tracks(db, json_path)
