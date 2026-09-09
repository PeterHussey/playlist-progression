"""Tests for eval metric helpers."""
import json
import sqlite3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts import eval_playlist as ev


def test_spearman_perfect_and_inverse():
    assert abs(ev.spearman([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) - 1.0) < 1e-9
    assert abs(ev.spearman([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) + 1.0) < 1e-9


def test_candidate_pool_spearman_uses_shared_pool():
    ess = {"a": 0.1, "b": 0.5, "c": 0.9, "d": 0.4, "e": 0.7}
    clap = {"a": 0.2, "b": 0.4, "c": 0.8, "d": 0.6, "e": 0.3}
    rho, n = ev.candidate_pool_spearman(ess, clap)
    assert n == 5
    assert -1.0 <= rho <= 1.0


def test_candidate_pool_spearman_needs_overlap():
    rho, n = ev.candidate_pool_spearman({"a": 0.1}, {"b": 0.9})
    assert (rho, n) == (None, 0)


def test_band_means_separates():
    entries = [{"band": "Near", "distance": 0.1}, {"band": "Near", "distance": 0.3},
               {"band": "Far", "distance": 0.8}]
    means = ev.band_means(entries)
    assert means["Near"] < means["Far"]


def test_anchor_adherence_counts_violations():
    meta = [{"mood_ok": True, "key_ok": True, "tempo_ok": True},
            {"mood_ok": True, "key_ok": False, "tempo_ok": True}]
    out = ev.anchor_adherence(meta)
    assert out == {"mood": 1.0, "key": 0.5, "tempo": 1.0}


def _sidecar(bpm=120.0, key="C", mode="major"):
    return {
        "version": "1.1", "duration_sec": 200.0,
        "tempo": {"bpm": bpm, "confidence": 0.9},
        "key": {"key": key, "mode": mode, "scale": f"{key} {mode}", "confidence": 2.0},
        "mood": {"sad": 0.8},
    }


def test_main_runs_both_paths(tmp_path, capsys):
    """eval main() prints legacy and clap sections without error."""
    db = tmp_path / "eval.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        """CREATE TABLE tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL, title TEXT, artist TEXT,
            duration_sec REAL, feature_json TEXT, clap_embedding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    for fp, title, bpm, clap, key, mode in [
        ("/m/seed.mp3", "Seed", 120.0, [1.0, 0.0], "C", "major"),
        ("/m/a.mp3", "Alpha", 125.0, [0.9, 0.1], "G", "major"),
        ("/m/b.mp3", "Beta", 130.0, [0.5, 0.5], "D", "major"),
        ("/m/c.mp3", "Gamma", 120.0, None, "C", "major"),
    ]:
        conn.execute(
            "INSERT INTO tracks (file_path, title, duration_sec, feature_json, clap_embedding)"
            " VALUES (?, ?, ?, ?, ?)",
            (fp, title, 200.0, json.dumps(_sidecar(bpm, key, mode)),
             json.dumps(clap) if clap is not None else None),
        )
    conn.commit()
    conn.close()

    ev.main(["--db", str(db), "--seed-title", "Seed", "--limit", "2"])
    out = capsys.readouterr().out
    assert "Legacy" in out
    assert "CLAP" in out
    assert "different scales" in out
    assert "candidate pool" in out
