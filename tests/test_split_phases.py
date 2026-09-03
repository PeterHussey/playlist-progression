"""Tests for split DSP/mood phases with NULL+retry on mood failure."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _sidecar(version="1.1", with_mood=True):
    """Helper: build a minimal feature sidecar dict."""
    d = {
        "version": version,
        "duration_sec": 200.0,
        "loudness": {"integrated": -10.0, "range": 5.0},
        "tempo": {"bpm": 120.0, "confidence": 0.9},
        "key": {"key": "C", "mode": "major", "scale": "C major", "confidence": 0.9},
        "spectral": {"centroid": 2000.0, "rolloff": 4000.0, "flatness": 0.02},
        "rhythm": {"danceability": 0.5, "onset_rate": 1.0},
    }
    if with_mood:
        d["mood"] = {
            "happy": 0.8,
            "sad": 0.1,
            "aggressive": 0.1,
            "relaxed": 0.5,
            "electronic": 0.2,
            "party": 0.6,
            "acoustic": 0.3,
        }
    return d


# ── timeout_for tests ────────────────────────────────────────────────


def test_timeout_for_phases(monkeypatch):
    """timeout_for returns correct defaults for 'dsp' and 'mood' phases."""
    from src.recommender.feature_extractor import timeout_for

    monkeypatch.delenv("EXTRACT_DSP_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("EXTRACT_MOOD_TIMEOUT_SEC", raising=False)
    assert timeout_for("dsp") == 60
    assert timeout_for("mood") == 180


def test_timeout_for_unknown_phase_raises():
    """timeout_for raises ValueError for unknown phase."""
    from src.recommender.feature_extractor import timeout_for

    try:
        timeout_for("unknown")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_timeout_for_explicit_overrides_env(monkeypatch):
    """Explicit arg overrides env and default."""
    from src.recommender.feature_extractor import timeout_for

    monkeypatch.setenv("EXTRACT_DSP_TIMEOUT_SEC", "90")
    monkeypatch.setenv("EXTRACT_MOOD_TIMEOUT_SEC", "200")
    assert timeout_for("dsp", explicit=30) == 30
    assert timeout_for("mood", explicit=45) == 45


def test_timeout_for_env_overrides_default(monkeypatch):
    """Env var overrides default when no explicit arg."""
    from src.recommender.feature_extractor import timeout_for

    monkeypatch.setenv("EXTRACT_DSP_TIMEOUT_SEC", "120")
    monkeypatch.setenv("EXTRACT_MOOD_TIMEOUT_SEC", "300")
    assert timeout_for("dsp") == 120
    assert timeout_for("mood") == 300


# ── _run_extraction no_mood / NULL+retry tests ──────────────────────


def test_mood_failure_keeps_dsp(tmp_path, monkeypatch):
    """When no_mood=True, feature_json is stored without mood key (NULL mood)."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from unittest.mock import patch

    from src.recommender.ingest_pipeline import init_database, _run_extraction
    from src.recommender.track import Track

    audio = tmp_path / "s.mp3"
    audio.touch()
    db = tmp_path / "t.db"
    conn = init_database(db)
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
        (str(audio), None, None, 0.0),
    )
    conn.commit()
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def fake_essentia(a, o, timeout=None, no_mood=False, mood_only=False):
        """Simulate extract_essentia writing a sidecar."""
        Path(o).write_text(json.dumps(_sidecar(with_mood=not no_mood)))

    with patch(
        "src.recommender.ingest_pipeline.extract_essentia", side_effect=fake_essentia
    ):
        track = Track(id=tid, file_path=audio)
        _run_extraction(conn, tid, audio, track, no_mood=True)

    row = conn.execute("SELECT feature_json FROM tracks WHERE id=?", (tid,)).fetchone()
    assert row is not None, "Row should exist"
    parsed = json.loads(row[0])
    assert "duration_sec" in parsed, "DSP features should be stored"
    assert "mood" not in parsed, "Mood key should be absent when no_mood=True"
    conn.close()


def test_mood_exception_keeps_dsp(tmp_path, monkeypatch):
    """When mood extraction raises, DSP is kept and mood is NULL."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from unittest.mock import patch

    from src.recommender.ingest_pipeline import init_database, _run_extraction
    from src.recommender.track import Track

    audio = tmp_path / "f.mp3"
    audio.touch()
    db = tmp_path / "t2.db"
    conn = init_database(db)
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
        (str(audio), None, None, 0.0),
    )
    conn.commit()
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    call_count = [0]

    def fake_essentia(a, o, timeout=None, no_mood=False, mood_only=False):
        call_count[0] += 1
        if mood_only:
            raise RuntimeError("mood extraction failed")
        Path(o).write_text(json.dumps(_sidecar(with_mood=False)))

    with patch(
        "src.recommender.ingest_pipeline.extract_essentia", side_effect=fake_essentia
    ):
        track = Track(id=tid, file_path=audio)
        _run_extraction(conn, tid, audio, track, no_mood=False)

    row = conn.execute("SELECT feature_json FROM tracks WHERE id=?", (tid,)).fetchone()
    assert row is not None
    parsed = json.loads(row[0])
    assert "duration_sec" in parsed, "DSP features should be stored"
    assert "mood" not in parsed, "Mood key should be absent after mood failure"
    conn.close()


def test_mood_success_merges_dsp(tmp_path, monkeypatch):
    """When mood succeeds, mood is merged into feature_json."""
    monkeypatch.chdir(tmp_path)
    import sqlite3
    from unittest.mock import patch

    from src.recommender.ingest_pipeline import init_database, _run_extraction
    from src.recommender.track import Track

    audio = tmp_path / "m.mp3"
    audio.touch()
    db = tmp_path / "t3.db"
    conn = init_database(db)
    conn.execute(
        "INSERT INTO tracks (file_path, title, artist, duration_sec) VALUES (?, ?, ?, ?)",
        (str(audio), None, None, 0.0),
    )
    conn.commit()
    tid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def fake_essentia(a, o, timeout=None, no_mood=False, mood_only=False):
        if mood_only:
            Path(o).write_text(json.dumps({"mood": _sidecar(with_mood=True)["mood"]}))
        else:
            Path(o).write_text(json.dumps(_sidecar(with_mood=False)))

    with patch(
        "src.recommender.ingest_pipeline.extract_essentia", side_effect=fake_essentia
    ):
        track = Track(id=tid, file_path=audio)
        _run_extraction(conn, tid, audio, track, no_mood=False)

    row = conn.execute("SELECT feature_json FROM tracks WHERE id=?", (tid,)).fetchone()
    assert row is not None
    parsed = json.loads(row[0])
    assert "duration_sec" in parsed
    assert "mood" in parsed, "Mood should be present when mood extraction succeeds"
    assert parsed["mood"]["happy"] == 0.8
    conn.close()


# ── _run_extraction accepts new keyword args ─────────────────────────


def test_run_extraction_accepts_new_kwargs():
    """_run_extraction signature includes dsp_timeout, mood_timeout, no_mood."""
    import inspect
    from src.recommender.ingest_pipeline import _run_extraction

    sig = inspect.signature(_run_extraction)
    params = list(sig.parameters.keys())
    assert "dsp_timeout" in params, "_run_extraction must accept 'dsp_timeout'"
    assert "mood_timeout" in params, "_run_extraction must accept 'mood_timeout'"
    assert "no_mood" in params, "_run_extraction must accept 'no_mood'"


def test_run_pipeline_accepts_new_kwargs():
    """run_pipeline signature includes dsp_timeout, mood_timeout, no_mood."""
    import inspect
    from src.recommender.ingest_pipeline import run_pipeline

    sig = inspect.signature(run_pipeline)
    params = list(sig.parameters.keys())
    assert "dsp_timeout" in params, "run_pipeline must accept 'dsp_timeout'"
    assert "mood_timeout" in params, "run_pipeline must accept 'mood_timeout'"
    assert "no_mood" in params, "run_pipeline must accept 'no_mood'"


# ── extract_essentia signature tests ────────────────────────────────


def test_extract_essentia_accepts_no_mood_and_mood_only():
    """extract_essentia signature includes no_mood and mood_only kwargs."""
    import inspect
    from src.recommender.feature_extractor import extract_essentia

    sig = inspect.signature(extract_essentia)
    params = list(sig.parameters.keys())
    assert "no_mood" in params, "extract_essentia must accept 'no_mood'"
    assert "mood_only" in params, "extract_essentia must accept 'mood_only'"
