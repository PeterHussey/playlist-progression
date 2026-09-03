"""Tests for batch extraction worker that reuses TF graphs."""

import json
import inspect
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Batch manifest schema tests ─────────────────────────────────────


def test_batch_manifest_schema(tmp_path, monkeypatch):
    """run_batch processes manifest, writes summary with ok/failed lists."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "a.mp3"), "output_path": str(tmp_path / "a.json")},
        {"audio_path": str(tmp_path / "bad.mp3"), "output_path": str(tmp_path / "bad.json")},
    ]))

    summary_data = {
        "ok": [str(tmp_path / "a.json")],
        "failed": [{"output": str(tmp_path / "bad.json"), "error": "boom"}],
    }

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        # Simulate the batch script: write summary file at summary_path, return rc=0
        # cmd is [python, script, --batch, manifest_path]
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")

    assert summary["ok"][0].endswith("a.json")
    assert summary["failed"][0]["output"].endswith("bad.json")


def test_batch_all_ok(tmp_path, monkeypatch):
    """When all tracks succeed, ok list has all, failed is empty."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "t1.mp3"), "output_path": str(tmp_path / "t1.json")},
        {"audio_path": str(tmp_path / "t2.mp3"), "output_path": str(tmp_path / "t2.json")},
    ]))

    summary_data = {
        "ok": [str(tmp_path / "t1.json"), str(tmp_path / "t2.json")],
        "failed": [],
    }

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")

    assert len(summary["ok"]) == 2
    assert len(summary["failed"]) == 0


def test_batch_all_fail(tmp_path, monkeypatch):
    """When all tracks fail, ok is empty, failed has all entries."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "x.mp3"), "output_path": str(tmp_path / "x.json")},
    ]))

    summary_data = {
        "ok": [],
        "failed": [{"output": str(tmp_path / "x.json"), "error": "fail"}],
    }

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")

    assert len(summary["ok"]) == 0
    assert len(summary["failed"]) == 1
    assert summary["failed"][0]["error"] == "fail"


def test_batch_empty_manifest(tmp_path, monkeypatch):
    """Empty manifest produces empty summary."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([]))

    summary_data = {"ok": [], "failed": []}

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")

    assert summary["ok"] == []
    assert summary["failed"] == []


def test_batch_summary_written(tmp_path, monkeypatch):
    """Summary JSON file is written at summary_path."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(tmp_path / "a.mp3"), "output_path": str(tmp_path / "a.json")},
    ]))

    summary_data = {"ok": [str(tmp_path / "a.json")], "failed": []}

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        summary_path = tmp_path / "summary.json"
        summary_path.write_text(json.dumps(summary_data))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        summary = fe.run_batch(manifest, tmp_path / "summary.json")

    assert (tmp_path / "summary.json").exists()
    loaded = json.loads((tmp_path / "summary.json").read_text())
    assert loaded == summary


def test_batch_timeout_raises(tmp_path, monkeypatch):
    """Batch worker timeout raises RuntimeError."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"audio_path": "a.mp3", "output_path": "a.json"}]))

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        try:
            fe.run_batch(manifest, tmp_path / "summary.json")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "timed out" in str(e)


def test_batch_nonzero_exit_raises(tmp_path, monkeypatch):
    """Batch worker non-zero exit raises RuntimeError."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"audio_path": "a.mp3", "output_path": "a.json"}]))

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        result = MagicMock()
        result.returncode = 1
        result.stdout = ""
        result.stderr = "something went wrong"
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        try:
            fe.run_batch(manifest, tmp_path / "summary.json")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "failed" in str(e)


def test_batch_missing_summary_raises(tmp_path, monkeypatch):
    """Batch worker missing summary file raises RuntimeError."""
    monkeypatch.chdir(tmp_path)
    from src.recommender import feature_extractor as fe

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"audio_path": "a.mp3", "output_path": "a.json"}]))

    def fake_subprocess_run(cmd, capture_output=False, text=False, timeout=None):
        # Script exits 0 but doesn't write summary (shouldn't happen but tests resilience)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    with patch.object(fe.subprocess, "run", side_effect=fake_subprocess_run):
        try:
            fe.run_batch(manifest, tmp_path / "summary.json")
            assert False, "Should have raised RuntimeError"
        except RuntimeError as e:
            assert "did not produce summary" in str(e)


# ── run_batch signature tests ───────────────────────────────────────


def test_run_batch_signature():
    """run_batch accepts manifest_path, summary_path, and optional timeout."""
    from src.recommender.feature_extractor import run_batch

    sig = inspect.signature(run_batch)
    params = list(sig.parameters.keys())
    assert "manifest_path" in params
    assert "summary_path" in params
    assert "timeout" in params


# ── Pipeline batch path tests ───────────────────────────────────────


def test_run_pipeline_accepts_batch_flag():
    """run_pipeline signature includes batch parameter."""
    from src.recommender.ingest_pipeline import run_pipeline

    sig = inspect.signature(run_pipeline)
    assert "batch" in sig.parameters


# ── CLI --batch flag tests ──────────────────────────────────────────


def test_cli_accepts_batch_flag():
    """CLI --batch argument is accepted."""
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert "--batch" in result.stdout
