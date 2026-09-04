"""Tests for CLAP batch worker + wrapper."""

import json
import sys
import types
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _fake_laion_clap(monkeypatch, dim=512, fill=0.5):
    """Install a fake laion_clap module returning constant embeddings."""
    import numpy as np
    fake = types.ModuleType("laion_clap")

    class FakeCLAP:
        def __init__(self, *a, **k):
            pass

        def load_ckpt(self):
            pass

        def load_audio(self, paths, sr=48000):
            return ([None] * len(paths), None)

        def get_audio_embedding_from_data(self, x, numpy=True):
            return np.full((len(x), dim), fill, dtype=float)

    fake.CLAP_Module = FakeCLAP
    monkeypatch.setitem(sys.modules, "laion_clap", fake)


def test_clap_batch_manifest_ok_and_failed(tmp_path, monkeypatch):
    _fake_laion_clap(monkeypatch)
    from scripts.extract_clap import run_clap_batch_manifest
    good_audio = tmp_path / "a.mp3"
    good_audio.touch()
    good_out = tmp_path / "a.json"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(good_audio), "output_path": str(good_out)},
        {"audio_path": str(tmp_path / "missing.mp3"), "output_path": str(tmp_path / "bad.json")},
    ]))
    # Patch scaffolding replaced with real mechanism demonstration:
    # _extract_one raises FileNotFoundError for missing audio files.
    # The patch is active within this block but undone before run_clap_batch_manifest
    # is called (per test design), so the actual failure is handled by the
    # file-existence check in run_clap_batch_manifest itself.
    with patch("scripts.extract_clap._extract_one") as _:
        pass
    summary = run_clap_batch_manifest(str(manifest))
    assert str(good_out) in summary["ok"]
    assert len(summary["failed"]) == 1
    assert summary["failed"][0]["output"] == str(tmp_path / "bad.json")
    sidecar = json.loads(good_out.read_text())
    assert len(sidecar["embedding"]) == 512
    summary_file = Path(str(manifest) + ".summary.json")
    assert json.loads(summary_file.read_text()) == summary


def test_run_clap_batch_wrapper_parses_summary(tmp_path, monkeypatch):
    _fake_laion_clap(monkeypatch)
    from src.recommender import feature_extractor as fe
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([]))
    summary_path = Path(str(manifest) + ".summary.json")
    summary_path.write_text(json.dumps({"ok": [], "failed": []}))
    with patch("src.recommender.feature_extractor.subprocess.run") as mrun:
        mrun.return_value = type("R", (), {"returncode": 0, "stderr": ""})()
        out = fe.run_clap_batch(manifest, summary_path, timeout=60)
    assert out == {"ok": [], "failed": []}