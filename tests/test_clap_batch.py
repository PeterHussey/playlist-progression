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

        def get_audio_embedding_from_filelist(self, x, use_tensor=False):
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


def test_clap_batch_cli_entry_point(tmp_path, monkeypatch):
    """Verify main() CLI entry point with --batch flag works end-to-end."""
    _fake_laion_clap(monkeypatch)
    from scripts.extract_clap import main, run_clap_batch_manifest
    import subprocess

    good_audio = tmp_path / "a.mp3"
    good_audio.touch()
    good_out = tmp_path / "a.json"
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([
        {"audio_path": str(good_audio), "output_path": str(good_out)},
    ]))

    # Invoke main() with argv mocked to ["extract_clap.py", "--batch", str(manifest)]
    with patch.object(sys, "argv", ["extract_clap.py", "--batch", str(manifest)]):
        main()

    # Sidecar should be written
    assert good_out.exists()
    sidecar = json.loads(good_out.read_text())
    assert len(sidecar["embedding"]) == 512
    assert sidecar["version"] == "1.0"
    assert sidecar["model"] == "clap-v1"

    # Summary should be written at manifest + ".summary.json"
    summary_file = Path(str(manifest) + ".summary.json")
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text())
    assert summary["ok"] == [str(good_out)]
    assert len(summary["failed"]) == 0


def test_clap_module_initialized_with_enable_fusion(tmp_path, monkeypatch):
    """Verify CLAP_Module is initialized with enable_fusion=True for variable-length audio."""
    calls = []
    fake = types.ModuleType("laion_clap")

    class RecordingFakeCLAP:
        def __init__(self, *a, **k):
            calls.append((a, k))

        def load_ckpt(self, *a, **k):
            pass

        def get_audio_embedding_from_filelist(self, x, use_tensor=False):
            import numpy as np
            return np.full((len(x), 512), 0.5, dtype=float)

    fake.CLAP_Module = RecordingFakeCLAP
    monkeypatch.setitem(sys.modules, "laion_clap", fake)

    from scripts.extract_clap import _extract_one, run_clap_batch_manifest

    good_audio = tmp_path / "a.mp3"
    good_audio.touch()
    out1 = tmp_path / "out1.json"
    _extract_one(str(good_audio), str(out1))
    assert len(calls) == 1
    assert calls[0][1].get("enable_fusion") is True

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps([{"audio_path": str(good_audio), "output_path": str(tmp_path / "out2.json")}]))
    run_clap_batch_manifest(str(manifest))
    assert len(calls) == 2
    assert calls[1][1].get("enable_fusion") is True

