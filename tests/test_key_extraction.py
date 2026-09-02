#!/usr/bin/env python3
"""Failing test: extract() must produce varied, confident keys."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

MUSIC_DIR = Path(__file__).resolve().parent.parent / "test-playlist-music"
EXTRACT = Path(__file__).resolve().parent.parent / "scripts" / "extract_essentia.py"

def extract_to_tmp(audio_path: Path) -> dict:
    """Run the production extract script and return parsed JSON."""
    import subprocess
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        subprocess.run(
            [sys.executable, str(EXTRACT), str(audio_path), str(tmp_path)],
            check=True, capture_output=True, timeout=60,
        )
        return json.loads(tmp_path.read_text())
    finally:
        tmp_path.unlink(missing_ok=True)

def test_key_is_not_constant():
    """Verify extract() produces varied keys with nonzero confidence."""
    files = sorted(MUSIC_DIR.glob("*.mp3"))[:3]
    assert len(files) >= 2, f"Need >= 2 test files, found {len(files)}"

    results = []
    for f in files:
        data = extract_to_tmp(f)
        key_info = data["key"]
        results.append((f.name, key_info["key"], key_info["scale"],
                        key_info["confidence"]))
        print(f"  {f.name}: {key_info['key']} {key_info['scale']} "
              f"conf={key_info['confidence']:.3f}")

    # At least two different keys detected
    keys = [r[1] for r in results]
    assert len(set(keys)) > 1, f"Expected varied keys, got constant: {keys}"

    # All confidences above threshold
    for name, key, scale, conf in results:
        assert conf > 0.5, f"{name}: confidence too low ({conf:.3f})"

    print("PASS: key extraction produces varied, confident results")

if __name__ == "__main__":
    print("Test: key extraction (via production extract_essentia.py)")
    test_key_is_not_constant()
