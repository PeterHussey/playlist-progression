# Mood Description Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 7 mood axes to the similarity space using Essentia TensorFlow classifiers.

**Architecture:** Modify extract_essentia.py to add mood extraction using MusiCNN models, update generate_playlist.py to include mood axes in distance calculations, and add TensorFlow as a required dependency.

**Tech Stack:** Python, Essentia, TensorFlow, SQLite

**Spec:** docs/superpowers/specs/2026-09-02-mood-description-design.md

## Global Constraints

- TensorFlow >= 2.10.0 required
- Models download automatically from Essentia servers on first use
- Audio resampled to 16kHz for MusiCNN models
- Graceful fallback if individual mood extraction fails (score = 0.0)
- All existing tests must continue to pass

---

## File Structure

**Modified Files:**
- `scripts/extract_essentia.py` - Add mood extraction function and integrate with existing extraction
- `generate_playlist.py:11-36` - Add mood axes to AXIS_NAMES and extract_features function
- `requirements.txt` - Add tensorflow dependency
- `docs/SCHEMA.md:104-119` - Update mood descriptor documentation

**New Files:**
- `tests/test_mood_extraction.py` - Unit tests for mood extraction functionality
- `models/` - Directory for downloaded mood classifier models (created automatically)

---

### Task 1: Add TensorFlow Dependency

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: None
- Produces: Updated requirements.txt with tensorflow dependency

- [ ] **Step 1: Update requirements.txt**

Add tensorflow dependency to requirements.txt:

```txt
# Core dependencies (stdlib only):
# - sqlite3, subprocess, json, dataclasses, pathlib, argparse
# - typing (stdlib since Python 3.5+)

# Metadata extraction (ID3/Vorbis tags)
tinytag>=1.10.0

# Mood classification (TensorFlow models)
tensorflow>=2.10.0
```

- [ ] **Step 2: Verify installation**

Run: `pip install -r requirements.txt`
Expected: TensorFlow installs successfully

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add tensorflow dependency for mood classification"
```

---

### Task 2: Create Mood Extraction Tests

**Files:**
- Create: `tests/test_mood_extraction.py`

**Interfaces:**
- Consumes: None (tests will mock dependencies)
- Produces: Test suite for mood extraction functionality

- [ ] **Step 1: Create test file with basic structure**

```python
#!/usr/bin/env python3
"""Tests for mood extraction functionality."""
import pytest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_mood_models_download():
    """Test that mood models can be downloaded."""
    from scripts.extract_essentia import download_mood_models
    
    models_dir = Path("test_models")
    try:
        moods = ["happy"]  # Test with just one mood
        download_mood_models(moods, models_dir)
        
        # Check that model file was created
        model_path = models_dir / "mood_happy-musicnn-msd-1.pb"
        assert model_path.exists()
    finally:
        # Cleanup
        if models_dir.exists():
            import shutil
            shutil.rmtree(models_dir)


def test_mood_extraction_with_mock():
    """Test mood extraction with mocked TensorFlow."""
    from unittest.mock import patch, MagicMock
    import numpy as np
    
    # Mock TensorFlow components
    with patch('essentia.standard.TensorflowPredictMusiCNN') as mock_predict:
        mock_predict.return_value = MagicMock(return_value=np.array([0.8]))
        
        from scripts.extract_essentia import extract_mood
        
        # Create dummy audio
        audio = np.random.randn(16000)  # 1 second at 16kHz
        
        # Test extraction
        scores = extract_mood(audio, models_dir=Path("test_models"))
        
        assert isinstance(scores, dict)
        assert "happy" in scores
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mood_extraction.py -v`
Expected: FAIL with "ImportError" or "ModuleNotFoundError" (functions don't exist yet)

- [ ] **Step 3: Commit**

```bash
git add tests/test_mood_extraction.py
git commit -m "test: add failing tests for mood extraction"
```

---

### Task 3: Implement Model Download Function

**Files:**
- Modify: `scripts/extract_essentia.py`

**Interfaces:**
- Consumes: None
- Produces: `download_mood_models(moods: list[str], models_dir: Path) -> None`

- [ ] **Step 1: Add model download function**

Add to `scripts/extract_essentia.py` before the `extract` function:

```python
import os
import urllib.request
from pathlib import Path


def download_mood_models(moods: list[str], models_dir: Path) -> None:
    """Download mood classification models from Essentia servers.
    
    Args:
        moods: List of mood names to download (e.g., ["happy", "sad"])
        models_dir: Directory to store downloaded models
        
    Raises:
        RuntimeError: If download fails for any model
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "https://essentia.upf.edu/models/classifiers"
    
    for mood in moods:
        model_file = f"mood_{mood}-musicnn-msd-1.pb"
        model_path = models_dir / model_file
        
        if model_path.exists():
            continue
            
        url = f"{base_url}/mood_{mood}/{model_file}"
        
        try:
            print(f"Downloading mood model for {mood}...")
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download mood model for {mood}: {e}")
```

- [ ] **Step 2: Run tests to verify model download works**

Run: `pytest tests/test_mood_extraction.py::test_mood_models_download -v`
Expected: PASS (downloads model to test_models directory)

- [ ] **Step 3: Clean up test artifacts**

Run: `rm -rf test_models`

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_essentia.py
git commit -m "feat: add mood model download function"
```

---

### Task 4: Implement Mood Extraction Function

**Files:**
- Modify: `scripts/extract_essentia.py`

**Interfaces:**
- Consumes: `download_mood_models` from Task 3
- Produces: `extract_mood(audio, models_dir: Path) -> dict[str, float]`

- [ ] **Step 1: Add mood extraction function**

Add to `scripts/extract_essentia.py` after `download_mood_models`:

```python
def extract_mood(audio, models_dir: Path = Path("models")) -> dict[str, float]:
    """Extract mood scores using MusiCNN classifiers.
    
    Args:
        audio: Audio signal (numpy array, 44.1kHz)
        models_dir: Directory containing mood models
        
    Returns:
        Dictionary mapping mood names to scores (0-1)
    """
    moods = ["happy", "sad", "aggressive", "relaxed", "electronic", "party", "acoustic"]
    scores = {}
    
    # Download models if missing
    try:
        download_mood_models(moods, models_dir)
    except RuntimeError as e:
        print(f"Warning: Failed to download mood models: {e}", file=sys.stderr)
        return {mood: 0.0 for mood in moods}
    
    # Resample audio to 16kHz for MusiCNN
    try:
        from essentia.standard import Resample
        audio_16k = Resample(inputSampleRate=44100, outputSampleRate=16000)(audio)
    except Exception as e:
        print(f"Warning: Audio resampling failed: {e}", file=sys.stderr)
        return {mood: 0.0 for mood in moods}
    
    # Extract mood scores
    for mood in moods:
        model_path = models_dir / f"mood_{mood}-musicnn-msd-1.pb"
        
        if not model_path.exists():
            print(f"Warning: Mood model not found for {mood}", file=sys.stderr)
            scores[mood] = 0.0
            continue
        
        try:
            from essentia.standard import TensorflowPredictMusiCNN
            activations = TensorflowPredictMusiCNN(graphFilename=str(model_path))(audio_16k)
            scores[mood] = float(activations.mean())
        except Exception as e:
            print(f"Warning: Mood extraction failed for {mood}: {e}", file=sys.stderr)
            scores[mood] = 0.0
    
    return scores
```

- [ ] **Step 2: Run tests to verify mood extraction works**

Run: `pytest tests/test_mood_extraction.py::test_mood_extraction_with_mock -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/extract_essentia.py
git commit -m "feat: add mood extraction function"
```

---

### Task 5: Integrate Mood Extraction into Main Extract Function

**Files:**
- Modify: `scripts/extract_essentia.py:32-116`

**Interfaces:**
- Consumes: `extract_mood` from Task 4
- Produces: Updated JSON output with "mood" section

- [ ] **Step 1: Add mood extraction to extract function**

Modify the `extract` function in `scripts/extract_essentia.py` to call `extract_mood`:

```python
def extract(audio_path: str, output_path: str) -> None:
    """Run Essentia extraction and write JSON sidecar."""
    # ... existing code up to building result ...
    
    # Build output
    result = {
        "version": "1.0",
        "duration_sec": round(duration, 2),
        "loudness": {
            "integrated": round(float(integrated), 2),
            "range": round(float(loud_range), 2),
        },
        "tempo": {
            "bpm": round(float(tempo), 1),
            "confidence": round(float(beats_confidence), 3),
        },
        "key": {
            "key": str(key),
            "mode": str(scale),
            "scale": f"{key} {scale}",
            "confidence": round(float(key_confidence), 3),
        },
        "spectral": {
            "centroid": round(float(c), 2),
            "rolloff": round(float(ro), 2),
            "flatness": round(float(fl), 4),
        },
        "rhythm": {
            "danceability": round(float(danceability), 4),
            "onset_rate": round(float(onset_rate), 2),
        },
    }
    
    # Add mood extraction
    try:
        mood_scores = extract_mood(audio)
        result["mood"] = {k: round(v, 4) for k, v in mood_scores.items()}
    except Exception as e:
        print(f"Warning: Mood extraction failed: {e}", file=sys.stderr)
        result["mood"] = {
            "happy": 0.0, "sad": 0.0, "aggressive": 0.0, "relaxed": 0.0,
            "electronic": 0.0, "party": 0.0, "acoustic": 0.0
        }
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
```

- [ ] **Step 2: Test with real audio file**

Run: `python scripts/extract_essentia.py <test_audio_file> /tmp/test_output.json`
Expected: JSON output includes "mood" section with 7 float values

- [ ] **Step 3: Verify JSON structure**

Run: `cat /tmp/test_output.json | python -m json.tool`
Expected: Valid JSON with mood section

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_essentia.py
git commit -m "feat: integrate mood extraction into main extraction pipeline"
```

---

### Task 6: Update generate_playlist.py for Mood Axes

**Files:**
- Modify: `generate_playlist.py:11-36`

**Interfaces:**
- Consumes: mood section from feature_json
- Produces: Updated AXIS_NAMES and extract_features function

- [ ] **Step 1: Update AXIS_NAMES**

Modify `generate_playlist.py` to include mood axes:

```python
AXIS_NAMES = [
    "duration_sec", "loudness.integrated", "loudness.range",
    "tempo.bpm", "tempo.confidence", "key.confidence",
    "spectral.centroid", "spectral.rolloff", "spectral.flatness",
    "rhythm.danceability", "rhythm.onset_rate",
    "mood.happy", "mood.sad", "mood.aggressive", "mood.relaxed",
    "mood.electronic", "mood.party", "mood.acoustic",
]
```

- [ ] **Step 2: Update extract_features function**

Modify `extract_features` in `generate_playlist.py` to include mood scores:

```python
def extract_features(row):
    raw = json.loads(row["feature_json"] or "{}")
    vec = []
    vec.append(float(raw.get("duration_sec", 0)))
    loud = raw.get("loudness", {})
    vec.append(float(loud.get("integrated", 0)))
    vec.append(float(loud.get("range", 0)))
    tempo = raw.get("tempo", {})
    vec.append(float(tempo.get("bpm", 0)))
    vec.append(float(tempo.get("confidence", 0)))
    key = raw.get("key", {})
    vec.append(float(key.get("confidence", 0)))
    spec = raw.get("spectral", {})
    vec.append(float(spec.get("centroid", 0)))
    vec.append(float(spec.get("rolloff", 0)))
    vec.append(float(spec.get("flatness", 0)))
    rhythm = raw.get("rhythm", {})
    vec.append(float(rhythm.get("danceability", 0)))
    vec.append(float(rhythm.get("onset_rate", 0)))
    
    # Add mood features
    mood = raw.get("mood", {})
    vec.append(float(mood.get("happy", 0)))
    vec.append(float(mood.get("sad", 0)))
    vec.append(float(mood.get("aggressive", 0)))
    vec.append(float(mood.get("relaxed", 0)))
    vec.append(float(mood.get("electronic", 0)))
    vec.append(float(mood.get("party", 0)))
    vec.append(float(mood.get("acoustic", 0)))
    
    return vec, raw
```

- [ ] **Step 3: Test playlist generation**

Run: `python generate_playlist.py`
Expected: Playlist generated with mood axes included in distance calculations

- [ ] **Step 4: Verify distance calculations include mood axes**

Check output for axis statistics including mood axes:
```
Axis mood.happy: mean=X.XX, stddev=X.XX
Axis mood.sad: mean=X.XX, stddev=X.XX
...
```

- [ ] **Step 5: Commit**

```bash
git add generate_playlist.py
git commit -m "feat: add mood axes to playlist generation"
```

---

### Task 7: Update Documentation

**Files:**
- Modify: `docs/SCHEMA.md:104-119`

**Interfaces:**
- Consumes: None
- Produces: Updated documentation reflecting mood extraction implementation

- [ ] **Step 1: Update SCHEMA.md mood section**

Update the mood descriptors section in `docs/SCHEMA.md` to reflect implementation:

```markdown
### highlevel.mood

Mood descriptors are neural-network-derived probability scores extracted using
Essentia's pre-trained MusiCNN TensorFlow classifiers. Each value represents
the model's confidence that the track belongs to a particular mood category.
Together they form a multi-dimensional mood space.

| Descriptor | Range | Branching Role |
|---|---|---|
| `highlevel.mood_happy` | 0–1 | **Joy axis.** High values indicate upbeat, major-key, optimistic tracks. Useful for building a "feel-good" progression or deliberately descending from joy into melancholy. |
| `highlevel.mood_sad` | 0–1 | **Melancholy axis.** High values indicate minor-key, slow, emotionally heavy tracks. Pairs with happy for emotional contrast transitions. |
| `highlevel.mood_aggressive` | 0–1 | **Intensity axis.** High values indicate loud, fast, distorted, or confrontational tracks. Drives transitions between calm and aggressive energy states. |
| `highlevel.mood_relaxed` | 0–1 | **Calm axis.** High values indicate gentle, slow, low-energy tracks. The complement to aggressive — useful for wind-down progressions at the end of a playlist. |
| `highlevel.mood_electronic` | 0–1 | **Synthetic axis.** High values indicate synthesised, produced, or electronically-generated sounds. Useful for genre-blending transitions between organic and electronic sections. |
| `highlevel.mood_party` | 0–1 | **Celebration axis.** High values indicate high-energy, social, danceable tracks. Overlaps with danceability but captures a broader social/energetic vibe rather than just rhythmic regularity. |
| `highlevel.mood_acoustic` | 0–1 | **Organic axis.** High values indicate naturally recorded, unplugged, or instrument-driven tracks. The complement to electronic — drives transitions between synthetic and acoustic textures. |

**Implementation Notes:**
- Requires TensorFlow >= 2.10.0
- Models download automatically from Essentia servers on first use
- Audio is resampled to 16kHz for MusiCNN models
- Extraction adds ~7-14 seconds per track
- Graceful fallback to 0.0 if extraction fails for any mood
```

- [ ] **Step 2: Commit**

```bash
git add docs/SCHEMA.md
git commit -m "docs: update mood descriptor documentation"
```

---

### Task 8: Integration Testing

**Files:**
- Modify: `tests/run_qa.sh` (add mood extraction verification)

**Interfaces:**
- Consumes: All previous tasks
- Produces: Updated QA script with mood extraction tests

- [ ] **Step 1: Add mood extraction test to QA script**

Add to `tests/run_qa.sh` after existing tests:

```bash
# Test mood extraction
echo "Testing mood extraction..."
if python -c "from scripts.extract_essentia import extract_mood; print('Mood extraction available')" 2>/dev/null; then
    echo "✓ Mood extraction function available"
else
    echo "✗ Mood extraction function not available"
    exit 1
fi
```

- [ ] **Step 2: Run full QA suite**

Run: `bash tests/run_qa.sh`
Expected: All tests pass including mood extraction verification

- [ ] **Step 3: Commit**

```bash
git add tests/run_qa.sh
git commit -m "test: add mood extraction verification to QA suite"
```

---

### Task 9: Final Verification

**Files:**
- None (verification only)

**Interfaces:**
- Consumes: All previous tasks
- Produces: Verified working mood extraction feature

- [ ] **Step 1: Clean and reinstall dependencies**

Run: `pip install -r requirements.txt --force-reinstall`
Expected: TensorFlow installs successfully

- [ ] **Step 2: Test full pipeline with real audio**

Run: `python run.py <music_directory> database/playlist.db`
Expected: Tracks ingested with mood scores in feature_json

- [ ] **Step 3: Verify mood scores in database**

Run: `sqlite3 database/playlist.db "SELECT feature_json FROM tracks LIMIT 1;" | python -m json.tool`
Expected: JSON includes "mood" section with 7 float values

- [ ] **Step 4: Generate playlist with mood axes**

Run: `python generate_playlist.py`
Expected: Playlist generated using mood axes in distance calculations

- [ ] **Step 5: Verify playlist includes mood information**

Check `playlist_summary.txt` for mood-related statistics

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete mood description feature implementation"
```

---

## Success Criteria

1. All 7 mood scores extracted for test tracks
2. Mood axes integrated into distance calculations
3. No regression in existing playlist generation
4. Extraction time acceptable (< 15 seconds per track)
5. Graceful handling of missing models or TensorFlow
6. All tests pass including new mood extraction tests
7. Documentation updated to reflect implementation