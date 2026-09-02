# Mood Description Feature Design

> **Date:** 2026-09-02  
> **Status:** Approved  
> **Scope:** Add 7 mood axes to similarity space using Essentia TensorFlow classifiers

## Overview

Add mood classification to the playlist progression prototype using Essentia's pre-trained TensorFlow classifiers. This adds 7 new dimensions to the similarity space: happy, sad, aggressive, relaxed, electronic, party, and acoustic.

## Requirements

1. Extract 7 mood scores (0-1) for each audio track
2. Use Essentia's pre-trained MusiCNN TensorFlow classifiers
3. TensorFlow becomes a required dependency
4. Models download automatically on first use
5. Mood scores integrate into existing distance calculations
6. Graceful fallback if extraction fails

## Architecture

### Components Modified

1. **`scripts/extract_essentia.py`** - Add mood extraction logic
2. **`generate_playlist.py`** - Add mood axes to AXIS_NAMES and extract_features
3. **`requirements.txt`** - Add tensorflow dependency
4. **`docs/SCHEMA.md`** - Update documentation

### New Components

1. **Model directory** - `models/` for downloaded mood classifiers
2. **Model download logic** - Automatic download from Essentia servers

## Implementation Details

### 1. Model Management

Create `models/` directory for storing downloaded mood classifier models. Each mood requires two files:
- `<mood>-musicnn-msd-1.pb` (TensorFlow graph)
- `<mood>-musicnn-msd-1.json` (metadata)

Models to download:
- `mood_happy-musicnn-msd-1.pb` + `.json`
- `mood_sad-musicnn-msd-1.pb` + `.json`
- `mood_aggressive-musicnn-msd-1.pb` + `.json`
- `mood_relaxed-musicnn-msd-1.pb` + `.json`
- `mood_electronic-musicnn-msd-1.pb` + `.json`
- `mood_party-musicnn-msd-1.pb` + `.json`
- `mood_acoustic-musicnn-msd-1.pb` + `.json`

Download URLs:
```
https://essentia.upf.edu/models/classifiers/mood_<mood>/mood_<mood>-musicnn-msd-1.pb
https://essentia.upf.edu/models/classifiers/mood_<mood>/mood_<mood>-musicnn-msd-1.json
```

### 2. Extraction Logic

Add to `extract_essentia.py`:

```python
def extract_mood(audio, models_dir="models"):
    """Extract mood scores using MusiCNN classifiers."""
    import os
    import urllib.request
    
    moods = ["happy", "sad", "aggressive", "relaxed", "electronic", "party", "acoustic"]
    scores = {}
    
    # Resample audio to 16kHz for MusiCNN
    from essentia.standard import Resample
    audio_16k = Resample(inputSampleRate=44100, outputSampleRate=16000)(audio)
    
    for mood in moods:
        model_path = os.path.join(models_dir, f"mood_{mood}-musicnn-msd-1.pb")
        
        # Download if missing
        if not os.path.exists(model_path):
            os.makedirs(models_dir, exist_ok=True)
            url = f"https://essentia.upf.edu/models/classifiers/mood_{mood}/mood_{mood}-musicnn-msd-1.pb"
            try:
                urllib.request.urlretrieve(url, model_path)
            except Exception as e:
                print(f"Warning: Failed to download mood model for {mood}: {e}", file=sys.stderr)
                scores[mood] = 0.0
                continue
        
        # Run prediction
        try:
            from essentia.standard import TensorflowPredictMusiCNN
            activations = TensorflowPredictMusiCNN(graphFilename=model_path)(audio_16k)
            scores[mood] = float(activations.mean())
        except Exception as e:
            print(f"Warning: Mood extraction failed for {mood}: {e}", file=sys.stderr)
            scores[mood] = 0.0
    
    return scores
```

### 3. Output Format

Add `"mood"` section to JSON output:

```json
{
  "version": "1.0",
  "duration_sec": ...,
  "loudness": { ... },
  "tempo": { ... },
  "key": { ... },
  "spectral": { ... },
  "rhythm": { ... },
  "mood": {
    "happy": 0.85,
    "sad": 0.12,
    "aggressive": 0.23,
    "relaxed": 0.67,
    "electronic": 0.45,
    "party": 0.78,
    "acoustic": 0.34
  }
}
```

### 4. Dependency Management

Add to `requirements.txt`:
```
tensorflow>=2.10.0
```

Note: TensorFlow installation may vary by platform. On macOS, TensorFlow supports both Intel and Apple Silicon. On Linux, GPU support is available.

### 5. Integration with generate_playlist.py

Update AXIS_NAMES to include mood axes:
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

Update extract_features to include mood scores:
```python
def extract_features(row):
    raw = json.loads(row["feature_json"] or "{}")
    vec = []
    # ... existing feature extraction ...
    
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

## Error Handling

1. **TensorFlow not installed:** Exit with clear error message and installation instructions
2. **Model download fails:** Log warning, continue with mood score = 0.0 for that mood
3. **Extraction fails:** Log warning, continue with mood score = 0.0 for that mood
4. **Missing mood scores:** extract_features defaults to 0.0 for missing mood values

## Testing Strategy

1. **Unit tests:** Test mood extraction with sample audio files
2. **Integration tests:** Test full pipeline with mood extraction enabled
3. **QA script updates:** Add mood extraction verification to tests/run_qa.sh
4. **Performance tests:** Measure extraction time with mood models

## Performance Considerations

- Each mood model adds ~1-2 seconds extraction time
- Total mood extraction: ~7-14 seconds per track
- Model download: ~20MB total (7 models × ~3MB each)
- Memory usage: ~500MB during extraction (TensorFlow overhead)

## Rollback Plan

If mood extraction causes issues:
1. Remove mood axes from AXIS_NAMES in generate_playlist.py
2. Remove mood extraction from extract_essentia.py
3. Keep mood section in JSON output (zeros) for backward compatibility
4. Document mood extraction as optional enhancement

## Success Criteria

1. All 7 mood scores extracted for test tracks
2. Mood axes integrated into distance calculations
3. No regression in existing playlist generation
4. Extraction time acceptable (< 15 seconds per track)
5. Graceful handling of missing models or TensorFlow