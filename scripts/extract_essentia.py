#!/usr/bin/env python3
"""
Extract Essentia DSP features from an audio file and write a JSON sidecar.

Usage:
    python3 extract_essentia.py <audio_path> <output_path>

Exit codes:
    0  — Success (output file written)
    1  — Runtime error (message on stderr)
    2  — Bad arguments

Output format (see INTEGRATION.md):
    {
      "version": "1.0",
      "duration_sec": ...,
      "loudness": { "integrated": ..., "range": ... },
      "tempo": { "bpm": ..., "confidence": ... },
      "key": { "key": "...", "mode": "...", "scale": "...", "confidence": ... },
      "spectral": { "centroid": ..., "bandwidth": ..., "rolloff": ..., "flatness": ... },
      "rhythm": { "danceability": ..., "onset_rate": ... }
    }
"""

import json
import os
import urllib.request
from pathlib import Path
import sys

# Essentia imports are deferred to runtime so import errors appear on stderr
# rather than crashing at module load time.

# Sidecar schema version. Bump when extraction output changes meaningfully
# (e.g. 1.0 -> 1.1: spectral descriptors switched from full-track FFT to
# frame-wise means). The ingest pipeline re-extracts rows whose stored
# version differs from this constant.
EXTRACTOR_VERSION = "1.1"


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
        meta_file = f"mood_{mood}-musicnn-msd-1.json"
        model_path = models_dir / model_file
        meta_path = models_dir / meta_file

        if model_path.exists() and meta_path.exists():
            continue

        mood_url = f"{base_url}/mood_{mood}"

        try:
            print(f"Downloading mood model for {mood}...")
            if not model_path.exists():
                urllib.request.urlretrieve(f"{mood_url}/{model_file}", model_path)
            if not meta_path.exists():
                urllib.request.urlretrieve(f"{mood_url}/{meta_file}", meta_path)
        except Exception as e:
            raise RuntimeError(f"Failed to download mood model for {mood}: {e}")


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
        
        # Determine which column holds the target mood class from model metadata
        target_idx = 0
        try:
            import json as _json
            meta_path = models_dir / f"mood_{mood}-musicnn-msd-1.json"
            meta = _json.loads(meta_path.read_text())
            classes = meta.get("classes") or []
            target_idx = next((i for i, c in enumerate(classes) if c.lower() == mood.lower()), 0)
        except Exception:
            pass  # fall back to column 0

        try:
            from essentia.standard import TensorflowPredictMusiCNN
            import numpy as np
            activations = TensorflowPredictMusiCNN(graphFilename=str(model_path))(audio_16k)
            arr = np.asarray(activations)
            scores[mood] = float(arr[:, target_idx].mean())
        except Exception as e:
            print(f"Warning: Mood extraction failed for {mood}: {e}", file=sys.stderr)
            scores[mood] = 0.0
    
    return scores


def compute_spectral_descriptors(audio) -> tuple[float, float, float]:
    """Frame-wise spectral means (centroid, rolloff, flatness).

    Correct Essentia usage is FrameGenerator -> Windowing -> Spectrum per
    frame, aggregating descriptor means. Calling Spectrum() on the full
    track buffer is a full-track FFT flaw (same class as the key bug fixed
    via KeyExtractor in ba7c8d3): it yields one global spectrum with
    exaggerated dynamic range (flatness≈0) instead of representative means.
    """
    import essentia.standard as es

    frame_gen = es.FrameGenerator(
        audio, frameSize=2048, hopSize=1024, startFromZero=True
    )
    window = es.Windowing(type="hann")
    spectrum = es.Spectrum()
    centroid_algo = es.Centroid(range=22050)
    rolloff_algo = es.RollOff(cutoff=0.85)
    flatness_algo = es.Flatness()

    centroids: list[float] = []
    rolloffs: list[float] = []
    flatnesses: list[float] = []

    for frame in frame_gen:
        spec = spectrum(window(frame))
        centroids.append(float(centroid_algo(spec)))
        rolloffs.append(float(rolloff_algo(spec)))
        flatnesses.append(float(flatness_algo(spec)))

    if not centroids:
        return (0.0, 0.0, 0.0)

    import numpy as np

    return (
        float(np.mean(centroids)),
        float(np.mean(rolloffs)),
        float(np.mean(flatnesses)),
    )


def extract_dsp(audio) -> dict:
    """Extract DSP-only features (loudness, tempo, spectral, key, rhythm, duration).

    Returns a dict WITHOUT mood key — mood is added separately.
    """
    import essentia.standard as es
    import numpy as np

    duration = len(audio) / 44100.0  # assume 44.1kHz

    # Loudness (LoudnessEBUR128 requires stereo; duplicate mono channel)
    stereo = np.column_stack((audio, audio))
    ebu = es.LoudnessEBUR128()
    _, _, integrated, loud_range = ebu(stereo)

    # Tempo and rhythm
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    tempo, beats, beats_confidence, *_ = rhythm_extractor(audio)
    danceability_result = es.Danceability()(audio)
    danceability = danceability_result[0] if isinstance(danceability_result, tuple) else danceability_result

    # Spectral descriptors (frame-wise means — not full-track FFT)
    c, ro, fl = compute_spectral_descriptors(audio)

    # Key (frame-wise via KeyExtractor)
    key, scale, key_confidence = es.KeyExtractor()(audio)

    # Onset rate
    onset_result = es.OnsetRate()(audio)
    onset_rate = onset_result[1] if isinstance(onset_result, tuple) else onset_result

    # Build output (NO mood key)
    result = {
        "version": EXTRACTOR_VERSION,
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
    return result


def extract(audio_path: str, output_path: str, include_mood: bool = True) -> None:
    """Run Essentia extraction and write JSON sidecar."""
    try:
        import essentia
        import essentia.standard as es
    except ImportError as e:
        print(f"Error: Essentia not installed: {e}", file=sys.stderr)
        sys.exit(1)

    # Load audio
    try:
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)

    if len(audio) == 0:
        print("Error: empty audio file", file=sys.stderr)
        sys.exit(1)

    result = extract_dsp(audio)

    # Add mood extraction if requested
    if include_mood:
        try:
            mood_scores = extract_mood(audio)
            result["mood"] = {k: round(v, 4) for k, v in mood_scores.items()}
        except Exception as e:
            print(f"Warning: Mood extraction failed: {e}", file=sys.stderr)
            # DO NOT write zeros: omit mood key so pipeline stores NULL+retry

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def extract_mood_only(audio_path: str, output_path: str) -> None:
    """Load existing sidecar, extract mood, merge, and rewrite.

    Used for --mood-only mode: retries mood extraction on a track
    that previously had mood NULL.
    """
    try:
        import essentia
        import essentia.standard as es
    except ImportError as e:
        print(f"Error: Essentia not installed: {e}", file=sys.stderr)
        sys.exit(1)

    # Load existing sidecar
    output = Path(output_path)
    if not output.exists():
        print(f"Error: sidecar not found for mood-only: {output_path}", file=sys.stderr)
        sys.exit(1)

    existing = json.loads(output.read_text())

    # Load audio
    try:
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
    except Exception as e:
        print(f"Error loading audio: {e}", file=sys.stderr)
        sys.exit(1)

    if len(audio) == 0:
        print("Error: empty audio file", file=sys.stderr)
        sys.exit(1)

    # Extract mood
    try:
        mood_scores = extract_mood(audio)
        existing["mood"] = {k: round(v, 4) for k, v in mood_scores.items()}
    except Exception as e:
        print(f"Warning: Mood extraction failed: {e}", file=sys.stderr)
        # DO NOT write zeros: omit mood key so pipeline stores NULL+retry

    with open(output_path, "w") as f:
        json.dump(existing, f, indent=2)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("audio_path", nargs="?")
    p.add_argument("output_path", nargs="?")
    p.add_argument("--prefetch", action="store_true", help="Download mood classification models")
    p.add_argument("--models-dir", default="models", help="Directory to store downloaded models")
    p.add_argument("--no-mood", action="store_true", help="Skip mood extraction (DSP only)")
    p.add_argument("--mood-only", action="store_true", help="Run only mood extraction on existing sidecar")
    args = p.parse_args()

    if args.prefetch:
        download_mood_models(
            ["happy", "sad", "aggressive", "relaxed", "electronic", "party", "acoustic"],
            Path(args.models_dir),
        )
        return

    if not args.audio_path or not args.output_path:
        print(
            f"Usage: {sys.argv[0]} <audio_path> <output_path> [--prefetch] [--no-mood] [--mood-only] [--models-dir DIR]",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.mood_only:
        extract_mood_only(args.audio_path, args.output_path)
    else:
        extract(args.audio_path, args.output_path, include_mood=not args.no_mood)


if __name__ == "__main__":
    main()
