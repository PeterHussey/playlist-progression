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
import sys

# Essentia imports are deferred to runtime so import errors appear on stderr
# rather than crashing at module load time.


def extract(audio_path: str, output_path: str) -> None:
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

    duration = len(audio) / 44100.0  # assume 44.1kHz

    # Loudness
    loudness = es.Loudness()
    integrated = loudness(audio)
    loud_range = es.LoudnessRange()(audio)

    # Tempo and rhythm
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
    tempo, beats, beats_confidence, _ = rhythm_extractor(audio)
    danceability = es.Danceability()(audio)

    # Key
    key_extractor = es.Key()
    key, scale, key_confidence = key_extractor(audio)

    # Spectral descriptors
    spectrum = es.Spectrum()
    centroid = es.Centroid(range=22050)
    bandwidth = es.Bandwidth(range=22050)
    rolloff = es.Rolloff(spectrum, cutoff=0.85)
    flatness = es.Flatness()

    spec = spectrum(audio)
    c = centroid(spec)
    bw = bandwidth(spec)
    ro = rolloff(spec)
    fl = flatness(spec)

    # Onset rate
    onset_rate = es.OnsetRate()(audio)

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
            "bandwidth": round(float(bw), 2),
            "rolloff": round(float(ro), 2),
            "flatness": round(float(fl), 4),
        },
        "rhythm": {
            "danceability": round(float(danceability), 4),
            "onset_rate": round(float(onset_rate), 2),
        },
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <audio_path> <output_path>",
            file=sys.stderr,
        )
        sys.exit(2)

    audio_path = sys.argv[1]
    output_path = sys.argv[2]

    extract(audio_path, output_path)


if __name__ == "__main__":
    main()
