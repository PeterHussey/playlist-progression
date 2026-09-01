#!/usr/bin/env python3
"""
Extract a CLAP embedding from an audio file and write a JSON sidecar.

Usage:
    python3 extract_clap.py <audio_path> <output_path>

Exit codes:
    0  — Success (output file written)
    1  — Runtime error (message on stderr)
    2  — Bad arguments

Output format (see INTEGRATION.md):
    {
      "version": "1.0",
      "model": "clap-v1",
      "embedding": [0.012, -0.034, ..., -0.021]  // 512 floats
    }
"""

import json
import sys


def extract(audio_path: str, output_path: str) -> None:
    """Run CLAP extraction and write JSON sidecar."""
    try:
        import laion_clap
    except ImportError as e:
        print(f"Error: CLAP not installed: {e}", file=sys.stderr)
        sys.exit(1)

    # Load model
    try:
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # loads default checkpoint
    except Exception as e:
        print(f"Error loading CLAP model: {e}", file=sys.stderr)
        sys.exit(1)

    # Get audio embedding
    try:
        audio_data, _ = model.load_audio([audio_path], sr=48000)
        embedding = model.get_audio_embedding_from_data(x=audio_data, numpy=True)
    except Exception as e:
        print(f"Error extracting CLAP embedding: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert to list of floats
    if hasattr(embedding, "tolist"):
        emb_list = embedding[0].tolist()  # batch dim
    else:
        emb_list = [float(x) for x in embedding[0]]

    # Validate dimension
    if len(emb_list) != 512:
        print(
            f"Error: unexpected embedding dimension: {len(emb_list)} (expected 512)",
            file=sys.stderr,
        )
        sys.exit(1)

    result = {
        "version": "1.0",
        "model": "clap-v1",
        "embedding": emb_list,
    }

    with open(output_path, "w") as f:
        json.dump(result, f)


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
