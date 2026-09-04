#!/usr/bin/env python3
"""
Extract a CLAP embedding from an audio file and write a JSON sidecar.

Usage:
    python3 extract_clap.py <audio_path> <output_path>
    python3 extract_clap.py --batch MANIFEST

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
from pathlib import Path


def _extract_one(audio_path: str, output_path: str) -> None:
    """Extract CLAP embedding for a single audio file.

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written

    Raises:
        ImportError: if laion_clap is not installed
        Exception: if model loading or embedding extraction fails
    """
    try:
        import laion_clap
    except ImportError as e:
        raise

    # Load model
    try:
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()  # loads default checkpoint
    except Exception as e:
        raise

    # Get audio embedding
    try:
        embedding = model.get_audio_embedding_from_filelist([audio_path])
    except Exception as e:
        raise

    # Convert to list of floats
    if hasattr(embedding, "tolist"):
        emb_list = embedding[0].tolist()  # batch dim
    else:
        emb_list = [float(x) for x in embedding[0]]

    # Validate dimension
    if len(emb_list) != 512:
        raise RuntimeError(f"unexpected embedding dimension: {len(emb_list)} (expected 512)")

    result = {
        "version": "1.0",
        "model": "clap-v1",
        "embedding": emb_list,
    }

    with open(output_path, "w") as f:
        json.dump(result, f)


def extract(audio_path: str, output_path: str) -> None:
    """Run CLAP extraction and write JSON sidecar (CLI wrapper).

    Args:
        audio_path: path to the input audio file
        output_path: path where the JSON sidecar will be written

    Exits:
        1 — on runtime error
    """
    try:
        _extract_one(audio_path, output_path)
    except Exception as e:
        print(f"Error extracting CLAP embedding: {e}", file=sys.stderr)
        sys.exit(1)


def run_clap_batch_manifest(manifest_path: str) -> dict:
    """Process a manifest of audio files, reusing one loaded CLAP model.

    Manifest JSON is a list of {audio_path, output_path}. Individual track
    failures are caught and recorded; the batch always exits 0.

    Args:
        manifest_path: path to manifest JSON (list of {audio_path, output_path})

    Returns:
        Summary dict: {"ok": [str], "failed": [{"output": str, "error": str}]}
    """
    try:
        import laion_clap
    except ImportError as e:
        print(f"Error: CLAP not installed: {e}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(Path(manifest_path).read_text())
    try:
        model = laion_clap.CLAP_Module(enable_fusion=False)
        model.load_ckpt()
    except Exception as e:
        print(f"Error loading CLAP model: {e}", file=sys.stderr)
        sys.exit(1)

    ok, failed = [], []
    for entry in manifest:
        try:
            audio_path = entry["audio_path"]
            output_path = entry["output_path"]

            # Check file existence before extraction
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # Get audio embedding
            embedding = model.get_audio_embedding_from_filelist([audio_path])

            # Convert to list of floats
            if hasattr(embedding, "tolist"):
                emb_list = embedding[0].tolist()
            else:
                emb_list = [float(x) for x in embedding[0]]

            # Validate dimension
            if len(emb_list) != 512:
                raise RuntimeError(f"unexpected embedding dimension: {len(emb_list)} (expected 512)")

            result = {
                "version": "1.0",
                "model": "clap-v1",
                "embedding": emb_list,
            }

            with open(output_path, "w") as f:
                json.dump(result, f)

            ok.append(output_path)
        except Exception as e:
            failed.append({"output": entry["output_path"], "error": str(e)})
            print(f"batch failed {entry['audio_path']}: {e}", file=sys.stderr)

    summary = {"ok": ok, "failed": failed}
    Path(str(manifest_path) + ".summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("audio_path", nargs="?", help="Input audio path (required for single-file mode)")
    p.add_argument("output_path", nargs="?", help="Output JSON sidecar path (required for single-file mode)")
    p.add_argument("--batch", metavar="MANIFEST", help="Process a batch manifest of audio files")
    args = p.parse_args()

    if args.batch:
        run_clap_batch_manifest(args.batch)
        return

    if not args.audio_path or not args.output_path:
        print(
            f"Usage: {sys.argv[0]} <audio_path> <output_path> [--batch MANIFEST]",
            file=sys.stderr,
        )
        sys.exit(2)

    extract(args.audio_path, args.output_path)


if __name__ == "__main__":
    main()
