#!/usr/bin/env python3
"""Download sentence-transformers model with extended timeout.

This script uses huggingface_hub to download the model with better timeout handling.
"""

import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed.")
    print("Install with: pip install huggingface_hub")
    sys.exit(1)


def download_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", timeout: int = 300):
    """Download model using huggingface_hub with extended timeout.

    Args:
        model_name: Full model name (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        timeout: Download timeout in seconds (default: 300 = 5 minutes)
    """
    print(f"Downloading model: {model_name}")
    print(f"Timeout: {timeout} seconds")
    print("This may take several minutes depending on your connection...\n")

    # Set environment variable for timeout
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)

    try:
        # Download the model
        cache_dir = snapshot_download(
            repo_id=model_name,
            cache_dir=None,  # Use default cache location
            resume_download=True,  # Resume if partially downloaded
            local_files_only=False
        )

        print(f"\n✓ Successfully downloaded model!")
        print(f"Cache location: {cache_dir}")

        # Verify the model can be loaded
        print("\nVerifying model can be loaded...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            test_embedding = model.encode("Test sentence")
            print(f"✓ Model verification successful! Embedding dimension: {len(test_embedding)}")
        except Exception as e:
            print(f"⚠ Model downloaded but verification failed: {e}")
            print("You may need to restart your Python process to use the model.")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative options:")
        print("1. Try again with longer timeout: python3 scripts/download_model.py --timeout 600")
        print("2. Use manual download instructions: cat scripts/download_model_instructions.md")
        print("3. Use huggingface-cli: huggingface-cli download sentence-transformers/all-MiniLM-L6-v2")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download sentence-transformers model")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Full model name (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download timeout in seconds (default: 300)"
    )

    args = parser.parse_args()

    success = download_model(args.model, args.timeout)
    sys.exit(0 if success else 1)

