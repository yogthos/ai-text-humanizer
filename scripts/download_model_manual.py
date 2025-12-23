#!/usr/bin/env python3
"""Manual download script for sentence-transformers model with increased timeout.

This script downloads the all-MiniLM-L6-v2 model with extended timeouts
to handle slow connections.
"""

import os
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError as e:
    print(f"Error: Missing required package: {e}")
    print("Please install: pip install sentence-transformers requests")
    sys.exit(1)


def download_model_with_retry(model_name: str = 'all-MiniLM-L6-v2', timeout: int = 60):
    """Download model with extended timeout and retry logic.

    Args:
        model_name: Name of the model to download
        timeout: Request timeout in seconds (default: 60)
    """
    print(f"Downloading model: {model_name}")
    print(f"Using timeout: {timeout} seconds")
    print("This may take several minutes depending on your connection...")

    # Configure retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )

    # Create session with retry adapter
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    try:
        # Load model with custom session and timeout
        # SentenceTransformer will use the requests session if we set it up
        print("\nInitializing model download...")

        # Set environment variable for longer timeout
        os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(timeout)

        # Download the model
        model = SentenceTransformer(model_name, request_timeout=timeout)

        print(f"\n✓ Successfully downloaded and loaded model: {model_name}")
        print(f"Model location: {model._model_card_vars.get('model_name', 'N/A')}")

        # Test the model
        test_text = "This is a test sentence."
        embedding = model.encode(test_text)
        print(f"✓ Model test successful! Embedding dimension: {len(embedding)}")

        return True

    except Exception as e:
        print(f"\n✗ Error downloading model: {e}")
        print("\nAlternative: Download manually from Hugging Face:")
        print(f"  1. Visit: https://huggingface.co/sentence-transformers/{model_name}")
        print(f"  2. Download all files to: ~/.cache/huggingface/hub/models--sentence-transformers--{model_name.replace('-', '--')}")
        print(f"  3. Or use: huggingface-cli download sentence-transformers/{model_name}")
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download sentence-transformers model manually")
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Model name to download (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)"
    )

    args = parser.parse_args()

    success = download_model_with_retry(args.model, args.timeout)
    sys.exit(0 if success else 1)

