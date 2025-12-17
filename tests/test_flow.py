"""Test script for flow planning and template selection.

NOTE: This module tests deprecated functionality that is no longer used
in the RAG-based pipeline. These tests are kept for reference but the
template selection system has been replaced by RAG-based dual retrieval.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("⚠ Note: Template selection tests are for deprecated functionality.")
print("  The pipeline now uses RAG-based dual retrieval instead.")
print("  These tests are kept for reference only.\n")

# Skip these tests by default since they test deprecated code
if __name__ == "__main__":
    print("Skipping deprecated template selection tests.")
    print("See test_atlas.py for RAG-based retrieval tests.")
    sys.exit(0)
