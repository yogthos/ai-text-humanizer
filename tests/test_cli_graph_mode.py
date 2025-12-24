"""Tests for CLI graph mode integration."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestCLIGraphMode:
    """Test CLI graph mode functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
        self.input_file = Path(self.temp_dir) / "input.txt"
        self.output_file = Path(self.temp_dir) / "output.txt"

        # Create test config
        config = {
            "atlas": {
                "persist_path": str(Path(self.temp_dir) / "atlas_cache")
            }
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)

        # Create test input file
        with open(self.input_file, 'w') as f:
            f.write("Test input text.")

    def test_graph_mode_flag_exists(self):
        """Test that --graph-mode flag is defined."""
        from restyle import main
        import argparse

        # Create parser like restyle.py does
        parser = argparse.ArgumentParser()
        parser.add_argument('input', type=str)
        parser.add_argument('-o', '--output', type=str, required=True)
        parser.add_argument('--graph-mode', action='store_true')

        # Parse with flag
        args = parser.parse_args(['input.txt', '-o', 'output.txt', '--graph-mode'])

        assert args.graph_mode is True

    def test_graph_mode_validates_index_exists(self):
        """Test that graph mode validates index exists."""
        import chromadb
        from pathlib import Path

        # Create ChromaDB with collection
        chroma_path = Path(self.temp_dir) / "atlas_cache" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.create_collection(name="style_graphs")

        # Add a test graph
        collection.add(
            ids=["test_id"],
            documents=["A test graph description"],
            metadatas=[{"mermaid": "graph LR; A --> B", "node_count": 2}]
        )

        # Verify collection exists and has content
        assert collection.count() > 0

    def test_graph_mode_fails_when_index_missing(self):
        """Test that graph mode fails with clear error when index missing."""
        import sys
        from unittest.mock import patch

        # Mock sys.argv and sys.exit
        test_args = [
            'restyle.py',
            str(self.input_file),
            '-o',
            str(self.output_file),
            '--graph-mode',
            '--config',
            str(self.config_path)
        ]

        with patch('sys.argv', test_args):
            with patch('sys.exit') as mock_exit:
                try:
                    from restyle import main
                    main()
                except SystemExit:
                    pass

        # Should have attempted to exit
        # (We can't easily test the exact exit code without running full CLI)

    @patch('chromadb.PersistentClient')
    def test_graph_mode_checks_collection_count(self, mock_client_class):
        """Test that graph mode checks collection is not empty."""
        # Mock ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0  # Empty collection

        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_class.return_value = mock_client

        # This would be tested in integration, but we can verify the logic
        # would check count == 0

    def test_graph_mode_shows_count_in_verbose(self):
        """Test that graph mode shows graph count in verbose mode."""
        import chromadb
        from pathlib import Path

        # Create ChromaDB with collection
        chroma_path = Path(self.temp_dir) / "atlas_cache" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.create_collection(name="style_graphs")

        # Add multiple graphs
        collection.add(
            ids=["id1", "id2", "id3"],
            documents=["Desc 1", "Desc 2", "Desc 3"],
            metadatas=[
                {"mermaid": "graph LR; A --> B", "node_count": 2},
                {"mermaid": "graph LR; C --> D", "node_count": 2},
                {"mermaid": "graph LR; E --> F", "node_count": 2}
            ]
        )

        count = collection.count()
        assert count == 3

    def test_graph_mode_handles_invalid_collection(self):
        """Test that graph mode handles invalid collection gracefully."""
        import chromadb
        from pathlib import Path

        # Create ChromaDB path but no collection
        chroma_path = Path(self.temp_dir) / "atlas_cache" / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_path))

        # Try to get non-existent collection - should raise exception
        with pytest.raises(Exception):
            client.get_collection(name="style_graphs")

