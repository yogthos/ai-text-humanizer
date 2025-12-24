"""Graph telemetry logger for debugging and visual inspection of graph matches."""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional


class GraphLogger:
    """Logs graph matches to a Markdown file for visual inspection and debugging."""

    def __init__(self, config_path: str = "config.json"):
        """Initialize the graph logger.

        Args:
            config_path: Path to configuration file.
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Get telemetry config
        graph_config = self.config.get("graph_pipeline", {})
        telemetry_config = graph_config.get("telemetry", {})

        # Get log path (default to logs/graph_debug.md)
        self.log_path = Path(telemetry_config.get("log_path", "logs/graph_debug.md"))

        # Create logs directory if it doesn't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize log file with header if it doesn't exist
        if not self.log_path.exists():
            with open(self.log_path, 'w', encoding='utf-8') as f:
                f.write("# Graph Pipeline Debug Log\n\n")
                f.write("This file contains graph matching decisions for debugging.\n\n")
                f.write("---\n\n")

    def log_match(
        self,
        paragraph_index: int,
        input_graph: Dict[str, Any],
        style_match: Dict[str, Any],
        final_text: str
    ) -> None:
        """Log a graph match to the debug file.

        Args:
            paragraph_index: Index of the paragraph in the document.
            input_graph: Input graph dictionary with 'mermaid', 'description', 'node_count', 'node_map'.
            style_match: Style match dictionary with 'style_mermaid', 'node_mapping', 'distance', 'style_metadata'.
            final_text: Generated text from the graph.
        """
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                # Write paragraph header
                f.write(f"## Paragraph {paragraph_index}\n\n")

                # Write input graph
                f.write("### Input Graph\n\n")
                f.write("```mermaid\n")
                f.write(f"{input_graph.get('mermaid', 'N/A')}\n")
                f.write("```\n\n")

                # Write matched style graph
                f.write("### Matched Style Graph\n\n")
                f.write("```mermaid\n")
                f.write(f"{style_match.get('style_mermaid', 'N/A')}\n")
                f.write("```\n\n")

                # Write metadata
                distance = style_match.get('distance', 'N/A')
                if isinstance(distance, (int, float)):
                    f.write(f"**Distance:** {distance:.4f}\n\n")
                else:
                    f.write(f"**Distance:** {distance}\n\n")

                input_node_count = input_graph.get('node_count', 'N/A')
                style_metadata = style_match.get('style_metadata', {})
                style_node_count = style_metadata.get('node_count', 'N/A')
                f.write(f"**Node Count:** Input={input_node_count}, Style={style_node_count}\n\n")

                # Write node mapping
                f.write("### Node Mapping\n\n")
                f.write("```json\n")
                node_mapping = style_match.get('node_mapping', {})
                f.write(json.dumps(node_mapping, indent=2))
                f.write("\n```\n\n")

                # Write generated text
                f.write("### Generated Text\n\n")
                f.write(f"> {final_text}\n\n")

                # Write separator
                f.write("---\n\n")

        except Exception as e:
            # Log warning but don't fail - telemetry is non-blocking
            print(f"Warning: Failed to write graph telemetry log: {e}")

