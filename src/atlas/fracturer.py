"""Semantic Fracturer for Dynamic Graph Fracturing.

Groups propositions into logical clusters that match the target style's density,
preserving causal chains and logical relationships.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.llm_provider import LLMProvider


class SemanticFracturer:
    """Groups propositions into logical clusters for sentence generation."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize the Semantic Fracturer.

        Args:
            llm_provider: LLM provider instance for grouping propositions.
        """
        self.llm_provider = llm_provider

    def _strip_markdown_code_blocks(self, text: str) -> str:
        """Strip markdown code blocks from text.

        Args:
            text: Text that may contain markdown code blocks.

        Returns:
            Text with code blocks removed.
        """
        # Remove ```json ... ``` blocks
        text = re.sub(r'```json\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove ``` ... ``` blocks (generic)
        text = re.sub(r'```\s*\n?(.*?)\n?```', r'\1', text, flags=re.DOTALL)
        # Remove any remaining ``` markers
        text = text.replace('```', '')
        return text.strip()

    def fracture(
        self,
        propositions: List[str],
        target_density: int = 4,
        max_density: int = 6,
        input_graph: Optional[Dict[str, Any]] = None
    ) -> List[List[int]]:
        """Group propositions into logical clusters.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per group.
            max_density: Maximum number of propositions per group (hard cap).
            input_graph: Optional global graph dictionary with 'mermaid' for graph-based clustering.

        Returns:
            List of lists of indices, where each inner list represents a cluster.
            Example: [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
        """
        if not propositions:
            return []

        if len(propositions) <= target_density:
            # If we have fewer propositions than target, return single group
            return [list(range(len(propositions)))]

        # Try graph-based fracturing first (if graph provided)
        if input_graph and input_graph.get('mermaid'):
            try:
                groups = self._fracture_with_graph(propositions, input_graph, target_density, max_density)
                if groups and self._validate_groups(groups, len(propositions)):
                    return groups
            except Exception as e:
                print(f"Warning: Graph-based fracturing failed: {e}, falling back to LLM")

        # Try LLM-based fracturing
        try:
            groups = self._fracture_with_llm(propositions, target_density, max_density)
            if groups and self._validate_groups(groups, len(propositions)):
                return groups
        except Exception as e:
            print(f"Warning: LLM fracturing failed: {e}, using fallback")

        # Fallback: simple fixed-size chunking
        return self._fallback_chunking(propositions, target_density)

    def _fracture_with_llm(
        self,
        propositions: List[str],
        target_density: int,
        max_density: int
    ) -> Optional[List[List[int]]]:
        """Use LLM to group propositions into logical clusters.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per group.
            max_density: Maximum number of propositions per group.

        Returns:
            List of lists of indices, or None if LLM call fails.
        """
        # Create indexed list for LLM
        indexed_propositions = [
            f"{i}: {prop}" for i, prop in enumerate(propositions)
        ]

        system_prompt = (
            "You are a Logic Architect. Your task is to group atomic facts "
            "into logical clusters that form coherent sentences. "
            "Preserve causal chains and logical relationships."
        )

        user_prompt = f"""Propositions:
{chr(10).join(indexed_propositions)}

Target Density: ~{target_density} propositions per group.
Max Density: {max_density} propositions per group.

Task: Group these propositions into logical clusters that form coherent sentences.
- Keep causal chains together (e.g., if proposition 1 causes proposition 2, keep them in the same group).
- Keep related facts together (e.g., conditions and consequences).
- Each group should have roughly {target_density} propositions (aim for {target_density}, but can vary from 2 to {max_density}).
- Do NOT break logical dependencies.

Output: JSON list of lists of indices (e.g., [[0, 1, 2], [3, 4], [5, 6, 7, 8]]).
Each inner list represents one sentence cluster."""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=500
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            groups = json.loads(response)

            # Validate format
            if not isinstance(groups, list):
                return None

            # Ensure all inner items are lists
            for group in groups:
                if not isinstance(group, list):
                    return None

            return groups

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Failed to parse LLM response for fracturing: {e}")
            return None

    def _validate_groups(self, groups: List[List[int]], total_propositions: int) -> bool:
        """Validate that groups cover all propositions exactly once.

        Args:
            groups: List of lists of indices.
            total_propositions: Total number of propositions.

        Returns:
            True if groups are valid, False otherwise.
        """
        if not groups:
            return False

        # Collect all indices
        all_indices = []
        for group in groups:
            if not isinstance(group, list):
                return False
            for idx in group:
                if not isinstance(idx, int):
                    return False
                if idx < 0 or idx >= total_propositions:
                    return False
                all_indices.append(idx)

        # Check that all indices are present exactly once
        if len(all_indices) != total_propositions:
            return False

        if set(all_indices) != set(range(total_propositions)):
            return False

        return True

    def _fracture_with_graph(
        self,
        propositions: List[str],
        input_graph: Dict[str, Any],
        target_density: int,
        max_density: int
    ) -> Optional[List[List[int]]]:
        """Use graph structure with Pre-Clustering Contraction to group propositions.

        Implements the Pre-Clustering Contraction algorithm:
        1. Parse mermaid to NetworkX graph
        2. Assign edge weights (semantic=100.0, strong rhetorical=2.0, weak=1.0)
        3. Contract semantic components into Super Nodes
        4. Cluster Super Nodes using community detection
        5. Expand back to original proposition indices

        Args:
            propositions: List of proposition strings
            input_graph: Graph dictionary with 'mermaid' string
            target_density: Target number of propositions per group
            max_density: Maximum number of propositions per group

        Returns:
            List of lists of indices, or None if graph parsing fails
        """
        try:
            import networkx as nx

            # Step 1: Parse mermaid to NetworkX graph
            mermaid = input_graph.get('mermaid', '')
            if not mermaid:
                return None

            # Parse mermaid edges
            edges = self._parse_mermaid_edges(mermaid)

            if not edges:
                return None

            # Build NetworkX graph
            G = nx.Graph()
            for source, target, label in edges:
                # Extract numeric index from node ID (P0 -> 0, P1 -> 1, etc.)
                source_idx = int(re.search(r'\d+', source).group()) if re.search(r'\d+', source) else None
                target_idx = int(re.search(r'\d+', target).group()) if re.search(r'\d+', target) else None

                if source_idx is not None and target_idx is not None:
                    # Ensure nodes exist
                    G.add_node(source_idx)
                    G.add_node(target_idx)

                    # Step 2: Assign edge weights
                    label_upper = label.upper()
                    if label_upper in ['PURPOSE', 'ORIGIN', 'COMPOSITION', 'ATTRIBUTION']:
                        weight = 100.0  # Semantic edges - unbreakable
                    elif label_upper in ['CONTRAST', 'CAUSALITY']:
                        weight = 2.0  # Strong rhetorical
                    else:
                        weight = 1.0  # Weak rhetorical (SEQUENCE, LIST, etc.)

                    G.add_edge(source_idx, target_idx, weight=weight, label=label)

            # Ensure all proposition indices are in the graph (even if isolated)
            for i in range(len(propositions)):
                if i not in G:
                    G.add_node(i)

            # Step 3: Pre-Clustering Contraction
            # Find connected components using ONLY semantic edges (weight=100.0)
            semantic_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('weight') == 100.0]
            semantic_subgraph = G.edge_subgraph(semantic_edges)
            super_nodes = list(nx.connected_components(semantic_subgraph))

            # Create mapping: super_node_id -> list of proposition indices
            super_node_map = {}
            isolated_nodes = set(range(len(propositions)))

            for super_idx, component in enumerate(super_nodes):
                prop_indices = sorted(list(component))
                super_node_map[super_idx] = prop_indices
                isolated_nodes -= component

            # Add isolated nodes as their own super nodes
            for node in isolated_nodes:
                super_idx = len(super_node_map)
                super_node_map[super_idx] = [node]

            # Step 4: Build contracted graph with Super Nodes
            G_contracted = nx.Graph()

            # Add super nodes as vertices
            for super_idx in super_node_map:
                G_contracted.add_node(f"Super_{super_idx}")

            # Add edges between super nodes (from original graph, excluding semantic edges)
            # If any proposition in Super_A is connected to any proposition in Super_B via non-semantic edge,
            # add an edge between Super_A and Super_B
            for super_a_idx, props_a in super_node_map.items():
                for super_b_idx, props_b in super_node_map.items():
                    if super_a_idx >= super_b_idx:
                        continue  # Avoid duplicate edges

                    # Check if there's a non-semantic edge between any prop in A and any prop in B
                    has_edge = False
                    for prop_a in props_a:
                        for prop_b in props_b:
                            if G.has_edge(prop_a, prop_b):
                                edge_data = G[prop_a][prop_b]
                                if edge_data.get('weight', 1.0) < 100.0:  # Non-semantic edge
                                    has_edge = True
                                    # Use the weight of the strongest edge between these super nodes
                                    weight = edge_data.get('weight', 1.0)
                                    G_contracted.add_edge(f"Super_{super_a_idx}", f"Super_{super_b_idx}", weight=weight)
                                    break
                        if has_edge:
                            break

            # Step 5: Cluster super nodes using community detection
            if len(G_contracted.nodes()) > 1:
                try:
                    # Use greedy modularity communities
                    communities = list(nx.community.greedy_modularity_communities(G_contracted))
                except Exception:
                    # Fallback: if community detection fails, treat each super node as its own cluster
                    communities = [{node} for node in G_contracted.nodes()]
            else:
                # Single super node
                communities = [set(G_contracted.nodes())]

            # Step 6: Expand back to proposition indices
            result_groups = []
            for community in communities:
                prop_indices = []
                for super_node in community:
                    # Extract super node index
                    super_idx = int(super_node.split('_')[1])
                    # Add all propositions from this super node
                    prop_indices.extend(super_node_map[super_idx])

                if prop_indices:
                    result_groups.append(sorted(prop_indices))

            # Validate groups cover all propositions
            all_covered = set()
            for group in result_groups:
                all_covered.update(group)

            if all_covered != set(range(len(propositions))):
                # Missing some propositions - add them as separate groups
                missing = set(range(len(propositions))) - all_covered
                for prop_idx in sorted(missing):
                    result_groups.append([prop_idx])

            return result_groups if result_groups else None

        except Exception as e:
            print(f"Warning: Graph-based fracturing failed: {e}")
            return None

    def _parse_mermaid_edges(self, mermaid: str) -> List[tuple]:
        """Parse Mermaid graph string to extract edges with labels.

        Args:
            mermaid: Mermaid graph string (e.g., "graph LR; P0 --cause--> P1; P1 --contrast--> P2")

        Returns:
            List of tuples (source, target, label) where label is the edge type
        """
        edges = []

        # Remove graph declaration if present
        mermaid_clean = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

        # Pattern 1: P0 --label--> P1 (labeled edge)
        labeled_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--([^-]+)-->\s*([A-Z_][A-Z0-9_]*)'
        labeled_matches = re.findall(labeled_edge_pattern, mermaid_clean)
        for match in labeled_matches:
            source, label, target = match
            # Normalize label (remove extra spaces, convert to uppercase)
            label = label.strip().upper()
            # Map common edge labels to standard types
            label_map = {
                'CAUSE': 'CAUSALITY',
                'CONTRAST': 'CONTRAST',
                'DEFINE': 'DEFINITION',
                'DEFINITION': 'DEFINITION',
                'SEQUENCE': 'SEQUENCE',
                'SUPPORT': 'SUPPORT',
                'CONDITION': 'CONDITIONAL',
                'CONDITIONAL': 'CONDITIONAL',
                'LIST': 'LIST',
                'ENUMERATION': 'LIST',
                # Semantic edge types
                'PURPOSE': 'PURPOSE',
                'ORIGIN': 'ORIGIN',
                'COMPOSITION': 'COMPOSITION',
                'ATTRIBUTION': 'ATTRIBUTION'
            }
            normalized_label = label_map.get(label, label)
            edges.append((source, target, normalized_label))

        # Pattern 2: P0 --> P1 (unlabeled edge, infer from context or use SEQUENCE)
        simple_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
        simple_matches = re.findall(simple_edge_pattern, mermaid_clean)
        for match in simple_matches:
            source, target = match
            # Check if this edge was already captured with a label
            if not any(e[0] == source and e[1] == target for e in edges):
                # Default to SEQUENCE for unlabeled edges
                edges.append((source, target, 'SEQUENCE'))

        return edges

    def _fallback_chunking(
        self,
        propositions: List[str],
        target_density: int
    ) -> List[List[int]]:
        """Fallback to simple fixed-size chunking.

        Args:
            propositions: List of proposition strings.
            target_density: Target number of propositions per chunk.

        Returns:
            List of lists of indices.
        """
        groups = []
        for i in range(0, len(propositions), target_density):
            group = list(range(i, min(i + target_density, len(propositions))))
            groups.append(group)
        return groups

