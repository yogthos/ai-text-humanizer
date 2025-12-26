"""Graph-based planning for sentence generation.

This module implements the GraphPlanner that converts semantic graphs
into sentence plans based on author rhythm and style profile.
"""

import re
import random
from typing import List, Optional, Dict, Any, Tuple
import networkx as nx

from src.planning.sentence_plan import SentenceNode, SentencePlan


class GraphPlanner:
    """Plans sentence structure from semantic graphs based on author rhythm."""

    def __init__(self, style_profile: Optional[Dict[str, Any]] = None):
        """Initialize GraphPlanner with style profile.

        Args:
            style_profile: Dictionary containing style metrics including:
                - structural_dna: Dict with avg_words_per_sentence
                - burstiness: Float (0.0-1.0) indicating sentence length variation
                If None, uses default values.
        """
        self.style_profile = style_profile or {}
        structural_dna = self.style_profile.get('structural_dna', {})
        self.avg_words_per_sentence = structural_dna.get('avg_words_per_sentence', 25.0)
        self.burstiness = self.style_profile.get('burstiness', 0.5)
        self.keywords = self.style_profile.get('keywords', [])[:15]  # Top 15 keywords

    def _generate_rhythm_pattern(self, total_props: int, burstiness: float) -> List[int]:
        """Generate rhythm pattern for clustering propositions.

        Args:
            total_props: Total number of propositions
            burstiness: Burstiness value (0.0-1.0)

        Returns:
            List of integers representing propositions per sentence
        """
        # Estimate propositions per sentence (assume ~10 words per prop)
        props_per_sentence = max(1, int(self.avg_words_per_sentence / 10))
        estimated_sentences = max(1, total_props // props_per_sentence)

        if burstiness > 0.6:  # High burstiness - jagged pattern
            pattern = []
            remaining = total_props
            # Generate varying sizes
            while remaining > 0:
                if remaining <= 2:
                    pattern.append(remaining)
                    break
                # Vary between 1 and props_per_sentence * 2
                size = random.randint(1, min(props_per_sentence * 2, remaining))
                pattern.append(size)
                remaining -= size
            return pattern
        else:  # Low burstiness - balanced pattern
            # Distribute evenly with slight variation
            base_size = total_props // estimated_sentences
            remainder = total_props % estimated_sentences
            pattern = [base_size] * estimated_sentences
            # Distribute remainder
            for i in range(remainder):
                pattern[i] += 1
            # Add slight variation (±1), ensuring no negative or zero values
            for i in range(len(pattern)):
                if pattern[i] > 1 and random.random() < 0.3:
                    variation = random.choice([-1, 1])
                    # Only apply variation if result is still positive
                    if pattern[i] + variation > 0:
                        pattern[i] += variation
            return pattern

    def _parse_mermaid_graph(
        self, mermaid: str, node_map: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Tuple[int, int, str]], Dict[int, str]]:
        """Parse mermaid graph to extract edges and node mapping.

        Args:
            mermaid: Mermaid graph string
            node_map: Optional dictionary mapping node IDs to proposition text

        Returns:
            Tuple of (edges, node_mapping) where:
                - edges: List of (source_idx, target_idx, label) tuples
                - node_mapping: Dict mapping index to proposition text
        """
        edges = []
        prop_map: Dict[int, str] = {}

        if not mermaid:
            return edges, prop_map

        # Remove graph declaration if present
        mermaid_clean = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

        # Pattern 1: P0 --label--> P1 (labeled edge)
        labeled_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--([^-]+)-->\s*([A-Z_][A-Z0-9_]*)'
        labeled_matches = re.findall(labeled_edge_pattern, mermaid_clean)
        for match in labeled_matches:
            source, label, target = match
            # Extract numeric index from node ID (P0 -> 0, P1 -> 1, etc.)
            source_match = re.search(r'\d+', source)
            target_match = re.search(r'\d+', target)
            if source_match and target_match:
                source_idx = int(source_match.group())
                target_idx = int(target_match.group())
                # Normalize label
                label = label.strip().upper()
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
                    'PURPOSE': 'PURPOSE',
                    'ORIGIN': 'ORIGIN',
                    'COMPOSITION': 'COMPOSITION',
                    'ATTRIBUTION': 'ATTRIBUTION'
                }
                normalized_label = label_map.get(label, label)
                edges.append((source_idx, target_idx, normalized_label))

                # Extract proposition text from node_map if available
                if node_map:
                    if source not in prop_map and source in node_map:
                        prop_map[source_idx] = node_map[source]
                    if target not in prop_map and target in node_map:
                        prop_map[target_idx] = node_map[target]

        # Pattern 2: P0 --> P1 (unlabeled edge)
        simple_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
        simple_matches = re.findall(simple_edge_pattern, mermaid_clean)
        for match in simple_matches:
            source, target = match
            source_match = re.search(r'\d+', source)
            target_match = re.search(r'\d+', target)
            if source_match and target_match:
                source_idx = int(source_match.group())
                target_idx = int(target_match.group())
                # Check if already captured
                if not any(e[0] == source_idx and e[1] == target_idx for e in edges):
                    edges.append((source_idx, target_idx, 'SEQUENCE'))
                    if node_map:
                        if source_idx not in prop_map and source in node_map:
                            prop_map[source_idx] = node_map[source]
                        if target_idx not in prop_map and target in node_map:
                            prop_map[target_idx] = node_map[target]

        return edges, prop_map

    def _determine_transition(self, edge_label: str, node_role: str) -> str:
        """Determine transition type from edge label.

        Args:
            edge_label: Edge label from graph (e.g., "CAUSALITY", "CONTRAST")
            node_role: Role of the node (e.g., "Contrast", "Definition")

        Returns:
            Transition type string
        """
        label_upper = edge_label.upper()
        if label_upper == 'CAUSALITY':
            return "Causal"
        elif label_upper == 'CONTRAST':
            return "Adversative"
        elif label_upper == 'SEQUENCE':
            return "Sequential"
        elif label_upper in ['PURPOSE', 'ATTRIBUTION', 'ORIGIN', 'COMPOSITION']:
            return "Semantic"
        else:
            return "Flow"

    def _determine_role(
        self,
        propositions: List[str],
        graph_edges: List[Tuple[int, int, str]],
        cluster_indices: List[int],
        paragraph_intent: str
    ) -> str:
        """Determine role of a sentence cluster.

        Args:
            propositions: List of all propositions
            graph_edges: List of (source, target, label) edges
            cluster_indices: Indices of propositions in this cluster
            paragraph_intent: Overall paragraph intent

        Returns:
            Role string (e.g., "Thesis", "Elaboration", "Contrast")
        """
        if not cluster_indices:
            return "Elaboration"

        first_idx = cluster_indices[0]

        # Check incoming edges to first proposition
        incoming_edges = [e for e in graph_edges if e[1] == first_idx]
        outgoing_edges = [e for e in graph_edges if e[0] == first_idx]

        # Analyze edge types
        has_contrast = any(e[2].upper() == 'CONTRAST' for e in incoming_edges + outgoing_edges)
        has_definition = any(e[2].upper() == 'DEFINITION' for e in incoming_edges + outgoing_edges)
        has_causality = any(e[2].upper() == 'CAUSALITY' for e in incoming_edges + outgoing_edges)

        # If first cluster and has definition, likely thesis
        if first_idx == 0 and has_definition:
            return "Thesis"
        elif has_contrast:
            return "Contrast"
        elif has_definition:
            return "Definition"
        elif has_causality:
            return "Elaboration"
        elif paragraph_intent == 'NARRATIVE':
            return "Narrative"
        else:
            return "Elaboration"

    def _get_topological_sort(
        self, edges: List[Tuple[int, int, str]], num_nodes: int
    ) -> List[int]:
        """Get topological sort of nodes preserving logical order.

        Args:
            edges: List of (source, target, label) edges
            num_nodes: Total number of nodes

        Returns:
            Ordered list of node indices
        """
        try:
            # Build directed graph
            G = nx.DiGraph()
            for i in range(num_nodes):
                G.add_node(i)

            for source, target, label in edges:
                G.add_edge(source, target)

            # Try topological sort
            try:
                # Use lexicographical_topological_sort to prefer lower indices (preserve original order)
                # when no dependency forces a specific order.
                sorted_nodes = list(nx.lexicographical_topological_sort(G))
                return sorted_nodes
            except nx.NetworkXError:
                # Cycle detected, use DFS-based ordering
                visited = set()
                result = []

                def dfs(node):
                    if node in visited:
                        return
                    visited.add(node)
                    # Visit neighbors first
                    for neighbor in sorted(G.successors(node)):
                        if neighbor not in visited:
                            dfs(neighbor)
                    result.append(node)

                for node in range(num_nodes):
                    if node not in visited:
                        dfs(node)

                return result
        except Exception:
            # Fallback to simple sequential order
            return list(range(num_nodes))

    def create_plan(
        self,
        propositions: List[str],
        input_graph: Optional[Dict[str, Any]] = None,
        paragraph_intent: str = "ARGUMENT",
        paragraph_signature: str = "DEFINITION"
    ) -> SentencePlan:
        """Create a sentence plan from propositions and semantic graph.

        Args:
            propositions: List of proposition strings
            input_graph: Optional graph dictionary with 'mermaid' and 'node_map'
            paragraph_intent: Overall paragraph intent (DEFINITION, ARGUMENT, NARRATIVE)
            paragraph_signature: Logical signature (CONTRAST, CAUSALITY, etc.)

        Returns:
            SentencePlan with ordered sentence nodes
        """
        total_props = len(propositions)
        if total_props == 0:
            return SentencePlan(nodes=[], paragraph_intent=paragraph_intent, paragraph_signature=paragraph_signature)

        # Parse graph if available
        edges: List[Tuple[int, int, str]] = []
        node_map: Dict[int, str] = {}
        if input_graph and input_graph.get('mermaid'):
            edges, node_map = self._parse_mermaid_graph(
                input_graph.get('mermaid', ''),
                input_graph.get('node_map', {})
            )

        # Generate rhythm pattern
        rhythm_pattern = self._generate_rhythm_pattern(total_props, self.burstiness)

        # Get topological sort
        sorted_indices = self._get_topological_sort(edges, total_props)

        # Cluster propositions with soft cap logic
        sentence_nodes: List[SentenceNode] = []
        visited = set()
        rhythm_idx = 0
        current_cluster: List[int] = []
        previous_cluster_end: Optional[int] = None

        for prop_idx in sorted_indices:
            if prop_idx in visited:
                continue

            # Check if we should close current cluster early (hard break detection)
            should_close_early = False
            if previous_cluster_end is not None and current_cluster:
                # Check edge between last node of previous cluster and current candidate
                connecting_edges = [
                    e for e in edges
                    if (e[0] == previous_cluster_end and e[1] == prop_idx) or
                       (e[1] == previous_cluster_end and e[0] == prop_idx)
                ]
                if connecting_edges:
                    edge_label = connecting_edges[0][2].upper()
                    # Hard break indicators: strong CONTRAST, or shift in intent type
                    if edge_label == 'CONTRAST' and len(current_cluster) >= 1:
                        should_close_early = True

            # Check semantic boundaries: never split semantic edges
            has_semantic_edge = False
            if current_cluster:
                last_in_cluster = current_cluster[-1]
                # Check if there's a semantic edge connecting last in cluster to current
                semantic_edges = [
                    e for e in edges
                    if ((e[0] == last_in_cluster and e[1] == prop_idx) or
                        (e[1] == last_in_cluster and e[0] == prop_idx)) and
                       e[2].upper() in ['PURPOSE', 'ATTRIBUTION', 'ORIGIN', 'COMPOSITION']
                ]
                if semantic_edges:
                    has_semantic_edge = True

            # Check parallelism: does this node share a target/source with the cluster?
            # Parallel nodes (e.g. P0->P2 and P1->P2) should stay together to avoid repetitive sentences
            is_parallel = False
            if current_cluster and not should_close_early:
                # Get common neighbors of cluster
                cluster_targets = set()
                cluster_sources = set()
                for cluster_node in current_cluster:
                    # Targets of cluster_node (where cluster_node is source)
                    # Store (target_idx, label)
                    targets = [e for e in edges if e[0] == cluster_node]
                    cluster_targets.update([(t[1], t[2]) for t in targets])
                    
                    # Sources of cluster_node (where cluster_node is target)
                    # Store (source_idx, label)
                    sources = [e for e in edges if e[1] == cluster_node]
                    cluster_sources.update([(s[0], s[2]) for s in sources])
                
                # Check current node neighbors
                curr_targets = set([(e[1], e[2]) for e in edges if e[0] == prop_idx])
                curr_sources = set([(e[0], e[2]) for e in edges if e[1] == prop_idx])
                
                # Intersection? (Parallel if they share a target or source with same label)
                if not cluster_targets.isdisjoint(curr_targets) or \
                   not cluster_sources.isdisjoint(curr_sources):
                    is_parallel = True

            # Determine if we've reached rhythm target (soft cap)
            rhythm_target = rhythm_pattern[rhythm_idx] if rhythm_idx < len(rhythm_pattern) else len(propositions)
            reached_target = len(current_cluster) >= rhythm_target

            # Add to current cluster if:
            # - We haven't reached target AND not closing early, OR
            # - There's a semantic edge (must keep together)
            # - It's parallel structure (must keep together to avoid repetition, within limits)
            if (not reached_target and not should_close_early) or has_semantic_edge or (is_parallel and len(current_cluster) < 6):
                current_cluster.append(prop_idx)
                visited.add(prop_idx)
            else:
                # Close current cluster and start new one
                if current_cluster:
                    # Create sentence node from cluster
                    cluster_props = [propositions[i] for i in current_cluster]
                    role = self._determine_role(propositions, edges, current_cluster, paragraph_intent)

                    # Determine transition from previous cluster
                    transition = "None"
                    if previous_cluster_end is not None and current_cluster:
                        # Find edge connecting previous to current
                        connecting = [
                            e for e in edges
                            if e[0] == previous_cluster_end and e[1] == current_cluster[0]
                        ]
                        if connecting:
                            transition = self._determine_transition(connecting[0][2], role)
                        else:
                            transition = "Flow"

                    target_length = int(len(current_cluster) * 10)  # Estimate 10 words per prop
                    # Use style profile avg if available
                    if self.avg_words_per_sentence:
                        target_length = int(self.avg_words_per_sentence)

                    node = SentenceNode(
                        id=f"S{len(sentence_nodes)}",
                        propositions=cluster_props,
                        role=role,
                        transition_type=transition,
                        target_length=target_length,
                        keywords=self.keywords[:3] if self.keywords else [],  # Use top 3 keywords
                        global_indices=current_cluster,
                        intended_subject=None
                    )
                    sentence_nodes.append(node)
                    previous_cluster_end = current_cluster[-1]
                    rhythm_idx += 1

                # Start new cluster
                current_cluster = [prop_idx]
                visited.add(prop_idx)

        # Handle remaining cluster
        if current_cluster:
            cluster_props = [propositions[i] for i in current_cluster]
            role = self._determine_role(propositions, edges, current_cluster, paragraph_intent)

            transition = "None"
            if previous_cluster_end is not None:
                connecting = [
                    e for e in edges
                    if e[0] == previous_cluster_end and e[1] == current_cluster[0]
                ]
                if connecting:
                    transition = self._determine_transition(connecting[0][2], role)
                else:
                    transition = "Flow"

            target_length = int(len(current_cluster) * 10)
            if self.avg_words_per_sentence:
                target_length = int(self.avg_words_per_sentence)

            node = SentenceNode(
                id=f"S{len(sentence_nodes)}",
                propositions=cluster_props,
                role=role,
                transition_type=transition,
                target_length=target_length,
                keywords=self.keywords[:3] if self.keywords else [],
                global_indices=current_cluster,
                intended_subject=None
            )
            sentence_nodes.append(node)

        # Phase 5: Plan Validation
        # Verify all propositions are included
        all_covered = set()
        for node in sentence_nodes:
            all_covered.update(node.global_indices)

        missing_props = set(range(total_props)) - all_covered
        if missing_props:
            # Add missing propositions as separate nodes
            for prop_idx in sorted(missing_props):
                node = SentenceNode(
                    id=f"S{len(sentence_nodes)}",
                    propositions=[propositions[prop_idx]],
                    role="Elaboration",
                    transition_type="Flow",
                    target_length=int(self.avg_words_per_sentence) if self.avg_words_per_sentence else 25,
                    keywords=self.keywords[:3] if self.keywords else [],
                    global_indices=[prop_idx],
                    intended_subject=None
                )
                sentence_nodes.append(node)

        return SentencePlan(
            nodes=sentence_nodes,
            paragraph_intent=paragraph_intent,
            paragraph_signature=paragraph_signature
        )

