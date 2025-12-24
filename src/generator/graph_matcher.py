"""Topological Graph Matcher.

Matches input logical graphs to author style graphs from ChromaDB
and maps input nodes to style nodes.
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.utils import embedding_functions

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.generator.llm_provider import LLMProvider


class TopologicalMatcher:
    """Matches input graphs to style graphs and maps nodes."""

    def __init__(self, config_path: str = "config.json", chroma_path: Optional[str] = None):
        """Initialize the Topological Matcher.

        Args:
            config_path: Path to configuration file.
            chroma_path: Optional custom path for ChromaDB. Defaults to atlas_cache/chroma.
        """
        self.config_path = config_path

        # Initialize ChromaDB client
        if chroma_path:
            self.chroma_path = Path(chroma_path)
        else:
            self.chroma_path = project_root / "atlas_cache" / "chroma"

        self.chroma_path.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB at {self.chroma_path}...")
        self.client = chromadb.PersistentClient(path=str(self.chroma_path))

        # Get collection
        collection_name = "style_graphs"
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            # Create collection with default embedding function
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=embedding_fn
            )
            print(f"Created new collection: {collection_name}")

        # Initialize LLM provider
        print("Initializing LLM provider...")
        self.llm_provider = LLMProvider(config_path=config_path)

        # Load config for potential future use
        with open(config_path, 'r') as f:
            self.config = json.load(f)

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

    def _parse_mermaid_nodes(self, mermaid: str) -> List[str]:
        """Parse Mermaid graph string to extract node names.

        Args:
            mermaid: Mermaid graph string (e.g., "graph LR; ROOT --> NODE1").

        Returns:
            List of unique node identifiers.
        """
        nodes = set()

        # Remove graph declaration if present
        mermaid = re.sub(r'^graph\s+\w+\s*;?\s*', '', mermaid, flags=re.IGNORECASE)

        # Pattern 1: ROOT --> NODE1 or ROOT --edge--> NODE1
        # Pattern 2: NODE1[Label] --> NODE2
        # Pattern 3: ROOT --> NODE1 --> NODE2 (chain)

        # Extract all node identifiers
        # Match node names (alphanumeric, underscore, can have brackets)
        node_pattern = r'([A-Z_][A-Z0-9_]*)(?:\[[^\]]*\])?'

        # Find all nodes in the graph
        matches = re.findall(node_pattern, mermaid)
        nodes.update(matches)

        # Also look for nodes in edge definitions
        # ROOT --label--> NODE1
        edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*--[^>]*-->\s*([A-Z_][A-Z0-9_]*)'
        edge_matches = re.findall(edge_pattern, mermaid)
        for match in edge_matches:
            nodes.add(match[0])
            nodes.add(match[1])

        # Also handle simple arrow notation: ROOT --> NODE1
        simple_edge_pattern = r'([A-Z_][A-Z0-9_]*)\s*-->\s*([A-Z_][A-Z0-9_]*)'
        simple_matches = re.findall(simple_edge_pattern, mermaid)
        for match in simple_matches:
            nodes.add(match[0])
            nodes.add(match[1])

        # Filter out invalid single-underscore nodes (parsing artifacts)
        nodes = {node for node in nodes if node != '_'}

        return sorted(list(nodes))

    def _determine_role(self, document_context: Optional[Dict[str, Any]]) -> Optional[str]:
        """Determine paragraph role from document context.

        Args:
            document_context: Dict with 'current_index' and 'total_paragraphs'.

        Returns:
            Role string: 'opener', 'body', 'closer', or None.
        """
        if document_context is None:
            return None

        current_index = document_context.get('current_index', 0)
        total_paragraphs = document_context.get('total_paragraphs', 1)

        if current_index == 0:
            return 'opener'
        elif current_index == total_paragraphs - 1:
            return 'closer'
        else:
            return 'body'

    def get_best_match(
        self,
        input_graph_data: Dict[str, Any],
        document_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get the best matching style graph for the input graph.

        Args:
            input_graph_data: Dict with 'description', 'mermaid', 'node_map', 'node_count'.
            document_context: Optional dict with 'current_index' and 'total_paragraphs' for role filtering.

        Returns:
            Dict with 'style_mermaid', 'node_mapping', 'style_metadata', and 'distance'.

        Raises:
            ValueError: If input_graph_data is missing required fields or no matches found.
        """
        # Validate input
        if 'description' not in input_graph_data:
            raise ValueError("input_graph_data must contain 'description' field")

        input_node_count = input_graph_data.get('node_count', len(input_graph_data.get('node_map', {})))
        input_intent = input_graph_data.get('intent')

        # Step A: Context Determination & Semantic Search
        target_role = self._determine_role(document_context)

        # Query ChromaDB with larger top_k for better reranking
        query_text = input_graph_data['description']
        top_k = 20  # Retrieve more candidates for intent-based reranking

        # Step A.1: Try query with role filter if target_role is set
        # Only use role filter if we're confident the metadata field exists
        results = None
        query_kwargs = {
            'query_texts': [query_text],
            'n_results': top_k
        }

        # Attempt role filtering only if target_role is set
        # Catch errors specifically to handle missing metadata fields
        if target_role:
            query_kwargs['where'] = {"paragraph_role": target_role}
            try:
                results = self.collection.query(**query_kwargs)
                # Validate results structure
                if (not results or
                    not results.get('ids') or
                    not results['ids'][0] or
                    len(results['ids'][0]) == 0):
                    results = None
            except Exception as e:
                # Metadata field might not exist or query syntax error
                print(f"Warning: Query with role filter '{target_role}' failed: {e}")
                results = None

        # Step A.2: Fallback to query without role filter
        if not results:
            query_kwargs.pop('where', None)  # Remove where clause
            try:
                results = self.collection.query(**query_kwargs)
            except Exception as e:
                print(f"Warning: Query without filter also failed: {e}")
                results = None

        if not results or not results.get('ids') or not results['ids'][0]:
            raise ValueError("No style graphs found in collection")

        # Step A.3: Extract candidates with distances
        ids = results['ids'][0]
        distances = results['distances'][0] if results.get('distances') else [0.0] * len(ids)
        metadatas = results['metadatas'][0] if results.get('metadatas') else [{}] * len(ids)
        documents = results['documents'][0] if results.get('documents') else [''] * len(ids)

        # Step A.4: Create candidate list and apply intent boosting
        candidates = []
        for i, graph_id in enumerate(ids):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')
            style_intent = metadata.get('intent')

            # Intent-based reranking: boost candidates with matching intent
            priority_score = distance
            intent_match = False
            if input_intent and style_intent:
                # Normalize intents for comparison (case-insensitive)
                if input_intent.upper() == style_intent.upper():
                    # Intent match: multiply distance by 0.5 (strong boost)
                    priority_score = distance * 0.5
                    intent_match = True

            candidates.append({
                'id': graph_id,
                'mermaid': metadata.get('mermaid', ''),
                'node_count': metadata.get('node_count', 0),
                'edge_types': metadata.get('edge_types', ''),
                'skeleton': metadata.get('skeleton', ''),
                'intent': style_intent,
                'paragraph_role': metadata.get('paragraph_role'),
                'original_text': metadata.get('original_text', ''),
                'distance': distance,
                'priority_score': priority_score,
                'intent_match': intent_match
            })

        # Step A.5: Sort by priority_score (lowest is best), then by distance as tiebreaker
        candidates.sort(key=lambda x: (x['priority_score'], x['distance']))

        # Step B: Topological Filtering (Isomorphism Check) with Intent Prioritization
        # Filter candidates that meet node count constraint
        valid_candidates = [c for c in candidates if c['node_count'] >= input_node_count]

        if not valid_candidates:
            # Overflow handling: pick largest available style graph
            if candidates:
                print(f"Warning: No style graph meets node count constraint ({input_node_count}). "
                      f"Using largest available graph ({max(c['node_count'] for c in candidates)} nodes).")
                # Sort by node_count descending, then by priority_score
                candidates.sort(key=lambda x: (-x['node_count'], x['priority_score']))
                selected_candidate = candidates[0]
            else:
                raise ValueError("No style graphs available in collection")
        else:
            # Prioritize intent matches: if we have candidates with matching intent, prefer them
            intent_matched = [c for c in valid_candidates if c.get('intent_match', False)]
            if intent_matched and input_intent:
                # Use the best intent-matched candidate (lowest priority_score)
                selected_candidate = intent_matched[0]
            else:
                # Pick the one with lowest priority_score (which equals distance if no intent match)
                selected_candidate = valid_candidates[0]

        # Step C: The Projection (Node Mapping)
        style_mermaid = selected_candidate['mermaid']
        style_node_count = selected_candidate['node_count']
        input_mermaid = input_graph_data.get('mermaid', '')
        input_node_map = input_graph_data.get('node_map', {})

        # Extract style node names
        style_nodes = self._parse_mermaid_nodes(style_mermaid)

        # Call LLM to map nodes
        system_prompt = (
            "You are a Graph Mapper. Your task is to fit user content into "
            "an author's structural template."
        )

        user_prompt = f"""Input Graph: {input_mermaid}
Style Graph: {style_mermaid}
Input Nodes: {json.dumps(input_node_map, indent=2)}
Input Node Count: {input_node_count}
Style Node Count: {style_node_count}

Task: Map the Input Nodes (P0, P1...) into the Style Graph slots.
- The Style Graph Structure is IMMUTABLE. You must fit content into it.
- If Style has more nodes than Input, mark excess Style nodes as 'UNUSED'.
- **CRITICAL: If the Input Graph has MORE nodes than the Style Graph, you must LOGICALLY GROUP multiple input nodes into a single Style slot.**
  Example: If Input has P0, P1, P2, P3 and Style has only ROOT and CLAIM, you might map:
  {{ 'ROOT': 'P0, P1', 'CLAIM': 'P2, P3' }}
  Do not drop content. All input nodes must be assigned.
- Preserve logical flow: if Input P0 causes P1, ensure Style mapping maintains causality.
- Group related propositions logically (e.g., if P0 and P1 are both conditions, they can share a slot).

Return JSON mapping: {{ 'StyleNodeA': 'P0', 'StyleNodeB': 'P1, P2', 'StyleNodeC': 'UNUSED' }}"""

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
            node_mapping = json.loads(response)

            # Validate that all style nodes are mapped
            if not isinstance(node_mapping, dict):
                raise ValueError("Node mapping must be a dictionary")

            # Ensure all style nodes are in the mapping
            for style_node in style_nodes:
                if style_node not in node_mapping:
                    print(f"Warning: Style node '{style_node}' not in mapping, adding as 'UNUSED'")
                    node_mapping[style_node] = 'UNUSED'

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: LLM node mapping failed: {e}")
            # Fallback: naive mapping
            node_mapping = {}
            input_keys = sorted(input_node_map.keys())
            for i, style_node in enumerate(style_nodes):
                if i < len(input_keys):
                    node_mapping[style_node] = input_keys[i]
                else:
                    node_mapping[style_node] = 'UNUSED'

        # Prepare return value
        return {
            'style_mermaid': style_mermaid,
            'node_mapping': node_mapping,
            'style_metadata': {
                'node_count': style_node_count,
                'edge_types': selected_candidate['edge_types'].split(',') if isinstance(selected_candidate['edge_types'], str) else selected_candidate['edge_types'],
                'skeleton': selected_candidate.get('skeleton', ''),
                'original_text': selected_candidate['original_text'],
                'paragraph_role': selected_candidate.get('paragraph_role'),
                'intent': selected_candidate.get('intent')
            },
            'distance': selected_candidate['distance'],
            'intent_match': selected_candidate.get('intent_match', False)
        }
