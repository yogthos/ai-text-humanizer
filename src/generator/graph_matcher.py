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

    def __init__(self, config_path: str = "config.json", chroma_path: Optional[str] = None, llm_provider: Optional[LLMProvider] = None):
        """Initialize the Topological Matcher.

        Args:
            config_path: Path to configuration file.
            chroma_path: Optional custom path for ChromaDB. Defaults to atlas_cache/chroma.
            llm_provider: Optional LLM provider instance. If None, creates a new one.
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

        # Initialize LLM provider (use provided one or create new)
        if llm_provider is not None:
            self.llm_provider = llm_provider
        else:
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

    def _select_diverse_candidates(
        self,
        candidates: List[Dict[str, Any]],
        top_k: int,
        input_intent: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Select diverse candidates covering different intents.

        Args:
            candidates: List of candidate dictionaries (already sorted by priority_score).
            top_k: Number of candidates to select.
            input_intent: Input intent for prioritization.

        Returns:
            List of top_k diverse candidates.
        """
        if len(candidates) <= top_k:
            return candidates[:top_k]

        # Group candidates by intent
        intent_groups = {}
        for candidate in candidates:
            intent = candidate.get('intent', 'UNKNOWN')
            if intent not in intent_groups:
                intent_groups[intent] = []
            intent_groups[intent].append(candidate)

        # Prioritize input intent if it exists
        selected = []
        seen_intents = set()

        # First, add best candidate with matching intent (if exists)
        if input_intent:
            for intent, group in intent_groups.items():
                if intent.upper() == input_intent.upper():
                    selected.append(group[0])
                    seen_intents.add(intent)
                    break

        # Then, add one candidate from each intent group (diversity)
        for intent, group in intent_groups.items():
            if intent not in seen_intents and len(selected) < top_k:
                selected.append(group[0])
                seen_intents.add(intent)

        # Fill remaining slots with best candidates (regardless of intent)
        remaining = top_k - len(selected)
        if remaining > 0:
            for candidate in candidates:
                if candidate not in selected and len(selected) < top_k:
                    selected.append(candidate)

        return selected[:top_k]

    def _synthesize_best_fit(
        self,
        input_graph: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Use LLM to select best skeleton or synthesize a fixed version.

        Args:
            input_graph: Input graph data with 'description', 'intent', 'mermaid', etc.
            candidates: List of candidate dictionaries.

        Returns:
            Selected candidate dictionary with potentially updated 'skeleton' field.
        """
        input_description = input_graph.get('description', '')
        input_intent = input_graph.get('intent', 'UNKNOWN')
        input_mermaid = input_graph.get('mermaid', '')

        # Format candidates for prompt
        candidates_text = []
        for i, candidate in enumerate(candidates):
            skeleton = candidate.get('skeleton', '')
            intent = candidate.get('intent', 'UNKNOWN')
            distance = candidate.get('distance', float('inf'))
            node_count = candidate.get('node_count', 0)
            candidates_text.append(
                f"Candidate {i}:\n"
                f"  Intent: {intent}\n"
                f"  Skeleton: {skeleton}\n"
                f"  Distance: {distance:.3f}\n"
                f"  Node Count: {node_count}"
            )

        system_prompt = """You are a Skeleton Selector and Synthesizer. Your task is to:
1. Identify logical mismatches between input logic and candidate skeletons.
2. Select the best base candidate.
3. If needed, rewrite the skeleton to fix connector mismatches."""

        user_prompt = f"""**Input Logic:**
{input_description}

**Input Intent:** {input_intent}

**Input Graph:** {input_mermaid}

**Candidates:**
{chr(10).join(candidates_text)}

**Task:**
1. **Identify Logical Mismatch:** Check if any candidate's skeleton has connectors that contradict the input logic.
   - Example: Input is "Definition" (explaining what X is), but skeleton has "However" (contrast connector).
   - Example: Input is "And" (addition), but skeleton has "But" (contrast).

2. **Select Best Base Candidate:**
   - Choose the candidate whose skeleton structure best matches the input logic.
   - Prioritize intent matches, but also consider structural fit.

3. **Rewrite the Skeleton (if needed):**
   - If the selected candidate's connectors contradict the input (e.g., Skeleton has "However" but Input is "And"),
     you MUST output a modified skeleton with the correct connectors.
   - Keep the same structural complexity and rhythm.
   - Only change connectors/logical words, not the overall sentence structure.
   - If no rewrite is needed, return the original skeleton as-is.

**Output JSON:**
{{
  "selected_index": 0,
  "revised_skeleton": "The original or modified skeleton text here..."
}}

**Rules:**
- `selected_index` must be between 0 and {len(candidates) - 1}.
- `revised_skeleton` must be a complete, grammatically valid sentence structure.
- If the skeleton is already correct, return it unchanged.
- Only modify connectors/logical words that contradict the input intent."""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=800
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)

            selected_index = result.get('selected_index', 0)
            revised_skeleton = result.get('revised_skeleton', '')

            # Validate selected_index
            if not isinstance(selected_index, int) or selected_index < 0 or selected_index >= len(candidates):
                print(f"Warning: Invalid selected_index {selected_index}, using 0")
                selected_index = 0

            # Get selected candidate
            selected_candidate = candidates[selected_index].copy()
            original_skeleton = selected_candidate.get('skeleton', '')

            # Update skeleton if revised
            if revised_skeleton and revised_skeleton.strip():
                selected_candidate['skeleton'] = revised_skeleton.strip()
                if verbose:
                    if revised_skeleton.strip() != original_skeleton:
                        print(f"     ✏️  Skeleton revised:")
                        print(f"        Original: {original_skeleton[:80]}...")
                        print(f"        Revised:  {revised_skeleton.strip()[:80]}...")
                    else:
                        print(f"     ✓ Skeleton unchanged: {original_skeleton[:80]}...")
            elif verbose:
                print(f"     ✓ Selected candidate {selected_index}: {original_skeleton[:80]}...")

            if verbose:
                selected_intent = selected_candidate.get('intent', 'UNKNOWN')
                selected_distance = selected_candidate.get('distance', float('inf'))
                print(f"     Selected: Intent={selected_intent}, Distance={selected_distance:.3f}")

            return selected_candidate

        except (json.JSONDecodeError, Exception) as e:
            print(f"Warning: Synthesis LLM call failed: {e}")
            # Fallback: return top candidate (lowest distance)
            return candidates[0]

    def synthesize_match(
        self,
        propositions: List[str],
        input_intent: str,
        document_context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Synthesize a custom blueprint from propositions using The Architect pattern.

        Args:
            propositions: List of input proposition strings.
            input_intent: Rhetorical intent (DEFINITION, ARGUMENT, NARRATIVE, etc.).
            document_context: Optional dict with 'current_index' and 'total_paragraphs' for role filtering.
            verbose: Enable verbose logging.

        Returns:
            Dict with 'style_metadata', 'node_mapping', and 'intent'.
        """
        if verbose:
            print(f"  🏗️  Architect: Synthesizing blueprint from {len(propositions)} propositions")
            print(f"     Input intent: {input_intent}")

        # Step A: Retrieve top_k=5 candidates from ChromaDB
        candidates = []
        try:
            # Create a description from propositions for semantic search
            description = f"{' '.join(propositions[:2])[:200]}..." if len(propositions) > 2 else ' '.join(propositions)

            # Determine target role for filtering
            target_role = self._determine_role(document_context)

            # Query ChromaDB
            query_kwargs = {
                'query_texts': [description],
                'n_results': 5
            }

            # Try role filtering if available
            has_role_field = False
            try:
                sample = self.collection.get(limit=1, include=['metadatas'])
                if sample.get('metadatas') and len(sample['metadatas']) > 0:
                    has_role_field = 'paragraph_role' in sample['metadatas'][0]
            except Exception:
                pass

            if target_role and has_role_field:
                query_kwargs['where'] = {"paragraph_role": target_role}
                try:
                    results = self.collection.query(**query_kwargs)
                    if results and results.get('ids') and results['ids'][0]:
                        candidates = self._extract_candidates_from_results(results)
                except Exception as e:
                    if verbose:
                        print(f"     ⚠ Role-filtered query failed: {e}, trying without filter")
                    query_kwargs.pop('where', None)

            # Fallback to unfiltered query if needed
            if not candidates:
                try:
                    query_kwargs.pop('where', None)
                    results = self.collection.query(**query_kwargs)
                    if results and results.get('ids') and results['ids'][0]:
                        candidates = self._extract_candidates_from_results(results)
                except Exception as e:
                    if verbose:
                        print(f"     ⚠ ChromaDB query failed: {e}, using empty candidate list")
                    candidates = []

        except Exception as e:
            if verbose:
                print(f"     ⚠ Error retrieving candidates: {e}, using empty candidate list")
            candidates = []

        if verbose:
            print(f"     Retrieved {len(candidates)} style candidates")

        # Step B: The Architect - Call LLM to construct custom blueprint
        # Format indexed propositions
        indexed_propositions = []
        for i, prop in enumerate(propositions):
            indexed_propositions.append(f"P{i}: {prop}")
        indexed_propositions_text = "\n".join(indexed_propositions)

        # Format candidates
        candidates_list = []
        if candidates:
            for i, candidate in enumerate(candidates):
                skeleton = candidate.get('skeleton', '')
                intent = candidate.get('intent', 'UNKNOWN')
                distance = candidate.get('distance', float('inf'))
                candidates_list.append(
                    f"Candidate {i+1}:\n"
                    f"  Intent: {intent}\n"
                    f"  Skeleton: {skeleton}\n"
                    f"  Distance: {distance:.3f}"
                )
        else:
            candidates_list.append("No style candidates available. Create a skeleton from scratch.")

        candidates_text = "\n\n".join(candidates_list)

        system_prompt = "You are a Structural Architect. Create a sentence skeleton that fits ALL input propositions while mimicking the style of the candidates."

        user_prompt = f"""INPUT FACTS:
{indexed_propositions_text}

STYLE CANDIDATES:
{candidates_text}

INSTRUCTIONS:
1. Select the Candidate that best matches the logic.
2. **MERGE REDUNDANCIES:** If multiple input propositions state the same fact, assign them to a SINGLE slot in the skeleton. Do not create repetitive structures.
3. **Create a Revised Skeleton** that:
   - Has a slot `[P#]` for EVERY unique input fact. DO NOT DROP FACTS, but merge duplicates.
   - Uses the author's connective style (conjunctions, punctuation).
   - Fits the logical flow (Definition vs Narrative).
4. Return JSON: {{'revised_skeleton': '...', 'rationale': '...'}}"""

        try:
            response = self.llm_provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_type="editor",
                require_json=True,
                temperature=0.3,
                max_tokens=800
            )

            # Strip markdown code blocks
            response = self._strip_markdown_code_blocks(response)

            # Parse JSON
            result = json.loads(response)
            revised_skeleton = result.get('revised_skeleton', '')
            rationale = result.get('rationale', '')

            if verbose:
                print(f"     ✓ Architect created skeleton: {revised_skeleton[:80]}...")
                if rationale:
                    print(f"     Rationale: {rationale[:100]}...")

        except (json.JSONDecodeError, Exception) as e:
            if verbose:
                print(f"     ⚠ Architect LLM call failed: {e}, using fallback skeleton")
            # Fallback: create simple skeleton with all propositions
            revised_skeleton = " ".join([f"[P{i}]" for i in range(len(propositions))])
            if len(propositions) > 1:
                revised_skeleton = " ".join([f"[P{i}]" if i == 0 else f"and [P{i}]" for i in range(len(propositions))])

        # Step C: Create blueprint with direct P0->P0 mapping
        node_mapping = {f'P{i}': f'P{i}' for i in range(len(propositions))}

        blueprint = {
            'style_metadata': {
                'skeleton': revised_skeleton,
                'node_count': len(propositions),
                'edge_types': [],
                'original_text': '',
                'paragraph_role': target_role,
                'intent': input_intent
            },
            'node_mapping': node_mapping,
            'intent': input_intent,
            'distance': 0.0  # No distance for synthesized blueprints
        }

        if verbose:
            print(f"     ✓ Blueprint created with {len(node_mapping)} node mappings")

        return blueprint

    def _extract_candidates_from_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract candidate dictionaries from ChromaDB query results.

        Args:
            results: ChromaDB query results dict.

        Returns:
            List of candidate dictionaries.
        """
        candidates = []
        ids = results.get('ids', [])
        if not ids or not ids[0]:
            return candidates

        distances = results.get('distances', [[]])[0] if results.get('distances') else [0.0] * len(ids[0])
        metadatas = results.get('metadatas', [[]])[0] if results.get('metadatas') else [{}] * len(ids[0])

        for i, graph_id in enumerate(ids[0]):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else float('inf')

            candidates.append({
                'id': graph_id,
                'mermaid': metadata.get('mermaid', ''),
                'node_count': metadata.get('node_count', 0),
                'edge_types': metadata.get('edge_types', ''),
                'skeleton': metadata.get('skeleton', ''),
                'intent': metadata.get('intent'),
                'paragraph_role': metadata.get('paragraph_role'),
                'original_text': metadata.get('original_text', ''),
                'distance': distance
            })

        return candidates

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
        document_context: Optional[Dict[str, Any]] = None,
        verbose: bool = False
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
        input_description = input_graph_data.get('description', '')

        if verbose:
            print(f"  🔍 Matching graph: {input_description[:80]}...")
            print(f"     Input intent: {input_intent}, Node count: {input_node_count}")

        # Step A: Context Determination & Semantic Search
        target_role = self._determine_role(document_context)

        if verbose and target_role:
            print(f"     Target role: {target_role}")

        # Query ChromaDB with larger top_k for diversity, then select top 5 diverse candidates
        query_text = input_graph_data['description']
        retrieval_k = 20  # Retrieve more candidates for diversity selection
        top_k = 5  # Final number of candidates to pass to synthesis

        # Step A.0: Check if metadata fields exist by sampling collection
        # This prevents errors when filtering on non-existent fields
        has_role_field = False
        has_intent_field = False
        try:
            sample = self.collection.get(limit=1, include=['metadatas'])
            if sample.get('metadatas') and len(sample['metadatas']) > 0:
                sample_meta = sample['metadatas'][0]
                has_role_field = 'paragraph_role' in sample_meta
                has_intent_field = 'intent' in sample_meta
        except Exception as e:
            # If we can't sample, assume fields don't exist (safer)
            print(f"Warning: Could not check metadata schema: {e}")

        # Step A.1: Try query with role filter if target_role is set AND field exists
        results = None
        query_kwargs = {
            'query_texts': [query_text],
            'n_results': retrieval_k  # Retrieve more for diversity
        }

        # Attempt role filtering only if target_role is set AND field exists in metadata
        if target_role and has_role_field:
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

        # Step A.2: Fallback to query without role filter (pure vector search)
        if not results:
            query_kwargs.pop('where', None)  # Remove where clause
            try:
                results = self.collection.query(**query_kwargs)
            except Exception as e:
                print(f"Warning: Query without filter also failed: {e}")
                # Last resort: try with minimal parameters
                try:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=min(retrieval_k, 10)  # Reduce retrieval_k for last attempt
                    )
                except Exception as e2:
                    print(f"Error: All query attempts failed. Last error: {e2}")
                    raise ValueError("No style graphs found in collection - ChromaDB query failed")

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

        if verbose:
            print(f"     Retrieved {len(candidates)} candidates from ChromaDB")
            if len(candidates) > 0:
                print(f"     Top 5 candidates:")
                for i, c in enumerate(candidates[:5]):
                    intent = c.get('intent', 'UNKNOWN')
                    distance = c.get('distance', float('inf'))
                    node_count = c.get('node_count', 0)
                    skeleton_preview = (c.get('skeleton', '')[:50] + '...') if len(c.get('skeleton', '')) > 50 else c.get('skeleton', '')
                    print(f"       {i+1}. Intent: {intent}, Distance: {distance:.3f}, Nodes: {node_count}, Skeleton: {skeleton_preview}")

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
                selected_candidates = candidates[:top_k]  # Take top_k for synthesis
            else:
                raise ValueError("No style graphs available in collection")
        else:
            # Select top_k diverse candidates (prioritize intent diversity)
            selected_candidates = self._select_diverse_candidates(valid_candidates, top_k, input_intent)

        if verbose:
            print(f"     Selected {len(selected_candidates)} diverse candidates for synthesis")
            intents = [c.get('intent', 'UNKNOWN') for c in selected_candidates]
            print(f"     Candidate intents: {', '.join(intents)}")

        # Step B.1: Synthesize best fit using LLM
        try:
            selected_candidate = self._synthesize_best_fit(input_graph_data, selected_candidates, verbose=verbose)
        except Exception as e:
            print(f"Warning: Synthesis failed: {e}, falling back to top candidate")
            # Safety fallback: use top candidate (lowest distance)
            if valid_candidates:
                selected_candidate = valid_candidates[0]
            elif candidates:
                selected_candidate = candidates[0]
            else:
                raise ValueError("No style graphs available in collection")

        # Step C: The Projection (Node Mapping)
        style_mermaid = selected_candidate['mermaid']
        style_node_count = selected_candidate['node_count']
        input_mermaid = input_graph_data.get('mermaid', '')
        input_node_map = input_graph_data.get('node_map', {})

        # Extract style node names
        style_nodes = self._parse_mermaid_nodes(style_mermaid)

        # Call LLM to map nodes
        if verbose:
            print(f"     📍 Mapping {len(input_node_map)} input nodes to {len(style_nodes)} style nodes")
            print(f"        Input nodes: {', '.join(input_node_map.keys())}")
            print(f"        Style nodes: {', '.join(style_nodes)}")

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

            if verbose:
                print(f"     ✓ Node mapping:")
                for style_node, input_ref in node_mapping.items():
                    if input_ref == 'UNUSED':
                        print(f"        {style_node} → UNUSED")
                    else:
                        print(f"        {style_node} → {input_ref}")

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
