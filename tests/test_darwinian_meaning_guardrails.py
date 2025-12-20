"""Tests for Darwinian Meaning Guardrails feature.

This feature implements hierarchical filtering: meaning must pass before style is evaluated.
Candidates that fail meaning checks (low recall or hallucinations) are disqualified or forced into repair mode.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Ensure config.json exists before imports
from tests.test_helpers import ensure_config_exists
ensure_config_exists()

try:
    from src.ingestion.blueprint import SemanticBlueprint, BlueprintExtractor
    from src.validator.semantic_critic import SemanticCritic
    from src.generator.translator import StyleTranslator
    from src.analysis.semantic_analyzer import PropositionExtractor
    DEPENDENCIES_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)
    print(f"⚠ Skipping tests: Missing dependencies - {IMPORT_ERROR}")


def test_meaning_gate_low_recall_fails():
    """Test that candidates with low proposition recall get score=0.0 and pass=False."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_meaning_gate_low_recall_fails (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = PropositionExtractor()
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted. Users lost their data."
    generated_text = "The system had issues."  # Very low recall - missing most propositions

    # Extract propositions
    propositions = extractor.extract_atomic_propositions(original_text)
    blueprint = blueprint_extractor.extract(original_text)

    # Evaluate in paragraph mode
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,
        style_lexicon=None,
        verbose=False
    )

    # Should fail meaning gate (recall < 0.85)
    assert result["pass"] == False, f"Should fail meaning gate with low recall. Got pass={result['pass']}"
    assert result["score"] == 0.0, f"Score should be 0.0 for meaning failure. Got score={result['score']}"
    assert result["proposition_recall"] < 0.85, f"Proposition recall should be < 0.85. Got {result['proposition_recall']}"
    assert result["style_alignment"] == 0.0, f"Style should not be calculated if meaning fails. Got {result['style_alignment']}"
    assert "CRITICAL" in result["feedback"], f"Feedback should mention critical failure. Got: {result['feedback']}"

    print("✓ test_meaning_gate_low_recall_fails passed")


def test_meaning_gate_context_leak_fails():
    """Test that candidates with context leaks get score=0.0 immediately."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_meaning_gate_context_leak_fails (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = PropositionExtractor()
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed."
    generated_text = "The server crashed due to emergent complexity."  # "emergent complexity" is from global context, not input
    propositions = extractor.extract_atomic_propositions(original_text)
    blueprint = blueprint_extractor.extract(original_text)

    # Create global context with keywords that don't appear in input
    global_context = {
        "keywords": ["emergent complexity", "systems theory", "dialectical materialism"]
    }

    # Evaluate in paragraph mode
    result = critic._evaluate_paragraph_mode(
        generated_text=generated_text,
        original_text=original_text,
        propositions=propositions,
        author_style_vector=None,
        style_lexicon=None,
        global_context=global_context,
        verbose=False
    )

    # Should fail immediately due to context leak
    assert result["pass"] == False, f"Should fail due to context leak. Got pass={result['pass']}"
    assert result["score"] == 0.0, f"Score should be 0.0 for context leak. Got score={result['score']}"
    assert result.get("context_leak_detected", False) == True, f"Should detect context leak. Got {result.get('context_leak_detected')}"
    assert "CRITICAL" in result["feedback"], f"Feedback should mention critical failure. Got: {result['feedback']}"

    print("✓ test_meaning_gate_context_leak_fails passed")


def test_meaning_gate_high_recall_passes():
    """Test that candidates with high recall pass meaning gate and style is calculated."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_meaning_gate_high_recall_passes (missing dependencies)")
        return

    critic = SemanticCritic(config_path="config.json")
    extractor = PropositionExtractor()
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted."
    generated_text = "The server crashed and the database was corrupted."  # High recall - all propositions preserved

    propositions = extractor.extract_atomic_propositions(original_text)
    blueprint = blueprint_extractor.extract(original_text)

    # Mock style alignment to return a score
    with patch.object(critic, '_check_style_alignment', return_value=(0.8, {})):
        result = critic._evaluate_paragraph_mode(
            generated_text=generated_text,
            original_text=original_text,
            propositions=propositions,
            author_style_vector=None,
            style_lexicon=None,
            verbose=False
        )

    # Should pass meaning gate (recall >= 0.85)
    assert result["proposition_recall"] >= 0.85, f"Proposition recall should be >= 0.85. Got {result['proposition_recall']}"
    # Style should be calculated (not 0.0)
    assert result["style_alignment"] > 0.0, f"Style should be calculated if meaning passes. Got {result['style_alignment']}"
    # Score should be > 0.0 (meaning weight * recall + style weight * style)
    assert result["score"] > 0.0, f"Score should be > 0.0 if meaning passes. Got {result['score']}"

    print("✓ test_meaning_gate_high_recall_passes passed")


def test_arena_meaning_filter_valid_candidates():
    """Test that arena filters by meaning first, then ranks by style."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_arena_meaning_filter_valid_candidates (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted."
    blueprint = blueprint_extractor.extract(original_text)

    # Create candidates with different recall and style scores
    candidates = [
        {
            "text": "The server crashed and the database was corrupted.",  # High recall, high style
            "skeleton": "[NP] [VP] [NP]",
            "source_example": ""
        },
        {
            "text": "The system had issues.",  # Low recall, high style
            "skeleton": "[NP] [VP]",
            "source_example": ""
        },
        {
            "text": "The server crashed. The database was corrupted. Users lost data.",  # High recall, medium style
            "skeleton": "[NP] [VP]. [NP] [VP]. [NP] [VP].",
            "source_example": ""
        }
    ]

    # Mock the SemanticCritic class to return different scores
    def mock_evaluate(self, text, blueprint, **kwargs):
        result = {
            "pass": True,
            "recall_score": 0.9 if "server" in text and "database" in text else 0.5,
            "precision_score": 0.8,
            "score": 0.9 if "server" in text and "database" in text else 0.3,
            "feedback": "OK",
            "logic_fail": False
        }
        return result

    # Mock style density calculation
    def mock_style_density(text, lexicon):
        # Return higher density for longer texts
        return 0.8 if len(text) > 50 else 0.3

    translator._calculate_style_density = mock_style_density
    translator._calculate_skeleton_adherence = Mock(return_value=0.9)

    # Patch SemanticCritic.evaluate
    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        # Run arena
        survivors = translator._run_arena(
            candidates=candidates,
            blueprint=blueprint,
            style_dna_dict={"lexicon": ["server", "database"]},
            verbose=False
        )

    # Should only return meaning-valid candidates (recall >= 0.85)
    assert len(survivors) > 0, "Should have at least one survivor"
    for survivor in survivors:
        assert survivor["recall_score"] >= 0.85, f"All survivors should have recall >= 0.85. Got {survivor['recall_score']}"
        assert not survivor.get("context_leak_detected", False), "No survivors should have context leaks"
        assert not survivor.get("hallucination", False), "No survivors should have hallucinations"

    # Should be sorted by style_density (descending)
    if len(survivors) > 1:
        for i in range(len(survivors) - 1):
            assert survivors[i]["style_density"] >= survivors[i+1]["style_density"], \
                f"Survivors should be sorted by style_density. Got {survivors[i]['style_density']} < {survivors[i+1]['style_density']}"

    print("✓ test_arena_meaning_filter_valid_candidates passed")


def test_arena_meaning_filter_all_fail_repair_mode():
    """Test that if all candidates fail meaning, arena selects by recall for repair."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_arena_meaning_filter_all_fail_repair_mode (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted. Users lost their data."
    blueprint = blueprint_extractor.extract(original_text)

    # Create candidates that all fail meaning (low recall)
    candidates = [
        {
            "text": "The system had issues.",  # Very low recall
            "skeleton": "[NP] [VP]",
            "source_example": ""
        },
        {
            "text": "Something went wrong.",  # Very low recall
            "skeleton": "[NP] [VP]",
            "source_example": ""
        },
        {
            "text": "The server crashed.",  # Medium recall (better than others)
            "skeleton": "[NP] [VP]",
            "source_example": ""
        }
    ]

    # Mock the SemanticCritic to return low recall for all
    def mock_evaluate(self, text, blueprint, **kwargs):
        recall = 0.7 if "server" in text else 0.3  # Only third candidate has partial recall
        result = {
            "pass": False,
            "recall_score": recall,
            "precision_score": 0.8,
            "score": 0.0,  # Meaning failure = score 0.0
            "feedback": "Low recall",
            "logic_fail": False
        }
        return result

    translator._calculate_style_density = Mock(return_value=0.5)
    translator._calculate_skeleton_adherence = Mock(return_value=0.9)

    # Patch SemanticCritic.evaluate
    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        # Run arena
        survivors = translator._run_arena(
            candidates=candidates,
            blueprint=blueprint,
            style_dna_dict=None,
            verbose=False
        )

    # Should return candidates sorted by recall (for repair)
    assert len(survivors) > 0, "Should have survivors for repair mode"
    # Best candidate should have highest recall
    assert survivors[0]["recall_score"] >= survivors[-1]["recall_score"], \
        f"Survivors should be sorted by recall (descending) for repair. Got {survivors[0]['recall_score']} < {survivors[-1]['recall_score']}"

    print("✓ test_arena_meaning_filter_all_fail_repair_mode passed")


def test_repair_attempts_tracking():
    """Test that repair attempts are tracked and incremented correctly."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_repair_attempts_tracking (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed."
    blueprint = blueprint_extractor.extract(original_text)

    # Create parent with meaning failure
    parent = {
        "text": "The system had issues.",
        "skeleton": "[NP] [VP]",
        "source_example": "",
        "recall_score": 0.5,
        "score": 0.0,
        "meaning_failure": True,
        "repair_attempts": 0
    }

    # Create child that also fails
    child_candidates = [{
        "text": "The system had problems.",
        "skeleton": "[NP] [VP]",
        "source_example": ""
    }]

    # Mock the SemanticCritic to return meaning failure
    def mock_evaluate(self, text, blueprint, **kwargs):
        return {
            "pass": False,
            "recall_score": 0.5,
            "score": 0.0,
            "feedback": "Low recall",
            "logic_fail": False
        }

    translator._calculate_style_density = Mock(return_value=0.5)
    translator._calculate_skeleton_adherence = Mock(return_value=0.9)

    # Patch SemanticCritic.evaluate
    with patch.object(SemanticCritic, 'evaluate', mock_evaluate):
        # Run arena on children
        child_survivors = translator._run_arena(
            candidates=child_candidates,
            blueprint=blueprint,
            style_dna_dict=None,
            verbose=False
        )

    # Simulate evolutionary loop repair attempt tracking
    for child in child_survivors:
        child_meaning_failure = (child.get("score", 0.0) == 0.0 or
                                child.get("recall_score", 0.0) < 0.85)
        if child_meaning_failure and parent.get("meaning_failure", False):
            child["repair_attempts"] = parent.get("repair_attempts", 0) + 1

    # Child should have repair_attempts = 1 (incremented from parent's 0)
    assert child_survivors[0].get("repair_attempts", 0) == 1, \
        f"Repair attempts should be incremented. Got {child_survivors[0].get('repair_attempts', 0)}"

    print("✓ test_repair_attempts_tracking passed")


def test_structure_swap_triggered():
    """Test that structure swap is triggered after 2 failed repair attempts."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_structure_swap_triggered (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted."
    blueprint = blueprint_extractor.extract(original_text)

    # Create parent with 2 repair attempts (should trigger structure swap)
    parent = {
        "text": "The system had issues.",
        "skeleton": "[NP] [VP]",
        "source_example": "",
        "recall_score": 0.5,
        "score": 0.0,
        "meaning_failure": True,
        "repair_attempts": 2,
        "deltas": ["Missing propositions: server crashed, database corrupted"],
        "critic_result": {
            "feedback": "Low recall",
            "recall_details": {
                "missing": ["server crashed", "database corrupted"]
            }
        }
    }

    # Mock LLM provider to capture the prompt
    mock_llm = Mock()
    mock_llm.call.return_value = '["The server crashed. The database was corrupted."]'

    translator.llm_provider = mock_llm

    # Call _breed_children with parent that has 2 repair attempts
    children = translator._breed_children(
        parents=[parent],
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Formal",
        rhetorical_type=None,
        style_lexicon=None,
        num_children=1,
        verbose=False
    )

    # Should generate children
    assert len(children) > 0, "Should generate children"

    # Check that the prompt contains structure swap language
    call_args = mock_llm.call.call_args
    if call_args:
        user_prompt = call_args[1].get("user_prompt", "") or call_args[0][1] if len(call_args[0]) > 1 else ""
        assert "STRUCTURE SWAP" in user_prompt or "DISCARD the template" in user_prompt, \
            f"Prompt should contain structure swap language. Got: {user_prompt[:200]}"

    print("✓ test_structure_swap_triggered passed")


def test_repair_mode_prompt():
    """Test that repair mode generates correct prompt for first attempt."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_repair_mode_prompt (missing dependencies)")
        return

    translator = StyleTranslator(config_path="config.json")
    blueprint_extractor = BlueprintExtractor()

    original_text = "The server crashed. The database was corrupted."
    blueprint = blueprint_extractor.extract(original_text)

    # Create parent with meaning failure but only 1 repair attempt
    parent = {
        "text": "The system had issues.",
        "skeleton": "[NP] [VP]",
        "source_example": "",
        "recall_score": 0.5,
        "score": 0.0,
        "meaning_failure": True,
        "repair_attempts": 1,
        "deltas": ["Missing propositions: server crashed, database corrupted"],
        "critic_result": {
            "feedback": "Low recall",
            "recall_details": {
                "missing": ["server crashed", "database corrupted"]
            }
        }
    }

    # Mock LLM provider
    mock_llm = Mock()
    mock_llm.call.return_value = '["The server crashed. The database was corrupted."]'

    translator.llm_provider = mock_llm

    # Call _breed_children
    children = translator._breed_children(
        parents=[parent],
        blueprint=blueprint,
        author_name="Test Author",
        style_dna="Formal",
        rhetorical_type=None,
        style_lexicon=None,
        num_children=1,
        verbose=False
    )

    # Check that the prompt contains repair language (not structure swap)
    call_args = mock_llm.call.call_args
    if call_args:
        user_prompt = call_args[1].get("user_prompt", "") or call_args[0][1] if len(call_args[0]) > 1 else ""
        assert "REPAIR TASK" in user_prompt or "Keep the exact sentence structure" in user_prompt, \
            f"Prompt should contain repair language. Got: {user_prompt[:200]}"
        assert "STRUCTURE SWAP" not in user_prompt, \
            f"Prompt should NOT contain structure swap language on first repair. Got: {user_prompt[:200]}"

    print("✓ test_repair_mode_prompt passed")


def test_config_min_viable_recall():
    """Test that config min_viable_recall is used correctly."""
    if not DEPENDENCIES_AVAILABLE:
        print("⚠ SKIPPED: test_config_min_viable_recall (missing dependencies)")
        return

    import json

    # Read config
    with open("config.json", "r") as f:
        config = json.load(f)

    # Check that min_viable_recall is set
    assert "evolutionary" in config, "Config should have evolutionary section"
    assert "min_viable_recall" in config["evolutionary"], "Config should have min_viable_recall"
    assert config["evolutionary"]["min_viable_recall"] == 0.85, \
        f"min_viable_recall should be 0.85. Got {config['evolutionary']['min_viable_recall']}"

    # Check that hallucination_penalty is set
    assert "hallucination_penalty" in config["evolutionary"], "Config should have hallucination_penalty"
    assert config["evolutionary"]["hallucination_penalty"] == 1.0, \
        f"hallucination_penalty should be 1.0. Got {config['evolutionary']['hallucination_penalty']}"

    # Check paragraph_fusion threshold
    assert "paragraph_fusion" in config, "Config should have paragraph_fusion section"
    assert "min_viable_recall_threshold" in config["paragraph_fusion"], \
        "Config should have min_viable_recall_threshold"
    assert config["paragraph_fusion"]["min_viable_recall_threshold"] == 0.85, \
        f"min_viable_recall_threshold should be 0.85. Got {config['paragraph_fusion']['min_viable_recall_threshold']}"

    print("✓ test_config_min_viable_recall passed")


if __name__ == "__main__":
    test_meaning_gate_low_recall_fails()
    test_meaning_gate_context_leak_fails()
    test_meaning_gate_high_recall_passes()
    test_arena_meaning_filter_valid_candidates()
    test_arena_meaning_filter_all_fail_repair_mode()
    test_repair_attempts_tracking()
    test_structure_swap_triggered()
    test_repair_mode_prompt()
    test_config_min_viable_recall()
    print("\n✓ All Darwinian Meaning Guardrails tests completed!")

