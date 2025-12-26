"""Unit tests for prompt builder."""

import pytest
from unittest.mock import MagicMock

from src.generation.prompt_builder import (
    PromptBuilder,
    MultiSentencePromptBuilder,
    GenerationPrompt,
    DEFAULT_TRANSITION_WORDS,
)
from src.models.plan import (
    SentencePlan,
    SentenceNode,
    SentenceRole,
    TransitionType,
)
from src.models.graph import PropositionNode, SemanticGraph
from src.ingestion.context_analyzer import GlobalContext, ParagraphContext


class TestGenerationPrompt:
    """Test GenerationPrompt dataclass."""

    def test_create_prompt(self):
        """Test creating a generation prompt."""
        prompt = GenerationPrompt(
            system_prompt="You are a writer.",
            user_prompt="Write a sentence."
        )

        assert prompt.system_prompt == "You are a writer."
        assert prompt.user_prompt == "Write a sentence."

    def test_to_messages(self):
        """Test conversion to message format."""
        prompt = GenerationPrompt(
            system_prompt="System message",
            user_prompt="User message"
        )

        messages = prompt.to_messages()

        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "System message"}
        assert messages[1] == {"role": "user", "content": "User message"}


class TestPromptBuilder:
    """Test PromptBuilder functionality."""

    @pytest.fixture
    def global_context(self):
        """Create a sample global context."""
        return GlobalContext(
            thesis="Main argument here.",
            intent="persuade",
            keywords=["key1", "key2"],
            perspective="third_person",
            style_dna="Clear, direct writing style.",
            author_name="Test Author",
            target_burstiness=0.25,
            target_sentence_length=15.0,
            top_vocabulary=["therefore", "however", "indeed"],
            total_paragraphs=3,
            processed_paragraphs=0
        )

    @pytest.fixture
    def builder(self, global_context):
        """Create a prompt builder."""
        return PromptBuilder(global_context)

    @pytest.fixture
    def sentence_node(self):
        """Create a sample sentence node."""
        prop = PropositionNode(
            id="p1",
            text="The concept evolves.",
            subject="concept",
            verb="evolves"
        )
        return SentenceNode(
            id="s1",
            propositions=[prop],
            role=SentenceRole.THESIS,
            transition=TransitionType.NONE,
            target_length=15,
            keywords=["evolves"]
        )

    @pytest.fixture
    def sentence_plan(self, sentence_node):
        """Create a sample sentence plan."""
        return SentencePlan(
            nodes=[sentence_node],
            paragraph_intent="ARGUMENT",
            paragraph_signature="SEQUENCE",
            paragraph_role="BODY"
        )

    @pytest.fixture
    def paragraph_context(self):
        """Create a sample paragraph context."""
        graph = SemanticGraph(
            nodes=[PropositionNode(id="p1", text="Test.", subject="S", verb="is")],
            edges=[]
        )
        return ParagraphContext(
            paragraph_idx=0,
            role="BODY",
            semantic_graph=graph,
            previous_summary="",
            sentence_count_target=3,
            total_propositions=1
        )

    def test_build_paragraph_prompt(self, builder, sentence_plan, paragraph_context):
        """Test building paragraph prompt."""
        prompt = builder.build_paragraph_prompt(sentence_plan, paragraph_context)

        assert isinstance(prompt, GenerationPrompt)
        assert prompt.system_prompt  # Has system prompt
        assert prompt.user_prompt  # Has user prompt
        assert "BODY" in prompt.user_prompt

    def test_build_sentence_prompt(self, builder, sentence_node, sentence_plan):
        """Test building sentence prompt."""
        prompt = builder.build_sentence_prompt(sentence_node, sentence_plan)

        assert isinstance(prompt, GenerationPrompt)
        # New example-driven prompt format - check for key elements
        assert "evolves" in prompt.user_prompt.lower()  # proposition content
        # Note: Removed explicit word count targets to reduce mechanical output

    def test_build_sentence_prompt_with_previous(self, builder, sentence_node, sentence_plan):
        """Test sentence prompt with previous sentence."""
        prompt = builder.build_sentence_prompt(
            sentence_node, sentence_plan,
            previous_sentence="This is the previous sentence."
        )

        assert "Previous sentence: This is the previous sentence." in prompt.user_prompt

    def test_build_revision_prompt(self, builder, sentence_node):
        """Test building revision prompt."""
        prompt = builder.build_revision_prompt(
            original_sentence="The concept slowly evolves over time.",
            feedback="Too long, reduce by 3 words.",
            sentence_node=sentence_node
        )

        assert "Revise this sentence" in prompt.user_prompt
        assert "Too long" in prompt.user_prompt
        assert "The concept slowly evolves" in prompt.user_prompt

    def test_system_prompt_contains_context(self, builder):
        """Test system prompt contains global context."""
        prompt = builder._build_system_prompt()

        assert "Test Author" in prompt
        assert "Clear, direct writing style" in prompt
        # Note: sentence length stats removed from system prompt to reduce mechanical output

    def test_format_sentence_spec(self, builder, sentence_node):
        """Test sentence specification formatting."""
        spec = builder._format_sentence_spec(sentence_node, 1)

        assert "Sentence 1" in spec
        assert "THESIS" in spec
        assert "15 words" in spec

    def test_transition_hint_causal(self, builder):
        """Test transition hints for causal relationship."""
        prop = PropositionNode(id="p1", text="Test.", subject="S", verb="is")
        node = SentenceNode(
            id="s1",
            propositions=[prop],
            transition=TransitionType.CAUSAL,
            target_length=10
        )

        plan = SentencePlan(nodes=[node], paragraph_role="BODY")
        prompt = builder.build_sentence_prompt(node, plan)

        # Should have descriptive hint about logical consequence (not explicit words)
        assert "logical consequence" in prompt.user_prompt.lower() or "consequence" in prompt.user_prompt.lower()


class TestMultiSentencePromptBuilder:
    """Test MultiSentencePromptBuilder functionality."""

    @pytest.fixture
    def global_context(self):
        """Create a sample global context."""
        return GlobalContext(
            thesis="Main argument.",
            intent="inform",
            keywords=["key"],
            perspective="third_person",
            style_dna="Direct style.",
            author_name="Author",
            target_burstiness=0.2,
            target_sentence_length=12.0,
            top_vocabulary=["word"],
            total_paragraphs=1,
            processed_paragraphs=0
        )

    @pytest.fixture
    def builder(self, global_context):
        """Create a multi-sentence prompt builder."""
        return MultiSentencePromptBuilder(global_context)

    @pytest.fixture
    def sentence_node(self):
        """Create a sample sentence node."""
        prop = PropositionNode(id="p1", text="Test prop.", subject="Test", verb="is")
        return SentenceNode(
            id="s1",
            propositions=[prop],
            target_length=12,
            keywords=["test"]
        )

    def test_build_alternatives_prompt(self, builder, sentence_node):
        """Test building alternatives prompt."""
        prompt = builder.build_alternatives_prompt(
            sentence_node,
            previous_sentence="Previous text.",
            num_alternatives=3
        )

        assert isinstance(prompt, GenerationPrompt)
        assert "3" in prompt.user_prompt
        assert "different versions" in prompt.user_prompt

    def test_build_scoring_prompt(self, builder, sentence_node):
        """Test building scoring prompt."""
        candidates = [
            "First option here.",
            "Second option here.",
            "Third option here."
        ]

        prompt = builder.build_scoring_prompt(candidates, sentence_node)

        assert "Score" in prompt.user_prompt
        assert "1. First option" in prompt.user_prompt
        assert "2. Second option" in prompt.user_prompt

    def test_build_scoring_prompt_with_exemplar(self, builder, sentence_node):
        """Test scoring prompt with style exemplar."""
        candidates = ["Option one.", "Option two."]

        prompt = builder.build_scoring_prompt(
            candidates, sentence_node,
            style_exemplar="This is an exemplar sentence."
        )

        assert "exemplar" in prompt.user_prompt.lower()
        assert "This is an exemplar sentence" in prompt.user_prompt


class TestPunctuationTracking:
    """Test punctuation pattern tracking."""

    @pytest.fixture
    def global_context(self):
        """Create a sample global context."""
        return GlobalContext(
            thesis="Main argument.",
            intent="inform",
            keywords=["key"],
            perspective="third_person",
            style_dna="Direct style.",
            author_name="Author",
            target_burstiness=0.2,
            target_sentence_length=12.0,
            top_vocabulary=["word"],
            total_paragraphs=1,
            processed_paragraphs=0
        )

    @pytest.fixture
    def builder(self, global_context):
        """Create a prompt builder."""
        return PromptBuilder(global_context)

    def test_punctuation_tracking(self, builder):
        """Test that punctuation is tracked in generated sentences."""
        builder.register_generated_sentence("This is a test; with semicolons.")
        builder.register_generated_sentence("This has (parentheses), commas, lots of them.")

        assert builder._used_punctuation["semicolons"] == 1
        assert builder._used_punctuation["parentheticals"] == 1
        assert builder._used_punctuation["commas"] >= 2

    def test_punctuation_reset(self, builder):
        """Test that punctuation tracking resets properly."""
        builder.register_generated_sentence("Test; semicolon.")
        builder.reset_tracking()

        assert builder._used_punctuation["semicolons"] == 0

    def test_em_dash_tracking(self, builder):
        """Test em-dash detection (both types)."""
        builder.register_generated_sentence("This—has an em-dash.")
        builder.register_generated_sentence("This--has double hyphens.")

        assert builder._used_punctuation["em_dashes"] == 2


class TestIntentClassification:
    """Test sentence intent and signature classification."""

    @pytest.fixture
    def global_context(self):
        """Create a sample global context."""
        return GlobalContext(
            thesis="Main argument.",
            intent="inform",
            keywords=["key"],
            perspective="third_person",
            style_dna="Direct style.",
            author_name="Author",
            target_burstiness=0.2,
            target_sentence_length=12.0,
            top_vocabulary=["word"],
            total_paragraphs=1,
            processed_paragraphs=0
        )

    @pytest.fixture
    def builder(self, global_context):
        """Create a prompt builder."""
        return PromptBuilder(global_context)

    def test_classify_interrogative(self, builder):
        """Test interrogative intent classification."""
        intent, _ = builder._classify_sentence_intent("What causes this effect?")
        assert intent == "INTERROGATIVE"

    def test_classify_argument(self, builder):
        """Test argument intent using semantic similarity."""
        # Use a sentence that is semantically closer to "arguing a point with support"
        intent, _ = builder._classify_sentence_intent(
            "I argue that this reasoning demonstrates my point with supporting evidence."
        )
        assert intent == "ARGUMENT"

    def test_classify_causality_signature(self, builder):
        """Test causality signature classification."""
        _, signature = builder._classify_sentence_intent(
            "The cause leads directly to this effect and consequence."
        )
        assert signature == "CAUSALITY"

    def test_classify_contrast_signature(self, builder):
        """Test contrast signature classification."""
        # Use a sentence semantically closer to "showing opposition"
        _, signature = builder._classify_sentence_intent(
            "These opposing views contradict and differ from each other."
        )
        assert signature == "CONTRAST"

    def test_classify_sequence_default(self, builder):
        """Test sequence signature for continuation."""
        _, signature = builder._classify_sentence_intent(
            "Next in the progression, we continue building on previous points."
        )
        assert signature == "SEQUENCE"


class TestTransitionWords:
    """Test transition word mappings."""

    def test_causal_transitions(self):
        """Test causal transition words exist."""
        words = DEFAULT_TRANSITION_WORDS[TransitionType.CAUSAL]
        assert "therefore" in words
        assert "thus" in words

    def test_adversative_transitions(self):
        """Test adversative transition words exist."""
        words = DEFAULT_TRANSITION_WORDS[TransitionType.ADVERSATIVE]
        assert "however" in words
        assert "but" in words

    def test_additive_transitions(self):
        """Test additive transition words exist."""
        words = DEFAULT_TRANSITION_WORDS[TransitionType.ADDITIVE]
        assert "moreover" in words
        assert "furthermore" in words

    def test_temporal_transitions(self):
        """Test temporal transition words exist."""
        words = DEFAULT_TRANSITION_WORDS[TransitionType.TEMPORAL]
        assert "then" in words
        assert "next" in words

    def test_none_transition_empty(self):
        """Test NONE transition has no words."""
        words = DEFAULT_TRANSITION_WORDS[TransitionType.NONE]
        assert words == []
