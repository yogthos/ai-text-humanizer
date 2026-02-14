"""Persona configurations for different authors.

Each author maps to a persona that forces subjective, non-robotic output.
The persona defines HOW the author would EXPERIENCE and REACT to information,
not just how they would explain it.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PersonaConfig:
    """Configuration for an author's persona/voice."""

    # Core identity
    archetype: str  # "Frantic Scholar", "Weary Correspondent"
    emotional_lens: str  # How they FEEL about information
    voice_mode: str  # "journal", "letter", "confession", "report"

    # Anti-robot constraints
    banned_starters: List[str] = field(default_factory=list)
    required_patterns: List[str] = field(default_factory=list)

    # Vocabulary themes (adjectives that fit the author's style)
    adjective_themes: List[str] = field(default_factory=list)

    # Sensory anchors (ground abstract ideas in physical details)
    sensory_anchors: List[str] = field(default_factory=list)

    # Emotional reactions to different content types
    reaction_to_complexity: str = "struggle with it"
    reaction_to_certainty: str = "assert it personally"
    reaction_to_uncertainty: str = "acknowledge the doubt"

    # Structural preferences
    prefer_fragments: bool = True
    prefer_em_dashes: bool = True
    prefer_exclamations: bool = False
    max_consecutive_long_sentences: int = 2


# Author-specific configurations
PERSONA_CONFIGS = {
    "default": PersonaConfig(
        archetype="Opinionated Diarist recording raw thoughts",
        emotional_lens="personal reaction over objective analysis",
        voice_mode="private diary",
        banned_starters=[
            "It is",
            "There are",
            "Moreover",
            "However",
            "Therefore",
            "Thus",
            "Furthermore",
            "It should be noted",
            "In conclusion",
        ],
        required_patterns=["fragment", "em_dash"],
        adjective_themes=[],  # Will be populated from author corpus
        sensory_anchors=["page", "thought", "moment", "silence"],
        reaction_to_complexity="wrestle with it visibly on the page",
        reaction_to_certainty="assert it with personal conviction",
        reaction_to_uncertainty="admit the doubt, probe it",
        prefer_fragments=True,
        prefer_em_dashes=True,
        prefer_exclamations=False,
        max_consecutive_long_sentences=2,
    ),
}


def get_persona_config(author: str) -> PersonaConfig:
    """Get persona configuration for an author.

    Falls back to default if author not found.
    """
    # Try exact match first
    if author in PERSONA_CONFIGS:
        return PERSONA_CONFIGS[author]

    # Try partial match (e.g., "Lovecraft" matches "H.P. Lovecraft")
    author_lower = author.lower()
    for key, config in PERSONA_CONFIGS.items():
        if author_lower in key.lower() or key.lower() in author_lower:
            return config

    # Fall back to default
    return PERSONA_CONFIGS["default"]
