"""Data structures for sentence planning."""

from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class SentenceNode:
    """Represents a single sentence in the plan."""

    id: str
    propositions: List[str]
    role: str  # e.g., "Thesis", "Elaboration", "Contrast", "Definition"
    transition_type: str  # e.g., "Causal", "Adversative", "Flow", "None"
    target_length: int  # Target word count from style profile
    keywords: List[str] = field(default_factory=list)  # Mandatory vocabulary
    global_indices: List[int] = field(default_factory=list)  # Original proposition indices
    style_template: Optional[str] = None  # Retrieved style candidate text for syntax template (RAG)
    intended_subject: Optional[str] = None  # Subject derived from propositions (for tracking)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "id": self.id,
            "propositions": self.propositions,
            "role": self.role,
            "transition_type": self.transition_type,
            "target_length": self.target_length,
            "keywords": self.keywords,
            "global_indices": self.global_indices,
            "style_template": self.style_template,
            "intended_subject": self.intended_subject
        }

    @classmethod
    def from_json(cls, data: dict) -> "SentenceNode":
        """Deserialize from JSON-compatible dict."""
        return cls(
            id=data["id"],
            propositions=data["propositions"],
            role=data["role"],
            transition_type=data["transition_type"],
            target_length=data["target_length"],
            keywords=data.get("keywords", []),
            global_indices=data.get("global_indices", []),
            style_template=data.get("style_template"),
            intended_subject=data.get("intended_subject")
        )

    def __repr__(self) -> str:
        return f"SentenceNode(id={self.id}, role={self.role}, props={len(self.propositions)}, length={self.target_length})"


@dataclass
class SentencePlan:
    """Represents a complete plan for a paragraph."""

    nodes: List[SentenceNode]
    paragraph_intent: str  # Overall paragraph intent (DEFINITION, ARGUMENT, NARRATIVE)
    paragraph_signature: str  # Logical signature (CONTRAST, CAUSALITY, etc.)

    def to_json(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "nodes": [node.to_json() for node in self.nodes],
            "paragraph_intent": self.paragraph_intent,
            "paragraph_signature": self.paragraph_signature
        }

    @classmethod
    def from_json(cls, data: dict) -> "SentencePlan":
        """Deserialize from JSON-compatible dict."""
        return cls(
            nodes=[SentenceNode.from_json(node_data) for node_data in data["nodes"]],
            paragraph_intent=data["paragraph_intent"],
            paragraph_signature=data["paragraph_signature"]
        )

    def __repr__(self) -> str:
        return f"SentencePlan(intent={self.paragraph_intent}, signature={self.paragraph_signature}, nodes={len(self.nodes)})"

