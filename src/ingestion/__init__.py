"""Ingestion module for text processing."""

from .proposition_extractor import PropositionExtractor, SVOTriple, PropositionNode

__all__ = [
    "PropositionExtractor",
    "PropositionNode",
    "SVOTriple",
]
