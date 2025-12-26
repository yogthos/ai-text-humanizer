"""Generation module for data-driven style transfer."""

from .evolutionary_generator import (
    EvolutionarySentenceGenerator,
    EvolutionaryParagraphGenerator,
    Candidate,
    GenerationState,
)
from .data_driven_generator import (
    DataDrivenStyleTransfer,
    TransferResult,
    DocumentTransferResult,
    create_transfer_pipeline,
    load_profile_and_create_transfer,
)

__all__ = [
    # Evolutionary generation
    "EvolutionarySentenceGenerator",
    "EvolutionaryParagraphGenerator",
    "Candidate",
    "GenerationState",
    # Data-driven transfer pipeline
    "DataDrivenStyleTransfer",
    "TransferResult",
    "DocumentTransferResult",
    "create_transfer_pipeline",
    "load_profile_and_create_transfer",
]
