"""Module 10: Lossless Speculative Decoding and Multi-Horizon Medusa Draft Heads."""

from src.module_10_speculative.speculative_decoding import (
    MedusaDraftHead,
    MedusaResidualBlock,
    SpeculativeStepResult,
    compute_residual_distribution,
    compute_speculative_speedup,
    speculative_decoding_step,
    verify_marginal_distribution_identity,
)

__all__ = [
    "MedusaDraftHead",
    "MedusaResidualBlock",
    "SpeculativeStepResult",
    "compute_residual_distribution",
    "compute_speculative_speedup",
    "speculative_decoding_step",
    "verify_marginal_distribution_identity",
]
