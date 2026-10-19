"""Module 16: Latent-Space Continuous Reasoning (Coconut & Recurrent Depth SLMs)."""

from src.module_16_latent_reasoning.coconut_reasoning import (
    AdaptivePonderingGate,
    LatentReasoningTrace,
    LatentThoughtReasoner,
    compare_discrete_cot_vs_latent_coconut,
)

__all__ = [
    "AdaptivePonderingGate",
    "LatentReasoningTrace",
    "LatentThoughtReasoner",
    "compare_discrete_cot_vs_latent_coconut",
]
