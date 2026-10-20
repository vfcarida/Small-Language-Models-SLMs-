"""Module 30: Differential Transformer (DiffAttn) for Attention-Noise Cancellation."""

from src.module_30_differential_attention.diff_attention import (
    DifferentialAttention,
    compute_lambda_init,
    measure_attention_snr,
)

__all__ = [
    "DifferentialAttention",
    "compute_lambda_init",
    "measure_attention_snr",
]
