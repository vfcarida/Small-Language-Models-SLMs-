"""Module 12: Selective State-Space Models (Mamba) and Hybrid SSM-Attention SLMs (Hymba / Jamba)."""

from src.module_12_hybrid_ssm.hybrid_ssm_block import (
    HybridMambaAttentionBlock,
    SelectiveSSMBlock,
    profile_hybrid_memory_scaling,
    selective_ssm_scan,
)

__all__ = [
    "HybridMambaAttentionBlock",
    "SelectiveSSMBlock",
    "profile_hybrid_memory_scaling",
    "selective_ssm_scan",
]
