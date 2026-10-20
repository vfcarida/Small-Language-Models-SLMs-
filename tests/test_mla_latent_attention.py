"""Unit tests for Module 27: Multi-Head Latent Attention (MLA) & Low-Rank KV Compression."""

import torch

from src.module_27_mla_latent_attention import (
    MultiHeadLatentAttention,
    profile_mla_kv_cache_savings,
)


def test_mla_matrix_absorption_matches_materialized_attention() -> None:
    torch.manual_seed(42)
    mla = MultiHeadLatentAttention(
        d_model=32,
        num_heads=4,
        d_head=8,
        d_latent_kv=12,
        d_rope=4,
    )
    h = torch.randn(2, 10, 32)
    y_std = mla.forward_standard(h, causal=True)
    y_abs = mla.forward_absorbed(h, causal=True)

    assert y_std.shape == (2, 10, 32)
    assert torch.allclose(y_std, y_abs, atol=1e-5)


def test_mla_compress_kv_dimensions() -> None:
    mla = MultiHeadLatentAttention(
        d_model=48,
        num_heads=6,
        d_head=8,
        d_latent_kv=16,
        d_rope=6,
    )
    h = torch.randn(3, 14, 48)
    c_kv, k_rope = mla.compress_kv(h)
    assert c_kv.shape == (3, 14, 16)
    assert k_rope.shape == (3, 1, 14, 6)


def test_profile_mla_kv_cache_savings_achieves_high_compression() -> None:
    stats = profile_mla_kv_cache_savings(
        num_heads=32,
        num_kv_heads_gqa=8,
        d_head=64,
        d_latent_kv=256,
        d_rope=64,
    )
    assert stats["mla_elements_per_token"] == 320.0
    assert stats["mha_elements_per_token"] == 4096.0
    assert stats["mla_vs_mha_compression_ratio"] > 10.0
    assert stats["mla_vs_gqa_compression_ratio"] > 2.5
