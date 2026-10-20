"""Unit tests for Module 30: Differential Transformer (DiffAttn)."""

import torch

from src.module_30_differential_attention import (
    DifferentialAttention,
    compute_lambda_init,
    measure_attention_snr,
)


def test_compute_lambda_init_monotonic_with_depth() -> None:
    l0 = compute_lambda_init(0)
    l5 = compute_lambda_init(5)
    l12 = compute_lambda_init(12)
    assert abs(l0 - 0.2) < 1e-6
    assert l0 < l5 < l12 < 0.8


def test_differential_attention_cancels_identical_common_mode_maps() -> None:
    torch.manual_seed(42)
    diff_attn = DifferentialAttention(d_model=32, num_heads=4, layer_idx=1)
    x = torch.randn(2, 6, 32)
    out, maps = diff_attn.forward_with_maps(x, causal=True)

    assert out.shape == (2, 6, 32)
    assert maps["diff_attn_map"].shape == (2, 4, 6, 6)
    expected_diff = maps["attn_primary"] - maps["lambda"] * maps["attn_noise"]
    assert torch.allclose(maps["diff_attn_map"], expected_diff, atol=1e-6)


def test_measure_attention_snr_quantifies_target_concentration() -> None:
    # Construct a synthetic attention map focused on key index 2
    attn = torch.full((1, 1, 4, 6), 0.02)
    attn[..., 2] = 0.90
    snr = measure_attention_snr(attn, target_indices=[2])
    assert snr > 20.0
