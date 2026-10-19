"""Unit tests for Module 05B Cayley Stiefel Manifold Rotation & INT4 KV-Cache Quantization."""

import pytest
import torch

from src.module_05_quantization.cayley_kv_quant import (
    build_normalized_hadamard,
    cayley_orthogonal_step,
    optimize_cayley_rotation,
    quantize_kv_cache_int4,
)


def test_cayley_orthogonal_step_preserves_stiefel_manifold() -> None:
    torch.manual_seed(42)
    dim = 32
    R0 = build_normalized_hadamard(dim)
    grad = torch.randn(dim, dim)

    R1 = cayley_orthogonal_step(R0, grad, lr=0.35)
    eye = torch.eye(dim)
    ortho_err = torch.max(torch.abs(R1.t() @ R1 - eye)).item()
    assert ortho_err < 1e-5


def test_optimize_cayley_rotation_int3_beats_naive_rtn() -> None:
    torch.manual_seed(42)
    d_out, d_in, seq_len = 32, 32, 16
    w = torch.randn(d_out, d_in) * 0.05
    # Extreme weight outlier in column 0 crushes quantization bins for columns 1..15 in group 0
    w[:, 0] *= 15.0
    x = torch.randn(seq_len, d_in)
    x[:, 0] *= 0.1

    best_R, stats = optimize_cayley_rotation(
        w, x, w_bit=3, q_group_size=16, steps=15, lr=0.25
    )
    assert best_R.shape == (d_in, d_in)
    assert stats["orthogonality_error"] < 1e-5
    assert stats["cayley_mse"] <= stats["hadamard_mse"]
    assert stats["cayley_mse"] < stats["rtn_mse"]


def test_quantize_kv_cache_int4_fidelity_and_compression() -> None:
    torch.manual_seed(7)
    k = torch.randn(2, 4, 32, 32)
    k[:, :, :, 0] *= 10.0  # Persistent Key channel outlier
    v = torch.randn(2, 4, 32, 32)

    k_q, v_q, stats = quantize_kv_cache_int4(k, v, group_size=16)
    assert k_q.shape == k.shape
    assert v_q.shape == v.shape
    assert stats["compression_ratio_vs_fp16"] == pytest.approx(2.67, abs=0.05)
    assert stats["attention_cosine_similarity"] > 0.99

    with pytest.raises(ValueError, match="must be 4D tensors"):
        quantize_kv_cache_int4(torch.randn(4, 32, 32), torch.randn(4, 32, 32))
