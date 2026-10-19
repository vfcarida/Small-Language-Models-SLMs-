"""
Unit tests for Module 12: State-Space & Hybrid Mamba-Transformer SLMs.
"""

import pytest
import torch

from src.module_12_hybrid_ssm.hybrid_ssm_block import (
    HybridMambaAttentionBlock,
    SelectiveSSMBlock,
    profile_hybrid_memory_scaling,
    selective_ssm_scan,
)


def test_selective_ssm_scan_causality_and_shapes():
    torch.manual_seed(42)
    b, l, d_inner, d_state = 2, 16, 32, 8
    u = torch.randn(b, l, d_inner)
    delta = torch.nn.functional.softplus(torch.randn(b, l, d_inner))
    A = -torch.exp(torch.randn(d_inner, d_state))
    B = torch.randn(b, l, d_state)
    C = torch.randn(b, l, d_state)
    D = torch.ones(d_inner)

    y, h_last = selective_ssm_scan(u, delta, A, B, C, D)
    assert y.shape == (b, l, d_inner)
    assert h_last.shape == (b, d_inner, d_state)

    # Verify strict causality: perturbing future token t=10 must not change output at t<=9
    u_perturbed = u.clone()
    u_perturbed[:, 10:, :] += 5.0
    y_perturbed, _ = selective_ssm_scan(u_perturbed, delta, A, B, C, D)
    assert torch.allclose(y[:, :10, :], y_perturbed[:, :10, :], atol=1e-6)
    assert not torch.allclose(y[:, 10:, :], y_perturbed[:, 10:, :], atol=1e-4)


def test_selective_ssm_block_recurrent_step_equivalence():
    torch.manual_seed(7)
    block = SelectiveSSMBlock(d_model=32, d_state=8, expand=2)
    x = torch.randn(2, 10, 32)
    y_seq, h_seq = block(x)
    assert y_seq.shape == (2, 10, 32)
    assert h_seq is not None and h_seq.shape == (2, 64, 8)

    # Step recurrently token-by-token and verify equivalence with full sequence scan
    h_step = None
    step_outputs = []
    for t in range(x.shape[1]):
        y_t, h_step = block.step_recurrent(x[:, t, :], state=h_step)
        step_outputs.append(y_t.unsqueeze(1))
    y_recurrent = torch.cat(step_outputs, dim=1)
    assert torch.allclose(y_seq, y_recurrent, atol=1e-5)
    assert torch.allclose(h_seq, h_step, atol=1e-5)


def test_hybrid_mamba_attention_block_and_memory_profile():
    torch.manual_seed(11)
    hybrid = HybridMambaAttentionBlock(
        d_model=64,
        num_q_heads=4,
        num_kv_heads=2,
        d_state=8,
    )
    x = torch.randn(2, 12, 64)
    out, final_state = hybrid(x)
    assert out.shape == (2, 12, 64)
    assert final_state.shape == (2, 128, 8)

    profile = profile_hybrid_memory_scaling(context_length=32768, num_layers=28, attention_layer_ratio=0.125)
    assert profile["hybrid_ssm_gqa_cache_mb"] < profile["gqa_cache_mb"] < profile["mha_cache_mb"]
    assert profile["reduction_vs_gqa_x"] >= 5.0
    assert profile["reduction_vs_mha_x"] >= 40.0

    with pytest.raises(ValueError, match="context_length"):
        profile_hybrid_memory_scaling(context_length=0)
