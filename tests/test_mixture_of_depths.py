"""
Unit tests for Module 20: Mixture-of-Depths (MoD) & Dynamic Token Routing.
"""

import pytest
import torch

from src.module_20_mixture_of_depths.mod_routing import (
    MixtureOfDepthsBlock,
    compute_mod_auxiliary_loss,
    profile_mod_compute_savings,
)


def test_mixture_of_depths_block_bypass_and_routing():
    torch.manual_seed(42)
    block = MixtureOfDepthsBlock(d_model=32, capacity_factor=0.25)
    x = torch.randn(2, 12, 32, requires_grad=True)

    out, telemetry = block(x)
    assert out.shape == (2, 12, 32)
    # Exactly K = int(0.25 * 12) = 3 tokens per sequence must be routed; 9 tokens bypass identically
    mask = telemetry["top_k_mask"]
    assert torch.all(mask.sum(dim=-1) == 3.0)

    # Verify that unrouted tokens (mask == 0) match the residual input x identically
    unrouted_diff = (out - x)[mask == 0.0]
    assert torch.allclose(unrouted_diff, torch.zeros_like(unrouted_diff), atol=1e-6)

    # Verify gradient flow through router and heavy block
    aux_loss = compute_mod_auxiliary_loss(telemetry["aux_logits"], mask)
    total_loss = out.pow(2).sum() + aux_loss
    total_loss.backward()
    assert block.router.weight.grad is not None
    assert block.causal_aux_predictor.weight.grad is not None


def test_profile_mod_compute_savings_monotonicity():
    prof_25 = profile_mod_compute_savings(seq_len=4096, capacity_factor=0.25, mod_interleave_step=1)
    prof_50 = profile_mod_compute_savings(seq_len=4096, capacity_factor=0.50, mod_interleave_step=1)

    assert prof_25["mod_gflops"] < prof_50["mod_gflops"] < prof_25["dense_gflops"]
    assert prof_25["flops_reduction_percent"] > 70.0
    assert prof_25["speedup_multiplier_x"] > 3.5

    with pytest.raises(ValueError, match="capacity_factor"):
        MixtureOfDepthsBlock(d_model=32, capacity_factor=0.0)
