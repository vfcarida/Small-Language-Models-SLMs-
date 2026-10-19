"""
Unit tests for Module 15: Ternary 1.58-Bit Native SLMs (BitNet b1.58).
"""

import pytest
import torch

from src.module_15_bitnet_ternary.bitnet_linear import (
    BitLinear,
    profile_bitnet_hardware_savings,
    quantize_activations_absmax_int8,
    quantize_weights_absmean_ternary,
)


def test_quantize_weights_absmean_ternary_values():
    torch.manual_seed(42)
    w = torch.randn(32, 64) * 2.5
    w_ternary, gamma, zero_frac = quantize_weights_absmean_ternary(w)

    unique_vals = set(w_ternary.unique().tolist())
    assert unique_vals.issubset({-1.0, 0.0, 1.0})
    assert gamma.item() > 0.0
    assert 0.1 < zero_frac < 0.9


def test_quantize_activations_absmax_int8_bounds():
    torch.manual_seed(7)
    x = torch.randn(4, 12, 32) * 5.0
    x_int, scale = quantize_activations_absmax_int8(x, bits=8)
    assert x_int.max().item() <= 127.0
    assert x_int.min().item() >= -127.0
    assert torch.all(scale > 0.0)


def test_bitlinear_ste_gradients_and_add_only_parity():
    torch.manual_seed(19)
    layer = BitLinear(in_features=32, out_features=16)
    x = torch.randn(2, 8, 32, requires_grad=True)

    y_ste = layer(x)
    assert y_ste.shape == (2, 8, 16)
    loss = y_ste.pow(2).sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.linalg.norm(layer.weight.grad) > 0.0

    # Verify exact numerical equivalence with multiplier-free add/subtract kernel
    y_add = layer.forward_add_only(x.detach())
    assert torch.allclose(y_ste.detach(), y_add, atol=1e-5)

    hw = profile_bitnet_hardware_savings(num_params_b=1.54)
    assert hw["bitnet_1_58b_vram_mb"] < hw["int4_vram_mb"] < hw["fp16_vram_mb"]
    assert hw["memory_reduction_vs_fp16_x"] > 9.0

    with pytest.raises(ValueError, match="num_params_b"):
        profile_bitnet_hardware_savings(num_params_b=0.0)
