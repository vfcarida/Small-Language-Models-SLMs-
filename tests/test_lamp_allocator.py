"""
Unit tests for Module 17: Layer-Adaptive Mixed-Precision (LAMP) & Hessian Bit Allocator.
"""

import pytest
import torch

from src.module_17_mixed_precision.lamp_allocator import (
    allocate_mixed_precision_bits,
    apply_lamp_mixed_precision,
    compute_layer_hessian_sensitivity,
)


def test_compute_layer_hessian_sensitivity_monotonicity():
    torch.manual_seed(42)
    w = torch.randn(32, 32)
    x_low = torch.randn(16, 32) * 0.5
    x_high = torch.randn(16, 32) * 3.0

    sens_low = compute_layer_hessian_sensitivity(w, x_low)
    sens_high = compute_layer_hessian_sensitivity(w, x_high)
    assert sens_high > sens_low > 0.0


def test_allocate_mixed_precision_bits_respects_budget_and_sensitivity():
    params = [1000, 1000, 1000, 1000]
    sensitivities = [100.0, 1.0, 1.0, 50.0]

    assigned, avg_bits = allocate_mixed_precision_bits(
        layer_num_params=params,
        layer_sensitivities=sensitivities,
        target_avg_bits=3.0,
        candidate_bits=(2, 3, 4, 8),
    )
    assert avg_bits <= 3.0 + 1e-6
    # Highest sensitivity layers (0 and 3) must receive >= bits than middle layers (1 and 2)
    assert assigned[0] >= assigned[1]
    assert assigned[3] >= assigned[2]

    with pytest.raises(ValueError, match="target_avg_bits"):
        allocate_mixed_precision_bits(params, sensitivities, target_avg_bits=1.5, candidate_bits=(2, 4))


def test_apply_lamp_mixed_precision_beats_uniform_mse():
    torch.manual_seed(11)
    scales = [6.0, 0.8, 0.7, 0.9, 0.8, 5.5]
    weights = [torch.randn(32, 32) * (0.2 * s) for s in scales]
    activations = [torch.randn(24, 32) * s for s in scales]

    q_weights, report = apply_lamp_mixed_precision(
        weights=weights,
        calib_activations=activations,
        target_avg_bits=3.0,
        candidate_bits=(2, 3, 4, 8),
    )
    assert len(q_weights) == len(weights)
    assert report.achieved_avg_bits <= report.target_avg_bits + 1e-6
    assert report.lamp_output_mse < report.uniform_output_mse
    assert report.mse_reduction_ratio > 1.2
