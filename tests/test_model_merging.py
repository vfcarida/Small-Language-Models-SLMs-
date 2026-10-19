"""
Unit tests for Module 21: Task-Arithmetic & Evolutionary Model Merging (TIES, DARE, SLERP).
"""

import pytest
import torch

from src.module_21_model_merging.task_arithmetic_merging import (
    dare_sparsify_delta,
    slerp_weights,
    ties_merge_task_vectors,
)


def test_dare_sparsify_delta_unbiased_expectation():
    torch.manual_seed(42)
    delta = torch.ones(200, 200) * 2.5
    sparse_delta = dare_sparsify_delta(delta, drop_rate=0.8, seed=123)

    zero_frac = float((sparse_delta == 0.0).to(torch.float32).mean().item())
    assert 0.75 < zero_frac < 0.85
    # Unbiased expectation: mean of rescaled tensor stays close to original mean (2.5)
    assert abs(sparse_delta.mean().item() - 2.5) < 0.15

    with pytest.raises(ValueError, match="drop_rate"):
        dare_sparsify_delta(delta, drop_rate=1.0)


def test_ties_merge_task_vectors_resolves_sign_conflicts():
    base = torch.zeros(4, 4)
    # Construct two opposing task vectors on coordinate (0,0) with +4.0 vs -1.0
    d1 = torch.zeros(4, 4)
    d2 = torch.zeros(4, 4)
    d1[0, 0] = 4.0
    d2[0, 0] = -1.0

    merged, report = ties_merge_task_vectors(base, [d1, d2], density=1.0, scaling=1.0)
    # Elected sign at (0,0) is +1 (since 4.0 - 1.0 = +3.0 > 0), so only d1[0,0]=4.0 is averaged!
    assert abs(merged[0, 0].item() - 4.0) < 1e-5
    assert report.sign_conflict_fraction > 0.0


def test_slerp_weights_endpoints_and_norm_preservation():
    torch.manual_seed(7)
    w0 = torch.randn(32, 32)
    w0 = (w0 / torch.linalg.norm(w0)) * 10.0
    w1 = torch.randn(32, 32)
    w1 = (w1 / torch.linalg.norm(w1)) * 10.0

    assert torch.allclose(slerp_weights(w0, w1, t=0.0), w0, atol=1e-5)
    assert torch.allclose(slerp_weights(w0, w1, t=1.0), w1, atol=1e-5)

    w_mid = slerp_weights(w0, w1, t=0.5)
    # Geodesic midpoint on sphere of radius 10 must preserve norm == 10.0 (unlike LERP!)
    assert abs(torch.linalg.norm(w_mid).item() - 10.0) < 1e-4
