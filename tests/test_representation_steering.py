"""
Unit tests for Module 23: Activation Steering & Representation Engineering (RepE & CAA).
"""

import pytest
import torch

from src.module_23_representation_steering.activation_steering import (
    ablate_concept_subspace,
    apply_activation_steering,
    extract_steering_vector,
)


def test_extract_steering_vector_pca_and_mean_diff():
    torch.manual_seed(42)
    d = 32
    true_dir = torch.randn(d)
    true_dir = true_dir / torch.linalg.norm(true_dir)

    base = torch.randn(24, d)
    pos = base + 3.0 * true_dir.unsqueeze(0) + torch.randn(24, d) * 0.1
    neg = base - 3.0 * true_dir.unsqueeze(0) + torch.randn(24, d) * 0.1

    v_pca, align_pca = extract_steering_vector(pos, neg, method="pca")
    v_mean, align_mean = extract_steering_vector(pos, neg, method="mean_diff")

    assert abs(torch.linalg.norm(v_pca).item() - 1.0) < 1e-5
    assert abs(torch.linalg.norm(v_mean).item() - 1.0) < 1e-5
    assert align_pca > 0.95
    assert align_mean > 0.95
    assert torch.dot(v_pca, true_dir).item() > 0.95


def test_apply_activation_steering_and_orthogonal_ablation():
    torch.manual_seed(11)
    h = torch.randn(2, 6, 32)
    v = torch.randn(32)
    v = v / torch.linalg.norm(v)

    orig_norms = torch.linalg.norm(h, dim=-1)
    steered = apply_activation_steering(h, v, coeff=2.0, preserve_norm=True)
    steered_norms = torch.linalg.norm(steered, dim=-1)
    # Norm preservation invariant
    assert torch.allclose(orig_norms, steered_norms, atol=1e-5)

    # Orthogonal subspace ablation must annihilate projection onto v
    ablated = ablate_concept_subspace(h, v)
    residual_dot = torch.matmul(ablated, v)
    assert torch.allclose(residual_dot, torch.zeros_like(residual_dot), atol=1e-5)

    with pytest.raises(ValueError, match="matching 2D shape"):
        extract_steering_vector(torch.randn(4, 8), torch.randn(5, 8))
