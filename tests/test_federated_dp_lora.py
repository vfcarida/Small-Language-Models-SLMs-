"""
Unit tests for Module 14: Federated & Differential-Privacy LoRA Aggregation.
"""

import pytest
import torch

from src.module_14_federated_dp_lora.dp_federated_lora import (
    aggregate_federated_lora,
    clip_and_noise_gradients,
    compute_gaussian_dp_sigma,
)


def test_compute_gaussian_dp_sigma_monotonicity():
    sigma_loose = compute_gaussian_dp_sigma(epsilon=4.0, delta=1e-5, sample_rate=0.01, num_steps=100)
    sigma_strict = compute_gaussian_dp_sigma(epsilon=1.0, delta=1e-5, sample_rate=0.01, num_steps=100)
    # Stricter privacy (smaller epsilon) requires higher noise multiplier sigma
    assert sigma_strict > sigma_loose > 0.0

    with pytest.raises(ValueError, match="epsilon"):
        compute_gaussian_dp_sigma(epsilon=0.0, delta=1e-5)


def test_clip_and_noise_gradients_norm_bound():
    torch.manual_seed(42)
    per_sample_grads = torch.randn(10, 16, 4) * 5.0
    max_norm = 1.5

    # Zero noise to test pure per-sample clipping bound
    priv_grad_no_noise, stats = clip_and_noise_gradients(
        per_sample_grads, max_grad_norm=max_norm, noise_multiplier=0.0
    )
    assert priv_grad_no_noise.shape == (16, 4)
    assert stats["max_clipped_grad_norm"] <= max_norm + 1e-5
    assert stats["clipped_fraction"] > 0.5

    # With Gaussian noise injection
    priv_grad_noisy, _ = clip_and_noise_gradients(
        per_sample_grads, max_grad_norm=max_norm, noise_multiplier=0.8, seed=99
    )
    assert not torch.allclose(priv_grad_no_noise, priv_grad_noisy)


def test_aggregate_federated_lora_eliminates_cross_term_error():
    torch.manual_seed(123)
    k_clients, rank, d_in, d_out = 4, 8, 32, 32
    shared_A = torch.randn(rank, d_in)
    distinct_As = [torch.randn(rank, d_in) for _ in range(k_clients)]
    client_Bs = [torch.randn(d_out, rank) for _ in range(k_clients)]

    # 1. Naive FedAvg with distinct A_k suffers from large cross-term error
    _, _, err_naive = aggregate_federated_lora(distinct_As, client_Bs, mode="naive_fedavg")
    assert err_naive > 1.0

    # 2. FFA-LoRA (frozen shared A_0) achieves exact zero cross-term error
    s_A_ffa, s_B_ffa, err_ffa = aggregate_federated_lora([shared_A] * k_clients, client_Bs, mode="ffa_lora")
    assert s_A_ffa.shape == (rank, d_in)
    assert s_B_ffa.shape == (d_out, rank)
    assert err_ffa < 1e-5

    # 3. SVD-Exact reduces reconstruction error vs naive FedAvg on distinct A_k
    s_A_svd, s_B_svd, err_svd = aggregate_federated_lora(distinct_As, client_Bs, mode="svd_exact")
    assert s_A_svd.shape == (rank, d_in)
    assert s_B_svd.shape == (d_out, rank)
    assert err_svd < err_naive
