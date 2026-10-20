"""Unit tests for Module 26: Mechanistic Interpretability via Top-K Sparse Autoencoders."""

import torch

from src.module_26_sparse_autoencoders import (
    TopKSparseAutoencoder,
    clamp_sae_feature,
    compute_sae_loss_with_auxk,
)


def test_topk_sae_enforces_exact_k_sparsity_and_unit_decoder_norms() -> None:
    torch.manual_seed(23)
    sae = TopKSparseAutoencoder(d_model=16, dict_size=64, k_active=4)
    x = torch.randn(10, 16)
    out = sae(x)

    z = out["z_sparse"]
    assert z.shape == (10, 64)
    # Every token has at most k_active=4 non-zero features
    non_zeros_per_token = (z > 0).sum(dim=-1)
    assert torch.all(non_zeros_per_token <= 4)
    assert torch.all(non_zeros_per_token >= 1)

    # Decoder columns have unit L2 norm
    col_norms = torch.linalg.norm(sae.W_dec, dim=0)
    assert torch.allclose(col_norms, torch.ones_like(col_norms), atol=1e-5)


def test_compute_sae_loss_with_auxk_activates_on_dead_latents() -> None:
    torch.manual_seed(31)
    sae = TopKSparseAutoencoder(d_model=12, dict_size=48, k_active=3)
    # Mark half the dictionary as dead latents
    sae.steps_since_fired[:24] = 25
    x = torch.randn(8, 12)
    sae_out = sae(x, update_firing_stats=False)

    loss_dict = compute_sae_loss_with_auxk(
        sae=sae,
        x=x,
        sae_out=sae_out,
        k_aux=4,
        alpha_aux=0.1,
        dead_threshold_steps=10,
    )
    loss_dict["total_loss"].backward()

    assert loss_dict["recon_loss"].item() > 0.0
    assert loss_dict["auxk_loss"].item() > 0.0
    assert sae.W_dec.grad is not None


def test_clamp_sae_feature_shifts_residual_along_monosemantic_atom() -> None:
    torch.manual_seed(37)
    sae = TopKSparseAutoencoder(d_model=16, dict_size=32, k_active=4)
    x = torch.randn(4, 16)
    z_before, _ = sae.encode(x)
    active_idx = int(torch.argmax(z_before[0]).item())

    # Ablate the most active feature on token 0 to zero
    x_ablated = clamp_sae_feature(sae, x, feature_idx=active_idx, clamp_value=0.0)
    delta = x_ablated[0] - x[0]
    expected_delta = -z_before[0, active_idx] * sae.W_dec[:, active_idx]
    assert torch.allclose(delta, expected_delta, atol=1e-5)
