"""Unit tests for Module 25: Matryoshka Representation Learning & Elastic Sub-Network Slicing."""

import torch

from src.module_25_matryoshka_minitron import (
    ElasticMatryoshkaLinear,
    compute_matryoshka_loss,
    rank_minitron_structural_importance,
)


def test_compute_matryoshka_loss_backpropagates_nested_prefixes() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(4, 6, 32, requires_grad=True)
    weight = torch.randn(12, 32, requires_grad=True)
    targets = torch.randint(0, 12, (4, 6))

    out = compute_matryoshka_loss(
        hidden_states=hidden,
        classifier_weight=weight,
        targets=targets,
        nesting_dims=[8, 16, 32],
    )
    loss = out["total_loss"]
    loss.backward()

    assert loss.item() > 0.0
    assert set(out["per_dim_losses"].keys()) == {8, 16, 32}
    assert hidden.grad is not None
    # Prefix channels [0:8] receive gradient from all 3 nested objectives
    prefix_grad_norm = hidden.grad[..., :8].norm().item()
    tail_grad_norm = hidden.grad[..., 16:32].norm().item()
    assert prefix_grad_norm > tail_grad_norm


def test_rank_minitron_structural_importance_orders_channels_and_layers() -> None:
    torch.manual_seed(11)
    cal = torch.randn(8, 10, 16) * 0.1
    cal[..., 5] *= 50.0  # Channel 5 is dominant
    cal[..., 12] *= 25.0  # Channel 12 is second most salient

    x_in_0 = torch.randn(8, 10, 16)
    x_out_0 = x_in_0.clone()  # Layer 0 is a near-identity block (BI ~ 0)
    x_in_1 = torch.randn(8, 10, 16)
    x_out_1 = -x_in_1.clone()  # Layer 1 strongly transforms activations (BI ~ 2)

    ranking = rank_minitron_structural_importance(
        calibration_activations=cal,
        layer_inputs=[x_in_0, x_in_1],
        layer_outputs=[x_out_0, x_out_1],
    )

    top_channels = ranking["channel_ranking"][:2].tolist()
    assert top_channels == [5, 12]
    assert ranking["layer_ranking"] == [1, 0]
    assert ranking["layer_block_influence"][1] > ranking["layer_block_influence"][0]


def test_elastic_matryoshka_linear_retains_top_channel_fidelity() -> None:
    torch.manual_seed(19)
    layer = ElasticMatryoshkaLinear(in_features=16, out_features=8, bias=False)
    x = torch.randn(5, 16) * 0.01
    x[:, [3, 9, 14, 1]] *= 100.0  # 4 dominant channels carry >99% of signal energy

    ranking_info = rank_minitron_structural_importance(x)
    full_out_before = layer(x)

    layer.reorder_by_importance(ranking_info["channel_ranking"])
    full_out_after = layer(x, active_in_features=16)
    sliced_out_top4 = layer(x, active_in_features=4)

    # Full-width output is mathematically invariant under column permutation
    assert torch.allclose(full_out_before, full_out_after, atol=1e-5)
    # 4-channel slice closely approximates full 16-channel output
    rel_err = torch.linalg.norm(full_out_after - sliced_out_top4) / torch.linalg.norm(full_out_after)
    assert rel_err.item() < 0.05
