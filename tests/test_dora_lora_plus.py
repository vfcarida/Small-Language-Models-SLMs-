"""Unit tests for Module 31: Weight-Decomposed Low-Rank Adaptation (DoRA) & LoRA+."""

import torch

from src.module_31_dora_lora_plus import (
    DoRALinear,
    build_lora_plus_param_groups,
)


def test_dora_linear_identity_at_zero_b_init_and_exact_merge() -> None:
    torch.manual_seed(17)
    dora = DoRALinear(in_features=24, out_features=12, rank=4, lora_alpha=8.0)
    # At init, lora_B is zero and magnitude == ||W_0||_row, so W_DoRA == W_0
    assert torch.allclose(dora.effective_weight(), dora.weight_base, atol=1e-5)

    # Perturb lora_B and magnitude, then verify merge_into_linear parity
    with torch.no_grad():
        dora.lora_B.normal_(0.0, 0.1)
        dora.magnitude.mul_(1.15)

    x = torch.randn(5, 24)
    y_unmerged = dora(x)
    merged_linear = dora.merge_into_linear()
    y_merged = merged_linear(x)

    assert torch.allclose(y_unmerged, y_merged, atol=1e-5)


def test_build_lora_plus_param_groups_assigns_asymmetric_learning_rates() -> None:
    dora = DoRALinear(in_features=16, out_features=8, rank=4)
    groups = build_lora_plus_param_groups(dora, base_lr=1e-4, lora_plus_lambda=12.0)
    lr_by_name = {g["group_name"]: g["lr"] for g in groups}

    assert abs(lr_by_name["lora_A"] - 1e-4) < 1e-10
    assert abs(lr_by_name["lora_B_fast"] - 1.2e-3) < 1e-10
    assert lr_by_name["lora_B_fast"] / lr_by_name["lora_A"] == 12.0
