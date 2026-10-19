import pytest
import torch
from src.module_06_alignment.dpo_trainer import (
    compute_dpo_loss,
    compute_simpo_loss,
    explain_dpo_math,
)


def test_compute_dpo_loss_rewards_and_accuracy():
    policy_chosen = torch.tensor([-10.0, -12.0])
    policy_rejected = torch.tensor([-20.0, -25.0])
    ref_chosen = torch.tensor([-12.0, -14.0])
    ref_rejected = torch.tensor([-18.0, -22.0])

    loss, chosen_r, rejected_r, acc = compute_dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )
    assert loss.item() > 0.0
    assert torch.all(chosen_r > rejected_r)
    assert acc == 1.0


def test_compute_simpo_loss_reference_free():
    policy_chosen = torch.tensor([-10.0, -15.0])
    policy_rejected = torch.tensor([-25.0, -30.0])
    chosen_lengths = torch.tensor([10, 15])
    rejected_lengths = torch.tensor([10, 15])

    loss, chosen_r, rejected_r, acc = compute_simpo_loss(
        policy_chosen, policy_rejected, chosen_lengths, rejected_lengths, beta=2.0, gamma=0.5
    )
    assert loss.item() > 0.0
    assert torch.all(chosen_r > rejected_r)
    assert acc == 1.0


def test_dpo_and_simpo_validation_errors():
    t = torch.tensor([-1.0])
    lengths = torch.tensor([5])
    with pytest.raises(ValueError, match="DPO beta must be strictly positive"):
        compute_dpo_loss(t, t, t, t, beta=0.0)
    with pytest.raises(ValueError, match="SimPO beta must be strictly positive"):
        compute_simpo_loss(t, t, lengths, lengths, beta=-1.0)
    with pytest.raises(ValueError, match="SimPO margin gamma must be non-negative"):
        compute_simpo_loss(t, t, lengths, lengths, beta=1.0, gamma=-0.1)


def test_explain_dpo_math_execution():
    summary = explain_dpo_math()
    assert summary["dpo_accuracy"] == 1.0
    assert summary["simpo_accuracy"] == 1.0
