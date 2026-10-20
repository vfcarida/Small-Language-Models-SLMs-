"""Unit tests for Module 29: Machine Unlearning & Targeted Knowledge Erasure (NPO & RMU)."""

import torch

from src.module_29_machine_unlearning import (
    compute_npo_unlearning_loss,
    compute_rmu_representation_loss,
)


def test_npo_gradient_weight_vanishes_as_sample_is_forgotten() -> None:
    ref_logps = torch.tensor([-2.0, -2.0, -2.0])
    # Case 1: Policy still remembers forget set (logp == ref_logp)
    pol_remember = torch.tensor([-2.0, -2.0, -2.0], requires_grad=True)
    out_remember = compute_npo_unlearning_loss(pol_remember, ref_logps, beta=0.5)

    # Case 2: Policy has already forgotten the sample (logp << ref_logp)
    pol_forgotten = torch.tensor([-25.0, -25.0, -25.0], requires_grad=True)
    out_forgotten = compute_npo_unlearning_loss(pol_forgotten, ref_logps, beta=0.5)

    assert abs(out_remember["mean_forget_gradient_weight"].item() - 0.5) < 1e-5
    assert out_forgotten["mean_forget_gradient_weight"].item() < 1e-4


def test_npo_unlearning_step_decreases_forget_logprob() -> None:
    pol_forget = torch.tensor([-1.5, -1.8], requires_grad=True)
    ref_forget = torch.tensor([-1.5, -1.8])
    pol_retain = torch.tensor([-1.2, -1.0], requires_grad=True)
    ref_retain = torch.tensor([-1.2, -1.0])

    res = compute_npo_unlearning_loss(
        policy_forget_logps=pol_forget,
        ref_forget_logps=ref_forget,
        beta=0.2,
        policy_retain_logps=pol_retain,
        ref_retain_logps=ref_retain,
    )
    res["total_loss"].backward()
    # Gradient w.r.t. pol_forget is positive so gradient descent lowers pol_forget
    assert pol_forget.grad is not None
    assert torch.all(pol_forget.grad > 0.0)


def test_rmu_representation_loss_steers_forget_to_random_control_vector() -> None:
    torch.manual_seed(29)
    h_forget = torch.randn(4, 16, requires_grad=True)
    h_retain = torch.randn(4, 16, requires_grad=True)
    h_ref_retain = h_retain.detach().clone()
    u = torch.randn(16)

    out = compute_rmu_representation_loss(
        policy_forget_hidden=h_forget,
        policy_retain_hidden=h_retain,
        ref_retain_hidden=h_ref_retain,
        random_unit_vector=u,
        steering_coeff=5.0,
        alpha_retain=10.0,
    )
    out["total_loss"].backward()

    assert out["forget_misdirection_loss"].item() > 0.0
    assert out["retain_drift_loss"].item() == 0.0
    assert h_forget.grad is not None
