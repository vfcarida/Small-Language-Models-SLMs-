"""
Module 29: Machine Unlearning & Targeted Knowledge Erasure (NPO & RMU)
======================================================================

Implements bounded-gradient Machine Unlearning algorithms for Small Language
Models (<4B parameters) to satisfy regulatory Right-to-be-Forgotten (GDPR/LGPD)
and AI TRiSM hazardous capability removal without catastrophic collapse:
1. ``compute_npo_unlearning_loss``: Negative Preference Optimization (Zhang et al.,
   COLM 2024) bounding forget-set gradient updates via a reference-anchored
   Bradley-Terry negative log-sigmoid objective plus retain-set KL regularization.
2. ``compute_rmu_representation_loss``: Representation Misdirection for Unlearning
   (Li et al., ICML 2024, WMDP) steering forget-set hidden activations toward a
   scaled random unit control vector $c \\cdot \\mathbf{u}$ while pinning retain-set
   representations to the frozen reference model.

References:
    - Zhang et al. (COLM 2024). "Negative Preference Optimization: From Catastrophic
      Collapse to Effective Unlearning." arXiv:2404.05868.
    - Li et al. (ICML 2024). "The WMDP Benchmark: Measuring and Reducing Malicious
      Use With Unlearning (RMU)." arXiv:2403.03218.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def compute_npo_unlearning_loss(
    policy_forget_logps: torch.Tensor,
    ref_forget_logps: torch.Tensor,
    beta: float = 0.1,
    policy_retain_logps: Optional[torch.Tensor] = None,
    ref_retain_logps: Optional[torch.Tensor] = None,
    retain_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute Negative Preference Optimization (NPO) unlearning loss (Zhang et al., 2024).

    Unlike naive Gradient Ascent ($\\mathcal{L}_{\\text{GA}} = \\log \\pi_\\theta(y_f \\mid x_f)$)
    whose gradient norm stays constant and causes catastrophic model collapse,
    NPO formulates unlearning as a lower-bounded DPO objective using only negative
    (forget) samples anchored against $\\pi_{\\text{ref}}$:

    $$\\mathcal{L}_{\\text{NPO}}(\\theta) = -\\frac{2}{\\beta} \\mathbb{E}_{(x_f, y_f) \\sim \\mathcal{D}_{\\text{forget}}}\\!\\left[\\log \\sigma\\!\\left(-\\beta \\log \\frac{\\pi_\\theta(y_f \\mid x_f)}{\\pi_{\\text{ref}}(y_f \\mid x_f)}\\right)\\right]$$

    As $\\pi_\\theta(y_f \\mid x_f) \\ll \\pi_{\\text{ref}}(y_f \\mid x_f)$, the per-sample gradient
    weight $\\sigma(\\beta (\\log \\pi_\\theta - \\log \\pi_{\\text{ref}})) \\to 0$, smoothly halting
    parameter updates once the target sample is forgotten.

    Args:
        policy_forget_logps: Sequence log-probabilities $\\log \\pi_\\theta(y_f \\mid x_f)$ of shape ``(B,)``.
        ref_forget_logps: Frozen reference log-probabilities $\\log \\pi_{\\text{ref}}(y_f \\mid x_f)$ of shape ``(B,)``.
        beta: Inverse temperature $\\beta > 0$ controlling divergence sensitivity.
        policy_retain_logps: Optional sequence log-probs on $\\mathcal{D}_{\\text{retain}}$ of shape ``(B_r,)``.
        ref_retain_logps: Optional reference log-probs on $\\mathcal{D}_{\\text{retain}}$ of shape ``(B_r,)``.
        retain_weight: Weight $\\lambda_{\\text{retain}} \\ge 0$ for retain-set NLL/margin preservation.

    Returns:
        Dictionary containing:
            - ``"total_loss"``: Combined NPO + retain loss tensor.
            - ``"npo_forget_loss"``: NPO forget objective.
            - ``"retain_loss"``: Retain utility preservation loss.
            - ``"mean_forget_gradient_weight"``: Detached mean adaptive weight $\\sigma(\\beta \\Delta \\log p) \\in (0, 1)$.
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be positive, got {beta}.")
    if policy_forget_logps.shape != ref_forget_logps.shape:
        raise ValueError("policy_forget_logps and ref_forget_logps must have matching shapes.")

    log_ratio_forget = policy_forget_logps - ref_forget_logps.detach()
    npo_forget_loss = -(2.0 / beta) * F.logsigmoid(-beta * log_ratio_forget).mean()

    with torch.no_grad():
        grad_weight = torch.sigmoid(beta * log_ratio_forget).mean()

    if policy_retain_logps is not None:
        if ref_retain_logps is not None:
            retain_loss = F.mse_loss(policy_retain_logps, ref_retain_logps.detach())
        else:
            retain_loss = -policy_retain_logps.mean()
    else:
        retain_loss = torch.zeros((), device=policy_forget_logps.device, dtype=policy_forget_logps.dtype)

    total_loss = npo_forget_loss + retain_weight * retain_loss
    return {
        "total_loss": total_loss,
        "npo_forget_loss": npo_forget_loss,
        "retain_loss": retain_loss,
        "mean_forget_gradient_weight": grad_weight,
    }


def compute_rmu_representation_loss(
    policy_forget_hidden: torch.Tensor,
    policy_retain_hidden: torch.Tensor,
    ref_retain_hidden: torch.Tensor,
    random_unit_vector: torch.Tensor,
    steering_coeff: float = 6.5,
    alpha_retain: float = 100.0,
) -> Dict[str, torch.Tensor]:
    """Compute Representation Misdirection for Unlearning (RMU) loss (Li et al., ICML 2024).

    Scrambles intermediate residual activations on $\\mathcal{D}_{\\text{forget}}$ toward a
    fixed random direction $c \\cdot \\mathbf{u}$ ($\\|\\mathbf{u}\\|_2 = 1$) while locking
    activations on $\\mathcal{D}_{\\text{retain}}$ to match the frozen reference model:

    $$\\mathcal{L}_{\\text{RMU}} = \\mathbb{E}_{x_f \\sim \\mathcal{D}_{\\text{forget}}}\\!\\left[\\|\\mathbf{h}_\\theta(x_f) - c \\cdot \\mathbf{u}\\|_2^2\\right] + \\alpha_{\\text{retain}} \\mathbb{E}_{x_r \\sim \\mathcal{D}_{\\text{retain}}}\\!\\left[\\|\\mathbf{h}_\\theta(x_r) - \\mathbf{h}_{\\text{ref}}(x_r)\\|_2^2\\right]$$

    Args:
        policy_forget_hidden: Policy activations on forget batch of shape ``(..., d_model)``.
        policy_retain_hidden: Policy activations on retain batch of shape ``(..., d_model)``.
        ref_retain_hidden: Frozen reference activations on retain batch of shape ``(..., d_model)``.
        random_unit_vector: Fixed random direction vector $\\mathbf{u}$ of shape ``(d_model,)``.
        steering_coeff: Scalar multiplier $c > 0$ scaling the target random direction.
        alpha_retain: Retain regularization coefficient $\\alpha_{\\text{retain}} \\ge 0$.

    Returns:
        Dictionary with ``"total_loss"``, ``"forget_misdirection_loss"``, and ``"retain_drift_loss"``.
    """
    d_model = policy_forget_hidden.shape[-1]
    if random_unit_vector.shape != (d_model,):
        raise ValueError(
            f"random_unit_vector must have shape ({d_model},), got {tuple(random_unit_vector.shape)}."
        )
    if policy_retain_hidden.shape != ref_retain_hidden.shape:
        raise ValueError("policy_retain_hidden and ref_retain_hidden must have identical shapes.")

    u_norm = random_unit_vector / torch.linalg.norm(random_unit_vector).clamp(min=1e-8)
    target_forget = (steering_coeff * u_norm).expand_as(policy_forget_hidden)

    forget_loss = F.mse_loss(policy_forget_hidden, target_forget.detach())
    retain_loss = F.mse_loss(policy_retain_hidden, ref_retain_hidden.detach())
    total_loss = forget_loss + alpha_retain * retain_loss

    return {
        "total_loss": total_loss,
        "forget_misdirection_loss": forget_loss,
        "retain_drift_loss": retain_loss,
    }


def main() -> None:
    """Run a deterministic demonstration of NPO and RMU unlearning objectives."""
    torch.manual_seed(42)
    pol_f = torch.tensor([-1.5, -2.0, -1.8], requires_grad=True)
    ref_f = torch.tensor([-1.4, -1.9, -1.7])
    npo = compute_npo_unlearning_loss(pol_f, ref_f, beta=0.2)
    print(
        f"[Module 29] NPO loss={npo['npo_forget_loss'].item():.4f} | "
        f"Forget gradient weight={npo['mean_forget_gradient_weight'].item():.4f}"
    )


if __name__ == "__main__":
    main()
