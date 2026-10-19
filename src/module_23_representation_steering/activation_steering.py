"""
==============================================================================
MODULE 23 — ACTIVATION STEERING & REPRESENTATION ENGINEERING (RepE / CAA)
==============================================================================
OBJECTIVE:
    Control SLM runtime behavior (conciseness, truthfulness, AI TRiSM safety
    compliance, or refusal/bias ablation) directly in the residual activation
    space with ZERO weight updates and ZERO extra prompt tokens:
    1. `extract_steering_vector`: Computes a unit-norm concept direction v_hat
       in R^(d_model) from contrastive residual-stream pairs {(h_i^+, h_i^-)}
       via Contrastive Activation Addition (`mean_diff`, Rimsky et al., ACL 2024,
       arXiv:2312.06681) or Representation Engineering First Principal Component
       (`pca`, Zou et al., 2023, arXiv:2310.01405).
    2. `apply_activation_steering`: Injects alpha * v_hat into residual hidden
       states with optional L2 norm preservation so downstream RMSNorm layers
       remain within their calibrated operating regime.
    3. `ablate_concept_subspace`: Exact orthogonal rank-1 projection
       h_ablated = h - (h . v_hat) * v_hat (Arditi et al., NeurIPS 2024,
       arXiv:2406.11717) removing a target concept direction across all tokens.

RUN COMMAND:
    python src/module_23_representation_steering/activation_steering.py
==============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Literal, Tuple

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_steering_vector(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    method: Literal["mean_diff", "pca"] = "pca",
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    Extract a unit-norm concept steering vector v_hat in R^(d_model) from
    contrastive residual activations `pos_activations` and `neg_activations`
    of shape (N, d_model):
        - `mean_diff` (CAA, Rimsky et al., ACL 2024):
          v = mean_i(h_i^+ - h_i^-),  v_hat = v / ||v||_2
        - `pca` (RepE Reading Vector, Zou et al., 2023):
          Leading right singular vector u_1 of centered difference matrix
          D_i = (h_i^+ - h_i^-), oriented so mean_i(D_i . u_1) > 0.

    Returns:
        Tuple of (unit_steering_vector, directional_separation_cosine).
    """
    if pos_activations.ndim != 2 or pos_activations.shape != neg_activations.shape:
        raise ValueError("pos_activations and neg_activations must have matching 2D shape (N, d_model).")
    if pos_activations.shape[0] < 2:
        raise ValueError("At least 2 contrastive pairs are required.")

    diffs = (pos_activations.float() - neg_activations.float())  # (N, d_model)
    mean_diff = diffs.mean(dim=0)

    if method == "mean_diff":
        raw_v = mean_diff
    elif method == "pca":
        # RepE Linear Artificial Tomography (Zou et al., 2023, Sec 3.1.2):
        # First right singular vector of difference vectors D_i = h_i^+ - h_i^-
        # maximizes sum_i (D_i . v)^2 along the principal contrastive axis.
        _, _, vh = torch.linalg.svd(diffs, full_matrices=False)
        raw_v = vh[0]
        if float(torch.dot(raw_v, mean_diff).item()) < 0.0:
            raw_v = -raw_v
    else:
        raise ValueError(f"Unsupported steering extraction method: {method}")

    v_hat = raw_v / torch.linalg.norm(raw_v).clamp(min=eps)

    # Compute mean cosine alignment across all contrastive difference pairs
    diff_norms = torch.linalg.norm(diffs, dim=-1, keepdim=True).clamp(min=eps)
    unit_diffs = diffs / diff_norms
    alignment = float((unit_diffs @ v_hat).mean().item())
    return v_hat.to(pos_activations.dtype), round(alignment, 4)


def apply_activation_steering(
    hidden_states: torch.Tensor,
    steering_vector: torch.Tensor,
    coeff: float = 1.5,
    preserve_norm: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Steer residual-stream hidden states `hidden_states` of shape (..., d_model)
    along `steering_vector` with multiplier `coeff`:
        h_raw = h + coeff * v_hat
        If preserve_norm=True:
            h_steered = h_raw * (||h||_2 / ||h_raw||_2)
    """
    if hidden_states.shape[-1] != steering_vector.shape[0]:
        raise ValueError("Last dimension of hidden_states must match steering_vector dimension.")

    h_f = hidden_states.float()
    v_f = steering_vector.float()
    v_hat = v_f / torch.linalg.norm(v_f).clamp(min=eps)

    h_steered = h_f + float(coeff) * v_hat
    if preserve_norm:
        orig_norm = torch.linalg.norm(h_f, dim=-1, keepdim=True)
        new_norm = torch.linalg.norm(h_steered, dim=-1, keepdim=True).clamp(min=eps)
        h_steered = h_steered * (orig_norm / new_norm)

    return h_steered.to(hidden_states.dtype)


def ablate_concept_subspace(
    hidden_states: torch.Tensor,
    concept_vector: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Perform exact orthogonal rank-1 directional ablation (Arditi et al., NeurIPS 2024,
    arXiv:2406.11717) removing all projection onto `concept_vector`:
        h_ablated = h - (h . v_hat) * v_hat
    Guarantees that <h_ablated, v_hat> == 0 for every token position.
    """
    if hidden_states.shape[-1] != concept_vector.shape[0]:
        raise ValueError("Last dimension of hidden_states must match concept_vector dimension.")

    h_f = hidden_states.float()
    v_f = concept_vector.float()
    v_hat = v_f / torch.linalg.norm(v_f).clamp(min=eps)

    proj_scalar = torch.matmul(h_f, v_hat).unsqueeze(-1)  # (..., 1)
    h_ablated = h_f - proj_scalar * v_hat
    return h_ablated.to(hidden_states.dtype)


def main() -> None:
    torch.manual_seed(42)
    d_model = 64
    true_concept_dir = torch.randn(d_model)
    true_concept_dir = true_concept_dir / torch.linalg.norm(true_concept_dir)

    # Synthesize 32 contrastive residual pairs (e.g., Truthful vs. Hallucinated)
    base_acts = torch.randn(32, d_model)
    pos_acts = base_acts + 2.0 * true_concept_dir.unsqueeze(0) + torch.randn(32, d_model) * 0.2
    neg_acts = base_acts - 2.0 * true_concept_dir.unsqueeze(0) + torch.randn(32, d_model) * 0.2

    v_pca, align_score = extract_steering_vector(pos_acts, neg_acts, method="pca")
    cosine_with_truth = float(torch.dot(v_pca, true_concept_dir).item())

    test_hidden = torch.randn(2, 8, d_model)
    steered = apply_activation_steering(test_hidden, v_pca, coeff=2.5, preserve_norm=True)
    ablated = ablate_concept_subspace(test_hidden, v_pca)
    max_residual_proj = float(torch.matmul(ablated, v_pca).abs().max().item())

    logger.info("=" * 72)
    logger.info("MODULE 23: ACTIVATION STEERING & REPRESENTATION ENGINEERING (RepE / CAA)")
    logger.info("=" * 72)
    logger.info(
        f"[Concept Extraction] Pairwise Separation Alignment={align_score:.4f} "
        f"| Recovery Cosine with Ground-Truth Direction={cosine_with_truth:.4f}"
    )
    logger.info(
        f"[Runtime Steering]   Projection Before={torch.matmul(test_hidden, v_pca).mean():.3f} "
        f"-> Steered={torch.matmul(steered, v_pca).mean():.3f} "
        f"| Orthogonal Ablation Residual={max_residual_proj:.2e} (Exact Zero Subspace)"
    )


if __name__ == "__main__":
    main()
