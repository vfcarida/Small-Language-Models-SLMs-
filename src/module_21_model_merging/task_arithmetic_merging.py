"""
==============================================================================
MODULE 21 — TASK-ARITHMETIC & EVOLUTIONARY MODEL MERGING (TIES / DARE / SLERP)
==============================================================================
OBJECTIVE:
    Compose multiple domain-specialized Small Language Models (e.g., Math +
    Code + Legal/SQL) derived from a shared base checkpoint theta_base into a
    single unified SLM with ZERO GPU training cost:
    1. `ties_merge_task_vectors`: TIES-Merging (Trim, Elect Sign, and Disjoint
       Merge, Yadav et al., NeurIPS 2023, arXiv:2306.01708) resolving magnitude
       redundancy and sign conflict across K task vectors tau_k = theta_k - theta_base.
    2. `dare_sparsify_delta`: Drop And REscale (DARE, Yu et al., ICML 2024,
       arXiv:2311.03099) random delta pruning with unbiased 1 / (1 - p) rescaling.
    3. `slerp_weights`: Spherical Linear Interpolation (SLERP) along the geodesic
       arc on the weight hypersphere between two homologous SLM checkpoints.

RUN COMMAND:
    python src/module_21_model_merging/task_arithmetic_merging.py
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelMergeReport:
    """Telemetry from TIES / DARE / SLERP model merging."""

    num_models: int
    density: float
    sign_conflict_fraction: float
    merged_delta_norm: float
    naive_avg_delta_norm: float


def dare_sparsify_delta(
    delta: torch.Tensor,
    drop_rate: float = 0.8,
    seed: int = 42,
) -> torch.Tensor:
    """
    Apply DARE (Drop And REscale, Yu et al., ICML 2024, arXiv:2311.03099) to a
    fine-tuned task delta tensor Delta_theta = theta_finetuned - theta_base:
        m ~ Bernoulli(1 - p)
        Delta_tilde = (m * Delta_theta) / (1 - p)

    Preserves exact expectation E[Delta_tilde] = Delta_theta while eliminating
    up to 90% of redundant parameter perturbations that cause cross-task interference.
    """
    if not (0.0 <= drop_rate < 1.0):
        raise ValueError("drop_rate must be in [0.0, 1.0).")
    if drop_rate == 0.0:
        return delta.clone()

    gen = torch.Generator(device=delta.device)
    gen.manual_seed(seed)
    keep_prob = 1.0 - drop_rate
    mask = (
        torch.rand(delta.shape, generator=gen, device=delta.device, dtype=torch.float32)
        < keep_prob
    ).to(delta.dtype)
    return (delta * mask) / keep_prob


def ties_merge_task_vectors(
    base_weight: torch.Tensor,
    task_deltas: Sequence[torch.Tensor],
    density: float = 0.3,
    scaling: float = 1.0,
) -> Tuple[torch.Tensor, ModelMergeReport]:
    """
    Merge K task vectors {tau_k}_{k=1}^K onto `base_weight` via TIES-Merging
    (Yadav et al., NeurIPS 2023, arXiv:2306.01708):
        1. Trim: For each task vector tau_k, retain only the top-`density` fraction
           of coordinates by magnitude |tau_{k, i}|, setting the rest to 0.
        2. Elect Sign: Compute aggregate sign gamma_i = sgn(sum_{k=1}^K tau_tilde_{k, i}).
        3. Disjoint Merge: For each coordinate i, average ONLY the non-zero trimmed
           values tau_tilde_{k, i} whose sign matches gamma_i.
        4. Output: theta_merged = theta_base + scaling * tau_merged.
    """
    num_tasks = len(task_deltas)
    if num_tasks == 0:
        raise ValueError("task_deltas cannot be empty.")
    if not (0.0 < density <= 1.0):
        raise ValueError("density must be in (0.0, 1.0].")

    stacked = torch.stack([d.float().reshape(-1) for d in task_deltas], dim=0)  # (K, N)
    num_params = stacked.shape[1]
    k_keep = max(1, int(round(density * num_params)))

    # Step 1: Trim redundant low-magnitude updates per task vector
    trimmed = torch.zeros_like(stacked)
    for k in range(num_tasks):
        row = stacked[k]
        _, top_idx = torch.topk(row.abs(), k=k_keep, largest=True, sorted=False)
        trimmed[k, top_idx] = row[top_idx]

    # Measure sign conflict rate across tasks before sign election
    has_pos = (trimmed > 0.0).any(dim=0)
    has_neg = (trimmed < 0.0).any(dim=0)
    conflict_fraction = float((has_pos & has_neg).to(torch.float32).mean().item())

    # Step 2: Elect aggregate sign by total magnitude mass
    elected_sign = torch.sign(trimmed.sum(dim=0))  # (N,) in {-1, 0, +1}

    # Step 3: Disjoint Merge over agreeing task vectors
    agree_mask = (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0.0)
    agree_counts = agree_mask.to(torch.float32).sum(dim=0).clamp(min=1.0)
    merged_flat = (trimmed * agree_mask.to(torch.float32)).sum(dim=0) / agree_counts

    naive_avg_flat = stacked.mean(dim=0)
    merged_delta = merged_flat.reshape(base_weight.shape)
    merged_weight = (base_weight.float() + scaling * merged_delta).to(base_weight.dtype)

    report = ModelMergeReport(
        num_models=num_tasks,
        density=density,
        sign_conflict_fraction=round(conflict_fraction, 4),
        merged_delta_norm=round(float(torch.linalg.norm(merged_flat).item()), 4),
        naive_avg_delta_norm=round(float(torch.linalg.norm(naive_avg_flat).item()), 4),
    )
    return merged_weight, report


def slerp_weights(
    w0: torch.Tensor,
    w1: torch.Tensor,
    t: float = 0.5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Spherical Linear Interpolation (SLERP) between two weight tensors w0 and w1
    at interpolation factor t in [0, 1]:
        Omega = arccos( (w0 . w1) / (||w0|| ||w1||) )
        SLERP(t; w0, w1) = (sin((1 - t)*Omega) / sin(Omega)) * w0
                         + (sin(t*Omega) / sin(Omega)) * w1
    Falls back smoothly to linear interpolation (LERP) when w0 and w1 are nearly
    collinear (sin(Omega) < eps).
    """
    if not (0.0 <= t <= 1.0):
        raise ValueError("Interpolation parameter t must be in [0.0, 1.0].")
    if w0.shape != w1.shape:
        raise ValueError("w0 and w1 must have identical shapes.")

    v0 = w0.float().reshape(-1)
    v1 = w1.float().reshape(-1)

    n0 = torch.linalg.norm(v0).clamp(min=eps)
    n1 = torch.linalg.norm(v1).clamp(min=eps)
    u0 = v0 / n0
    u1 = v1 / n1

    dot = torch.clamp(torch.dot(u0, u1), min=-1.0 + eps, max=1.0 - eps)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)

    if float(sin_omega.abs().item()) < 1e-5:
        interp_dir = (1.0 - t) * u0 + t * u1
    else:
        coeff0 = torch.sin((1.0 - t) * omega) / sin_omega
        coeff1 = torch.sin(t * omega) / sin_omega
        interp_dir = coeff0 * u0 + coeff1 * u1

    target_norm = (1.0 - t) * n0 + t * n1
    out = (interp_dir * target_norm).reshape(w0.shape)
    return out.to(w0.dtype)


def main() -> None:
    torch.manual_seed(42)
    base_w = torch.randn(64, 64)
    # 3 specialist task vectors (e.g., Math, Code, Legal) with conflicting noise
    deltas = [torch.randn(64, 64) * 0.15 for _ in range(3)]
    dare_deltas = [dare_sparsify_delta(d, drop_rate=0.5, seed=10 + i) for i, d in enumerate(deltas)]

    merged_w, report = ties_merge_task_vectors(base_w, dare_deltas, density=0.4, scaling=1.0)
    slerp_w = slerp_weights(base_w + deltas[0], base_w + deltas[1], t=0.5)

    logger.info("=" * 72)
    logger.info("MODULE 21: TASK-ARITHMETIC & EVOLUTIONARY MODEL MERGING (TIES / DARE / SLERP)")
    logger.info("=" * 72)
    logger.info(
        f"[TIES + DARE Merge] Merged {report.num_models} SLMs (density={report.density:.0%}) "
        f"| Resolved Sign Conflicts={report.sign_conflict_fraction:.1%} "
        f"| TIES Norm={report.merged_delta_norm:.3f} vs. Naive Avg Norm={report.naive_avg_delta_norm:.3f}"
    )
    logger.info(
        f"[SLERP Geodesic]    ||w0||={torch.linalg.norm(base_w + deltas[0]):.2f}, "
        f"||w1||={torch.linalg.norm(base_w + deltas[1]):.2f} -> "
        f"||SLERP(0.5)||={torch.linalg.norm(slerp_w):.2f} (Preserved Hypersphere Norm)"
    )


if __name__ == "__main__":
    main()
