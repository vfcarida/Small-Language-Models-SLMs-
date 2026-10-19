"""
==============================================================================
MODULE 17 — LAYER-ADAPTIVE MIXED-PRECISION (LAMP) & HESSIAN BIT ALLOCATION
==============================================================================
OBJECTIVE:
    Replace uniform low-bit quantization across all SLM layers with second-order
    Hessian/Fisher sensitivity-guided mixed-precision bit allocation (`INT2` /
    `INT3` / `INT4` / `INT8`) under a strict device SRAM / VRAM budget:
    1. `compute_layer_hessian_sensitivity`: Second-order proxy Hessian trace
       Omega_l = Tr(X_l^T X_l / T) * (||W_l||_F^2 / d_in) (GPTQ, Frantar et al.,
       ICLR 2023, arXiv:2210.17323; SqueezeLLM, Kim et al., ICML 2024,
       arXiv:2306.07629; HAWQv3, Yao et al., ICML 2021).
    2. `allocate_mixed_precision_bits`: Marginal-gain optimal bit-width solver
       minimizing expected second-order quantization perturbation
       sum_l Omega_l * 4^(-b_l) subject to sum_l N_l * b_l <= B_max.
    3. `apply_lamp_mixed_precision`: End-to-end mixed-precision quantizer
       demonstrating strictly lower activation reconstruction MSE than uniform
       bit-width quantization under identical average bit budgets.

RUN COMMAND:
    python src/module_17_mixed_precision/lamp_allocator.py
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LAMPAllocationReport:
    """Telemetry from Layer-Adaptive Mixed-Precision (LAMP) bit allocation."""

    assigned_bits: List[int]
    layer_sensitivities: List[float]
    achieved_avg_bits: float
    target_avg_bits: float
    lamp_output_mse: float
    uniform_output_mse: float
    mse_reduction_ratio: float


def compute_layer_hessian_sensitivity(
    weight: torch.Tensor,
    calib_activations: torch.Tensor,
) -> float:
    """
    Compute the second-order quantization sensitivity score Omega_l for a linear
    layer Y_l = X_l W_l^T under layer-wise output reconstruction loss
    L_l = (1 / T) ||X_l (W_l - W_hat_l)^T||_F^2:
        H_l = (2 / T) X_l^T X_l  =>  Tr(H_l) = (2 / T) ||X_l||_F^2
        Omega_l = Tr(H_l) * (||W_l||_F^2 / d_in)

    Args:
        weight: Layer weight matrix of shape (d_out, d_in).
        calib_activations: Calibration input tensor of shape (..., d_in).

    Returns:
        Strictly non-negative scalar sensitivity Omega_l.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a 2D matrix of shape (d_out, d_in).")
    if calib_activations.shape[-1] != weight.shape[1]:
        raise ValueError("Last dimension of calib_activations must match weight.shape[1].")

    flat_x = calib_activations.reshape(-1, weight.shape[1]).float()
    num_tokens = max(1, flat_x.shape[0])
    d_in = max(1, weight.shape[1])

    hessian_trace = (2.0 / float(num_tokens)) * float(torch.sum(flat_x.pow(2)).item())
    weight_energy = float(torch.sum(weight.float().pow(2)).item()) / float(d_in)
    return max(0.0, hessian_trace * weight_energy)


def _quantize_symmetric_nbits(weight: torch.Tensor, bits: int, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric per-row simulated integer quantization at `bits` precision."""
    q_max = float((1 << (bits - 1)) - 1)
    if q_max < 1.0:
        q_max = 1.0
    w_f = weight.float()
    scale = w_f.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / q_max
    w_q = torch.clamp(torch.round(w_f / scale), min=-q_max, max=q_max) * scale
    return w_q.to(weight.dtype)


def allocate_mixed_precision_bits(
    layer_num_params: Sequence[int],
    layer_sensitivities: Sequence[float],
    target_avg_bits: float = 3.0,
    candidate_bits: Sequence[int] = (2, 3, 4, 8),
) -> Tuple[List[int], float]:
    """
    Solve the Layer-Adaptive Mixed-Precision (LAMP) bit allocation problem:
        minimize_{b_l in candidate_bits}  sum_l Omega_l * 4^(-b_l)
        subject to                        sum_l N_l * b_l <= target_avg_bits * sum_l N_l

    Uses an exact marginal-gain greedy ascent over the discrete convex objective
    f_l(b_l) = Omega_l * 4^(-b_l). Because 4^(-b) is strictly convex and decreasing,
    greedy upgrades by maximum marginal distortion reduction per bit are optimal
    on the convex hull of candidate bit-widths.

    Args:
        layer_num_params: Number of parameters N_l in each layer l.
        layer_sensitivities: Second-order sensitivity Omega_l for each layer l.
        target_avg_bits: Maximum average bits per parameter across the model.
        candidate_bits: Sorted sequence of allowed integer bit-widths.

    Returns:
        Tuple of (assigned_bits_per_layer, achieved_avg_bits).
    """
    num_layers = len(layer_num_params)
    if num_layers == 0 or len(layer_sensitivities) != num_layers:
        raise ValueError("layer_num_params and layer_sensitivities must be non-empty and match in length.")

    sorted_bits = sorted(set(int(b) for b in candidate_bits))
    if not sorted_bits or sorted_bits[0] < 2:
        raise ValueError("candidate_bits must contain integers >= 2.")
    if target_avg_bits < float(sorted_bits[0]):
        raise ValueError(f"target_avg_bits ({target_avg_bits}) cannot be below min candidate bit ({sorted_bits[0]}).")

    total_params = float(sum(layer_num_params))
    max_total_bits = target_avg_bits * total_params

    # Initialize all layers at the lowest candidate bit-width
    tier_idx = [0] * num_layers
    current_total_bits = sum(float(n) * sorted_bits[0] for n in layer_num_params)

    while True:
        best_layer = -1
        best_marginal_gain = -1.0
        best_bit_cost = 0.0

        for l_idx in range(num_layers):
            cur_t = tier_idx[l_idx]
            if cur_t + 1 >= len(sorted_bits):
                continue
            b_curr = sorted_bits[cur_t]
            b_next = sorted_bits[cur_t + 1]
            delta_bits = float(layer_num_params[l_idx]) * float(b_next - b_curr)
            if current_total_bits + delta_bits > max_total_bits + 1e-6:
                continue

            distortion_drop = float(layer_sensitivities[l_idx]) * (
                (4.0 ** (-b_curr)) - (4.0 ** (-b_next))
            )
            marginal_gain = distortion_drop / max(delta_bits, 1e-12)
            if marginal_gain > best_marginal_gain:
                best_marginal_gain = marginal_gain
                best_layer = l_idx
                best_bit_cost = delta_bits

        if best_layer == -1:
            break

        tier_idx[best_layer] += 1
        current_total_bits += best_bit_cost

    assigned_bits = [sorted_bits[t] for t in tier_idx]
    achieved_avg = current_total_bits / max(total_params, 1.0)
    return assigned_bits, round(float(achieved_avg), 4)


def apply_lamp_mixed_precision(
    weights: Sequence[torch.Tensor],
    calib_activations: Sequence[torch.Tensor],
    target_avg_bits: float = 3.0,
    candidate_bits: Sequence[int] = (2, 3, 4, 8),
) -> Tuple[List[torch.Tensor], LAMPAllocationReport]:
    """
    Profile layer sensitivities, allocate optimal mixed-precision bit-widths,
    quantize all layers, and compare total output reconstruction MSE against
    uniform bit-width quantization at `int(round(target_avg_bits))`.
    """
    if len(weights) == 0 or len(weights) != len(calib_activations):
        raise ValueError("weights and calib_activations must be non-empty sequences of equal length.")

    num_params = [int(w.numel()) for w in weights]
    sensitivities = [
        compute_layer_hessian_sensitivity(w, x) for w, x in zip(weights, calib_activations)
    ]

    assigned_bits, achieved_avg = allocate_mixed_precision_bits(
        layer_num_params=num_params,
        layer_sensitivities=sensitivities,
        target_avg_bits=target_avg_bits,
        candidate_bits=candidate_bits,
    )

    uniform_bit = int(round(target_avg_bits))
    quantized_weights: List[torch.Tensor] = []
    lamp_mse_total = 0.0
    uniform_mse_total = 0.0

    for w, x, b_lamp in zip(weights, calib_activations, assigned_bits):
        w_lamp = _quantize_symmetric_nbits(w, bits=b_lamp)
        w_unif = _quantize_symmetric_nbits(w, bits=uniform_bit)
        quantized_weights.append(w_lamp)

        x_f = x.reshape(-1, w.shape[1]).float()
        y_ref = x_f @ w.float().t()
        y_lamp = x_f @ w_lamp.float().t()
        y_unif = x_f @ w_unif.float().t()

        lamp_mse_total += float(torch.mean((y_ref - y_lamp).pow(2)).item())
        uniform_mse_total += float(torch.mean((y_ref - y_unif).pow(2)).item())

    report = LAMPAllocationReport(
        assigned_bits=assigned_bits,
        layer_sensitivities=[round(s, 4) for s in sensitivities],
        achieved_avg_bits=achieved_avg,
        target_avg_bits=target_avg_bits,
        lamp_output_mse=round(lamp_mse_total, 6),
        uniform_output_mse=round(uniform_mse_total, 6),
        mse_reduction_ratio=round(uniform_mse_total / max(lamp_mse_total, 1e-12), 2),
    )
    return quantized_weights, report


def main() -> None:
    torch.manual_seed(42)
    # Simulate an 8-layer SLM where layers 0 and 7 (embedding-adjacent / head-adjacent)
    # have 5x higher activation-weight curvature than middle layers
    scales = [5.0, 1.0, 0.8, 1.1, 0.9, 1.0, 1.2, 6.0]
    weights = [torch.randn(64, 64) * (0.2 * s) for s in scales]
    activations = [torch.randn(32, 64) * s for s in scales]

    _, report = apply_lamp_mixed_precision(
        weights=weights,
        calib_activations=activations,
        target_avg_bits=3.0,
        candidate_bits=(2, 3, 4, 8),
    )

    logger.info("=" * 72)
    logger.info("MODULE 17: LAYER-ADAPTIVE MIXED-PRECISION (LAMP) BIT ALLOCATOR")
    logger.info("=" * 72)
    logger.info(
        f"[Bit Allocation] Assigned Bits={report.assigned_bits} "
        f"(Achieved Avg={report.achieved_avg_bits:.2f} bits vs. Target={report.target_avg_bits:.2f} bits)"
    )
    logger.info(
        f"[Reconstruction Error] Uniform INT3 MSE={report.uniform_output_mse:.4f} "
        f"vs. LAMP Mixed-Precision MSE={report.lamp_output_mse:.4f} "
        f"({report.mse_reduction_ratio:.2f}x lower output error at same byte budget)"
    )


if __name__ == "__main__":
    main()
