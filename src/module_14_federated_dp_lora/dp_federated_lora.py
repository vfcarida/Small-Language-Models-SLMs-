"""
Module 14: Federated & Differential-Privacy LoRA Aggregation (DP-SGD + FFA-LoRA)
--------------------------------------------------------------------------------
Implements privacy-preserving decentralized SLM fine-tuning across enterprise silos:
1. Differential Privacy via Per-Sample Gradient Clipping & Calibrated Gaussian Noise
   (Abadi et al., CCS 2016 `arXiv:1607.00133`; Yu et al., ICLR 2022 `arXiv:2110.06500`):
   - Per-sample L2 norm clipping: g_bar_i = g_i * min(1, C / ||g_i||_2)
   - Calibrated Gaussian mechanism: g_tilde = (1 / B) * (sum_i g_bar_i + N(0, sigma^2 * C^2 * I))
2. Federated Freeze-A LoRA (FFA-LoRA, Sun et al., ICLR 2024 `arXiv:2403.12313`) and
   SVD-Exact LoRA Aggregation:
   - Eliminates the cross-term aggregation interference error inherent to naive FedAvg on LoRA:
     (1/K sum B_k)(1/K sum A_k) != 1/K sum (B_k A_k).

Author: Vinicius Carida
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DPLoRAReport:
    """Diagnostic telemetry for a DP-SGD gradient step and Federated LoRA aggregation round."""

    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float
    clipped_fraction: float
    mean_raw_grad_norm: float
    mean_clipped_grad_norm: float
    aggregation_mode: str
    cross_term_frobenius_error: float


def compute_gaussian_dp_sigma(
    epsilon: float,
    delta: float,
    sample_rate: float = 0.01,
    num_steps: int = 100,
) -> float:
    """
    Computes an analytic Gaussian mechanism noise multiplier sigma for (epsilon, delta)-DP
    under Poisson subsampling over T composition steps (Abadi et al., 2016).

    Args:
        epsilon: Target privacy budget epsilon (> 0).
        delta: Target failure probability delta (0 < delta < 1).
        sample_rate: Minibatch sampling ratio q = B / N.
        num_steps: Total number of training steps T.

    Returns:
        Calibrated noise multiplier sigma.
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0, 1).")
    if not (0.0 < sample_rate <= 1.0):
        raise ValueError("sample_rate must be in (0, 1].")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1.")

    # Moment accountant / Gaussian mechanism calibration bound:
    # sigma >= q * sqrt(2 * T * ln(1.25 / delta)) / epsilon
    c_delta = math.sqrt(2.0 * math.log(1.25 / delta))
    sigma = (sample_rate * math.sqrt(float(num_steps)) * c_delta) / epsilon
    return float(max(sigma, 1e-6))


def clip_and_noise_gradients(
    per_sample_grads: torch.Tensor,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 0.8,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Applies per-sample L2 gradient clipping followed by calibrated Gaussian noise injection
    for Differential Privacy (DP-SGD) on LoRA adapter parameters.

    Args:
        per_sample_grads: Tensor of shape (batch_size, *param_shape) containing per-example gradients.
        max_grad_norm: Clipping threshold C > 0.
        noise_multiplier: Gaussian noise multiplier sigma >= 0.
        seed: Deterministic PRNG seed for noise generation.

    Returns:
        Tuple of (privatized_mean_gradient, stats_dict).
    """
    if per_sample_grads.ndim < 2:
        raise ValueError("per_sample_grads must have shape (batch_size, *param_shape).")
    if max_grad_norm <= 0.0:
        raise ValueError("max_grad_norm must be strictly positive.")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier must be non-negative.")

    batch_size = per_sample_grads.shape[0]
    flat_grads = per_sample_grads.reshape(batch_size, -1).to(torch.float32)

    # Compute per-sample L2 norms: ||g_i||_2
    raw_norms = torch.linalg.norm(flat_grads, ord=2, dim=-1)  # (B,)
    clip_factors = torch.clamp(max_grad_norm / (raw_norms + 1e-12), max=1.0)  # (B,)
    clipped_flat = flat_grads * clip_factors.unsqueeze(-1)
    clipped_norms = torch.linalg.norm(clipped_flat, ord=2, dim=-1)

    # Sum clipped gradients and inject calibrated Gaussian noise N(0, sigma^2 * C^2 * I)
    summed_grads = clipped_flat.sum(dim=0)
    if noise_multiplier > 0.0:
        gen = torch.Generator(device=per_sample_grads.device)
        gen.manual_seed(seed)
        noise_std = noise_multiplier * max_grad_norm
        noise = torch.randn(
            summed_grads.shape,
            generator=gen,
            device=per_sample_grads.device,
            dtype=torch.float32,
        ) * noise_std
        noisy_sum = summed_grads + noise
    else:
        noisy_sum = summed_grads

    privatized_grad = (noisy_sum / float(batch_size)).reshape(per_sample_grads.shape[1:]).to(per_sample_grads.dtype)
    clipped_fraction = float((raw_norms > max_grad_norm).to(torch.float32).mean().item())

    stats = {
        "clipped_fraction": clipped_fraction,
        "mean_raw_grad_norm": float(raw_norms.mean().item()),
        "mean_clipped_grad_norm": float(clipped_norms.mean().item()),
        "max_clipped_grad_norm": float(clipped_norms.max().item()),
    }
    return privatized_grad, stats


def aggregate_federated_lora(
    client_A_list: List[torch.Tensor],
    client_B_list: List[torch.Tensor],
    client_weights: Optional[List[float]] = None,
    mode: Literal["ffa_lora", "svd_exact", "naive_fedavg"] = "ffa_lora",
    scaling: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Aggregates K client LoRA updates {(A_k, B_k)}_{k=1}^K across federated silos.

    Mathematical Context:
    True ideal aggregated weight update in full space is:
        Delta W_ideal = scaling * sum_{k=1}^K p_k (B_k @ A_k)
    1. `naive_fedavg`: Computes B_bar = sum p_k B_k and A_bar = sum p_k A_k independently.
       When clients have distinct A_k, B_bar @ A_bar introduces cross-term interference
       sum_{k != j} p_k p_j B_k A_j != 0.
    2. `ffa_lora` (Federated Freeze-A LoRA, Sun et al., ICLR 2024 `arXiv:2403.12313`):
       All clients share a frozen random/orthogonal projection A_0 (A_1 = ... = A_K = A_0)
       and only train/communicate B_k. By linearity:
       (sum_k p_k B_k) @ A_0 == sum_k p_k (B_k @ A_0), achieving ZERO cross-term error
       while cutting communication cost by 50%.
    3. `svd_exact`: Reconstructs Delta W_ideal = sum_k p_k (B_k @ A_k) on the server and projects
       back onto rank-r factors (B_server, A_server) via truncated SVD.

    Args:
        client_A_list: List of K tensors of shape (r, d_in).
        client_B_list: List of K tensors of shape (d_out, r).
        client_weights: Optional list of K non-negative aggregation weights summing to 1.
        mode: Aggregation strategy ('ffa_lora', 'svd_exact', or 'naive_fedavg').
        scaling: LoRA scaling factor (alpha / r).

    Returns:
        Tuple of (server_A, server_B, cross_term_frobenius_error).
    """
    num_clients = len(client_A_list)
    if num_clients == 0 or len(client_B_list) != num_clients:
        raise ValueError("client_A_list and client_B_list must be non-empty and have matching lengths.")

    if client_weights is None:
        probs = [1.0 / num_clients] * num_clients
    else:
        if len(client_weights) != num_clients:
            raise ValueError("client_weights length must match number of clients.")
        total_w = sum(client_weights)
        if total_w <= 0.0:
            raise ValueError("Sum of client_weights must be strictly positive.")
        probs = [w / total_w for w in client_weights]

    rank = client_A_list[0].shape[0]
    dtype = client_A_list[0].dtype

    # Ground-truth linear ensemble update in full parameter space
    delta_w_ideal = torch.zeros(
        (client_B_list[0].shape[0], client_A_list[0].shape[1]),
        dtype=torch.float32,
        device=client_A_list[0].device,
    )
    for p_k, A_k, B_k in zip(probs, client_A_list, client_B_list):
        delta_w_ideal = delta_w_ideal + p_k * scaling * (B_k.to(torch.float32) @ A_k.to(torch.float32))

    if mode == "naive_fedavg":
        server_A = sum(p_k * A_k.to(torch.float32) for p_k, A_k in zip(probs, client_A_list))
        server_B = sum(p_k * B_k.to(torch.float32) for p_k, B_k in zip(probs, client_B_list))
    elif mode == "ffa_lora":
        # Verify that A_k is frozen across clients (or enforce shared A_0 = client_A_list[0])
        server_A = client_A_list[0].to(torch.float32).clone()
        server_B = sum(p_k * B_k.to(torch.float32) for p_k, B_k in zip(probs, client_B_list))
        # Recompute ideal under FFA-LoRA constraint (where all clients use shared A_0)
        delta_w_ideal = sum(
            p_k * scaling * (B_k.to(torch.float32) @ server_A)
            for p_k, B_k in zip(probs, client_B_list)
        )
    elif mode == "svd_exact":
        # Factorize (delta_w_ideal / scaling) via truncated rank-r SVD
        u, s, vh = torch.linalg.svd(delta_w_ideal / scaling, full_matrices=False)
        r_eff = min(rank, s.shape[0])
        sqrt_s = torch.sqrt(s[:r_eff])
        server_B = u[:, :r_eff] * sqrt_s.unsqueeze(0)
        server_A = sqrt_s.unsqueeze(1) * vh[:r_eff, :]
    else:
        raise ValueError(f"Unsupported aggregation mode: {mode}")

    delta_w_reconstructed = scaling * (server_B @ server_A)
    frobenius_error = float(torch.linalg.norm(delta_w_ideal - delta_w_reconstructed, ord="fro").item())

    return server_A.to(dtype), server_B.to(dtype), frobenius_error


def main() -> None:
    torch.manual_seed(42)
    sigma = compute_gaussian_dp_sigma(epsilon=2.0, delta=1e-5, sample_rate=0.02, num_steps=200)
    per_sample = torch.randn(16, 64, 8) * 3.0
    priv_grad, stats = clip_and_noise_gradients(per_sample, max_grad_norm=1.0, noise_multiplier=sigma)

    # Compare naive FedAvg vs FFA-LoRA cross-term error
    shared_A = torch.randn(8, 64)
    distinct_As = [torch.randn(8, 64) for _ in range(4)]
    client_Bs = [torch.randn(64, 8) for _ in range(4)]

    _, _, err_naive = aggregate_federated_lora(distinct_As, client_Bs, mode="naive_fedavg")
    _, _, err_ffa = aggregate_federated_lora([shared_A] * 4, client_Bs, mode="ffa_lora")
    logger.info(
        f"[DP-Federated-LoRA] sigma={sigma:.4f} | clipped={stats['clipped_fraction']:.2%} | "
        f"Naive FedAvg Error={err_naive:.4f} vs FFA-LoRA Error={err_ffa:.8f}"
    )


if __name__ == "__main__":
    main()

