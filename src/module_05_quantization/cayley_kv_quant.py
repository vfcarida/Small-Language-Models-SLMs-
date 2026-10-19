"""
==============================================================================
MODULE 05B — CAYLEY STIEFEL MANIFOLD ROTATION (INT3/INT2) & INT4 KV-CACHE QUANT
==============================================================================
OBJECTIVE:
    1. Implement Cayley Transform SGD on the Stiefel Orthogonal Manifold
       (R^T R = I) from SpinQuant (Liu et al., ICLR 2025, arXiv:2405.16406)
       and QuaRot (Ashkboos et al., NeurIPS 2024, arXiv:2404.00456) to optimize
       rotation matrices for extreme sub-4-bit (INT3 / INT2) weight quantization.
    2. Implement Asymmetric Per-Channel Key & Per-Token Value INT4 KV-Cache
       Quantization (KVQuant, Hooper et al., NeurIPS 2024; KIVI, ICML 2024) for
       long-context edge inference.

RUN COMMAND:
    python src/module_05_quantization/cayley_kv_quant.py
==============================================================================
"""

import logging
import math
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.module_05_quantization.awq_quantization import pseudo_quantize_tensor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_normalized_hadamard(dim: int, device: torch.device | None = None) -> torch.Tensor:
    """Construct an orthonormal Sylvester-Hadamard matrix H in R^(dim x dim) with H^T H = I."""
    if dim <= 0 or (dim & (dim - 1)) != 0:
        raise ValueError(f"Hadamard dimension must be a positive power of 2. Got: {dim}")
    H = torch.tensor([[1.0]], device=device, dtype=torch.float32)
    for _ in range(int(math.log2(dim))):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(dim)


def cayley_orthogonal_step(
    R: torch.Tensor,
    grad_R: torch.Tensor,
    lr: float = 0.1,
) -> torch.Tensor:
    """
    Execute a single Riemannian Cayley transform update on the Stiefel manifold
    St(d, d) = {R in R^(d x d) : R^T R = I} (SpinQuant, Liu et al., ICLR 2025):

        1. Skew-symmetric tangent projection:
               W = G R^T - R G^T   (satisfies W^T = -W)
        2. Cayley retraction:
               R_next = (I + (lr / 2) * W)^(-1) (I - (lr / 2) * W) R

    By construction, (I + A)^(-1)(I - A) is strictly orthogonal for any
    skew-symmetric matrix A, guaranteeing R_next^T R_next = I up to FP precision.
    """
    if R.dim() != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Expected square 2D orthogonal matrix R, got shape {R.shape}")

    d = R.shape[0]
    eye = torch.eye(d, device=R.device, dtype=torch.float32)
    R_fp32 = R.float()
    G_fp32 = grad_R.float()

    # Project Euclidean gradient onto the skew-symmetric tangent space
    W = G_fp32 @ R_fp32.t() - R_fp32 @ G_fp32.t()
    w_norm = torch.linalg.norm(W, ord="fro").clamp(min=1.0)
    W_unit = W / w_norm

    lhs = eye + 0.5 * lr * W_unit
    rhs = (eye - 0.5 * lr * W_unit) @ R_fp32
    R_next = torch.linalg.solve(lhs, rhs)
    return R_next.to(R.dtype)


def ste_pseudo_quantize(
    weight: torch.Tensor,
    w_bit: int = 3,
    q_group_size: int = 32,
) -> torch.Tensor:
    """
    Scale-differentiable group-wise simulated quantization using the true
    Straight-Through Estimator (STE) on the rounding operator `round(u)`
    (SpinQuant, Liu et al., ICLR 2025):
        - Group min/max scales `s(W R)` remain fully differentiable so
          Riemannian gradients on the Stiefel manifold directly shrink
          group dynamic range and outlier peaks.
        - `round(u)` uses `u + (round(u) - u).detach()`.
    """
    out_features, in_features = weight.shape
    w_grouped = weight.reshape(-1, q_group_size).float()

    max_val = w_grouped.amax(dim=1, keepdim=True)
    min_val = w_grouped.amin(dim=1, keepdim=True)
    max_int = float((2 ** w_bit) - 1)
    scales = (max_val - min_val).clamp(min=1e-8) / max_int
    zeros_raw = -min_val / scales
    zeros = zeros_raw + (torch.round(zeros_raw) - zeros_raw).detach()
    zeros = torch.clamp(zeros, 0.0, max_int)

    u = w_grouped / scales + zeros
    u_round = u + (torch.round(u) - u).detach()
    q_w = torch.clamp(u_round, 0.0, max_int)
    dequant = (q_w - zeros) * scales
    return dequant.reshape(out_features, in_features).to(weight.dtype)


def optimize_cayley_rotation(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    w_bit: int = 3,
    q_group_size: int = 32,
    steps: int = 25,
    lr: float = 0.15,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Optimize an orthogonal rotation matrix R in St(d_in, d_in) via Cayley SGD
    to minimize sub-4-bit (e.g., INT3/INT2) layer output reconstruction error:

        min_{R^T R = I} || (X R) Q_b(W R)^T - X W^T ||_2^2

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]:
            - `best_R`: Optimized orthogonal matrix of shape `(d_in, d_in)`.
            - `metrics`: Dictionary with `'rtn_mse'`, `'hadamard_mse'`,
              `'cayley_mse'`, and `'orthogonality_error'`.
    """
    w_fp32 = weight.float()
    x_fp32 = inputs.reshape(-1, inputs.shape[-1]).float().to(weight.device)
    d_in = w_fp32.shape[1]

    with torch.no_grad():
        target_out = x_fp32 @ w_fp32.t()
        w_rtn, _, _ = pseudo_quantize_tensor(
            w_fp32, w_bit=w_bit, q_group_size=q_group_size, zero_point=True
        )
        rtn_mse = F.mse_loss(x_fp32 @ w_rtn.t(), target_out).item()

    if (d_in & (d_in - 1)) == 0:
        R_had = build_normalized_hadamard(d_in, device=weight.device)
    else:
        R_had = torch.eye(d_in, device=weight.device, dtype=torch.float32)

    with torch.no_grad():
        w_h, _, _ = pseudo_quantize_tensor(
            w_fp32 @ R_had, w_bit=w_bit, q_group_size=q_group_size, zero_point=True
        )
        hadamard_mse = F.mse_loss((x_fp32 @ R_had) @ w_h.t(), target_out).item()

    best_R = R_had.clone() if hadamard_mse <= rtn_mse else torch.eye(d_in, device=weight.device)
    best_mse = min(hadamard_mse, rtn_mse)

    for R_init in (R_had, torch.eye(d_in, device=weight.device, dtype=torch.float32)):
        R = R_init.clone()
        for step in range(steps):
            R_var = R.detach().requires_grad_(True)
            w_rot = w_fp32 @ R_var
            x_rot = x_fp32 @ R_var
            w_rot_q = ste_pseudo_quantize(w_rot, w_bit=w_bit, q_group_size=q_group_size)
            recon_loss = F.mse_loss(x_rot @ w_rot_q.t(), target_out)
            kurtosis_reg = 0.002 * torch.mean(w_rot ** 4)
            total_loss = recon_loss + kurtosis_reg
            total_loss.backward()

            with torch.no_grad():
                step_lr = lr * (0.90 ** step)
                R = cayley_orthogonal_step(R, R_var.grad, lr=step_lr)
                w_eval, _, _ = pseudo_quantize_tensor(
                    w_fp32 @ R, w_bit=w_bit, q_group_size=q_group_size, zero_point=True
                )
                curr_mse = F.mse_loss((x_fp32 @ R) @ w_eval.t(), target_out).item()
                if curr_mse < best_mse:
                    best_mse = curr_mse
                    best_R = R.clone()

    eye = torch.eye(d_in, device=weight.device, dtype=torch.float32)
    ortho_err = torch.max(torch.abs(best_R.t() @ best_R - eye)).item()

    return best_R.to(weight.dtype), {
        "rtn_mse": round(rtn_mse, 8),
        "hadamard_mse": round(hadamard_mse, 8),
        "cayley_mse": round(best_mse, 8),
        "orthogonality_error": float(ortho_err),
    }


def quantize_kv_cache_int4(
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """
    Asymmetric INT4 KV-Cache Quantization (KVQuant, Hooper et al., NeurIPS 2024;
    KIVI, Liu et al., ICML 2024):
        - Key states `K` of shape `(B, H, S, D)` exhibit persistent outlier
          channels along `D`, so `K` is quantized **per-channel** (grouped along
          the sequence dimension `S`).
        - Value states `V` of shape `(B, H, S, D)` are quantized **per-token**
          (grouped along the head dimension `D`).

    Requires `S` to be divisible by `group_size` (for Keys) and `D` to be
    divisible by `group_size` (for Values).
    """
    if key_states.dim() != 4 or value_states.dim() != 4:
        raise ValueError("key_states and value_states must be 4D tensors (B, H, S, D).")
    if key_states.shape != value_states.shape:
        raise ValueError("key_states and value_states must have identical shapes.")

    b, h, s, d = key_states.shape
    if s % group_size != 0 or d % group_size != 0:
        raise ValueError(
            f"Sequence length ({s}) and head_dim ({d}) must be divisible by group_size ({group_size})."
        )

    # 1. Per-channel Key quantization: transpose (B, H, S, D) -> (B*H*D, S)
    k_perm = key_states.permute(0, 1, 3, 2).contiguous().view(-1, s)
    k_q, _, _ = pseudo_quantize_tensor(
        k_perm, w_bit=4, q_group_size=group_size, zero_point=True
    )
    k_dequant = k_q.view(b, h, d, s).permute(0, 1, 3, 2).contiguous()

    # 2. Per-token Value quantization: reshape (B, H, S, D) -> (B*H*S, D)
    v_flat = value_states.contiguous().view(-1, d)
    v_q, _, _ = pseudo_quantize_tensor(
        v_flat, w_bit=4, q_group_size=group_size, zero_point=True
    )
    v_dequant = v_q.view(b, h, s, d).contiguous()

    with torch.no_grad():
        key_mse = F.mse_loss(k_dequant.float(), key_states.float()).item()
        val_mse = F.mse_loss(v_dequant.float(), value_states.float()).item()

        # Verify attention output fidelity with a random query tensor Q
        q = torch.randn_like(key_states, dtype=torch.float32)
        scale = 1.0 / math.sqrt(d)
        attn_ref = F.softmax((q @ key_states.float().transpose(-2, -1)) * scale, dim=-1) @ value_states.float()
        attn_quant = F.softmax((q @ k_dequant.float().transpose(-2, -1)) * scale, dim=-1) @ v_dequant.float()
        attn_cos = F.cosine_similarity(
            attn_ref.flatten().unsqueeze(0),
            attn_quant.flatten().unsqueeze(0),
        ).item()

    # Effective bits: 4 bits + (16-bit scale + 16-bit zero_point) / group_size
    eff_bits = 4.0 + (32.0 / group_size)
    compression_ratio = 16.0 / eff_bits

    return (
        k_dequant,
        v_dequant,
        {
            "effective_bits": round(eff_bits, 2),
            "compression_ratio_vs_fp16": round(compression_ratio, 2),
            "key_mse": round(key_mse, 8),
            "value_mse": round(val_mse, 8),
            "attention_cosine_similarity": round(attn_cos, 6),
        },
    )


def main() -> None:
    torch.manual_seed(42)
    d_out, d_in, seq_len = 64, 64, 32
    w = torch.randn(d_out, d_in) * 0.05
    # Weight outlier in channel 0 crushes resolution of channels 1..15 in group 0
    w[:, 0] *= 15.0
    x = torch.randn(seq_len, d_in)
    x[:, 0] *= 0.1  # Low activation on column 0, normal activations on crushed columns 1..15

    _, cayley_stats = optimize_cayley_rotation(
        w, x, w_bit=3, q_group_size=16, steps=20, lr=0.25
    )

    k = torch.randn(2, 4, 64, 64)
    k[:, :, :, 0] *= 12.0  # Persistent Key channel outlier
    v = torch.randn(2, 4, 64, 64)
    _, _, kv_stats = quantize_kv_cache_int4(k, v, group_size=16)

    logger.info("=" * 72)
    logger.info("MODULE 05B: CAYLEY STIEFEL INT3 & KV-CACHE INT4 TELEMETRY")
    logger.info("=" * 72)
    logger.info(
        f"[INT3 Sub-4-Bit] Naive RTN MSE={cayley_stats['rtn_mse']:.6f} "
        f"-> Hadamard MSE={cayley_stats['hadamard_mse']:.6f} "
        f"-> Learned Cayley MSE={cayley_stats['cayley_mse']:.6f} "
        f"| Ortho Error={cayley_stats['orthogonality_error']:.2e}"
    )
    logger.info(
        f"[INT4 KV-Cache] Compression={kv_stats['compression_ratio_vs_fp16']:.2f}x "
        f"| Key MSE={kv_stats['key_mse']:.6f} | Value MSE={kv_stats['value_mse']:.6f} "
        f"| Attention CosSim={kv_stats['attention_cosine_similarity']:.6f}"
    )


if __name__ == "__main__":
    main()
