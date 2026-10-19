"""
==============================================================================
MODULE 05 — AWQ & ROTATION-SMOOTHED QUANTIZATION
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 30 minutes (reading) + calibration run

OBJECTIVE:
    Implement Activation-Aware Weight Quantization (AWQ, Lin et al., MLSys 2024
    Best Paper Award) and orthogonal rotation outlier smoothing (SpinQuant,
    Liu et al., ICLR 2025). Includes both glass-box pure-PyTorch mathematical
    implementations for deterministic verification and production AutoAWQ
    INT4 GEMM kernel compilation.

CONCEPTS COVERED:
    1. Salient Weight Channels and Activation Outliers.
    2. Group-Wise Asymmetric & Symmetric INT4 Pseudo-Quantization.
    3. AWQ Optimal Per-Channel Scaling Grid Search: s = s_X^alpha.
    4. Orthogonal Hadamard Rotation Outlier Smoothing (SpinQuant invariance).

REFERENCES:
    - Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
      and Acceleration", MLSys 2024 Best Paper Award (arXiv:2306.00978)
    - Liu et al., "SpinQuant: LLM quantization with learned rotations",
      ICLR 2025 (arXiv:2405.16406)

RUN COMMAND:
    python src/module_05_quantization/awq_quantization.py
==============================================================================
"""

import argparse
import logging
import math
import os
from typing import Dict, Tuple

import torch
from transformers import AutoTokenizer

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pseudo_quantize_tensor(
    weight: torch.Tensor,
    w_bit: int = 4,
    q_group_size: int = 128,
    zero_point: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs group-wise simulated quantization (quantize-then-dequantize) on a
    2D weight matrix to evaluate quantization error in pure PyTorch.

    MATHEMATICAL FORMULATION:
        For each group g of size `q_group_size` along `in_features`:
        - Asymmetric (`zero_point=True`):
            scale = (max(W_g) - min(W_g)) / (2^b - 1)
            zp = round(-min(W_g) / scale)
            W_q = clamp(round(W_g / scale) + zp, 0, 2^b - 1)
            W_hat = (W_q - zp) * scale
        - Symmetric (`zero_point=False`):
            q_max = 2^(b - 1) - 1
            scale = max(|W_g|) / q_max
            W_q = clamp(round(W_g / scale), -q_max, q_max)
            W_hat = W_q * scale

    Args:
        weight: 2D weight tensor of shape `(out_features, in_features)`.
        w_bit: Quantization bit-width (e.g., 4 for INT4, 8 for INT8).
        q_group_size: Number of contiguous weights sharing a scale factor.
        zero_point: Whether to use asymmetric zero-point quantization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            `(dequantized_weight, scales, zeros)`.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {weight.shape}")
    out_features, in_features = weight.shape
    if q_group_size <= 0 or in_features % q_group_size != 0:
        raise ValueError(
            f"in_features ({in_features}) must be divisible by positive q_group_size ({q_group_size})."
        )

    w_grouped = weight.reshape(-1, q_group_size).float()

    if zero_point:
        max_val = w_grouped.amax(dim=1, keepdim=True)
        min_val = w_grouped.amin(dim=1, keepdim=True)
        max_int = (2 ** w_bit) - 1
        scales = (max_val - min_val).clamp(min=1e-8) / max_int
        zeros = torch.clamp(torch.round(-min_val / scales), 0, max_int)
        q_w = torch.clamp(torch.round(w_grouped / scales) + zeros, 0, max_int)
        dequant = (q_w - zeros) * scales
    else:
        max_val = w_grouped.abs().amax(dim=1, keepdim=True)
        max_int = (2 ** (w_bit - 1)) - 1
        scales = max_val.clamp(min=1e-8) / max_int
        zeros = torch.zeros_like(scales)
        q_w = torch.clamp(torch.round(w_grouped / scales), -max_int, max_int)
        dequant = q_w * scales

    return (
        dequant.reshape(out_features, in_features).to(weight.dtype),
        scales.reshape(out_features, -1),
        zeros.reshape(out_features, -1),
    )


def compute_awq_scales(
    weight: torch.Tensor,
    inputs: torch.Tensor,
    w_bit: int = 4,
    q_group_size: int = 128,
    n_grid: int = 20,
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    """
    Searches for the optimal AWQ per-channel scaling vector s = s_X^alpha via
    grid search over alpha in [0, 1] (Lin et al., MLSys 2024, Section 3.2).

    MATHEMATICAL FORMULATION:
        Given weight matrix W in R^(d_out x d_in) and calibration activations
        X in R^(T x d_in), let the per-channel mean activation magnitude be:
            s_X = mean_t(|X_{t, :}|) in R^(d_in)
        We parameterize the per-channel protection scale as s(alpha) = s_X^alpha,
        normalized by sqrt(max(s) * min(s)), and solve:
            alpha* = argmin_{alpha in [0, 1]} || (Q(W * diag(s)) * diag(s)^(-1)) X^T - W X^T ||_F^2

    Args:
        weight: 2D weight tensor `(d_out, d_in)`.
        inputs: Calibration input activations `(num_tokens, d_in)` or `(batch, seq_len, d_in)`.
        w_bit: Target bit-width (default 4).
        q_group_size: Quantization group size.
        n_grid: Number of grid search steps in `[0.0, 1.0]`.

    Returns:
        Tuple[torch.Tensor, float, Dict[str, float]]:
            - `best_scales`: Optimal per-channel scale vector of shape `(d_in,)`.
            - `best_alpha`: Optimal grid exponent in `[0.0, 1.0]`.
            - `metrics`: Dictionary containing `'rtn_mse'` (alpha=0 baseline) and `'awq_mse'`.
    """
    x_flat = inputs.reshape(-1, inputs.shape[-1]).float().to(weight.device)
    w_float = weight.float()

    with torch.no_grad():
        orig_out = x_flat @ w_float.t()
        act_scales = x_flat.abs().mean(dim=0).clamp(min=1e-6)

        best_error = float("inf")
        best_alpha = 0.0
        best_scales = torch.ones_like(act_scales)
        rtn_mse = 0.0

        for i in range(n_grid + 1):
            alpha = i / float(n_grid)
            scales = act_scales.pow(alpha)
            scales = scales / (torch.sqrt(scales.max() * scales.min()).clamp(min=1e-8))

            w_scaled = w_float * scales.unsqueeze(0)
            w_q, _, _ = pseudo_quantize_tensor(
                w_scaled, w_bit=w_bit, q_group_size=q_group_size, zero_point=True
            )
            w_dequant = w_q / scales.unsqueeze(0)

            quant_out = x_flat @ w_dequant.t()
            mse = torch.mean((orig_out - quant_out) ** 2).item()

            if i == 0:
                rtn_mse = mse

            if mse < best_error:
                best_error = mse
                best_alpha = alpha
                best_scales = scales.clone()

    return (
        best_scales.to(weight.dtype),
        best_alpha,
        {"rtn_mse": rtn_mse, "awq_mse": best_error},
    )


def apply_hadamard_rotation_smoothing(
    weight: torch.Tensor,
    inputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies an orthonormal Hadamard rotation H (H^T H = I) to weights and activations
    to disperse extreme outlier channels across all dimensions prior to quantization,
    demonstrating the SpinQuant invariance principle (Liu et al., ICLR 2025):
        (X H) (W H)^T = X (H H^T) W^T = X W^T.

    Args:
        weight: 2D weight tensor `(d_out, d_in)` where `d_in` is a power of 2.
        inputs: Activation tensor `(..., d_in)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: `(rotated_weight, rotated_inputs)`.
    """
    d_in = weight.shape[1]
    if d_in <= 0 or (d_in & (d_in - 1)) != 0:
        raise ValueError(f"Hadamard rotation requires in_features to be a power of 2. Got: {d_in}")

    H = torch.tensor([[1.0]], device=weight.device, dtype=torch.float32)
    num_steps = int(math.log2(d_in))
    for _ in range(num_steps):
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
    H = H / math.sqrt(d_in)

    rotated_w = (weight.float() @ H).to(weight.dtype)
    rotated_x = (inputs.float() @ H).to(inputs.dtype)
    return rotated_w, rotated_x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AWQ Quantization CLI")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/awq_model")
    return parser.parse_args()


def run_awq_compilation(model_id: str, output_dir: str, config: dict) -> None:
    """
    Executes Activation-Aware Weight Quantization using `autoawq` if available,
    falling back to a mathematical demonstration when CUDA/AutoAWQ is absent.
    """
    logger.info("=" * 60)
    logger.info(f"COMPILING AWQ INT4 MODEL FOR: {model_id}")
    logger.info("=" * 60)

    quant_cfg = config.get("quantization", {})
    quant_config = {
        "zero_point": quant_cfg.get("zero_point", True),
        "q_group_size": quant_cfg.get("q_group_size", 128),
        "w_bit": quant_cfg.get("w_bit", 4),
        "version": quant_cfg.get("version", "GEMM"),
    }
    logger.info(f"Quantization Settings: {quant_config}")

    try:
        from awq import AutoAWQForCausalLM

        logger.info("Loading original FP16 weights via AutoAWQ...")
        model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        logger.info("Running activation-aware calibration profiling...")
        model.quantize(tokenizer, quant_config=quant_config)

        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving compiled 4-bit model to: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("AWQ quantization completed successfully.")
    except ImportError:
        logger.warning(
            "Optional package 'autoawq' is not installed in the current environment. "
            "Running glass-box PyTorch AWQ salient-channel grid search verification instead."
        )
        torch.manual_seed(42)
        w = torch.randn(64, 128)
        x = torch.randn(32, 128)
        x[:, 0] *= 25.0  # Inject synthetic activation outlier channel
        _, best_alpha, metrics = compute_awq_scales(w, x, w_bit=4, q_group_size=32)
        logger.info(
            f"Verified AWQ Grid Search — Optimal alpha={best_alpha:.2f} | "
            f"RTN MSE={metrics['rtn_mse']:.4f} -> AWQ MSE={metrics['awq_mse']:.4f}"
        )
    except Exception as exc:
        logger.error(f"AWQ compilation failed: {exc}")


def main() -> None:
    config = load_config()
    args = parse_args()
    model_id = args.model_id or config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    run_awq_compilation(model_id, args.output_dir, config)


if __name__ == "__main__":
    main()
