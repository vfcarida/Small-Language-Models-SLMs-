"""
Module 30: Differential Transformer (DiffAttn) for Attention-Noise Cancellation
================================================================================

Implements the Differential Attention mechanism (Ye et al., Microsoft Research /
ICLR 2025) for Small Language Models (<4B parameters):
1. ``DifferentialAttention``: Computes two separate softmax attention maps over
   split query/key heads and subtracts the second map weighted by a learnable
   reparameterized scalar $\\lambda$:
   $$\\text{DiffAttn}(X) = \\left(\\text{softmax}\\!\\left(\\frac{Q_1 K_1^\\top}{\\sqrt{d_h}}\\right) - \\lambda\\,\\text{softmax}\\!\\left(\\frac{Q_2 K_2^\\top}{\\sqrt{d_h}}\\right)\\right) V$$
   followed by per-head ``RMSNorm`` scaled by $(1 - \\lambda_{\\text{init}})$.
2. ``measure_attention_snr``: Quantifies the Signal-to-Noise Ratio (SNR) of
   attention allocation on needle/target spans vs. distractor context tokens.

References:
    - Ye et al. (Microsoft Research / ICLR 2025). "Differential Transformer."
      arXiv:2410.05258.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lambda_init(layer_idx: int) -> float:
    """Compute depth-dependent initial differential coefficient $\\lambda_{\\text{init}}(\\ell)$.

    Following Ye et al. (ICLR 2025):
    $$\\lambda_{\\text{init}}(\\ell) = 0.8 - 0.6 \\cdot \\exp(-0.3 \\cdot \\ell)$$
    for 0-indexed layer $\\ell \\ge 0$.
    """
    return 0.8 - 0.6 * math.exp(-0.3 * max(0, int(layer_idx)))


class DifferentialAttention(nn.Module):
    """Differential Multi-Head Attention cancelling common-mode context noise.

    Splits queries and keys into pairs $(Q_1, Q_2)$ and $(K_1, K_2)$ of dimension
    $d_h$ per head, while values $V$ have dimension $2 d_h$ per head. Common-mode
    attention noise is cancelled via differential subtraction:

    $$\\lambda = \\exp(\\boldsymbol{\\lambda}_{q_1} \\cdot \\boldsymbol{\\lambda}_{k_1}) - \\exp(\\boldsymbol{\\lambda}_{q_2} \\cdot \\boldsymbol{\\lambda}_{k_2}) + \\lambda_{\\text{init}}$$

    Args:
        d_model: Model hidden dimension $d$.
        num_heads: Number of differential heads $h$ (must divide ``d_model // 2``).
        layer_idx: Layer index $\\ell \\ge 0$ used to set $\\lambda_{\\text{init}}(\\ell)$.
        eps: Numerical floor for per-head RMSNorm.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        layer_idx: int = 0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive.")
        if d_model % (2 * num_heads) != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by 2 * num_heads ({2 * num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // (2 * num_heads)
        self.layer_idx = layer_idx
        self.lambda_init = compute_lambda_init(layer_idx)
        self.eps = eps

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # Reparameterized lambda vectors (Ye et al., 2025)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.d_head))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.d_head))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.d_head))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.d_head))
        self.head_rms_weight = nn.Parameter(torch.ones(2 * self.d_head))

    def effective_lambda(self) -> torch.Tensor:
        """Compute scalar differential weight $\\lambda$."""
        l1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
        l2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
        return l1 - l2 + self.lambda_init

    def forward_with_maps(
        self,
        x: torch.Tensor,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute Differential Attention output and return intermediate attention maps."""
        bsz, seq_len, _ = x.shape
        q = self.W_Q(x).view(bsz, seq_len, self.num_heads, 2, self.d_head)
        k = self.W_K(x).view(bsz, seq_len, self.num_heads, 2, self.d_head)
        v = self.W_V(x).view(bsz, seq_len, self.num_heads, 2 * self.d_head).transpose(1, 2)

        q1 = q[:, :, :, 0, :].transpose(1, 2)  # (B, H, T, d_head)
        q2 = q[:, :, :, 1, :].transpose(1, 2)
        k1 = k[:, :, :, 0, :].transpose(1, 2)
        k2 = k[:, :, :, 1, :].transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head)
        s1 = torch.matmul(q1, k1.transpose(-2, -1)) * scale
        s2 = torch.matmul(q2, k2.transpose(-2, -1)) * scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
            )
            s1 = s1.masked_fill(mask, float("-inf"))
            s2 = s2.masked_fill(mask, float("-inf"))

        a1 = F.softmax(s1, dim=-1)
        a2 = F.softmax(s2, dim=-1)
        lam = self.effective_lambda()
        diff_map = a1 - lam * a2  # (B, H, T, T)

        head_out = torch.matmul(diff_map, v)  # (B, H, T, 2 * d_head)
        # Per-head RMSNorm scaled by (1 - lambda_init)
        rms = torch.rsqrt(head_out.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = head_out * rms * self.head_rms_weight * (1.0 - self.lambda_init)

        merged = normed.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.W_O(merged)
        return out, {
            "attn_primary": a1,
            "attn_noise": a2,
            "diff_attn_map": diff_map,
            "lambda": lam,
        }

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Standard forward returning ``(B, T, d_model)``."""
        out, _ = self.forward_with_maps(x, causal=causal)
        return out


def measure_attention_snr(
    attn_map: torch.Tensor,
    target_indices: Sequence[int],
    eps: float = 1e-8,
) -> float:
    """Measure Target-to-Distractor Signal-to-Noise Ratio (SNR) on an attention map.

    Given an attention map ``(..., T_query, T_key)`` and target key positions
    $\\mathcal{T} \\subset \\{0, \\dots, T_{\\text{key}} - 1\\}$, computes the ratio of mean
    absolute attention mass on target positions to mean absolute mass on non-target
    distractor positions.
    """
    seq_k = attn_map.shape[-1]
    if len(target_indices) == 0 or len(target_indices) >= seq_k:
        raise ValueError("target_indices must be a non-empty proper subset of key positions.")
    target_mask = torch.zeros(seq_k, dtype=torch.bool, device=attn_map.device)
    for idx in target_indices:
        target_mask[int(idx)] = True

    abs_map = attn_map.abs()
    signal = abs_map[..., target_mask].mean().item()
    noise = abs_map[..., ~target_mask].mean().item()
    return float(signal / max(noise, eps))


def main() -> None:
    """Run a deterministic demonstration of Differential Attention noise cancellation."""
    torch.manual_seed(42)
    diff_attn = DifferentialAttention(d_model=32, num_heads=4, layer_idx=2)
    x = torch.randn(2, 8, 32)
    out, maps = diff_attn.forward_with_maps(x)
    print(
        f"[Module 30] DiffAttn output shape={tuple(out.shape)} | "
        f"lambda={maps['lambda'].item():.4f}"
    )


if __name__ == "__main__":
    main()
