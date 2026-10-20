"""
Module 27: Multi-Head Latent Attention (MLA) & Low-Rank KV Joint Compression
============================================================================

Implements DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) for Small Language
Models (<4B parameters) to achieve >80% KV-cache reduction while preserving full
Multi-Head Attention rank expressivity:
1. ``MultiHeadLatentAttention``: Joint low-rank down-projection
   $\\mathbf{c}_t^{KV} = \\mathbf{W}_{DKV} \\mathbf{h}_t \\in \\mathbb{R}^{d_c}$ ($d_c \\ll n_h d_h$),
   decoupled Rotary Position Embedding (RoPE) key/query heads, and exact
   **weight-absorbed inference** (``forward_absorbed``) that scores attention
   directly against $\\mathbf{c}_t^{KV}$ without materializing full key/value tensors.
2. ``profile_mla_kv_cache_savings``: Analytical KV-cache footprint comparison across
   standard MHA, Grouped-Query Attention (GQA), and MLA.

References:
    - Liu et al. (DeepSeek-AI, 2024). "DeepSeek-V2: A Strong, Economical, and
      Efficient Mixture-of-Experts Language Model." arXiv:2405.04434.
    - DeepSeek-AI (2024). "DeepSeek-V3 Technical Report." arXiv:2412.19437.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_half_rotation_rope(x: torch.Tensor) -> torch.Tensor:
    """Apply deterministic Rotary Position Embedding (RoPE) to ``(B, H, T, D_r)``."""
    bsz, n_heads, seq_len, d_rope = x.shape
    if d_rope % 2 != 0:
        raise ValueError(f"d_rope must be even for RoPE rotation, got {d_rope}.")
    half = d_rope // 2
    positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
    freq_idx = torch.arange(half, device=x.device, dtype=x.dtype).unsqueeze(0)
    inv_freq = 1.0 / (10000.0 ** (2.0 * freq_idx / d_rope))
    angles = positions * inv_freq  # (T, half)
    cos = torch.cos(angles).unsqueeze(0).unsqueeze(0)  # (1, 1, T, half)
    sin = torch.sin(angles).unsqueeze(0).unsqueeze(0)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-V2/V3 Multi-Head Latent Attention (MLA) with Matrix Absorption.

    Given input hidden states $\\mathbf{h}_t \\in \\mathbb{R}^d$:
    - Low-rank KV latent: $\\mathbf{c}_t^{KV} = \\mathbf{W}_{DKV} \\mathbf{h}_t \\in \\mathbb{R}^{d_c}$
    - Content keys/values: $\\mathbf{k}_{t, i}^C = \\mathbf{W}_{UK, i} \\mathbf{c}_t^{KV} \\in \\mathbb{R}^{d_h}$,
      $\\mathbf{v}_{t, i}^C = \\mathbf{W}_{UV, i} \\mathbf{c}_t^{KV} \\in \\mathbb{R}^{d_h}$
    - Shared decoupled RoPE key: $\\mathbf{k}_t^R = \\text{RoPE}(\\mathbf{W}_{KR} \\mathbf{h}_t) \\in \\mathbb{R}^{d_r}$
    - Content & RoPE queries: $\\mathbf{q}_{t, i}^C = \\mathbf{W}_{UQ, i} \\mathbf{h}_t \\in \\mathbb{R}^{d_h}$,
      $\\mathbf{q}_{t, i}^R = \\text{RoPE}(\\mathbf{W}_{QR, i} \\mathbf{h}_t) \\in \\mathbb{R}^{d_r}$

    During inference (`forward_absorbed`), $\\mathbf{W}_{UK, i}$ is absorbed into
    $\\mathbf{W}_{UQ, i}$ so that $(\\mathbf{q}_{t, i}^C)^\\top \\mathbf{k}_{s, i}^C =
    (\\mathbf{W}_{UK, i}^\\top \\mathbf{q}_{t, i}^C)^\\top \\mathbf{c}_s^{KV}$, caching only
    $(\\mathbf{c}_s^{KV}, \\mathbf{k}_s^R)$ of total dimension $d_c + d_r \\ll 2 n_h d_h$.

    Args:
        d_model: Hidden dimension $d$.
        num_heads: Number of attention heads $n_h$.
        d_head: Per-head content dimension $d_h$.
        d_latent_kv: Compressed KV latent dimension $d_c$.
        d_rope: Decoupled rotary positional key/query dimension $d_r$ (must be even).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_head: int,
        d_latent_kv: int,
        d_rope: int = 8,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_head <= 0 or d_latent_kv <= 0:
            raise ValueError("All MLA dimensions must be positive.")
        if d_rope <= 0 or d_rope % 2 != 0:
            raise ValueError(f"d_rope must be a positive even integer, got {d_rope}.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_latent_kv = d_latent_kv
        self.d_rope = d_rope

        # Query projections (content + decoupled RoPE)
        self.W_UQ = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_QR = nn.Linear(d_model, num_heads * d_rope, bias=False)

        # Joint KV down-projection & shared decoupled RoPE key
        self.W_DKV = nn.Linear(d_model, d_latent_kv, bias=False)
        self.W_KR = nn.Linear(d_model, d_rope, bias=False)

        # Up-projections from KV latent c_t^{KV}
        self.W_UK = nn.Linear(d_latent_kv, num_heads * d_head, bias=False)
        self.W_UV = nn.Linear(d_latent_kv, num_heads * d_head, bias=False)

        self.W_O = nn.Linear(num_heads * d_head, d_model, bias=False)

    def compress_kv(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compress hidden states ``(B, T, d_model)`` into ``(c_kv, k_rope)`` cache."""
        bsz, seq_len, _ = h.shape
        c_kv = self.W_DKV(h)  # (B, T, d_latent_kv)
        k_rope_raw = self.W_KR(h).unsqueeze(1)  # (B, 1, T, d_rope)
        k_rope = _apply_half_rotation_rope(k_rope_raw)  # (B, 1, T, d_rope)
        return c_kv, k_rope

    def forward_standard(self, h: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Compute MLA by explicitly materializing content keys and values."""
        bsz, seq_len, _ = h.shape
        c_kv, k_rope = self.compress_kv(h)

        q_c = self.W_UQ(h).view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        q_r_raw = self.W_QR(h).view(bsz, seq_len, self.num_heads, self.d_rope).transpose(1, 2)
        q_r = _apply_half_rotation_rope(q_r_raw)

        k_c = self.W_UK(c_kv).view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v_c = self.W_UV(c_kv).view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.d_head + self.d_rope)
        scores_c = torch.matmul(q_c, k_c.transpose(-2, -1))
        scores_r = torch.matmul(q_r, k_rope.transpose(-2, -1))
        scores = (scores_c + scores_r) * scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out_heads = torch.matmul(attn, v_c)  # (B, H, T, d_head)
        out_flat = out_heads.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.W_O(out_flat)

    def forward_absorbed(self, h: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Compute MLA via weight absorption directly over compressed latent ``c_kv``.

        Absorbs $\\mathbf{W}_{UK}$ into query space and $\\mathbf{W}_{UV}$ into output
        projection so $\\mathbf{k}_t^C, \\mathbf{v}_t^C \\in \\mathbb{R}^{n_h d_h}$ are
        never materialized in the KV cache.
        """
        bsz, seq_len, _ = h.shape
        c_kv, k_rope = self.compress_kv(h)  # c_kv: (B, T, d_c), k_rope: (B, 1, T, d_r)

        q_c = self.W_UQ(h).view(bsz, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        q_r_raw = self.W_QR(h).view(bsz, seq_len, self.num_heads, self.d_rope).transpose(1, 2)
        q_r = _apply_half_rotation_rope(q_r_raw)

        # Reshape W_UK.weight of shape (H * d_head, d_c) -> (H, d_head, d_c)
        w_uk = self.W_UK.weight.view(self.num_heads, self.d_head, self.d_latent_kv)
        # Absorbed query in latent space: q_absorbed[b, h, t, c] = sum_d q_c[b, h, t, d] * w_uk[h, d, c]
        q_absorbed = torch.einsum("bhtd,hdc->bhtc", q_c, w_uk)  # (B, H, T, d_c)

        scale = 1.0 / math.sqrt(self.d_head + self.d_rope)
        scores_c = torch.matmul(q_absorbed, c_kv.unsqueeze(1).transpose(-2, -1))
        scores_r = torch.matmul(q_r, k_rope.transpose(-2, -1))
        scores = (scores_c + scores_r) * scale

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=h.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)  # (B, H, T, T)
        # Aggregate latent context directly in d_latent_kv space: (B, H, T, d_c)
        latent_ctx = torch.matmul(attn, c_kv.unsqueeze(1))

        # Project latent context through W_UV per head: (H, d_head, d_c)
        w_uv = self.W_UV.weight.view(self.num_heads, self.d_head, self.d_latent_kv)
        out_heads = torch.einsum("bhtc,hdc->bhtd", latent_ctx, w_uv)
        out_flat = out_heads.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.W_O(out_flat)

    def forward(self, h: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """Default forward uses matrix-absorbed MLA execution."""
        return self.forward_absorbed(h, causal=causal)


def profile_mla_kv_cache_savings(
    num_heads: int = 32,
    num_kv_heads_gqa: int = 8,
    d_head: int = 64,
    d_latent_kv: int = 256,
    d_rope: int = 64,
    seq_len: int = 8192,
    num_layers: int = 24,
    bytes_per_element: int = 2,
) -> Dict[str, float]:
    """Compare per-token and full-sequence KV cache footprints across MHA, GQA, and MLA.

    - **MHA:** $2 \\cdot n_h \\cdot d_h$ elements per token per layer.
    - **GQA:** $2 \\cdot n_{kv} \\cdot d_h$ elements per token per layer.
    - **MLA:** $d_c + d_r$ elements per token per layer (caching only $\\mathbf{c}_t^{KV}$
      and shared $\\mathbf{k}_t^R$).
    """
    mha_elem = 2 * num_heads * d_head
    gqa_elem = 2 * num_kv_heads_gqa * d_head
    mla_elem = d_latent_kv + d_rope

    factor = seq_len * num_layers * bytes_per_element / (1024.0 * 1024.0)
    mha_mb = mha_elem * factor
    gqa_mb = gqa_elem * factor
    mla_mb = mla_elem * factor

    return {
        "mha_elements_per_token": float(mha_elem),
        "gqa_elements_per_token": float(gqa_elem),
        "mla_elements_per_token": float(mla_elem),
        "mha_cache_mb": mha_mb,
        "gqa_cache_mb": gqa_mb,
        "mla_cache_mb": mla_mb,
        "mla_vs_mha_compression_ratio": mha_elem / mla_elem,
        "mla_vs_gqa_compression_ratio": gqa_elem / mla_elem,
    }


def main() -> None:
    """Run a deterministic verification of MLA matrix absorption and KV savings."""
    torch.manual_seed(42)
    mla = MultiHeadLatentAttention(
        d_model=64, num_heads=8, d_head=8, d_latent_kv=16, d_rope=4
    )
    h = torch.randn(2, 12, 64)
    y_std = mla.forward_standard(h)
    y_abs = mla.forward_absorbed(h)
    max_diff = (y_std - y_abs).abs().max().item()
    stats = profile_mla_kv_cache_savings()
    print(
        f"[Module 27] MLA absorption max_diff={max_diff:.2e} | "
        f"KV compression vs MHA={stats['mla_vs_mha_compression_ratio']:.2f}x"
    )


if __name__ == "__main__":
    main()
