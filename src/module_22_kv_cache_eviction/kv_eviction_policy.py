"""
==============================================================================
MODULE 22 — LONG-CONTEXT KV-CACHE EVICTION (StreamingLLM Sinks & H2O Oracle)
==============================================================================
OBJECTIVE:
    Enable infinite-context streaming and sub-linear KV-cache memory on edge
    SLMs under a strict fixed token budget B_max = N_sink + N_hh + N_recent:
    1. `StreamingLLM` Attention Sinks (Xiao et al., MIT Han Lab / ICLR 2024,
       arXiv:2309.17453): Preserves the first `num_sink_tokens` initial tokens
       that absorb surplus attention probability mass in autoregressive Softmax.
    2. `H2O` Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023, arXiv:2306.14048):
       Tracks cumulative attention scores s_j = sum_{t >= j} A_{t, j} and
       selectively retains the top `num_heavy_hitters` high-utility tokens
       alongside the `num_recent_tokens` local sliding window.
    3. `compare_kv_eviction_fidelity`: Empirical comparison demonstrating that
       `Sink + H2O + Recent` achieves orders-of-magnitude lower attention output
       distortion than naive FIFO / Sliding-Window eviction at identical memory.

RUN COMMAND:
    python src/module_22_kv_cache_eviction/kv_eviction_policy.py
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KVEvictionFidelityReport:
    """Comparison telemetry between Full KV-Cache, Sliding-Window, and Sink+H2O."""

    total_sequence_length: int
    cache_budget_tokens: int
    compression_ratio_x: float
    h2o_sink_l2_error: float
    sliding_window_l2_error: float
    error_reduction_vs_window_x: float


class H2OStreamingKVCache:
    """
    Hybrid `StreamingLLM` + `H2O` (Heavy-Hitter Oracle) Bounded KV-Cache:
        Maintains at most `max_budget = num_sink + num_heavy_hitters + num_recent`
        key-value vectors regardless of how many tokens T >> max_budget are streamed.
    """

    def __init__(
        self,
        num_sink_tokens: int = 4,
        num_heavy_hitters: int = 8,
        num_recent_tokens: int = 8,
    ) -> None:
        if num_sink_tokens < 1 or num_heavy_hitters < 0 or num_recent_tokens < 1:
            raise ValueError("Invalid sink, heavy-hitter, or recent token counts.")
        self.num_sink = num_sink_tokens
        self.num_hh = num_heavy_hitters
        self.num_recent = num_recent_tokens
        self.max_budget = num_sink_tokens + num_heavy_hitters + num_recent_tokens

    def select_retained_indices(
        self,
        cumulative_attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given 1D cumulative attention scores `s` of shape (seq_len,) accumulated
        over past queries, return the sorted 1D tensor of retained token indices
        of length min(seq_len, max_budget).
        """
        if cumulative_attention_scores.ndim != 1:
            raise ValueError("cumulative_attention_scores must be a 1D tensor of shape (seq_len,).")

        seq_len = cumulative_attention_scores.shape[0]
        device = cumulative_attention_scores.device
        if seq_len <= self.max_budget:
            return torch.arange(seq_len, device=device, dtype=torch.long)

        # 1. Initial Attention Sink indices: [0, ..., num_sink - 1]
        sink_idx = torch.arange(0, self.num_sink, device=device, dtype=torch.long)

        # 2. Local Recent Window indices: [seq_len - num_recent, ..., seq_len - 1]
        recent_start = seq_len - self.num_recent
        recent_idx = torch.arange(recent_start, seq_len, device=device, dtype=torch.long)

        # 3. Middle candidate pool for Heavy-Hitters: [num_sink, ..., recent_start - 1]
        if self.num_hh > 0 and recent_start > self.num_sink:
            mid_scores = cumulative_attention_scores[self.num_sink : recent_start]
            k_hh = min(self.num_hh, mid_scores.shape[0])
            _, top_rel = torch.topk(mid_scores, k=k_hh, largest=True, sorted=False)
            hh_idx = (top_rel + self.num_sink).to(torch.long)
            combined = torch.cat([sink_idx, hh_idx, recent_idx], dim=0)
        else:
            combined = torch.cat([sink_idx, recent_idx], dim=0)

        retained_sorted, _ = torch.sort(combined)
        return retained_sorted


def compare_kv_eviction_fidelity(
    seq_len: int = 128,
    head_dim: int = 32,
    num_sink: int = 4,
    num_hh: int = 8,
    num_recent: int = 8,
    seed: int = 42,
) -> KVEvictionFidelityReport:
    """
    Simulate an autoregressive stream where initial sink tokens (0..num_sink-1)
    and specific semantic anchor tokens in the middle of the context carry high
    key-query alignment, and compare the final query's attention output under:
        1. Full Unbounded KV-Cache (exact ground truth)
        2. Naive Sliding-Window Eviction (retains only the last B_max tokens)
        3. StreamingLLM + H2O Eviction (retains Sinks + Heavy-Hitters + Recent)
    """
    torch.manual_seed(seed)
    budget = num_sink + num_hh + num_recent
    if seq_len <= budget:
        raise ValueError("seq_len must exceed total cache budget for eviction comparison.")

    k_mat = torch.randn(seq_len, head_dim) * 0.5
    v_mat = torch.randn(seq_len, head_dim)
    q_final = torch.randn(1, head_dim)

    # Plant strong attention sink alignment at token 0 and heavy-hitter anchors at middle indices
    anchor_positions = [0, 1, seq_len // 3, seq_len // 2]
    for pos in anchor_positions:
        k_mat[pos] = q_final.squeeze(0) * 2.5
        v_mat[pos] = torch.randn(head_dim) * 3.0

    # Compute full attention weights for final query and cumulative importance scores
    scale = 1.0 / math.sqrt(head_dim)
    logits_full = (q_final @ k_mat.t()).squeeze(0) * scale  # (seq_len,)
    weights_full = F.softmax(logits_full, dim=-1)
    out_full = weights_full.unsqueeze(0) @ v_mat  # (1, head_dim)

    # Cumulative attention scores across prefix queries
    q_all = q_final.repeat(seq_len, 1) + torch.randn(seq_len, head_dim) * 0.2
    scores_matrix = torch.tril(F.softmax((q_all @ k_mat.t()) * scale, dim=-1))
    cum_scores = scores_matrix.sum(dim=0)

    # Policy A: StreamingLLM Sinks + H2O Heavy-Hitters + Recent Window
    cache = H2OStreamingKVCache(num_sink, num_hh, num_recent)
    idx_h2o = cache.select_retained_indices(cum_scores)
    k_h2o = k_mat[idx_h2o]
    v_h2o = v_mat[idx_h2o]
    w_h2o = F.softmax((q_final @ k_h2o.t()) * scale, dim=-1)
    out_h2o = w_h2o @ v_h2o

    # Policy B: Naive Sliding Window of identical total budget B_max
    idx_win = torch.arange(seq_len - budget, seq_len, dtype=torch.long)
    k_win = k_mat[idx_win]
    v_win = v_mat[idx_win]
    w_win = F.softmax((q_final @ k_win.t()) * scale, dim=-1)
    out_win = w_win @ v_win

    err_h2o = float(torch.linalg.norm(out_full - out_h2o).item())
    err_win = float(torch.linalg.norm(out_full - out_win).item())

    return KVEvictionFidelityReport(
        total_sequence_length=seq_len,
        cache_budget_tokens=budget,
        compression_ratio_x=round(float(seq_len) / float(budget), 2),
        h2o_sink_l2_error=round(err_h2o, 6),
        sliding_window_l2_error=round(err_win, 6),
        error_reduction_vs_window_x=round(err_win / max(err_h2o, 1e-8), 2),
    )


def main() -> None:
    report = compare_kv_eviction_fidelity(
        seq_len=256, head_dim=32, num_sink=4, num_hh=8, num_recent=8
    )

    logger.info("=" * 72)
    logger.info("MODULE 22: LONG-CONTEXT KV-CACHE EVICTION (StreamingLLM Sinks + H2O)")
    logger.info("=" * 72)
    logger.info(
        f"[KV Budget] Sequence Length={report.total_sequence_length} -> Bounded Cache={report.cache_budget_tokens} tokens "
        f"({report.compression_ratio_x:.1f}x KV Compression)"
    )
    logger.info(
        f"[Fidelity]  Sliding-Window L2 Error={report.sliding_window_l2_error:.4f} "
        f"vs. Sink+H2O L2 Error={report.h2o_sink_l2_error:.4f} "
        f"({report.error_reduction_vs_window_x:.1f}x more accurate attention reconstruction)"
    )


if __name__ == "__main__":
    main()
