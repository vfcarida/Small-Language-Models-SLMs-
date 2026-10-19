"""
==============================================================================
MODULE 16 — LATENT-SPACE CONTINUOUS REASONING (Coconut & RECURRENT DEPTH SLMs)
==============================================================================
OBJECTIVE:
    Replace verbose discrete Chain-of-Thought (`<think>...</think>`) token chains
    with compact continuous hidden-state thought recurrence (`Coconut`, Hao et al.,
    Meta FAIR 2024, arXiv:2412.06769; `Recurrent Depth`, Geiping et al., 2025,
    arXiv:2502.05171):
    1. `AdaptivePonderingGate`: Measures step-to-step predictive distribution
       convergence via symmetric KL divergence D_KL(p^{(k)} || p^{(k-1)}) and
       dynamically halts continuous recurrence once thoughts stabilize (< tau).
    2. `LatentThoughtReasoner`: Recurrent latent-space reasoning block that
       iteratively refines a continuous thought vector h^{(k)} in R^(d_model)
       before emitting discrete vocabulary logits.
    3. `compare_discrete_cot_vs_latent_coconut`: Analytical profiler contrasting
       KV-cache memory growth and autoregressive decoding latency of discrete CoT
       vs. Coconut latent pondering.

RUN COMMAND:
    python src/module_16_latent_reasoning/coconut_reasoning.py
==============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatentReasoningTrace:
    """Telemetry from a Coconut / Recurrent-Depth latent pondering episode."""

    steps_executed: int
    max_steps: int
    halted_early: bool
    kl_trajectory: List[float]
    final_confidence: float


class AdaptivePonderingGate:
    """
    Dynamic halting gate for Recurrent-Depth / Coconut latent reasoning
    (Geiping et al., 2025, arXiv:2502.05171):
    Computes the symmetric KL divergence between consecutive step distributions
    p^{(k)} and p^{(k-1)} over the vocabulary and triggers early halting when
    the divergence falls below `kl_threshold`.
    """

    @staticmethod
    def compute_step_kl_divergence(
        current_logits: torch.Tensor,
        previous_logits: torch.Tensor,
        eps: float = 1e-8,
    ) -> float:
        """
        Compute mean symmetric KL divergence across the batch:
            0.5 * (KL(p_curr || p_prev) + KL(p_prev || p_curr))
        """
        p_curr = F.softmax(current_logits.float(), dim=-1).clamp(min=eps)
        p_prev = F.softmax(previous_logits.float(), dim=-1).clamp(min=eps)

        kl_fwd = torch.sum(p_curr * (torch.log(p_curr) - torch.log(p_prev)), dim=-1)
        kl_rev = torch.sum(p_prev * (torch.log(p_prev) - torch.log(p_curr)), dim=-1)
        sym_kl = 0.5 * (kl_fwd + kl_rev)
        return float(sym_kl.mean().item())


class LatentThoughtReasoner(nn.Module):
    """
    Coconut / Recurrent-Depth Continuous Latent Reasoner:
        Given an initial prompt summary state h^{(0)} in R^(batch x d_model),
        iteratively refines the continuous thought state:
            h^{(k)} = LayerNorm(h^{(k-1)} + alpha_k * CoreBlock(h^{(k-1)}, h^{(0)}))
        Unlike discrete CoT—which projects every intermediate step through a
        bottleneck one-hot vocabulary token—latent thoughts preserve full
        d_model superposition across multiple candidate reasoning paths.
    """

    def __init__(
        self,
        d_model: int = 64,
        vocab_size: int = 256,
        max_latent_steps: int = 6,
        kl_threshold: float = 1e-3,
    ) -> None:
        super().__init__()
        if max_latent_steps < 1:
            raise ValueError("max_latent_steps must be >= 1.")
        if kl_threshold < 0.0:
            raise ValueError("kl_threshold must be non-negative.")

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_latent_steps = max_latent_steps
        self.kl_threshold = kl_threshold

        self.thought_adapter = nn.Linear(d_model * 2, d_model, bias=False)
        self.thought_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, bias=False),
        )
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        prompt_hidden: torch.Tensor,
        enable_early_exit: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, LatentReasoningTrace]:
        """
        Execute up to `max_latent_steps` continuous thought recurrences.

        Args:
            prompt_hidden: Input tensor of shape (batch, seq_len, d_model) or (batch, d_model).
            enable_early_exit: Whether to halt early when step-to-step KL < kl_threshold.

        Returns:
            Tuple of (final_logits, final_thought_state, trace).
        """
        anchor = prompt_hidden[:, -1, :] if prompt_hidden.ndim == 3 else prompt_hidden
        thought = self.norm(anchor)

        prev_logits: Optional[torch.Tensor] = None
        kl_history: List[float] = []
        halted_early = False
        steps_run = 0

        for step_idx in range(1, self.max_latent_steps + 1):
            steps_run = step_idx
            # Damped residual refinement conditioned on original prompt anchor
            combined = torch.cat([thought, anchor], dim=-1)
            delta = self.thought_mlp(F.silu(self.thought_adapter(combined)))
            damping = 1.0 / float(step_idx**1.5)
            thought = self.norm(thought + damping * delta)

            curr_logits = self.lm_head(thought)
            if prev_logits is not None:
                kl_val = AdaptivePonderingGate.compute_step_kl_divergence(curr_logits, prev_logits)
                kl_history.append(kl_val)
                if enable_early_exit and kl_val < self.kl_threshold:
                    halted_early = True
                    prev_logits = curr_logits
                    break
            prev_logits = curr_logits

        assert prev_logits is not None
        probs = F.softmax(prev_logits.float(), dim=-1)
        max_conf = float(probs.max(dim=-1).values.mean().item())

        trace = LatentReasoningTrace(
            steps_executed=steps_run,
            max_steps=self.max_latent_steps,
            halted_early=halted_early,
            kl_trajectory=kl_history,
            final_confidence=round(max_conf, 4),
        )
        return prev_logits, thought, trace


def compare_discrete_cot_vs_latent_coconut(
    discrete_cot_tokens: int = 256,
    latent_thought_steps: int = 16,
    answer_tokens: int = 32,
    kv_bytes_per_token: float = 57344.0,  # e.g., 28 layers * 2 * 2 KV heads * 128 dim * 2 bytes
    ms_per_discrete_token: float = 12.5,
    ms_per_latent_step: float = 6.0,      # No LM head softmax sampling or tokenizer round-trip
) -> Dict[str, float]:
    """
    Compare memory and latency of standard Discrete Chain-of-Thought vs.
    Coconut Continuous Latent Reasoning (Hao et al., 2024).
    """
    if discrete_cot_tokens < 1 or latent_thought_steps < 1:
        raise ValueError("Token and step counts must be >= 1.")

    discrete_total_tokens = discrete_cot_tokens + answer_tokens
    # Recurrent-depth latent steps reuse or compress state without appending verbose CoT tokens
    latent_total_kv_tokens = latent_thought_steps + answer_tokens

    discrete_kv_mb = (discrete_total_tokens * kv_bytes_per_token) / (1024**2)
    latent_kv_mb = (latent_total_kv_tokens * kv_bytes_per_token) / (1024**2)

    discrete_latency_ms = discrete_total_tokens * ms_per_discrete_token
    latent_latency_ms = (
        latent_thought_steps * ms_per_latent_step + answer_tokens * ms_per_discrete_token
    )

    return {
        "discrete_cot_kv_mb": round(discrete_kv_mb, 2),
        "latent_coconut_kv_mb": round(latent_kv_mb, 2),
        "kv_cache_reduction_x": round(discrete_kv_mb / max(latent_kv_mb, 1e-6), 2),
        "discrete_cot_latency_ms": round(discrete_latency_ms, 2),
        "latent_coconut_latency_ms": round(latent_latency_ms, 2),
        "latency_speedup_x": round(discrete_latency_ms / max(latent_latency_ms, 1e-6), 2),
    }


def main() -> None:
    torch.manual_seed(42)
    reasoner = LatentThoughtReasoner(
        d_model=64,
        vocab_size=256,
        max_latent_steps=8,
        kl_threshold=1e-4,
    )
    prompt_h = torch.randn(2, 10, 64)
    logits, thought, trace = reasoner(prompt_h, enable_early_exit=True)
    bench = compare_discrete_cot_vs_latent_coconut(discrete_cot_tokens=256, latent_thought_steps=16)

    logger.info("=" * 72)
    logger.info("MODULE 16: LATENT-SPACE CONTINUOUS REASONING (Coconut / Recurrent Depth)")
    logger.info("=" * 72)
    logger.info(
        f"[Adaptive Pondering] Executed {trace.steps_executed}/{trace.max_steps} steps "
        f"(Halted Early={trace.halted_early}) | Final KL={trace.kl_trajectory[-1]:.2e}"
    )
    logger.info(
        f"[CoT vs. Coconut] Discrete CoT KV={bench['discrete_cot_kv_mb']:.2f}MB ({bench['discrete_cot_latency_ms']:.0f}ms) "
        f"vs. Coconut KV={bench['latent_coconut_kv_mb']:.2f}MB ({bench['latent_coconut_latency_ms']:.0f}ms) "
        f"-> {bench['kv_cache_reduction_x']:.2f}x KV reduction, {bench['latency_speedup_x']:.2f}x speedup"
    )


if __name__ == "__main__":
    main()
