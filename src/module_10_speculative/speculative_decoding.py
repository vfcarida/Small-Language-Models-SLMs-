"""
==============================================================================
MODULE 10 — LOSSLESS SPECULATIVE DECODING & MEDUSA MULTI-TOKEN DRAFT HEADS
==============================================================================
OBJECTIVE:
    Use a compact quantized SLM (0.5B-1.5B) or lightweight Medusa residual
    heads to accelerate autoregressive generation by 2.0x-3.5x with provably
    lossless preservation of the target model's output distribution:
    1. Exact Speculative Rejection Sampling (`speculative_decoding_step`)
       (Leviathan et al., ICML 2023; Chen et al., DeepMind 2023).
    2. Residual Distribution Normalization `norm(max(0, p(x) - q(x)))` and
       marginal distribution identity verification.
    3. Analytical & Empirical Wall-Clock Speedup Calculator.
    4. Parallel Multi-Horizon Medusa Draft Heads (`MedusaDraftHead`, Cai et al.,
       ICML 2024).

RUN COMMAND:
    python src/module_10_speculative/speculative_decoding.py
==============================================================================
"""

from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeculativeStepResult:
    """Telemetry and emitted token sequence from a single speculative step."""

    emitted_tokens: List[int]
    num_accepted_draft_tokens: int
    lookahead_k: int
    acceptance_rate: float
    mean_acceptance_prob: float
    all_accepted_bonus: bool


def compute_residual_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the rejection-adjusted residual distribution:
        p_res(x) = max(0, p_target(x) - q_draft(x)) / sum_v max(0, p_target(v) - q_draft(v))

    THEOREM (Leviathan et al., ICML 2023, Appendix A):
        Sampling x ~ q(x), accepting with probability min(1, p(x)/q(x)), and
        otherwise sampling x' ~ p_res(x') yields marginal probability:
            P(X = x) = min(q(x), p(x)) + (1 - sum_v min(q(v), p(v))) * p_res(x)
                     = min(q(x), p(x)) + max(0, p(x) - q(x))
                     = p(x).
        Thus speculative decoding is strictly lossless for any draft model q.
    """
    diff = torch.clamp(target_probs.float() - draft_probs.float(), min=0.0)
    total = diff.sum(dim=-1, keepdim=True)
    # Fallback to target_probs when p_target == q_draft everywhere
    fallback_mask = total <= eps
    normed = diff / torch.clamp(total, min=eps)
    return torch.where(fallback_mask, target_probs.float(), normed)


def verify_marginal_distribution_identity(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> float:
    """
    Analytically verify the lossless speculative identity across the vocabulary:
        P_marginal(x) = min(q(x), p(x)) + (1 - beta) * p_res(x) == p(x)
    Returns the maximum absolute discrepancy max_x |P_marginal(x) - p(x)|.
    """
    p = target_probs.float()
    q = draft_probs.float()
    min_pq = torch.minimum(q, p)
    beta = min_pq.sum(dim=-1, keepdim=True)
    p_res = compute_residual_distribution(p, q)
    marginal = min_pq + (1.0 - beta) * p_res
    return float(torch.max(torch.abs(marginal - p)).item())


def speculative_decoding_step(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    uniform_rands: Optional[torch.Tensor] = None,
    seed: int = 42,
) -> SpeculativeStepResult:
    """
    Execute one lossless speculative decoding verification step of lookahead `K`.

    Args:
        draft_tokens: 1D tensor of shape `(K,)` containing candidate token IDs
            sampled autoregressively from the SLM draft model.
        draft_probs: 2D tensor of shape `(K, vocab_size)` with SLM draft
            probabilities `q_1, ..., q_K`.
        target_probs: 2D tensor of shape `(K + 1, vocab_size)` with target LLM
            probabilities `p_1, ..., p_{K+1}` computed in a single parallel pass.
        uniform_rands: Optional 1D tensor of shape `(K,)` in `[0, 1)` for
            deterministic unit testing of acceptance/rejection branches.
        seed: Random seed for residual/bonus sampling.
    """
    if draft_tokens.dim() != 1:
        raise ValueError("draft_tokens must be a 1D tensor of shape (K,).")
    k = int(draft_tokens.shape[0])
    if draft_probs.shape[0] != k or target_probs.shape[0] != k + 1:
        raise ValueError(
            f"Expected draft_probs of shape ({k}, V) and target_probs of shape ({k + 1}, V)."
        )

    gen = torch.Generator(device=draft_probs.device)
    gen.manual_seed(seed)

    if uniform_rands is None:
        u_samples = torch.rand(k, generator=gen, device=draft_probs.device)
    else:
        u_samples = uniform_rands.to(draft_probs.device)

    emitted: List[int] = []
    accepted_count = 0
    acc_probs: List[float] = []

    for i in range(k):
        tok_id = int(draft_tokens[i].item())
        q_val = max(float(draft_probs[i, tok_id].item()), 1e-12)
        p_val = float(target_probs[i, tok_id].item())
        accept_prob = min(1.0, p_val / q_val)
        acc_probs.append(accept_prob)

        if float(u_samples[i].item()) <= accept_prob:
            emitted.append(tok_id)
            accepted_count += 1
        else:
            p_res = compute_residual_distribution(target_probs[i], draft_probs[i])
            resampled_id = int(torch.multinomial(p_res, num_samples=1, generator=gen).item())
            emitted.append(resampled_id)
            break

    all_accepted = accepted_count == k
    if all_accepted:
        bonus_id = int(
            torch.multinomial(target_probs[k].float(), num_samples=1, generator=gen).item()
        )
        emitted.append(bonus_id)

    mean_acc_prob = sum(acc_probs) / max(len(acc_probs), 1)
    return SpeculativeStepResult(
        emitted_tokens=emitted,
        num_accepted_draft_tokens=accepted_count,
        lookahead_k=k,
        acceptance_rate=round(accepted_count / max(k, 1), 4),
        mean_acceptance_prob=round(mean_acc_prob, 4),
        all_accepted_bonus=all_accepted,
    )


def compute_speculative_speedup(
    acceptance_rate: float,
    lookahead_k: int = 4,
    draft_to_target_cost_ratio: float = 0.05,
) -> Dict[str, float]:
    """
    Compute theoretical expected tokens per target step and net wall-clock
    speedup factor (Leviathan et al., ICML 2023, Equation 3):
        E[tokens] = (1 - alpha^(K + 1)) / (1 - alpha)
        Speedup   = E[tokens] / (1 + K * c)
    where `alpha in [0, 1]` is the token acceptance rate and `c = T_draft / T_target`
    is the relative latency ratio of the SLM draft model vs. the target LLM.
    """
    if not (0.0 <= acceptance_rate <= 1.0):
        raise ValueError("acceptance_rate must lie in [0.0, 1.0].")
    if lookahead_k < 1:
        raise ValueError("lookahead_k must be >= 1.")
    if draft_to_target_cost_ratio < 0.0:
        raise ValueError("draft_to_target_cost_ratio must be non-negative.")

    if abs(1.0 - acceptance_rate) < 1e-8:
        expected_tokens = float(lookahead_k + 1)
    else:
        expected_tokens = (1.0 - (acceptance_rate ** (lookahead_k + 1))) / (
            1.0 - acceptance_rate
        )

    step_cost = 1.0 + lookahead_k * draft_to_target_cost_ratio
    wallclock_speedup = expected_tokens / step_cost

    return {
        "acceptance_rate": round(acceptance_rate, 4),
        "lookahead_k": float(lookahead_k),
        "draft_to_target_cost_ratio": round(draft_to_target_cost_ratio, 4),
        "expected_tokens_per_step": round(expected_tokens, 4),
        "wallclock_speedup_x": round(wallclock_speedup, 4),
    }


class MedusaResidualBlock(nn.Module):
    """Single residual SiLU block `h_out = h + W * SiLU(h)` for Medusa draft heads."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.linear(F.silu(hidden_states))


class MedusaDraftHead(nn.Module):
    """
    Multi-Horizon Speculative Draft Head (Cai et al., ICML 2024, "Medusa: Simple
    LLM Inference Acceleration Framework with Multiple Decoding Heads").

    Attaches `num_heads = K` lightweight residual blocks to the final hidden
    state `h_t` of a base model to predict tokens `[t+2, t+3, ..., t+K+1]` in a
    single parallel forward pass.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        vocab_size: int = 512,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        self.blocks = nn.ModuleList(
            [MedusaResidualBlock(hidden_size) for _ in range(num_heads)]
        )
        self.lm_heads = nn.ModuleList(
            [nn.Linear(hidden_size, vocab_size, bias=False) for _ in range(num_heads)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute draft logits for all `num_heads` lookahead positions.

        Args:
            hidden_states: Tensor of shape `(batch, seq_len, hidden_size)`.

        Returns:
            Tensor of shape `(num_heads, batch, seq_len, vocab_size)`.
        """
        head_logits = [
            lm_head(block(hidden_states))
            for block, lm_head in zip(self.blocks, self.lm_heads)
        ]
        return torch.stack(head_logits, dim=0)


def main() -> None:
    torch.manual_seed(42)
    vocab_size, k = 128, 4
    p_target = F.softmax(torch.randn(k + 1, vocab_size) * 1.5, dim=-1)
    # Simulate a well-aligned 1.5B SLM draft distribution close to the target
    q_draft = F.softmax(torch.log(p_target[:k] + 1e-8) + torch.randn(k, vocab_size) * 0.25, dim=-1)
    draft_tokens = torch.argmax(q_draft, dim=-1)

    max_ident_err = verify_marginal_distribution_identity(p_target[:k], q_draft)
    step_res = speculative_decoding_step(draft_tokens, q_draft, p_target, seed=42)
    speedup = compute_speculative_speedup(
        acceptance_rate=0.82, lookahead_k=4, draft_to_target_cost_ratio=0.04
    )

    logger.info("=" * 72)
    logger.info("MODULE 10: LOSSLESS SPECULATIVE DECODING & MEDUSA TELEMETRY")
    logger.info("=" * 72)
    logger.info(
        f"Marginal Identity Error: {max_ident_err:.2e} (Exact Lossless Distribution Guarantee)"
    )
    logger.info(
        f"Speculative Step (K={k}): Accepted={step_res.num_accepted_draft_tokens}/{k} "
        f"| Emitted Tokens={len(step_res.emitted_tokens)} ({step_res.emitted_tokens})"
    )
    logger.info(
        f"Speedup Projection (alpha=0.82, K=4, c=0.04): "
        f"E[tokens/step]={speedup['expected_tokens_per_step']:.2f} "
        f"| Net Speedup={speedup['wallclock_speedup_x']:.2f}x"
    )


if __name__ == "__main__":
    main()
