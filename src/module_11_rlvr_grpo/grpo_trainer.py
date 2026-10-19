"""
==============================================================================
MODULE 11 — REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS (RLVR / GRPO)
==============================================================================
OBJECTIVE:
    Incentivize multi-step chain-of-thought reasoning and deterministic schema
    compliance in Small Language Models (<4B parameters) without a memory-heavy
    Value Critic network, grounded in DeepSeek-R1 (arXiv:2501.12948) and
    DeepSeekMath / GRPO (Shao et al., 2024, arXiv:2402.03300):
    1. Critic-Free Group Relative Advantage Estimation (`compute_group_advantages`).
    2. Schulman's Unbiased Non-Negative Per-Token KL Estimator (`compute_schulman_kl`).
    3. Clipped Surrogate GRPO Objective (`compute_grpo_loss`).
    4. Deterministic Verifiable Reward Oracles (`VerifiableRewardOracle`) for
       `<think>...</think><answer>...</answer>` formatting, exact math answers,
       and strict JSON schema validation.

RUN COMMAND:
    python src/module_11_rlvr_grpo/grpo_trainer.py
==============================================================================
"""

import json
import logging
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_group_advantages(
    rewards: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute critic-free group-normalized advantages across `G` completions
    sampled for the same prompt (Shao et al., 2024, Equation 4):
        A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)

    Supports 1D tensors of shape `(G,)` or 2D tensors of shape `(B, G)`.
    """
    if rewards.numel() < 2 or rewards.shape[-1] < 2:
        raise ValueError("GRPO requires group size G >= 2 to normalize advantages.")

    r_fp32 = rewards.float()
    mean_r = r_fp32.mean(dim=-1, keepdim=True)
    std_r = r_fp32.std(dim=-1, unbiased=False, keepdim=True)
    advantages = (r_fp32 - mean_r) / (std_r + eps)
    return advantages.to(rewards.dtype)


def compute_schulman_kl(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Schulman's unbiased, strictly non-negative per-token KL divergence
    estimator used in GRPO (Shao et al., 2024, Equation 5):
        D_KL(pi_theta || pi_ref) = exp(log pi_ref - log pi_theta)
                                 - (log pi_ref - log pi_theta) - 1 >= 0
    """
    log_ratio = ref_logprobs.float() - policy_logprobs.float()
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return torch.clamp(kl, min=0.0)


def compute_grpo_loss(
    policy_logprobs: torch.Tensor,
    old_policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    rewards: torch.Tensor,
    completion_mask: Optional[torch.Tensor] = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the Group Relative Policy Optimization (GRPO) loss over a group of
    `G` completions of sequence length `T`:

        L_GRPO(theta) = - 1/G sum_{i=1}^G 1/|o_i| sum_{t=1}^{|o_i|} [
            min(rho_{i,t} A_i, clip(rho_{i,t}, 1-eps, 1+eps) A_i)
            - beta * D_KL(pi_theta || pi_ref)_{i,t}
        ]

    Args:
        policy_logprobs: Current policy token log-probabilities `(G, T)`.
        old_policy_logprobs: Rollout behavior policy token log-probabilities `(G, T)`.
        ref_logprobs: Frozen reference policy token log-probabilities `(G, T)`.
        rewards: Scalar verifiable rewards for each completion `(G,)`.
        completion_mask: Optional binary mask `(G, T)` (`1` for valid completion
            tokens, `0` for padding).
        clip_eps: PPO probability ratio clipping parameter (default `0.2`).
        kl_beta: Reference KL regularization coefficient (default `0.04`).

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: `(loss, telemetry)`.
    """
    if policy_logprobs.dim() != 2:
        raise ValueError("policy_logprobs must be a 2D tensor of shape (G, T).")
    g, t = policy_logprobs.shape
    if old_policy_logprobs.shape != (g, t) or ref_logprobs.shape != (g, t):
        raise ValueError("policy_logprobs, old_policy_logprobs, and ref_logprobs must share shape (G, T).")
    if rewards.shape != (g,):
        raise ValueError(f"Expected rewards of shape ({g},), got {rewards.shape}.")

    mask = (
        torch.ones_like(policy_logprobs, dtype=torch.float32)
        if completion_mask is None
        else completion_mask.float()
    )

    advantages = compute_group_advantages(rewards).unsqueeze(-1)  # (G, 1)
    log_ratio = policy_logprobs - old_policy_logprobs.detach()
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    per_token_surrogate = torch.minimum(surr1, surr2)

    per_token_kl = compute_schulman_kl(policy_logprobs, ref_logprobs.detach())
    per_token_obj = per_token_surrogate - kl_beta * per_token_kl

    token_counts = mask.sum(dim=-1).clamp(min=1.0)
    per_seq_obj = (per_token_obj * mask).sum(dim=-1) / token_counts
    loss = -per_seq_obj.mean()

    with torch.no_grad():
        clipped_mask = (torch.abs(ratio - 1.0) > clip_eps).float() * mask
        clip_frac = (clipped_mask.sum() / mask.sum().clamp(min=1.0)).item()
        mean_kl = ((per_token_kl * mask).sum() / mask.sum().clamp(min=1.0)).item()
        mean_surr = ((per_token_surrogate * mask).sum() / mask.sum().clamp(min=1.0)).item()

    return loss, {
        "loss": round(float(loss.item()), 6),
        "policy_surrogate": round(float(mean_surr), 6),
        "mean_kl": round(float(mean_kl), 6),
        "clip_fraction": round(float(clip_frac), 4),
        "mean_reward": round(float(rewards.float().mean().item()), 4),
        "reward_std": round(float(rewards.float().std(unbiased=False).item()), 4),
    }


class VerifiableRewardOracle:
    """
    Deterministic, rule-based reward verifiers for RLVR / GRPO training
    (DeepSeek-R1, 2025, Section 2.2.2 "Rule-based Reward Modeling"):
        1. Reasoning Format Verifier (`<think>...</think><answer>...</answer>`)
        2. Exact Math / Symbolic Answer Verifier
        3. Agentic JSON Schema Verifier
    """

    _THINK_ANSWER_PATTERN = re.compile(
        r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
        re.DOTALL,
    )
    _ANSWER_TAG_PATTERN = re.compile(r"<answer>(?P<answer>.*?)</answer>", re.DOTALL)
    _BOXED_PATTERN = re.compile(r"\\boxed\{(?P<answer>[^{}]+)\}")

    @classmethod
    def verify_reasoning_format(cls, completion: str) -> float:
        """
        Return `1.0` if the completion strictly follows `<think>...</think><answer>...</answer>`
        with non-empty reasoning and answer content; `0.25` if both tags appear
        somewhere with content; `0.0` otherwise.
        """
        strict_match = cls._THINK_ANSWER_PATTERN.match(completion)
        if strict_match:
            think_body = strict_match.group("think").strip()
            ans_body = strict_match.group("answer").strip()
            if len(think_body) > 0 and len(ans_body) > 0:
                return 1.0
        if "<think>" in completion and "</think>" in completion and "<answer>" in completion and "</answer>" in completion:
            return 0.25
        return 0.0

    @classmethod
    def extract_answer(cls, completion: str) -> str:
        """Extract final answer string from `<answer>...</answer>` or `\\boxed{...}`."""
        tag_match = cls._ANSWER_TAG_PATTERN.search(completion)
        if tag_match:
            return tag_match.group("answer").strip()
        boxed_match = cls._BOXED_PATTERN.search(completion)
        if boxed_match:
            return boxed_match.group("answer").strip()
        return completion.strip()

    @classmethod
    def verify_math_answer(
        cls,
        completion: str,
        ground_truth: str,
        tol: float = 1e-6,
    ) -> float:
        """
        Verify whether the extracted answer matches `ground_truth` numerically
        (within `tol`) or via normalized string comparison. Returns `1.0` or `0.0`.
        """
        candidate = cls.extract_answer(completion).replace(",", "").strip()
        target = ground_truth.replace(",", "").strip()

        try:
            cand_val = float(candidate)
            targ_val = float(target)
            return 1.0 if abs(cand_val - targ_val) <= tol else 0.0
        except ValueError:
            return 1.0 if candidate.lower() == target.lower() else 0.0

    @classmethod
    def verify_json_schema(
        cls,
        completion: str,
        required_keys: Sequence[str],
    ) -> float:
        """
        Verify whether the extracted answer is valid JSON containing all
        `required_keys`. Returns `1.0` if valid and complete, `0.0` otherwise.
        """
        candidate = cls.extract_answer(completion)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(parsed, dict):
            return 0.0
        return 1.0 if all(k in parsed for k in required_keys) else 0.0

    @classmethod
    def score_math_group(
        cls,
        completions: Sequence[str],
        ground_truth: str,
        format_weight: float = 0.25,
        accuracy_weight: float = 0.75,
    ) -> Dict[str, List[float]]:
        """
        Score a group of `G` completions combining format reward and exact math
        accuracy reward, and return raw rewards plus GRPO group advantages.
        """
        raw_rewards: List[float] = []
        for comp in completions:
            r_fmt = cls.verify_reasoning_format(comp)
            r_acc = cls.verify_math_answer(comp, ground_truth)
            raw_rewards.append(round(format_weight * r_fmt + accuracy_weight * r_acc, 4))

        r_tensor = torch.tensor(raw_rewards, dtype=torch.float32)
        adv_tensor = compute_group_advantages(r_tensor)
        return {
            "rewards": raw_rewards,
            "advantages": [round(float(a), 4) for a in adv_tensor.tolist()],
        }


def main() -> None:
    torch.manual_seed(42)
    group_completions = [
        "<think>12 * 15 = 180, plus 20 is 200.</think>\n<answer>200</answer>",
        "<think>Let me calculate 12 * 15 = 180. Adding 20 gives 195.</think>\n<answer>195</answer>",
        "The answer is 200.",
        "I am not sure, maybe 150.",
    ]
    scored = VerifiableRewardOracle.score_math_group(group_completions, ground_truth="200")

    g, seq_len = len(group_completions), 16
    old_lp = -torch.rand(g, seq_len) * 1.5
    ref_lp = old_lp - 0.05
    policy_lp = (old_lp + torch.randn(g, seq_len) * 0.05).requires_grad_(True)
    rewards = torch.tensor(scored["rewards"], dtype=torch.float32)

    loss, metrics = compute_grpo_loss(policy_lp, old_lp, ref_lp, rewards)
    loss.backward()

    logger.info("=" * 72)
    logger.info("MODULE 11: RLVR & GRPO (DEEPSEEK-R1 STYLE) REASONING ALIGNMENT")
    logger.info("=" * 72)
    logger.info(f"Group Verifiable Rewards:    {scored['rewards']}")
    logger.info(f"Normalized GRPO Advantages:  {scored['advantages']}")
    logger.info(
        f"GRPO Step Telemetry: Loss={metrics['loss']:.4f} | Surrogate={metrics['policy_surrogate']:.4f} "
        f"| Mean KL={metrics['mean_kl']:.6f} | ClipFrac={metrics['clip_fraction']*100:.1f}%"
    )


if __name__ == "__main__":
    main()
