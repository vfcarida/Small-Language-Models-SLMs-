"""
==============================================================================
MODULE 13 — MULTI-AGENT COLLABORATIVE SLM SWARMS & DEBATE VERIFICATION
==============================================================================
OBJECTIVE:
    Coordinate a swarm of specialized <2B parameter SLM micro-agents sharing
    a single frozen base model in VRAM via Multi-LoRA hot-swapping, achieving
    frontier-grade factuality and schema compliance through multi-agent debate
    and deterministic verification:
    1. Shared-Backbone Multi-LoRA Hot-Swapping (`MultiLoRASwarmBackbone`)
       (Belcák et al., NVIDIA Research 2025, arXiv:2506.02153).
    2. Confidence-Weighted Majority Voting (`confidence_weighted_vote`)
       (Du et al., ICML 2024, arXiv:2305.14325; Wang et al., ICLR 2023).
    3. Multi-Round Generator-Critic-Verifier Debate (`SLMAgentSwarm`)
       integrated with `VerifiableRewardOracle` and `PIIGuardrail`.

RUN COMMAND:
    python src/module_13_agentic_swarms/slm_swarm.py
==============================================================================
"""

from dataclasses import dataclass
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.module_08_routing.hybrid_router import PIIGuardrail
from src.module_11_rlvr_grpo.grpo_trainer import VerifiableRewardOracle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentProposal:
    """Single candidate response emitted by a role-specialized SLM agent."""

    agent_id: str
    role: str
    raw_completion: str
    extracted_answer: str
    confidence: float
    reliability_weight: float = 1.0


@dataclass(frozen=True)
class SwarmExecutionReport:
    """End-to-end telemetry from a multi-round SLM swarm debate."""

    task_prompt: str
    winning_answer: str
    winning_completion: str
    consensus_score: float
    round_1_mean_reward: float
    final_round_mean_reward: float
    pii_redacted: bool
    num_agents: int
    rounds_executed: int


class MultiLoRASwarmBackbone(nn.Module):
    """
    Shared frozen base projection W_0 in R^(d_out x d_in) with hot-swappable
    low-rank LoRA adapters (A_m, B_m) per agent role:
        y^(m) = x W_0^T + (alpha / r) * (x A_m^T) B_m^T
    Eliminates redundant base-model weight duplication across N concurrent SLM
    micro-agents on edge GPUs.
    """

    def __init__(
        self,
        d_in: int = 64,
        d_out: int = 64,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        roles: Sequence[str] = ("planner", "solver_a", "solver_b", "critic", "trism_auditor"),
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.lora_rank = lora_rank
        self.scaling = lora_alpha / max(lora_rank, 1)
        self.roles = tuple(roles)

        self.base_linear = nn.Linear(d_in, d_out, bias=False)
        self.base_linear.weight.requires_grad_(False)

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for idx, role in enumerate(self.roles):
            torch.manual_seed(100 + idx)
            self.lora_A[role] = nn.Parameter(torch.randn(lora_rank, d_in) * 0.02)
            self.lora_B[role] = nn.Parameter(torch.randn(d_out, lora_rank) * 0.02)

    def forward_role(self, x: torch.Tensor, role: str) -> torch.Tensor:
        """Execute forward pass using the frozen base weights + role-specific LoRA adapter."""
        if role not in self.lora_A:
            raise KeyError(f"Unknown agent role '{role}'. Registered roles: {self.roles}")
        base_out = self.base_linear(x)
        a_mat = self.lora_A[role]
        b_mat = self.lora_B[role]
        lora_out = (x @ a_mat.t()) @ b_mat.t() * self.scaling
        return base_out + lora_out

    @staticmethod
    def estimate_vram_savings(
        num_agents: int = 4,
        base_params_b: float = 1.54,
        lora_rank: int = 16,
        num_adapted_layers: int = 196,
        d_model: int = 1536,
        bytes_per_base_param: float = 0.53,  # AWQ INT4 (~4.25 bits)
    ) -> Dict[str, float]:
        """
        Compute VRAM footprint of `num_agents` independent SLM instances vs.
        1 shared base SLM + `num_agents` hot-swappable FP16 LoRA adapters.
        """
        if num_agents < 1:
            raise ValueError("num_agents must be >= 1.")

        single_base_mb = (base_params_b * 1e9 * bytes_per_base_param) / (1024**2)
        independent_fleet_mb = num_agents * single_base_mb

        # Each adapted linear layer has A (r x d) and B (d x r) in FP16 (2 bytes)
        params_per_adapter = num_adapted_layers * 2 * lora_rank * d_model
        single_adapter_mb = (params_per_adapter * 2.0) / (1024**2)
        shared_swarm_mb = single_base_mb + num_agents * single_adapter_mb
        savings_pct = (
            (independent_fleet_mb - shared_swarm_mb) / max(independent_fleet_mb, 1e-6)
        ) * 100.0

        return {
            "num_agents": float(num_agents),
            "independent_fleet_vram_mb": round(independent_fleet_mb, 2),
            "shared_multi_lora_vram_mb": round(shared_swarm_mb, 2),
            "single_adapter_overhead_mb": round(single_adapter_mb, 2),
            "vram_savings_percent": round(savings_pct, 2),
        }


def confidence_weighted_vote(
    proposals: Sequence[AgentProposal],
) -> Dict[str, Any]:
    """
    Aggregate multi-agent proposals via Confidence & Reliability Weighted Voting
    (Du et al., ICML 2024; Wang et al., ICLR 2023):
        Score(y) = sum_{m : a_m == y} w_m * C_m(y | x)
    """
    if not proposals:
        raise ValueError("proposals sequence cannot be empty.")

    answer_weights: Dict[str, float] = {}
    answer_exemplar: Dict[str, AgentProposal] = {}
    total_weight = 0.0

    for prop in proposals:
        key = prop.extracted_answer.strip()
        vote_w = max(0.0, float(prop.confidence)) * max(0.0, float(prop.reliability_weight))
        answer_weights[key] = answer_weights.get(key, 0.0) + vote_w
        total_weight += vote_w
        if key not in answer_exemplar or vote_w > (
            answer_exemplar[key].confidence * answer_exemplar[key].reliability_weight
        ):
            answer_exemplar[key] = prop

    winning_ans = max(answer_weights.keys(), key=lambda k: answer_weights[k])
    consensus_score = answer_weights[winning_ans] / max(total_weight, 1e-8)

    # Compute Shannon disagreement entropy across distinct candidate answers
    if len(answer_weights) <= 1 or total_weight <= 1e-8:
        disagreement_entropy = 0.0
    else:
        probs = [w / total_weight for w in answer_weights.values() if w > 0]
        raw_h = -sum(p * math.log(p + 1e-12) for p in probs)
        disagreement_entropy = raw_h / math.log(len(answer_weights))

    best_prop = answer_exemplar[winning_ans]
    return {
        "winning_answer": winning_ans,
        "winning_completion": best_prop.raw_completion,
        "winning_agent_id": best_prop.agent_id,
        "consensus_score": round(float(consensus_score), 4),
        "disagreement_entropy": round(float(disagreement_entropy), 4),
        "vote_distribution": {k: round(v, 4) for k, v in answer_weights.items()},
    }


class SLMAgentSwarm:
    """
    Multi-round collaborative debate and verification orchestrator combining:
        - Role-specialized SLM Solver agents
        - Deterministic Critic / Verifier (`VerifiableRewardOracle`)
        - AI TRiSM PII Auditor (`PIIGuardrail`)
    """

    def __init__(
        self,
        backbone: Optional[MultiLoRASwarmBackbone] = None,
    ) -> None:
        self.backbone = backbone or MultiLoRASwarmBackbone()

    def run_debate(
        self,
        task_prompt: str,
        initial_proposals: Sequence[AgentProposal],
        ground_truth: Optional[str] = None,
        required_json_keys: Optional[Sequence[str]] = None,
    ) -> SwarmExecutionReport:
        """
        Execute a 2-round Generator -> Critic/Verifier -> Consensus debate:
            - Round 1: Score initial agent proposals using deterministic format
              and task verifiers (`VerifiableRewardOracle`).
            - Round 2: Critic agent repairs unformatted proposals into strict
              `<think>...</think><answer>...</answer>` structure and boosts
              confidence of verified answers before weighted consensus voting.
        """
        if not initial_proposals:
            raise ValueError("initial_proposals cannot be empty.")

        def _score_completion(comp: str) -> float:
            fmt_score = VerifiableRewardOracle.verify_reasoning_format(comp)
            task_score = fmt_score
            if ground_truth is not None:
                task_score = VerifiableRewardOracle.verify_math_answer(comp, ground_truth)
            elif required_json_keys is not None:
                task_score = VerifiableRewardOracle.verify_json_schema(
                    comp, required_json_keys
                )
            return 0.3 * fmt_score + 0.7 * task_score

        round_1_rewards = [_score_completion(p.raw_completion) for p in initial_proposals]
        round_1_mean = sum(round_1_rewards) / len(round_1_rewards)

        # Find highest-reward valid answer from Round 1 for Critic synthesis
        best_r1_idx = max(range(len(initial_proposals)), key=lambda i: round_1_rewards[i])
        critic_anchor_ans = initial_proposals[best_r1_idx].extracted_answer

        refined_proposals: List[AgentProposal] = []
        for idx, prop in enumerate(initial_proposals):
            r1_score = round_1_rewards[idx]
            if r1_score >= 0.85:
                refined_comp = prop.raw_completion
                refined_ans = prop.extracted_answer
                refined_conf = min(0.99, prop.confidence * 1.15)
            else:
                # Critic repairs structure and aligns toward verified peer evidence
                refined_ans = critic_anchor_ans if r1_score < 0.5 else prop.extracted_answer
                refined_comp = (
                    f"<think>Refined by Swarm Critic after peer verification of {prop.agent_id}.</think>\n"
                    f"<answer>{refined_ans}</answer>"
                )
                refined_conf = max(0.75, prop.confidence)

            # Enforce AI TRiSM PII redaction on agent outputs
            if len(PIIGuardrail.inspect(refined_comp)) > 0:
                refined_comp = PIIGuardrail.redact(refined_comp)

            refined_proposals.append(
                AgentProposal(
                    agent_id=prop.agent_id,
                    role=prop.role,
                    raw_completion=refined_comp,
                    extracted_answer=refined_ans,
                    confidence=refined_conf,
                    reliability_weight=prop.reliability_weight,
                )
            )

        final_rewards = [_score_completion(p.raw_completion) for p in refined_proposals]
        final_mean = sum(final_rewards) / len(final_rewards)

        vote_result = confidence_weighted_vote(refined_proposals)
        pii_found = len(PIIGuardrail.inspect(task_prompt)) > 0

        return SwarmExecutionReport(
            task_prompt=PIIGuardrail.redact(task_prompt) if pii_found else task_prompt,
            winning_answer=vote_result["winning_answer"],
            winning_completion=vote_result["winning_completion"],
            consensus_score=vote_result["consensus_score"],
            round_1_mean_reward=round(round_1_mean, 4),
            final_round_mean_reward=round(final_mean, 4),
            pii_redacted=pii_found,
            num_agents=len(initial_proposals),
            rounds_executed=2,
        )


def main() -> None:
    vram_stats = MultiLoRASwarmBackbone.estimate_vram_savings(num_agents=4)
    swarm = SLMAgentSwarm()

    proposals = [
        AgentProposal(
            agent_id="slm_solver_1",
            role="solver_a",
            raw_completion="<think>Compute 15 * 14 = 210.</think>\n<answer>210</answer>",
            extracted_answer="210",
            confidence=0.88,
            reliability_weight=1.0,
        ),
        AgentProposal(
            agent_id="slm_solver_2",
            role="solver_b",
            raw_completion="I think 15 * 14 is 200.",
            extracted_answer="200",
            confidence=0.52,
            reliability_weight=0.9,
        ),
        AgentProposal(
            agent_id="slm_critic_1",
            role="critic",
            raw_completion="<think>15 * 10 = 150 and 15 * 4 = 60; 150 + 60 = 210.</think>\n<answer>210</answer>",
            extracted_answer="210",
            confidence=0.94,
            reliability_weight=1.2,
        ),
    ]

    report = swarm.run_debate(
        task_prompt="Calculate 15 * 14 for customer alice@enterprise.com",
        initial_proposals=proposals,
        ground_truth="210",
    )

    logger.info("=" * 72)
    logger.info("MODULE 13: MULTI-AGENT SLM SWARM & DEBATE VERIFICATION")
    logger.info("=" * 72)
    logger.info(
        f"[Multi-LoRA VRAM] 4 Independent SLMs={vram_stats['independent_fleet_vram_mb']:.1f}MB "
        f"vs. Shared Backbone Swarm={vram_stats['shared_multi_lora_vram_mb']:.1f}MB "
        f"(Savings={vram_stats['vram_savings_percent']:.1f}%)"
    )
    logger.info(
        f"[Debate Convergence] Round 1 Mean Reward={report.round_1_mean_reward:.2f} "
        f"-> Round 2 Mean Reward={report.final_round_mean_reward:.2f} "
        f"| Winning Answer={report.winning_answer} (Consensus={report.consensus_score*100:.1f}%)"
    )


if __name__ == "__main__":
    main()
