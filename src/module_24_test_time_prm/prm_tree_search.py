"""
Module 24: Test-Time Compute Scaling via Process Reward Models (PRM) & Tree Search
==================================================================================

Implements step-level verification and optimal test-time compute scaling for
Small Language Models (<4B parameters):
1. ``ProcessRewardScorer``: Neural and rule-assisted step-level verifier scoring
   intermediate reasoning steps $s_t$ with $r_t = P(\\text{step } t \\text{ valid} \\mid q, s_{1:t})$
   and aggregating trajectories via ``min`` (bottleneck), ``prod`` (joint probability),
   or ``mean_log`` (geometric mean).
2. ``best_of_n_weighted_voting``: Best-of-$N$ trajectory selection and PRM-weighted
   majority voting (PRM-Weighted Self-Consistency) (Lightman et al., ICLR 2024;
   Snell et al., UC Berkeley 2024).
3. ``step_level_beam_search``: Step-by-step beam search with beam width $B$ and
   branching factor $M$ guided by intermediate PRM scores.

References:
    - Lightman et al. (ICLR 2024 / OpenAI). "Let's Verify Step by Step." arXiv:2305.20050.
    - Snell et al. (UC Berkeley, 2024). "Scaling LLM Test-Time Compute Optimally
      can be More Effective than Scaling Model Parameters." arXiv:2408.03314.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn

AggregationMode = Literal["min", "prod", "mean_log"]


class ProcessRewardScorer(nn.Module):
    """Step-level Process Reward Model (PRM) head and trajectory aggregator.

    Maps hidden representations $\\mathbf{h}_t \\in \\mathbb{R}^d$ at the end of
    each reasoning step $t \\in \\{1, \\dots, T\\}$ to a scalar validity probability
    $r_t = \\sigma(\\mathbf{w}_{\\text{prm}}^\\top \\text{GELU}(\\mathbf{W}_1 \\mathbf{h}_t) + b) \\in (0, 1)$.

    Args:
        hidden_dim: Hidden dimension $d$ of the SLM backbone.
        eps: Numerical floor for probabilities before logarithmic operations.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}.")
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.step_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def score_step_embeddings(self, step_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute step-level correctness probabilities $r_t \\in (0, 1)$.

        Args:
            step_embeddings: Tensor of shape ``(..., num_steps, hidden_dim)``.

        Returns:
            Step probability tensor of shape ``(..., num_steps)`` in $(0, 1)$.
        """
        if step_embeddings.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"Expected last dimension {self.hidden_dim}, got {step_embeddings.shape[-1]}."
            )
        logits = self.step_head(step_embeddings).squeeze(-1)
        return torch.sigmoid(logits)

    def aggregate_step_scores(
        self,
        step_scores: torch.Tensor,
        mode: AggregationMode = "min",
    ) -> torch.Tensor:
        """Aggregate step probabilities $r_{1:T}$ into a trajectory-level PRM score.

        - ``"min"``: $\\text{PRM}(\\tau) = \\min_{1 \\le t \\le T} r_t$ (Lightman et al., 2024
          bottleneck verifier — a chain is only as valid as its weakest step).
        - ``"prod"``: $\\text{PRM}(\\tau) = \\prod_{t=1}^T r_t$ (autoregressive joint validity).
        - ``"mean_log"``: $\\text{PRM}(\\tau) = \\exp\\!\\left(\\frac{1}{T}\\sum_{t=1}^T \\log r_t\\right)$
          (length-normalized geometric mean).

        Args:
            step_scores: Step probabilities in $(0, 1]$ of shape ``(..., num_steps)``.
            mode: One of ``"min"``, ``"prod"``, or ``"mean_log"``.

        Returns:
            Trajectory score tensor of shape ``(...)`` in $(0, 1]$.
        """
        if step_scores.ndim < 1 or step_scores.shape[-1] == 0:
            raise ValueError("step_scores must have at least 1 step along the last axis.")
        clamped = step_scores.clamp(min=self.eps, max=1.0)
        if mode == "min":
            return clamped.min(dim=-1).values
        if mode == "prod":
            return torch.exp(torch.log(clamped).sum(dim=-1))
        if mode == "mean_log":
            return torch.exp(torch.log(clamped).mean(dim=-1))
        raise ValueError(
            f"Unsupported aggregation mode '{mode}'. Expected 'min', 'prod', or 'mean_log'."
        )


def best_of_n_weighted_voting(
    candidate_answers: Sequence[str],
    trajectory_step_scores: Sequence[Sequence[float]],
    aggregation_mode: AggregationMode = "min",
) -> Dict[str, object]:
    """Select the optimal response via Best-of-$N$ and PRM-Weighted Voting.

    Given $N$ sampled reasoning trajectories $\\{\\tau_1, \\dots, \\tau_N\\}$ with extracted
    final answers $a_i = \\text{ans}(\\tau_i)$ and step-level PRM probabilities
    $\\mathbf{r}^{(i)} = (r_1^{(i)}, \\dots, r_{T_i}^{(i)})$:

    - **Best-of-$N$ (BoN):** $a_{\\text{BoN}} = a_{i^*}$ where $i^* = \\arg\\max_i \\text{PRM}(\\tau_i)$.
    - **PRM-Weighted Voting (Snell et al., 2024):**
      $$a_{\\text{vote}} = \\arg\\max_{a \\in \\mathcal{A}} \\sum_{i : a_i = a} \\text{PRM}(\\tau_i)$$

    Args:
        candidate_answers: Sequence of $N$ final answer strings extracted from each trajectory.
        trajectory_step_scores: Sequence of $N$ lists of step-level PRM scores in $(0, 1]$.
        aggregation_mode: Trajectory aggregation rule (``"min"``, ``"prod"``, or ``"mean_log"``).

    Returns:
        Dictionary containing:
            - ``"best_of_n_answer"``: Answer from the single highest-scoring trajectory.
            - ``"best_of_n_index"``: Index $i^*$ of the highest-scoring trajectory.
            - ``"prm_weighted_vote_answer"``: Answer maximizing cumulative PRM weight.
            - ``"unweighted_majority_answer"``: Standard self-consistency majority vote.
            - ``"trajectory_scores"``: List of aggregated scores $\\text{PRM}(\\tau_i)$.
            - ``"answer_weight_map"``: Mapping from unique answer string to total PRM weight.
    """
    if len(candidate_answers) == 0:
        raise ValueError("candidate_answers must be non-empty.")
    if len(candidate_answers) != len(trajectory_step_scores):
        raise ValueError(
            "candidate_answers and trajectory_step_scores must have the same length."
        )

    eps = 1e-6
    traj_scores: List[float] = []
    for steps in trajectory_step_scores:
        if len(steps) == 0:
            raise ValueError("Each trajectory must have at least one step score.")
        clamped = [max(eps, min(1.0, float(s))) for s in steps]
        if aggregation_mode == "min":
            score = min(clamped)
        elif aggregation_mode == "prod":
            score = math.exp(sum(math.log(s) for s in clamped))
        elif aggregation_mode == "mean_log":
            score = math.exp(sum(math.log(s) for s in clamped) / len(clamped))
        else:
            raise ValueError(f"Unsupported aggregation_mode '{aggregation_mode}'.")
        traj_scores.append(score)

    best_idx = max(range(len(traj_scores)), key=lambda i: traj_scores[i])
    best_of_n_answer = candidate_answers[best_idx]

    weight_map: Dict[str, float] = {}
    count_map: Dict[str, int] = {}
    for ans, score in zip(candidate_answers, traj_scores):
        weight_map[ans] = weight_map.get(ans, 0.0) + score
        count_map[ans] = count_map.get(ans, 0) + 1

    prm_weighted_answer = max(weight_map.items(), key=lambda kv: kv[1])[0]
    unweighted_answer = max(count_map.items(), key=lambda kv: kv[1])[0]

    return {
        "best_of_n_answer": best_of_n_answer,
        "best_of_n_index": best_idx,
        "prm_weighted_vote_answer": prm_weighted_answer,
        "unweighted_majority_answer": unweighted_answer,
        "trajectory_scores": traj_scores,
        "answer_weight_map": weight_map,
    }


def step_level_beam_search(
    initial_prompt: str,
    step_expander: Callable[[List[str], int], List[Tuple[str, float]]],
    beam_width: int = 2,
    branching_factor: int = 2,
    max_steps: int = 3,
    aggregation_mode: AggregationMode = "min",
) -> Dict[str, object]:
    """Execute PRM-guided Step-Level Beam Search over multi-step reasoning paths.

    At each reasoning depth $t \\in \\{1, \\dots, T_{\\max}\\}$:
    1. Each active beam state $s_{1:t-1}$ expands into $M$ candidate next steps
       via ``step_expander(steps_so_far, branching_factor)`` returning pairs
       ``(step_text, prm_step_score)``.
    2. Candidate partial trajectories are scored by aggregating their step scores
       $r_{1:t}$ under ``aggregation_mode``.
    3. The top-$B$ (``beam_width``) partial trajectories are retained for depth $t+1$.

    Args:
        initial_prompt: Problem statement anchoring the root node.
        step_expander: Callable taking ``(trajectory_steps, branching_factor)`` and
            returning a list of ``(next_step_string, step_prm_probability)`` tuples.
        beam_width: Number of active beams $B$ retained per depth.
        branching_factor: Number of candidate continuations $M$ sampled per beam.
        max_steps: Number of reasoning steps $T_{\\max}$ to expand.
        aggregation_mode: Step aggregation rule (``"min"``, ``"prod"``, or ``"mean_log"``).

    Returns:
        Dictionary containing:
            - ``"best_trajectory"``: List of reasoning steps in the highest-scoring beam.
            - ``"best_score"``: Aggregated PRM score of the winning beam.
            - ``"active_beams"``: All $B$ surviving trajectories sorted by score descending.
            - ``"total_step_evaluations"``: Total PRM step evaluations executed.
    """
    if beam_width <= 0 or branching_factor <= 0 or max_steps <= 0:
        raise ValueError("beam_width, branching_factor, and max_steps must be positive.")

    eps = 1e-6

    def _agg(scores: List[float]) -> float:
        clamped = [max(eps, min(1.0, float(s))) for s in scores]
        if aggregation_mode == "min":
            return min(clamped)
        if aggregation_mode == "prod":
            return math.exp(sum(math.log(s) for s in clamped))
        if aggregation_mode == "mean_log":
            return math.exp(sum(math.log(s) for s in clamped) / len(clamped))
        raise ValueError(f"Unsupported aggregation_mode '{aggregation_mode}'.")

    # Each beam entry is (steps_list, step_scores_list, aggregated_score)
    beams: List[Tuple[List[str], List[float], float]] = [([initial_prompt], [], 1.0)]
    total_evals = 0

    for _ in range(max_steps):
        candidates: List[Tuple[List[str], List[float], float]] = []
        for steps_so_far, scores_so_far, _ in beams:
            expansions = step_expander(steps_so_far, branching_factor)
            for step_text, step_score in expansions[:branching_factor]:
                total_evals += 1
                new_steps = steps_so_far + [step_text]
                new_scores = scores_so_far + [float(step_score)]
                agg_score = _agg(new_scores)
                candidates.append((new_steps, new_scores, agg_score))

        candidates.sort(key=lambda item: item[2], reverse=True)
        beams = candidates[:beam_width]

    best_steps, best_step_scores, best_score = beams[0]
    return {
        "best_trajectory": best_steps,
        "best_step_scores": best_step_scores,
        "best_score": best_score,
        "active_beams": [
            {"steps": s, "step_scores": sc, "score": agg} for s, sc, agg in beams
        ],
        "total_step_evaluations": total_evals,
    }


def main() -> None:
    """Run a deterministic demonstration of PRM-weighted voting and step-level beam search."""
    res = best_of_n_weighted_voting(
        candidate_answers=["41", "41", "41", "42", "42"],
        trajectory_step_scores=[
            [0.9, 0.1, 0.8],
            [0.85, 0.12, 0.7],
            [0.8, 0.08, 0.75],
            [0.95, 0.92, 0.96],
            [0.91, 0.94, 0.90],
        ],
    )
    print(
        f"[Module 24] Unweighted Majority: {res['unweighted_majority_answer']} | "
        f"PRM-Weighted Vote: {res['prm_weighted_vote_answer']}"
    )


if __name__ == "__main__":
    main()

