"""Unit tests for Module 24: Test-Time Compute Scaling via PRM & Step-Level Beam Search."""

import torch

from src.module_24_test_time_prm import (
    ProcessRewardScorer,
    best_of_n_weighted_voting,
    step_level_beam_search,
)


def test_process_reward_scorer_shapes_and_aggregations() -> None:
    torch.manual_seed(42)
    scorer = ProcessRewardScorer(hidden_dim=16)
    embeddings = torch.randn(3, 4, 16)
    step_probs = scorer.score_step_embeddings(embeddings)

    assert step_probs.shape == (3, 4)
    assert torch.all((step_probs > 0.0) & (step_probs < 1.0))

    s_min = scorer.aggregate_step_scores(step_probs, mode="min")
    s_prod = scorer.aggregate_step_scores(step_probs, mode="prod")
    s_mean = scorer.aggregate_step_scores(step_probs, mode="mean_log")

    assert s_min.shape == (3,)
    assert torch.all(s_prod <= s_min + 1e-6)
    assert torch.all(s_mean >= s_min - 1e-6)


def test_best_of_n_weighted_voting_overrides_low_quality_majority() -> None:
    # Three trajectories answer "41" with a flawed intermediate step (0.1),
    # while two trajectories answer "42" with consistently verified steps (0.95).
    answers = ["41", "41", "41", "42", "42"]
    step_scores = [
        [0.9, 0.1, 0.8],
        [0.85, 0.12, 0.7],
        [0.8, 0.08, 0.75],
        [0.95, 0.92, 0.96],
        [0.91, 0.94, 0.90],
    ]
    res = best_of_n_weighted_voting(answers, step_scores, aggregation_mode="min")

    assert res["unweighted_majority_answer"] == "41"
    assert res["best_of_n_answer"] == "42"
    assert res["prm_weighted_vote_answer"] == "42"
    assert res["answer_weight_map"]["42"] > res["answer_weight_map"]["41"]


def test_step_level_beam_search_prunes_flawed_branches() -> None:
    def deterministic_expander(steps_so_far, branching_factor):
        depth = len(steps_so_far)
        if "bad_step" in steps_so_far[-1]:
            return [(f"step_{depth}_cont_a", 0.2), (f"step_{depth}_cont_b", 0.15)]
        return [
            (f"good_step_{depth}", 0.92),
            (f"bad_step_{depth}", 0.15),
        ][:branching_factor]

    search_out = step_level_beam_search(
        initial_prompt="Solve 2x + 4 = 10",
        step_expander=deterministic_expander,
        beam_width=2,
        branching_factor=2,
        max_steps=3,
        aggregation_mode="min",
    )

    assert len(search_out["best_trajectory"]) == 4  # prompt + 3 steps
    assert all("good_step" in s for s in search_out["best_trajectory"][1:])
    assert abs(search_out["best_score"] - 0.92) < 1e-6
    assert search_out["total_step_evaluations"] > 0
