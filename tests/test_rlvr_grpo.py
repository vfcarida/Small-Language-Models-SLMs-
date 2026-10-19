"""Unit tests for Module 11 Reinforcement Learning with Verifiable Rewards (RLVR / GRPO)."""

import pytest
import torch

from src.module_11_rlvr_grpo.grpo_trainer import (
    VerifiableRewardOracle,
    compute_group_advantages,
    compute_grpo_loss,
    compute_schulman_kl,
)


def test_compute_group_advantages_zero_mean_unit_std() -> None:
    rewards = torch.tensor([1.0, 0.25, 0.0, 0.75], dtype=torch.float32)
    adv = compute_group_advantages(rewards)
    assert adv.shape == (4,)
    assert float(adv.mean().item()) == pytest.approx(0.0, abs=1e-5)
    assert float(adv.std(unbiased=False).item()) == pytest.approx(1.0, abs=1e-4)

    with pytest.raises(ValueError, match="group size G >= 2"):
        compute_group_advantages(torch.tensor([1.0]))


def test_compute_schulman_kl_non_negative_and_identity() -> None:
    lp = torch.tensor([[-1.2, -0.5, -2.1], [-0.8, -1.5, -0.3]])
    kl_zero = compute_schulman_kl(lp, lp)
    assert torch.allclose(kl_zero, torch.zeros_like(lp), atol=1e-6)

    ref_lp = lp - 0.4
    kl_pos = compute_schulman_kl(lp, ref_lp)
    assert torch.all(kl_pos > 0.0)


def test_compute_grpo_loss_gradients_and_telemetry() -> None:
    torch.manual_seed(42)
    g, t = 4, 10
    old_lp = -torch.rand(g, t)
    ref_lp = old_lp - 0.1
    policy_lp = (old_lp + torch.randn(g, t) * 0.05).requires_grad_(True)
    rewards = torch.tensor([1.0, 0.25, 0.0, 0.5])

    loss, metrics = compute_grpo_loss(policy_lp, old_lp, ref_lp, rewards)
    loss.backward()

    assert policy_lp.grad is not None
    assert torch.isfinite(loss)
    assert metrics["mean_kl"] >= 0.0
    assert 0.0 <= metrics["clip_fraction"] <= 1.0


def test_verifiable_reward_oracle_format_math_and_json() -> None:
    valid_comp = "<think>First multiply 6 * 7 = 42.</think>\n<answer>42</answer>"
    bad_comp = "The answer is 41."

    assert VerifiableRewardOracle.verify_reasoning_format(valid_comp) == 1.0
    assert VerifiableRewardOracle.verify_reasoning_format(bad_comp) == 0.0
    assert VerifiableRewardOracle.verify_math_answer(valid_comp, "42") == 1.0
    assert VerifiableRewardOracle.verify_math_answer(bad_comp, "42") == 0.0

    json_comp = '<think>Formatting JSON.</think><answer>{"intent": "billing", "confidence": 0.95}</answer>'
    assert VerifiableRewardOracle.verify_json_schema(json_comp, ["intent", "confidence"]) == 1.0
    assert VerifiableRewardOracle.verify_json_schema(json_comp, ["missing_key"]) == 0.0

    scored = VerifiableRewardOracle.score_math_group([valid_comp, bad_comp], ground_truth="42")
    assert scored["rewards"] == [1.0, 0.0]
    assert scored["advantages"][0] > scored["advantages"][1]
