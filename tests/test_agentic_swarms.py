"""
Unit tests for Module 13: Multi-Agent Collaborative SLM Swarms & Debate Verification.
"""

import pytest
import torch

from src.module_13_agentic_swarms.slm_swarm import (
    AgentProposal,
    MultiLoRASwarmBackbone,
    SLMAgentSwarm,
    confidence_weighted_vote,
)


def test_multi_lora_swarm_backbone_and_vram_savings():
    backbone = MultiLoRASwarmBackbone(d_in=32, d_out=32, lora_rank=4)
    x = torch.randn(2, 6, 32)
    out_planner = backbone.forward_role(x, "planner")
    out_critic = backbone.forward_role(x, "critic")
    assert out_planner.shape == (2, 6, 32)
    assert out_critic.shape == (2, 6, 32)
    assert not torch.allclose(out_planner, out_critic)

    with pytest.raises(KeyError, match="Unknown agent role"):
        backbone.forward_role(x, "nonexistent_role")

    vram_stats = MultiLoRASwarmBackbone.estimate_vram_savings(num_agents=4)
    assert vram_stats["shared_multi_lora_vram_mb"] < vram_stats["independent_fleet_vram_mb"]
    assert vram_stats["vram_savings_percent"] > 65.0


def test_confidence_weighted_vote_consensus():
    proposals = [
        AgentProposal("a1", "solver_a", "<think>ok</think><answer>42</answer>", "42", 0.90, 1.0),
        AgentProposal("a2", "solver_b", "maybe 41", "41", 0.40, 0.8),
        AgentProposal("a3", "critic", "<think>verified</think><answer>42</answer>", "42", 0.95, 1.2),
    ]
    res = confidence_weighted_vote(proposals)
    assert res["winning_answer"] == "42"
    assert res["consensus_score"] > 0.80
    assert 0.0 <= res["disagreement_entropy"] <= 1.0


def test_slm_agent_swarm_debate_and_pii_guardrail():
    swarm = SLMAgentSwarm()
    proposals = [
        AgentProposal(
            agent_id="solver_1",
            role="solver_a",
            raw_completion="<think>Compute 12 * 12 = 144.</think>\n<answer>144</answer>",
            extracted_answer="144",
            confidence=0.92,
            reliability_weight=1.0,
        ),
        AgentProposal(
            agent_id="solver_2",
            role="solver_b",
            raw_completion="Unstructured guess: 140 for user john.doe@corp.com",
            extracted_answer="140",
            confidence=0.45,
            reliability_weight=0.9,
        ),
    ]
    report = swarm.run_debate(
        task_prompt="Compute 12 * 12 for client john.doe@corp.com",
        initial_proposals=proposals,
        ground_truth="144",
    )
    assert report.winning_answer == "144"
    assert report.final_round_mean_reward > report.round_1_mean_reward
    assert report.pii_redacted is True
    assert "john.doe@corp.com" not in report.task_prompt
