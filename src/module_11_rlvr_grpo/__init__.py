"""Module 11: Reinforcement Learning with Verifiable Rewards (RLVR) and Group Relative Policy Optimization (GRPO)."""

from src.module_11_rlvr_grpo.grpo_trainer import (
    VerifiableRewardOracle,
    compute_group_advantages,
    compute_grpo_loss,
    compute_schulman_kl,
)

__all__ = [
    "VerifiableRewardOracle",
    "compute_group_advantages",
    "compute_grpo_loss",
    "compute_schulman_kl",
]
