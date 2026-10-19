"""Module 13: Multi-Agent Collaborative SLM Swarms, Multi-LoRA Hot-Swapping, and Debate Verification."""

from src.module_13_agentic_swarms.slm_swarm import (
    AgentProposal,
    MultiLoRASwarmBackbone,
    SLMAgentSwarm,
    SwarmExecutionReport,
    confidence_weighted_vote,
)

__all__ = [
    "AgentProposal",
    "MultiLoRASwarmBackbone",
    "SLMAgentSwarm",
    "SwarmExecutionReport",
    "confidence_weighted_vote",
]
