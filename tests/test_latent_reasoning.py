"""
Unit tests for Module 16: Latent-Space Continuous Reasoning (Coconut & Recurrent Depth).
"""

import pytest
import torch

from src.module_16_latent_reasoning.coconut_reasoning import (
    AdaptivePonderingGate,
    LatentThoughtReasoner,
    compare_discrete_cot_vs_latent_coconut,
)


def test_adaptive_pondering_gate_kl_properties():
    logits = torch.randn(4, 128)
    kl_zero = AdaptivePonderingGate.compute_step_kl_divergence(logits, logits)
    assert abs(kl_zero) < 1e-6

    logits_shifted = logits + torch.randn_like(logits) * 2.0
    kl_pos = AdaptivePonderingGate.compute_step_kl_divergence(logits, logits_shifted)
    assert kl_pos > 0.01


def test_latent_thought_reasoner_early_exit_and_full_recurrence():
    torch.manual_seed(42)
    # High KL threshold forces early exit after step 2
    reasoner_early = LatentThoughtReasoner(
        d_model=32, vocab_size=64, max_latent_steps=8, kl_threshold=10.0
    )
    x = torch.randn(2, 6, 32)
    logits, thought, trace = reasoner_early(x, enable_early_exit=True)
    assert logits.shape == (2, 64)
    assert thought.shape == (2, 32)
    assert trace.halted_early is True
    assert trace.steps_executed == 2

    # Disabling early exit runs all max_latent_steps
    _, _, trace_full = reasoner_early(x, enable_early_exit=False)
    assert trace_full.halted_early is False
    assert trace_full.steps_executed == 8
    assert len(trace_full.kl_trajectory) == 7


def test_compare_discrete_cot_vs_latent_coconut_profiler():
    stats = compare_discrete_cot_vs_latent_coconut(
        discrete_cot_tokens=256, latent_thought_steps=16, answer_tokens=32
    )
    assert stats["latent_coconut_kv_mb"] < stats["discrete_cot_kv_mb"]
    assert stats["kv_cache_reduction_x"] >= 5.0
    assert stats["latency_speedup_x"] >= 5.0

    with pytest.raises(ValueError, match="Token and step counts"):
        compare_discrete_cot_vs_latent_coconut(discrete_cot_tokens=0)
