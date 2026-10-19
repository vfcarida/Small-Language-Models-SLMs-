"""Unit tests for Module 10 Lossless Speculative Decoding & Medusa Multi-Token Draft Heads."""

import pytest
import torch
import torch.nn.functional as F

from src.module_10_speculative.speculative_decoding import (
    MedusaDraftHead,
    compute_residual_distribution,
    compute_speculative_speedup,
    speculative_decoding_step,
    verify_marginal_distribution_identity,
)


def test_residual_distribution_and_lossless_marginal_identity() -> None:
    torch.manual_seed(42)
    p = F.softmax(torch.randn(6, 64) * 1.4, dim=-1)
    q = F.softmax(torch.randn(6, 64) * 1.1, dim=-1)

    p_res = compute_residual_distribution(p, q)
    assert p_res.shape == (6, 64)
    assert torch.all(p_res >= 0.0)
    assert torch.allclose(p_res.sum(dim=-1), torch.ones(6), atol=1e-5)

    max_err = verify_marginal_distribution_identity(p, q)
    assert max_err < 1e-6


def test_speculative_decoding_step_all_accepted_and_early_rejection() -> None:
    vocab_size, k = 32, 4
    draft_tokens = torch.tensor([2, 5, 8, 11], dtype=torch.long)
    q_draft = F.softmax(torch.randn(k, vocab_size), dim=-1)
    # 1. Identical target distribution on first K positions -> acceptance prob = 1.0
    p_target = torch.cat([q_draft, F.softmax(torch.randn(1, vocab_size), dim=-1)], dim=0)

    res_all = speculative_decoding_step(
        draft_tokens,
        q_draft,
        p_target,
        uniform_rands=torch.tensor([0.5, 0.5, 0.5, 0.5]),
        seed=42,
    )
    assert res_all.all_accepted_bonus is True
    assert res_all.num_accepted_draft_tokens == k
    assert len(res_all.emitted_tokens) == k + 1
    assert res_all.emitted_tokens[:k] == [2, 5, 8, 11]

    # 2. Force rejection at index 1 by setting p_target[1, draft_tokens[1]] = 0.0
    p_reject = p_target.clone()
    p_reject[1, 5] = 0.0
    p_reject[1] = p_reject[1] / p_reject[1].sum()

    res_rej = speculative_decoding_step(
        draft_tokens,
        q_draft,
        p_reject,
        uniform_rands=torch.tensor([0.2, 0.9, 0.2, 0.2]),
        seed=42,
    )
    assert res_rej.all_accepted_bonus is False
    assert res_rej.num_accepted_draft_tokens == 1
    assert len(res_rej.emitted_tokens) == 2
    assert res_rej.emitted_tokens[0] == 2
    assert res_rej.emitted_tokens[1] != 5


def test_compute_speculative_speedup_and_medusa_draft_head() -> None:
    stats = compute_speculative_speedup(
        acceptance_rate=0.80, lookahead_k=4, draft_to_target_cost_ratio=0.05
    )
    assert stats["expected_tokens_per_step"] == pytest.approx(3.3616, abs=1e-3)
    assert stats["wallclock_speedup_x"] > 2.5

    medusa = MedusaDraftHead(hidden_size=32, vocab_size=100, num_heads=3)
    h = torch.randn(2, 8, 32)
    logits = medusa(h)
    assert logits.shape == (3, 2, 8, 100)
