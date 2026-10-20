"""Unit tests for Module 32: Statistical Watermarking (KGW Green-List Logit Bias)."""

import torch

from src.module_32_statistical_watermarking import (
    apply_kgw_watermark_logits,
    detect_kgw_watermark_zscore,
    partition_green_red_vocab,
)


def test_partition_green_red_vocab_is_deterministic_and_exact_size() -> None:
    mask1 = partition_green_red_vocab(prev_token_id=42, vocab_size=200, gamma=0.25)
    mask2 = partition_green_red_vocab(prev_token_id=42, vocab_size=200, gamma=0.25)
    assert torch.equal(mask1, mask2)
    assert int(mask1.sum().item()) == 50


def test_kgw_watermark_distinguishes_watermarked_from_unwatermarked_sequences() -> None:
    torch.manual_seed(99)
    vocab_size = 100
    gamma = 0.25

    # Generate 80 tokens with KGW watermark bias delta=4.0
    wm_tokens = [1]
    for _ in range(80):
        logits = torch.randn(vocab_size) * 0.5
        biased = apply_kgw_watermark_logits(logits, [wm_tokens[-1]], gamma=gamma, delta=4.0)
        next_tok = int(torch.argmax(biased).item())
        wm_tokens.append(next_tok)

    # Generate 80 unwatermarked random tokens
    unwm_tokens = torch.randint(0, vocab_size, (81,)).tolist()

    det_wm = detect_kgw_watermark_zscore(wm_tokens, vocab_size=vocab_size, gamma=gamma)
    det_unwm = detect_kgw_watermark_zscore(unwm_tokens, vocab_size=vocab_size, gamma=gamma)

    assert det_wm["is_watermarked"] is True
    assert det_wm["z_score"] > 6.0
    assert det_unwm["is_watermarked"] is False
    assert det_wm["z_score"] > det_unwm["z_score"] + 4.0


def test_kgw_watermark_batch_logits_and_input_validation() -> None:
    batch_logits = torch.zeros(3, 64)
    biased = apply_kgw_watermark_logits(batch_logits, prev_token_ids=[5, 10, 15], gamma=0.25, delta=2.5)
    assert biased.shape == (3, 64)
    # Each row has exactly 16 green tokens biased to +2.5
    assert torch.all((biased == 2.5).sum(dim=-1) == 16)

