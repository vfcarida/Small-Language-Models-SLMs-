"""
Unit tests for Module 19: Cross-Tokenizer Knowledge Distillation via Optimal Transport (ULD & MinED).
"""

import pytest
import torch

from src.module_19_cross_tokenizer_kd.optimal_transport_kd import (
    CrossTokenizerDistiller,
    align_cross_tokenizer_vocab,
    compute_uld_wasserstein_loss,
    levenshtein_distance,
)


def test_levenshtein_distance_and_vocab_alignment():
    assert levenshtein_distance("quant", "quant") == 0
    assert levenshtein_distance("quant", "quantize") == 3

    student_vocab = ["▁the", "▁model", "▁slm"]
    teacher_vocab = ["Ġthe", "Ġmodel", "Ġmodels", "Ġslm"]
    proj, exact_rate = align_cross_tokenizer_vocab(student_vocab, teacher_vocab)

    assert proj.shape == (4, 3)
    # Each row of proj sums to 1.0 (probability-conserving projection)
    assert torch.allclose(proj.sum(dim=-1), torch.ones(4))
    assert exact_rate == 0.75


def test_compute_uld_wasserstein_loss_mismatched_vocab_sizes():
    torch.manual_seed(42)
    # Identical sorted rank distributions even with padded zeros must yield zero ULD loss
    z = torch.tensor([[10.0, 5.0, -20.0]])
    z_padded = torch.tensor([[10.0, 5.0, -20.0, -100.0, -100.0]])
    loss_same = compute_uld_wasserstein_loss(z, z_padded, temperature=1.0)
    assert loss_same.item() < 1e-4

    # Differentiable across mismatched vocabulary sizes (|V_S|=8 vs |V_T|=16)
    s_logits = torch.randn(2, 5, 8, requires_grad=True)
    t_logits = torch.randn(2, 5, 16)
    loss = compute_uld_wasserstein_loss(s_logits, t_logits, temperature=1.5)
    loss.backward()
    assert s_logits.grad is not None
    assert torch.linalg.norm(s_logits.grad) > 0.0


def test_cross_tokenizer_distiller_hybrid_loss():
    student_vocab = ["▁hello", "▁world", "▁edge", "▁ai"]
    teacher_vocab = ["Ġhello", "Ġworld", "Ġedge", "Ġcloud", "Ġai", "<eos>"]

    distiller = CrossTokenizerDistiller(
        student_tokens=student_vocab,
        teacher_tokens=teacher_vocab,
        uld_weight=0.4,
        temperature=1.5,
    )
    s_logits = torch.randn(2, 4, len(student_vocab), requires_grad=True)
    t_logits = torch.randn(2, 4, len(teacher_vocab))

    loss, metrics = distiller.compute_loss(s_logits, t_logits)
    assert loss.item() > 0.0
    assert metrics["uld_wasserstein_loss"] >= 0.0
    assert metrics["mined_projected_kl_loss"] >= 0.0

    with pytest.raises(ValueError, match="uld_weight"):
        CrossTokenizerDistiller(student_vocab, teacher_vocab, uld_weight=1.5)
