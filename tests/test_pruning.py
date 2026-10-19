import pytest
import torch
import torch.nn as nn
from src.module_04_pruning.magnitude_pruning import apply_magnitude_pruning
from src.module_04_pruning.wanda_pruning import (
    compute_wanda_scores,
    execute_wanda_pruning,
    rank_sequences_by_zipf_diversity,
)
from src.utils.metrics import calculate_sparsity


class ToySLMBlock(nn.Module):
    def __init__(self, d_in: int = 16, d_out: int = 8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def test_magnitude_pruning_unstructured():
    torch.manual_seed(42)
    model = ToySLMBlock(d_in=16, d_out=8)
    apply_magnitude_pruning(model, sparsity_ratio=0.5, sparsity_type="unstructured")
    stats = calculate_sparsity(model)
    assert abs(stats["sparsity_percent"] - 50.0) <= 2.0


def test_magnitude_pruning_2_4_semi_structured():
    torch.manual_seed(42)
    model = ToySLMBlock(d_in=16, d_out=8)
    apply_magnitude_pruning(model, sparsity_ratio=0.5, sparsity_type="2:4")
    W = model.proj.weight.data
    grouped = (W == 0).view(-1, 4).sum(dim=1)
    # Every group of 4 consecutive weights must have at least 2 zeros
    assert torch.all(grouped == 2)
    assert calculate_sparsity(model)["sparsity_percent"] == 50.0


def test_compute_wanda_scores_protects_outlier_activations():
    # Weight 0 has tiny magnitude (0.1) but huge activation norm (100.0 -> score 10.0)
    # Weight 1 has larger magnitude (1.0) but small activation norm (1.0 -> score 1.0)
    W = torch.tensor([[0.1, 1.0, 0.5, 0.2]])
    X_norm = torch.tensor([100.0, 1.0, 1.0, 1.0])
    scores = compute_wanda_scores(W, X_norm)
    assert scores[0, 0] == pytest.approx(10.0)
    assert scores[0, 1] == pytest.approx(1.0)
    assert scores[0, 0] > scores[0, 1]


def test_execute_wanda_pruning_per_output_row():
    torch.manual_seed(42)
    model = ToySLMBlock(d_in=16, d_out=8)
    calib = [torch.randn(4, 16) for _ in range(3)]
    execute_wanda_pruning(model, calib, sparsity_ratio=0.5, sparsity_type="unstructured")
    W = model.proj.weight.data
    # Verify per-output row guarantee: EVERY row has exactly 50% (8 of 16) zeros
    zeros_per_row = (W == 0).sum(dim=1)
    assert torch.all(zeros_per_row == 8)


def test_execute_wanda_pruning_2_4_semi_structured():
    torch.manual_seed(42)
    model = ToySLMBlock(d_in=16, d_out=8)
    calib = [torch.randn(4, 16) for _ in range(2)]
    execute_wanda_pruning(model, calib, sparsity_ratio=0.5, sparsity_type="2:4")
    W = model.proj.weight.data
    zeros_per_quad = (W == 0).view(-1, 4).sum(dim=1)
    assert torch.all(zeros_per_quad == 2)


def test_rank_sequences_by_zipf_diversity():
    repetitive_seq = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1]])
    diverse_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    ranked = rank_sequences_by_zipf_diversity([repetitive_seq, diverse_seq])
    assert torch.equal(ranked[0], diverse_seq)
    assert torch.equal(ranked[1], repetitive_seq)
