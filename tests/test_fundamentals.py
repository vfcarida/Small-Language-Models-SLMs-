import os
import torch
from src.module_00_fundamentals.attention_visualizer import (
    demonstrate_self_attention_math,
    visualize_attention_heatmap,
    explain_mha_vs_gqa,
)


def test_demonstrate_self_attention_math_properties():
    weights, tokens = demonstrate_self_attention_math()
    assert len(tokens) == 4
    assert weights.shape == (4, 4)
    # Row-stochastic property: each row of post-softmax attention weights sums to 1.0
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    assert torch.all(weights >= 0.0)


def test_visualize_attention_heatmap_creates_file(tmp_path):
    weights, tokens = demonstrate_self_attention_math()
    out_file = tmp_path / "attn.png"
    visualize_attention_heatmap(weights, tokens, save_path=str(out_file))
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_explain_mha_vs_gqa_runs_cleanly():
    explain_mha_vs_gqa()
