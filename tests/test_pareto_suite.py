"""Unit tests for the Cross-Architecture Empirical Pareto Suite (eval/benchmarks/pareto_suite.py)."""

import json
from pathlib import Path

from eval.benchmarks.pareto_suite import (
    BenchmarkObservation,
    compute_pareto_frontier,
    dominates,
    export_pareto_artifacts,
    format_pareto_markdown_table,
    get_reference_pareto_matrix,
    run_live_compression_ablation,
)


def test_dominates_and_pareto_frontier() -> None:
    obs_a = BenchmarkObservation(
        model_family="ModelA",
        compression_profile="AWQ-INT4",
        vram_mb=1100.0,
        ttft_ms=22.0,
        throughput_tps=170.0,
        arc_challenge=55.0,
        gsm8k=60.0,
        mmlu=60.0,
        cost_per_1m_tokens_usd=0.05,
    )
    obs_b = BenchmarkObservation(
        model_family="ModelB",
        compression_profile="FP16",
        vram_mb=3200.0,
        ttft_ms=45.0,
        throughput_tps=90.0,
        arc_challenge=52.0,
        gsm8k=55.0,
        mmlu=54.0,
        cost_per_1m_tokens_usd=0.14,
    )
    assert dominates(obs_a, obs_b) is True
    assert dominates(obs_b, obs_a) is False

    frontier = compute_pareto_frontier([obs_a, obs_b])
    assert len(frontier) == 1
    assert frontier[0].identifier == "ModelA (AWQ-INT4)"


def test_reference_pareto_matrix_within_8gb_budget() -> None:
    matrix = get_reference_pareto_matrix()
    assert len(matrix) >= 10

    edge_frontier = compute_pareto_frontier(matrix, max_vram_mb=8192.0)
    assert len(edge_frontier) >= 2
    for obs in edge_frontier:
        assert obs.vram_mb <= 8192.0
    frontier_ids = {obs.identifier for obs in edge_frontier}
    assert "Qwen2.5-1.5B (AWQ-INT4-SpinQuant)" in frontier_ids
    assert "Llama-3.1-70B-Instruct (Cloud-API)" not in frontier_ids


def test_live_compression_ablation_awq_and_spinquant_beat_rtn() -> None:
    ablation = run_live_compression_ablation(
        in_features=64, out_features=64, seq_len=32, group_size=16, seed=42
    )
    by_profile = {row["profile"]: row for row in ablation}

    assert "FP16/BF16-Dense" in by_profile
    assert "Wanda-2:4-SemiStructured" in by_profile
    assert "RTN-INT4-Naive" in by_profile
    assert "AWQ-INT4-SalientScaled" in by_profile
    assert "AWQ-INT4-SpinQuant-Hadamard" in by_profile

    assert by_profile["Wanda-2:4-SemiStructured"]["sparsity_ratio"] == 0.5
    # Both AWQ salient scaling and SpinQuant Hadamard smoothing must reduce INT4 MSE vs. naive RTN
    assert by_profile["AWQ-INT4-SalientScaled"]["output_mse"] < by_profile["RTN-INT4-Naive"]["output_mse"]
    assert by_profile["AWQ-INT4-SpinQuant-Hadamard"]["output_mse"] < by_profile["RTN-INT4-Naive"]["output_mse"]
    assert by_profile["AWQ-INT4-SpinQuant-Hadamard"]["cosine_similarity"] > 0.99


def test_export_pareto_artifacts(tmp_path: Path) -> None:
    payload = export_pareto_artifacts(tmp_path)
    json_path = tmp_path / "pareto_frontier.json"
    md_path = tmp_path / "pareto_frontier.md"

    assert json_path.exists()
    assert md_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(loaded["edge_8gb_pareto_frontier"]) == len(payload["edge_8gb_pareto_frontier"])
    md_table = format_pareto_markdown_table(get_reference_pareto_matrix(), max_vram_mb=8192.0)
    assert "**Pareto Optimal**" in md_table
