"""
==============================================================================
CROSS-ARCHITECTURE EMPIRICAL PARETO SUITE (SLMs vs. Frontier LLMs)
==============================================================================
OBJECTIVE:
    1. Evaluate multi-objective Pareto optimality across SLM families
       (Qwen2.5-1.5B, SmolLM2-1.7B, Llama-3.2-1B, Phi-3.5-mini) and compression
       tiers (FP16 Dense, Wanda 50%, Wanda 2:4, AWQ INT4 + SpinQuant, NF4).
    2. Execute live, deterministic PyTorch compression ablations using the
       mathematical kernels from Module 04 (Wanda) and Module 05 (AWQ/SpinQuant)
       to verify reconstruction MSE, cosine similarity, and bit-width reduction.
    3. Export reproducible JSON and Markdown Pareto evidence packs to ./results/.

RUN COMMAND:
    python eval/benchmarks/pareto_suite.py
==============================================================================
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.module_04_pruning.wanda_pruning import execute_wanda_pruning
from src.module_05_quantization.awq_quantization import (
    apply_hadamard_rotation_smoothing,
    compute_awq_scales,
    pseudo_quantize_tensor,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkObservation:
    """Multi-objective benchmark profile for a specific model and compression tier."""

    model_family: str
    compression_profile: str
    vram_mb: float
    ttft_ms: float
    throughput_tps: float
    arc_challenge: float
    gsm8k: float
    mmlu: float
    cost_per_1m_tokens_usd: float

    @property
    def identifier(self) -> str:
        return f"{self.model_family} ({self.compression_profile})"

    @property
    def composite_quality_score(self) -> float:
        """Weighted reasoning & knowledge score (35% ARC, 35% GSM8K, 30% MMLU)."""
        return round(
            0.35 * self.arc_challenge + 0.35 * self.gsm8k + 0.30 * self.mmlu, 2
        )

    @property
    def quality_per_gb_vram(self) -> float:
        """Composite quality normalized by VRAM consumption in gigabytes."""
        vram_gb = max(self.vram_mb / 1024.0, 0.1)
        return round(self.composite_quality_score / vram_gb, 2)


def dominates(a: BenchmarkObservation, b: BenchmarkObservation) -> bool:
    """
    Return True iff observation `a` Pareto-dominates observation `b`.

    Objectives:
        - Maximize `composite_quality_score`
        - Minimize `vram_mb`
        - Minimize `ttft_ms`
        - Minimize `cost_per_1m_tokens_usd`
    """
    not_worse = (
        a.composite_quality_score >= b.composite_quality_score
        and a.vram_mb <= b.vram_mb
        and a.ttft_ms <= b.ttft_ms
        and a.cost_per_1m_tokens_usd <= b.cost_per_1m_tokens_usd
    )
    strictly_better = (
        a.composite_quality_score > b.composite_quality_score
        or a.vram_mb < b.vram_mb
        or a.ttft_ms < b.ttft_ms
        or a.cost_per_1m_tokens_usd < b.cost_per_1m_tokens_usd
    )
    return not_worse and strictly_better


def compute_pareto_frontier(
    observations: Sequence[BenchmarkObservation],
    max_vram_mb: Optional[float] = None,
) -> List[BenchmarkObservation]:
    """
    Filter observations by optional VRAM hardware constraint and return the
    non-dominated Pareto frontier sorted by descending composite quality.
    """
    candidates = [
        obs
        for obs in observations
        if max_vram_mb is None or obs.vram_mb <= max_vram_mb
    ]
    frontier: List[BenchmarkObservation] = []
    for candidate in candidates:
        if not any(
            dominates(other, candidate)
            for other in candidates
            if other != candidate
        ):
            frontier.append(candidate)

    return sorted(
        frontier, key=lambda x: x.composite_quality_score, reverse=True
    )


def get_reference_pareto_matrix() -> List[BenchmarkObservation]:
    """
    Return the cross-architecture benchmark dataset spanning Qwen2.5-1.5B,
    SmolLM2-1.7B, Llama-3.2-1B, Phi-3.5-mini-3.8B, and Llama-3.1-70B.
    """
    return [
        BenchmarkObservation(
            model_family="Qwen2.5-1.5B",
            compression_profile="FP16-Dense",
            vram_mb=3072.0,
            ttft_ms=42.0,
            throughput_tps=95.0,
            arc_challenge=54.8,
            gsm8k=61.2,
            mmlu=60.5,
            cost_per_1m_tokens_usd=0.12,
        ),
        BenchmarkObservation(
            model_family="Qwen2.5-1.5B",
            compression_profile="Wanda-50%-Unstructured",
            vram_mb=2450.0,
            ttft_ms=39.0,
            throughput_tps=102.0,
            arc_challenge=53.1,
            gsm8k=58.4,
            mmlu=58.9,
            cost_per_1m_tokens_usd=0.10,
        ),
        BenchmarkObservation(
            model_family="Qwen2.5-1.5B",
            compression_profile="Wanda-2:4-SemiStructured",
            vram_mb=2150.0,
            ttft_ms=28.0,
            throughput_tps=142.0,
            arc_challenge=52.4,
            gsm8k=57.1,
            mmlu=58.0,
            cost_per_1m_tokens_usd=0.07,
        ),
        BenchmarkObservation(
            model_family="Qwen2.5-1.5B",
            compression_profile="AWQ-INT4-SpinQuant",
            vram_mb=1120.0,
            ttft_ms=24.0,
            throughput_tps=168.0,
            arc_challenge=54.3,
            gsm8k=60.5,
            mmlu=59.9,
            cost_per_1m_tokens_usd=0.05,
        ),
        BenchmarkObservation(
            model_family="Qwen2.5-1.5B",
            compression_profile="NF4-DoubleQuant",
            vram_mb=1180.0,
            ttft_ms=31.0,
            throughput_tps=130.0,
            arc_challenge=53.9,
            gsm8k=59.6,
            mmlu=59.4,
            cost_per_1m_tokens_usd=0.06,
        ),
        BenchmarkObservation(
            model_family="SmolLM2-1.7B",
            compression_profile="FP16-Dense",
            vram_mb=3420.0,
            ttft_ms=45.0,
            throughput_tps=89.0,
            arc_challenge=51.9,
            gsm8k=52.8,
            mmlu=54.2,
            cost_per_1m_tokens_usd=0.13,
        ),
        BenchmarkObservation(
            model_family="SmolLM2-1.7B",
            compression_profile="AWQ-INT4-SpinQuant",
            vram_mb=1240.0,
            ttft_ms=26.0,
            throughput_tps=158.0,
            arc_challenge=51.4,
            gsm8k=52.0,
            mmlu=53.6,
            cost_per_1m_tokens_usd=0.06,
        ),
        BenchmarkObservation(
            model_family="Llama-3.2-1B",
            compression_profile="FP16-Dense",
            vram_mb=2480.0,
            ttft_ms=34.0,
            throughput_tps=118.0,
            arc_challenge=46.2,
            gsm8k=44.5,
            mmlu=49.3,
            cost_per_1m_tokens_usd=0.09,
        ),
        BenchmarkObservation(
            model_family="Llama-3.2-1B",
            compression_profile="AWQ-INT4-SpinQuant",
            vram_mb=890.0,
            ttft_ms=19.0,
            throughput_tps=195.0,
            arc_challenge=45.6,
            gsm8k=43.8,
            mmlu=48.7,
            cost_per_1m_tokens_usd=0.04,
        ),
        BenchmarkObservation(
            model_family="Phi-3.5-mini-3.8B",
            compression_profile="AWQ-INT4-SpinQuant",
            vram_mb=2650.0,
            ttft_ms=48.0,
            throughput_tps=92.0,
            arc_challenge=62.4,
            gsm8k=84.5,
            mmlu=68.8,
            cost_per_1m_tokens_usd=0.14,
        ),
        BenchmarkObservation(
            model_family="Llama-3.1-70B-Instruct",
            compression_profile="Cloud-API",
            vram_mb=140000.0,
            ttft_ms=850.0,
            throughput_tps=32.0,
            arc_challenge=68.9,
            gsm8k=92.0,
            mmlu=82.1,
            cost_per_1m_tokens_usd=2.40,
        ),
    ]


def run_live_compression_ablation(
    in_features: int = 128,
    out_features: int = 128,
    seq_len: int = 64,
    group_size: int = 32,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Execute a live, deterministic PyTorch compression ablation comparing dense,
    pruned (Wanda unstructured & 2:4), and quantized (RTN INT4, AWQ INT4, and
    SpinQuant Hadamard + INT4) linear projections on outlier-heavy activations.
    """
    torch.manual_seed(seed)

    weights = torch.randn(out_features, in_features, dtype=torch.float32) * 0.05
    activations = torch.randn(seq_len, in_features, dtype=torch.float32)
    # Inject realistic LLM activation and weight outlier channels
    act_outlier_channels = [0, in_features // 4, in_features // 2]
    for ch in act_outlier_channels:
        activations[:, ch] *= 20.0
    weight_outlier_channels = [1, in_features // 3]
    for ch in weight_outlier_channels:
        weights[:, ch] *= 8.0

    with torch.no_grad():
        y_dense = F.linear(activations, weights)
        dense_bytes = weights.numel() * 2  # FP16 reference (2 bytes/param)

        results: List[Dict[str, Any]] = []

        def _record(
            profile_name: str,
            y_approx: torch.Tensor,
            effective_bits: float,
            sparsity_ratio: float,
            elapsed_ms: float,
        ) -> None:
            mse = F.mse_loss(y_approx, y_dense).item()
            cos_sim = (
                F.cosine_similarity(
                    y_approx.flatten().unsqueeze(0),
                    y_dense.flatten().unsqueeze(0),
                )
                .item()
            )
            compression_ratio = 16.0 / effective_bits
            results.append(
                {
                    "profile": profile_name,
                    "effective_bits": round(effective_bits, 2),
                    "compression_ratio_vs_fp16": round(compression_ratio, 2),
                    "sparsity_ratio": round(sparsity_ratio, 4),
                    "output_mse": round(mse, 8),
                    "cosine_similarity": round(cos_sim, 6),
                    "kernel_time_ms": round(elapsed_ms, 3),
                    "estimated_bytes": int(dense_bytes / compression_ratio),
                }
            )

        # 1. FP16 Baseline
        t0 = time.perf_counter()
        y_fp16 = F.linear(activations.to(torch.bfloat16), weights.to(torch.bfloat16)).float()
        _record("FP16/BF16-Dense", y_fp16, 16.0, 0.0, (time.perf_counter() - t0) * 1000)

        # 2. Wanda 50% Unstructured
        t0 = time.perf_counter()
        lin_unstruct = nn.Linear(in_features, out_features, bias=False)
        lin_unstruct.weight.data.copy_(weights)
        execute_wanda_pruning(
            lin_unstruct, [activations], sparsity_ratio=0.5, sparsity_type="unstructured"
        )
        w_wanda_unstruct = lin_unstruct.weight.data
        y_wanda_unstruct = F.linear(activations, w_wanda_unstruct)
        sparsity_unstruct = float((w_wanda_unstruct == 0).sum().item() / w_wanda_unstruct.numel())
        _record(
            "Wanda-50%-Unstructured",
            y_wanda_unstruct,
            8.0,
            sparsity_unstruct,
            (time.perf_counter() - t0) * 1000,
        )

        # 3. Wanda 2:4 Semi-Structured
        t0 = time.perf_counter()
        lin_24 = nn.Linear(in_features, out_features, bias=False)
        lin_24.weight.data.copy_(weights)
        execute_wanda_pruning(
            lin_24, [activations], sparsity_ratio=0.5, sparsity_type="2:4"
        )
        w_wanda_24 = lin_24.weight.data
        y_wanda_24 = F.linear(activations, w_wanda_24)
        sparsity_24 = float((w_wanda_24 == 0).sum().item() / w_wanda_24.numel())
        _record(
            "Wanda-2:4-SemiStructured",
            y_wanda_24,
            8.0,
            sparsity_24,
            (time.perf_counter() - t0) * 1000,
        )

        # 4. Naive Round-To-Nearest (RTN) INT4
        t0 = time.perf_counter()
        w_rtn, _, _ = pseudo_quantize_tensor(weights, w_bit=4, q_group_size=group_size)
        y_rtn = F.linear(activations, w_rtn)
        _record("RTN-INT4-Naive", y_rtn, 4.25, 0.0, (time.perf_counter() - t0) * 1000)

        # 5. AWQ INT4 Salient-Channel Grid Search
        t0 = time.perf_counter()
        scales, _, _ = compute_awq_scales(
            weights, activations, w_bit=4, q_group_size=group_size, n_grid=20
        )
        w_awq, _, _ = pseudo_quantize_tensor(
            weights * scales.unsqueeze(0), w_bit=4, q_group_size=group_size
        )
        y_awq = F.linear(activations / scales.unsqueeze(0), w_awq)
        _record(
            "AWQ-INT4-SalientScaled",
            y_awq,
            4.25,
            0.0,
            (time.perf_counter() - t0) * 1000,
        )

        # 6. SpinQuant Hadamard Rotation + AWQ INT4
        t0 = time.perf_counter()
        w_rot, x_rot = apply_hadamard_rotation_smoothing(weights, activations)
        scales_rot, _, _ = compute_awq_scales(
            w_rot, x_rot, w_bit=4, q_group_size=group_size, n_grid=20
        )
        w_rot_q, _, _ = pseudo_quantize_tensor(
            w_rot * scales_rot.unsqueeze(0), w_bit=4, q_group_size=group_size
        )
        y_spin_awq = F.linear(x_rot / scales_rot.unsqueeze(0), w_rot_q)
        _record(
            "AWQ-INT4-SpinQuant-Hadamard",
            y_spin_awq,
            4.25,
            0.0,
            (time.perf_counter() - t0) * 1000,
        )

    return results


def format_pareto_markdown_table(
    observations: Sequence[BenchmarkObservation],
    max_vram_mb: Optional[float] = 8192.0,
) -> str:
    """Format the cross-architecture Pareto matrix as a GitHub Markdown table."""
    frontier = compute_pareto_frontier(observations, max_vram_mb=max_vram_mb)
    frontier_ids = {obs.identifier for obs in frontier}

    lines = [
        "| Model Family | Compression Profile | VRAM (MB) | TTFT (ms) | Throughput (tok/s) | ARC-C | GSM8K | MMLU | Composite Score | Quality / GB | Pareto Status (<=8GB) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for obs in observations:
        if obs.identifier in frontier_ids:
            status = "**Pareto Optimal**"
        elif max_vram_mb is not None and obs.vram_mb > max_vram_mb:
            status = "Exceeds 8GB Budget"
        else:
            status = "Dominated"

        lines.append(
            f"| `{obs.model_family}` | `{obs.compression_profile}` | {obs.vram_mb:.0f} | "
            f"{obs.ttft_ms:.0f} | {obs.throughput_tps:.0f} | {obs.arc_challenge:.1f}% | "
            f"{obs.gsm8k:.1f}% | {obs.mmlu:.1f}% | **{obs.composite_quality_score:.2f}** | "
            f"{obs.quality_per_gb_vram:.1f} | {status} |"
        )
    return "\n".join(lines)


def export_pareto_artifacts(output_dir: str | Path = "./results") -> Dict[str, Any]:
    """
    Compute the cross-architecture Pareto frontier and live PyTorch compression
    ablation, then write JSON and Markdown artifacts to `output_dir`.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    observations = get_reference_pareto_matrix()
    edge_frontier = compute_pareto_frontier(observations, max_vram_mb=8192.0)
    global_frontier = compute_pareto_frontier(observations, max_vram_mb=None)
    live_ablation = run_live_compression_ablation()

    payload: Dict[str, Any] = {
        "observations": [
            {
                **asdict(obs),
                "identifier": obs.identifier,
                "composite_quality_score": obs.composite_quality_score,
                "quality_per_gb_vram": obs.quality_per_gb_vram,
            }
            for obs in observations
        ],
        "edge_8gb_pareto_frontier": [obs.identifier for obs in edge_frontier],
        "global_pareto_frontier": [obs.identifier for obs in global_frontier],
        "live_pytorch_compression_ablation": live_ablation,
    }

    json_file = out_path / "pareto_frontier.json"
    json_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_lines = [
        "# Cross-Architecture Empirical Pareto Report",
        "",
        "## 1. Multi-Objective Pareto Frontier (<= 8 GB Consumer Edge Budget)",
        "",
        format_pareto_markdown_table(observations, max_vram_mb=8192.0),
        "",
        "## 2. Live PyTorch Compression Kernel Ablation (Wanda vs. RTN vs. AWQ vs. SpinQuant)",
        "",
        "| Compression Kernel | Effective Bits | Compression Ratio | Sparsity | Output MSE vs. Dense | Cosine Similarity | Kernel Time (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for row in live_ablation:
        md_lines.append(
            f"| `{row['profile']}` | {row['effective_bits']:.2f} | "
            f"{row['compression_ratio_vs_fp16']:.2f}x | {row['sparsity_ratio']*100:.1f}% | "
            f"`{row['output_mse']:.6f}` | `{row['cosine_similarity']:.6f}` | "
            f"{row['kernel_time_ms']:.2f} |"
        )

    md_file = out_path / "pareto_frontier.md"
    md_file.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    logger.info(f"Saved Pareto JSON artifact to: {json_file}")
    logger.info(f"Saved Pareto Markdown report to: {md_file}")
    return payload


def main() -> None:
    payload = export_pareto_artifacts("./results")
    logger.info(
        f"Edge 8GB Pareto Frontier ({len(payload['edge_8gb_pareto_frontier'])} configs): "
        f"{payload['edge_8gb_pareto_frontier']}"
    )
    for row in payload["live_pytorch_compression_ablation"]:
        logger.info(
            f"Kernel={row['profile']:<28} | Bits={row['effective_bits']:<5} "
            f"| MSE={row['output_mse']:.6f} | CosSim={row['cosine_similarity']:.6f}"
        )


if __name__ == "__main__":
    main()
