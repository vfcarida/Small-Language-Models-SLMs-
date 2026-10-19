"""
==============================================================================
MODULE 07 — BENCHMARK RUNNER (EleutherAI lm-evaluation-harness)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 15 minutes (reading) + execution

OBJECTIVE:
    Execute standardized, reproducible evaluations for Small Language Models
    using the EleutherAI LM Evaluation Harness (`lm_eval`). Supports both live
    CLI execution on GPU hosts and deterministic offline simulation for CPU CI
    verification.

CONCEPTS COVERED:
    1. Few-shot and Zero-shot standardized evaluation protocols.
    2. Core SLM benchmarks: MMLU (general knowledge) and GSM8K (math reasoning).
    3. Structured JSON metric persistence for downstream reporting.

REFERENCES:
    - Gao et al., "A framework for few-shot language model evaluation",
      Zenodo / EleutherAI, 2023
    - Hendrycks et al., "Measuring Massive Multitask Language Understanding (MMLU)",
      ICLR 2021 (arXiv:2009.03300)

RUN COMMAND:
    python src/module_07_evaluation/benchmark_runner.py --model_id Qwen/Qwen2.5-1.5B --tasks mmlu,gsm8k
==============================================================================
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_lm_eval(
    model_id: str,
    tasks: str,
    num_fewshot: int,
    batch_size: int,
    output_dir: str,
    simulate: bool = True,
) -> Dict[str, Any]:
    """
    Executes or simulates the EleutherAI `lm_eval` benchmark suite and writes
    `results.json` into `output_dir`.

    Args:
        model_id: Hugging Face model ID or local checkpoint path.
        tasks: Comma-separated benchmark task identifiers (e.g., `"mmlu,gsm8k"`).
        num_fewshot: Number of few-shot examples per prompt.
        batch_size: Evaluation batch size.
        output_dir: Directory where `results.json` will be stored.
        simulate: If False and `lm_eval` is installed in PATH, runs the live subprocess.

    Returns:
        Dict[str, Any]: Evaluation result dictionary written to `results.json`.
    """
    logger.info("=" * 60)
    logger.info("ELEUTHERAI LM EVALUATION HARNESS")
    logger.info(f"Model ID: {model_id} | Tasks: {tasks} | Few-shot: {num_fewshot}")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    model_args = f"pretrained={model_id},dtype=bfloat16"

    cmd = [
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--num_fewshot",
        str(num_fewshot),
        "--batch_size",
        str(batch_size),
        "--output_path",
        output_dir,
    ]

    logger.info(f"Harness command: {' '.join(cmd)}")
    result_path = os.path.join(output_dir, "results.json")

    if not simulate and shutil.which("lm_eval") is not None:
        logger.info("Executing live lm_eval subprocess...")
        subprocess.run(cmd, check=True)
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)

    logger.info("Running in deterministic offline simulation mode...")
    is_finetuned = "merged" in model_id.lower() or "sft" in model_id.lower() or "finetuned" in output_dir.lower()
    mmlu_score = 0.478 if is_finetuned else 0.452
    gsm8k_score = 0.354 if is_finetuned else 0.201

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    results_dict: Dict[str, Any] = {}
    for task_name in task_list:
        if "gsm8k" in task_name.lower():
            results_dict[task_name] = {"exact_match,none": gsm8k_score}
        else:
            results_dict[task_name] = {"acc,none": mmlu_score}

    payload: Dict[str, Any] = {
        "results": results_dict,
        "config": {
            "model": model_id,
            "tasks": task_list,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "simulated": True,
        },
    }

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info(f"Evaluation metrics saved to: {result_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="EleutherAI LM-Eval Harness Runner")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results/eval_output")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute live lm_eval subprocess instead of offline simulation.",
    )
    args = parser.parse_args()

    run_lm_eval(
        model_id=args.model_id,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        simulate=not args.live,
    )


if __name__ == "__main__":
    main()
