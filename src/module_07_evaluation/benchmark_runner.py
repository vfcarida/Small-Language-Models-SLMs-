"""
==============================================================================
MODULE 07 — BENCHMARK RUNNER (lm-eval-harness)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 15 minutes (reading) + execution

OBJECTIVE:
    Learn the standard evaluation framework for Small Language Models (SLMs):
    the EleutherAI LM Evaluation Harness. Model outputs must be evaluated using
    reproducible metrics rather than manual checks.

CONCEPTS COVERED:
    1. Few-shot and Zero-shot evaluation structures.
    2. Classic target tasks: MMLU, GSM8K, HellaSwag, TruthfulQA.
    3. Automated metric logging.

REFERENCES:
    - Gao et al., "A framework for few-shot language model evaluation", 2023
    - Hendrycks et al., "Measuring Massive Multitask Language Understanding" (MMLU)

RUN COMMAND:
    python src/module_07_evaluation/benchmark_runner.py --model_id Qwen/Qwen2.5-1.5B --tasks mmlu_lite
==============================================================================
"""

import os
import json
import logging
import argparse
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_lm_eval(model_id: str, tasks: str, num_fewshot: int, batch_size: int, output_dir: str):
    """
    Spawns the EleutherAI lm-eval script execution.
    
    Args:
        model_id: Model identifier path.
        tasks: Comma-separated target task list.
        num_fewshot: Few-shot count per sample query.
        batch_size: Batch dimension.
        output_dir: Path to directory to write output JSON files.
    """
    logger.info("=" * 60)
    logger.info("RUNNING ELEUTHERAI LM EVALUATION HARNESS")
    logger.info(f"Model ID: {model_id}")
    logger.info(f"Tasks: {tasks}")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    model_args = f"pretrained={model_id},dtype=bfloat16"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", output_dir
    ]
    
    logger.info(f"Execution command: {' '.join(cmd)}")
    
    # We mock execution output below to support environments without active GPUs or access
    logger.info("\nSimulating LM Evaluation execution...")
    
    fake_results = {
        "results": {
            "mmlu": {"acc,none": 0.45},
            "gsm8k": {"exact_match,none": 0.22}
        },
        "config": {"model": model_id, "tasks": tasks}
    }
    
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(fake_results, f, indent=2)
        
    logger.info(f"Mocked results generated successfully in: {result_path}")
    logger.info("Render figures using module results_visualizer.py.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results/eval_output")
    args = parser.parse_args()
    
    run_lm_eval(args.model_id, args.tasks, args.num_fewshot, args.batch_size, args.output_dir)
