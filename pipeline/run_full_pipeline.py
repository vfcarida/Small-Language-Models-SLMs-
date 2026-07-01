"""
==============================================================================
PIPELINE END-TO-END — SLM PROJECT DELIVERY
==============================================================================
LEVEL: 🔴 Production
ESTIMATED EXECUTION TIME: ~4 hours (on 8GB GPU)

OBJECTIVE:
    Orchestrates the entire SLM project delivery workflow:
    1. Environment setup and verification.
    2. Zero-shot baseline evaluation.
    3. SFT + LoRA training on Alpaca instructions.
    4. Adapter weight merging.
    5. Evaluation of fine-tuned weights.
    6. Performance visualizations.

RUN COMMAND:
    python pipeline/run_full_pipeline.py
==============================================================================
"""

import os
import sys
import logging
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.config import load_config, ensure_output_dirs
from src.utils.metrics import get_vram_usage_gb
from src.module_02_sft.sft_lora import train_sft_lora, merge_lora_adapter
from src.module_07_evaluation.benchmark_runner import run_lm_eval
from src.module_07_evaluation.results_visualizer import plot_training_loss, plot_benchmark_radar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("🚀 INITIATING SLM END-TO-END PRODUCTION PIPELINE".center(80))
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # --------------------------------------------------------------------------
    # STAGE 0: SETUP
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 0/5] RUNTIME ENVIRONMENT SETUP")
    config = load_config()
    ensure_output_dirs(config)
    
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    logger.info(f"Target Model ID: {model_id}")
    logger.info(f"Initial VRAM Allocation: {get_vram_usage_gb():.2f} GB")
    
    # --------------------------------------------------------------------------
    # STAGE 1: BASELINE EVALUATION
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 1/5] EVALUATING BASELINE MODEL PERFORMANCE")
    baseline_dir = "./results/eval_baseline"
    run_lm_eval(
        model_id=model_id,
        tasks="mmlu,gsm8k",
        num_fewshot=0,
        batch_size=4,
        output_dir=baseline_dir
    )
    
    # --------------------------------------------------------------------------
    # STAGE 2: SUPERVISED FINE-TUNING (SFT + LoRA)
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 2/5] INITIATING TRAINING STAGES (SFT + LoRA)")
    logger.info("Starting training runs. This operation is resource-intensive...")
    
    adapter_dir = train_sft_lora(config=config)
    logger.info(f"Training completed successfully. Adapters written to: {adapter_dir}")
    
    # --------------------------------------------------------------------------
    # STAGE 3: MERGING ADAPTER WEIGHTS
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 3/5] MERGING LORA WEIGHTS WITH BASE PARAMETERS")
    merged_dir = "./results/sft_merged_model"
    
    merge_lora_adapter(
        base_model_id=model_id,
        adapter_path=adapter_dir,
        output_dir=merged_dir
    )
    
    # --------------------------------------------------------------------------
    # STAGE 4: POST-TRAINING EVALUATION
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 4/5] ASSESSING FINE-TUNED MODEL CAPABILITIES")
    eval_finetuned_dir = "./results/eval_finetuned"
    
    run_lm_eval(
        model_id=merged_dir,
        tasks="mmlu,gsm8k",
        num_fewshot=0,
        batch_size=4,
        output_dir=eval_finetuned_dir
    )
    
    # --------------------------------------------------------------------------
    # STAGE 5: VISUALIZATIONS AND GRAPH GENERATION
    # --------------------------------------------------------------------------
    logger.info("\n[STAGE 5/5] GENERATING PERFORMANCE METRICS VISUALIZATIONS")
    os.makedirs("./results/figures", exist_ok=True)
    
    plot_training_loss(
        metrics_path=os.path.join(adapter_dir, "trainer_state.json"),
        output_path="./results/figures/training_loss.png"
    )
    
    plot_benchmark_radar(
        results_path=os.path.join(eval_finetuned_dir, "results.json"),
        output_path="./results/figures/benchmark_radar.png"
    )
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE EXECUTION COMPLETED SUCCESSFULLY!".center(80))
    logger.info(f"Total time elapsed: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("=" * 80)
    logger.info("\nNext step: Run 'python pipeline/generate_report.py' to generate details.")


if __name__ == "__main__":
    main()
