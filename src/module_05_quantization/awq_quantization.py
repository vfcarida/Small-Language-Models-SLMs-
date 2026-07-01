"""
==============================================================================
MODULE 05 — AWQ QUANTIZATION (Activation-aware Weight Quantization)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 30 minutes (reading) + calibration run

OBJECTIVE:
    Learn how to perform post-training 4-bit weights quantization using AWQ.
    AWQ generates highly optimized CUDA kernels to accelerate inference and
    reduce memory footprints in production edge environments.

CONCEPTS COVERED:
    1. Salient Weights and Activation Outliers.
    2. Per-channel scaling vs. Round-to-Nearest quant.
    3. Group size, bit settings, and zero point parameters.

REFERENCES:
    - Lin et al., "AWQ: Activation-aware Weight Quantization for LLM 
      Compression and Acceleration", arXiv:2306.00978, 2023

RUN COMMAND:
    python src/module_05_quantization/awq_quantization.py
==============================================================================
"""

import logging
import os
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="AWQ Quantization CLI")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/awq_model")
    return parser.parse_args()


def run_awq_compilation(model_id: str, output_dir: str, config: dict):
    """
    Executes Activation-aware Weight Quantization.
    
    AWQ THEORY:
        About 1% of model weights are highly salient. Standard uniform rounding
        often causes numerical degradation. AWQ calculates activation magnitudes
        passing through channels, applies scaling factors to protect salient channels,
        and scales down activations during inference, preserving knowledge.
    """
    logger.info("=" * 60)
    logger.info(f"COMPILING AWQ INT4 MODEL FOR: {model_id}")
    logger.info("=" * 60)
    
    quant_cfg = config.get("quantization", {})
    
    quant_config = {
        "zero_point": quant_cfg.get("zero_point", True),
        "q_group_size": quant_cfg.get("q_group_size", 128),
        "w_bit": quant_cfg.get("w_bit", 4),
        "version": quant_cfg.get("version", "GEMM")
    }
    
    logger.info(f"Quantization Settings: {quant_config}")
    
    try:
        logger.info("\nLoading original FP16 weights...")
        model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        logger.info("\nStarting activation-aware profiling (running calibration passes)...")
        model.quantize(tokenizer, quant_config=quant_config)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\nSaving compiled 4-bit model to: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("✅ AWQ quantization completed successfully.")
        
    except Exception as e:
        logger.error(f"AWQ compilation failed: {e}")
        logger.error("Verify that the autoawq package is compiled and matched to your active CUDA device.")


def main():
    config = load_config()
    args = parse_args()
    
    model_id = args.model_id or config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    run_awq_compilation(model_id, args.output_dir, config)


if __name__ == "__main__":
    main()
