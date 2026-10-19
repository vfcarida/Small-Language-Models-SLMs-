"""
==============================================================================
MODULE 05 — DYNAMIC ON-THE-FLY QUANTIZATION (bitsandbytes INT8 & NF4)
==============================================================================
LEVEL: 🟢 Beginner
ESTIMATED TIME: 15 minutes

OBJECTIVE:
    Demonstrate dynamic 8-bit (LLM.int8()) and 4-bit NormalFloat (NF4) quantization
    using `bitsandbytes`, alongside theoretical memory footprint calculators that
    quantify VRAM savings across FP32, BF16, INT8, and NF4 precisions.

CONCEPTS COVERED:
    1. FP16/BF16 vs. LLM.int8() Mixed-Precision Decomposition.
    2. 4-bit NormalFloat (NF4) Information-Theoretic Quantile Quantization.
    3. Double Quantization (quantizing the quantization constants).
    4. Memory Bandwidth vs. Dynamic Dequantization Overhead Trade-offs.

REFERENCES:
    - Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers
      at Scale", NeurIPS 2022 (arXiv:2208.07339)
    - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
      NeurIPS 2023 (arXiv:2305.14314)

RUN COMMAND:
    python src/module_05_quantization/bitsandbytes_quant.py
==============================================================================
"""

import logging
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from src.utils.metrics import get_model_size_mb, get_vram_usage_gb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def estimate_quantized_memory_mb(
    num_parameters: int,
    bit_width: int = 4,
    q_group_size: int = 64,
    double_quant: bool = True,
) -> Dict[str, float]:
    """
    Computes the analytical memory footprint (in MB) of a model under a given
    bit-width and blockwise quantization configuration (Dettmers et al., 2023).

    MATHEMATICAL FORMULATION:
        - Raw weight bits = N * bit_width
        - Single quantization FP32 scale overhead = 32 / q_group_size bits per parameter
          (for q_group_size=64, this adds 0.5 bits/param).
        - Double quantization (8-bit scales with group size 256) reduces scale overhead
          to: 8 / q_group_size + 32 / (q_group_size * 256) bits per parameter
          (for q_group_size=64, this reduces overhead from 0.5 to ~0.127 bits/param).

    Args:
        num_parameters: Total number of model parameters N.
        bit_width: Target bit-width (`32`, `16`, `8`, or `4`).
        q_group_size: Block size for quantization scales.
        double_quant: Whether double quantization is enabled for 4-bit NF4.

    Returns:
        Dict[str, float]: Breakdown containing `'effective_bits_per_param'`,
        `'weight_memory_mb'`, `'scale_overhead_mb'`, and `'total_memory_mb'`.
    """
    if num_parameters <= 0:
        raise ValueError(f"num_parameters must be positive. Got: {num_parameters}")
    if bit_width not in (4, 8, 16, 32):
        raise ValueError(f"Unsupported bit_width: {bit_width}. Choose from 4, 8, 16, 32.")

    weight_bytes = num_parameters * (bit_width / 8.0)

    if bit_width in (16, 32):
        overhead_bits_per_param = 0.0
    elif bit_width == 8:
        overhead_bits_per_param = 16.0 / max(q_group_size, 1)
    else:
        if double_quant:
            overhead_bits_per_param = (8.0 / q_group_size) + (32.0 / (q_group_size * 256.0))
        else:
            overhead_bits_per_param = 32.0 / q_group_size

    overhead_bytes = num_parameters * (overhead_bits_per_param / 8.0)
    total_mb = (weight_bytes + overhead_bytes) / (1024.0 ** 2)

    return {
        "effective_bits_per_param": round(bit_width + overhead_bits_per_param, 4),
        "weight_memory_mb": round(weight_bytes / (1024.0 ** 2), 2),
        "scale_overhead_mb": round(overhead_bytes / (1024.0 ** 2), 2),
        "total_memory_mb": round(total_mb, 2),
    }


def evaluate_quantization(model_id: str, bit_target: int) -> None:
    """
    Loads the target model in the requested precision (16-bit, 8-bit, or 4-bit NF4)
    and logs empirical VRAM consumption.
    """
    logger.info(f"\n{'-' * 40}")
    logger.info(f"LOADING MODEL IN {bit_target}-BIT PRECISION")
    logger.info(f"{'-' * 40}")

    vram_before = get_vram_usage_gb()

    if bit_target == 16:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
    elif bit_target == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quant_config
        )
    elif bit_target == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quant_config
        )
    else:
        raise ValueError(f"Unsupported bit_target: {bit_target}")

    vram_after = get_vram_usage_gb()
    size_mb = get_model_size_mb(model)

    logger.info(f"Allocated VRAM Delta: {vram_after - vram_before:.2f} GB")
    logger.info(f"Parameter Memory Footprint: {size_mb / 1024:.2f} GB")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    logger.info("=" * 60)
    logger.info("MODULE 05: BITSANDBYTES QUANTIZATION (ON-THE-FLY)")
    logger.info("=" * 60)

    model_id = "Qwen/Qwen2.5-1.5B"
    num_params = 1_540_000_000

    for bits in (16, 8, 4):
        est = estimate_quantized_memory_mb(num_params, bit_width=bits)
        logger.info(
            f"Analytical Estimate ({bits}-bit): {est['total_memory_mb']:.1f} MB "
            f"({est['effective_bits_per_param']:.3f} bits/param)"
        )

    if not torch.cuda.is_available():
        logger.info("CUDA GPU not detected; skipping live bitsandbytes hardware allocation.")
        return

    evaluate_quantization(model_id, 16)
    evaluate_quantization(model_id, 8)
    evaluate_quantization(model_id, 4)

    logger.info("\nSUMMARY: bitsandbytes NF4 is ideal for rapid zero-calibration loading and QLoRA.")
    logger.info("For maximum edge inference throughput with fused INT4 GEMM kernels, use awq_quantization.py.")


if __name__ == "__main__":
    main()
