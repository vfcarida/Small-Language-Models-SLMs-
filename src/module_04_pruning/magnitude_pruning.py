"""
==============================================================================
MODULE 04 — MAGNITUDE PRUNING (Unstructured & 2:4 Semi-Structured Baseline)
==============================================================================
LEVEL: 🟡 Intermediate
ESTIMATED TIME: 15 minutes

OBJECTIVE:
    Implement classical Magnitude Pruning across linear transformer layers in both
    unstructured and NVIDIA 2:4 semi-structured formats. Serves as the analytical
    baseline for evaluating activation-aware methods such as Wanda.

CONCEPTS COVERED:
    1. Unstructured Weight Magnitude Pruning (Han et al., NeurIPS 2015).
    2. The Lottery Ticket Hypothesis (Frankle & Carbin, ICLR 2019).
    3. NVIDIA 2:4 Semi-Structured Sparsity for Sparse Tensor Core acceleration.
    4. Why pure weight magnitude fails in LLMs with emergent activation outliers.

REFERENCES:
    - Han et al., "Learning both Weights and Connections for Efficient Neural
      Networks", NeurIPS 2015 (arXiv:1506.02626)
    - Frankle & Carbin, "The Lottery Ticket Hypothesis: Finding Sparse, Trainable
      Neural Networks", ICLR 2019 (arXiv:1803.03635)

RUN COMMAND:
    python src/module_04_pruning/magnitude_pruning.py
==============================================================================
"""

import logging
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.utils.metrics import calculate_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SparsityType = Literal["unstructured", "2:4"]


def apply_magnitude_pruning(
    model: nn.Module,
    sparsity_ratio: float = 0.5,
    sparsity_type: SparsityType = "unstructured",
) -> nn.Module:
    """
    Applies weight magnitude pruning across all `nn.Linear` layers in the model.

    MATHEMATICAL THEORY:
        Magnitude pruning ranks weights solely by their absolute value S_ij = |W_ij|:
        - In "unstructured" mode, the lowest `sparsity_ratio` fraction of weights
          within each layer are zeroed out.
        - In "2:4" semi-structured mode (supported natively by NVIDIA Ampere/Hopper
          Sparse Tensor Cores), every contiguous block of 4 weights along the input
          dimension has exactly 2 weights with the smallest |W_ij| zeroed out,
          guaranteeing an exact 50% hardware-acceleratable sparsity pattern.

    LIMITATION IN TRANSFORMERS:
        Because Transformers develop large emergent activation outliers in specific
        hidden dimensions, a small weight W_ij multiplied by a massive activation X_j
        still produces a critical output signal. Ignoring ||X_j||_2 degrades perplexity,
        motivating Wanda (see `wanda_pruning.py`).

    Args:
        model: PyTorch neural network containing `nn.Linear` modules.
        sparsity_ratio: Fraction of weights to prune in `[0.0, 1.0)` for unstructured mode.
        sparsity_type: `"unstructured"` or `"2:4"` semi-structured sparsity.

    Returns:
        nn.Module: The pruned model (modified in-place and returned).
    """
    if not (0.0 <= sparsity_ratio < 1.0):
        raise ValueError(f"sparsity_ratio must be in [0.0, 1.0). Got: {sparsity_ratio}")

    logger.info(
        f"Applying Magnitude Pruning (mode={sparsity_type}, target_sparsity={sparsity_ratio * 100:.1f}%)"
    )

    linear_layers = [
        module for _, module in model.named_modules() if isinstance(module, nn.Linear)
    ]
    logger.info(f"Identified {len(linear_layers)} linear projection layers.")

    for module in linear_layers:
        W = module.weight.data
        abs_W = torch.abs(W)

        if sparsity_type == "2:4":
            in_features = abs_W.shape[1]
            if in_features % 4 != 0:
                raise ValueError(
                    f"2:4 semi-structured pruning requires in_features divisible by 4. Got {in_features}."
                )
            grouped = abs_W.view(-1, 4)
            _, smallest_indices = torch.topk(grouped, k=2, dim=1, largest=False)
            mask_grouped = torch.ones_like(grouped)
            mask_grouped.scatter_(1, smallest_indices, 0.0)
            mask = mask_grouped.view_as(W)
            module.weight.data.mul_(mask)
        elif sparsity_type == "unstructured":
            if sparsity_ratio == 0.0:
                continue
            flat_W = torch.flatten(abs_W)
            k = int(sparsity_ratio * flat_W.numel())
            if k <= 0:
                continue
            threshold, _ = torch.kthvalue(flat_W, k)
            mask = (abs_W > threshold).to(W.dtype)
            # If ties at threshold cause slightly fewer zeros than k, handle exact quantile when distinct
            if (mask == 0).sum().item() == 0:
                mask = (abs_W >= threshold).to(W.dtype)
            module.weight.data.mul_(mask)
        else:
            raise ValueError(f"Unsupported sparsity_type: '{sparsity_type}'. Use 'unstructured' or '2:4'.")

    logger.info("Magnitude pruning completed successfully.")
    return model


def main() -> None:
    logger.info("=" * 60)
    logger.info("MODULE 04: MAGNITUDE PRUNING BASELINE")
    logger.info("=" * 60)

    model_id = "Qwen/Qwen2.5-1.5B"
    logger.info(f"Loading model {model_id}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    logger.info("\nSparsity BEFORE pruning:")
    calculate_sparsity(model)

    apply_magnitude_pruning(model, sparsity_ratio=0.5, sparsity_type="unstructured")

    logger.info("\nSparsity AFTER pruning:")
    calculate_sparsity(model)

    logger.info("\n HARDWARE ACCELERATION NOTE:")
    logger.info("Unstructured sparsity zeroes parameters without shrinking dense matrix dimensions.")
    logger.info("Use sparsity_type='2:4' on NVIDIA Ampere/Hopper GPUs for 2x Sparse Tensor Core speedups,")
    logger.info("and see wanda_pruning.py for activation-aware importance scoring.")


if __name__ == "__main__":
    main()
