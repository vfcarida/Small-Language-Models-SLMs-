"""
==============================================================================
MODULE 04 — WANDA PRUNING (Pruning by Weights and Activations)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 45 minutes (reading) + execution

OBJECTIVE:
    Implement the Wanda (Pruning by Weights and Activations) algorithm from
    Sun et al. (ICLR 2024). Wanda induces high sparsity in Small Language Models
    in a single forward pass without any weight updates or Hessian inversions,
    supporting both per-output unstructured pruning and NVIDIA 2:4 semi-structured
    sparsity.

CONCEPTS COVERED:
    1. Emergent Large-Magnitude Activation Features in Transformers.
    2. Wanda Importance Metric: S_ij = |W_ij| * ||X_j||_2.
    3. Per-Output Comparison Group Pruning (row-wise across input channels).
    4. NVIDIA 2:4 Semi-Structured Hardware Sparsity.
    5. Zipf-Diversified Calibration Sequence Selection.

REFERENCES:
    - Sun et al., "A Simple and Effective Pruning Approach for Large Language
      Models (Wanda)", ICLR 2024 (arXiv:2306.11695)
    - Frantar & Alistarh, "SparseGPT: Massive Language Models Can Be Accurately
      Pruned in One-Shot", ICML 2023 (arXiv:2301.00774)

RUN COMMAND:
    python src/module_04_pruning/wanda_pruning.py
==============================================================================
"""

import logging
from typing import Dict, List, Literal

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import load_config
from src.utils.metrics import calculate_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SparsityType = Literal["unstructured", "2:4"]


def rank_sequences_by_zipf_diversity(sequences: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Ranks candidate token sequences by lexical entropy and unique-token coverage
    to counter calibration dilution on long-tail Zipfian vocabularies.

    Args:
        sequences: List of 2D token ID tensors of shape `(1, seq_len)`.

    Returns:
        List[torch.Tensor]: Sequences ordered from highest to lowest lexical diversity.
    """
    scored = []
    for seq in sequences:
        flat = seq.view(-1)
        unique_tokens, counts = torch.unique(flat, return_counts=True)
        probs = counts.float() / max(flat.numel(), 1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        scored.append((entropy, len(unique_tokens), seq))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored]


class ZipfCalibrationLoader:
    """
    Calibration data loader that extracts candidate windows and prioritizes
    segments with high lexical diversity across the Zipfian token distribution.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        dataset_name: str = "wikitext",
        split: str = "train",
        num_samples: int = 32,
        seq_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        self.num_samples = num_samples
        self.seq_len = seq_len

    def get_batches(self) -> List[torch.Tensor]:
        logger.info(f"Generating {self.num_samples} Zipf-diversified calibration batches...")
        text = "\n\n".join([t for t in self.dataset["text"][:1500] if t.strip()])
        encodings = self.tokenizer(text, return_tensors="pt")

        candidates: List[torch.Tensor] = []
        max_candidates = self.num_samples * 2
        for i in range(max_candidates):
            start_idx = i * (self.seq_len // 2)
            if start_idx + self.seq_len > encodings.input_ids.shape[1]:
                break
            inp = encodings.input_ids[:, start_idx : start_idx + self.seq_len]
            candidates.append(inp)

        ranked = rank_sequences_by_zipf_diversity(candidates)
        return ranked[: self.num_samples]


def compute_wanda_scores(weight: torch.Tensor, activation_norms: torch.Tensor) -> torch.Tensor:
    """
    Computes the Wanda importance score matrix S_ij = |W_ij| * ||X_j||_2.

    Args:
        weight: Linear layer weight matrix W of shape `(d_out, d_in)`.
        activation_norms: Input channel L2 norms ||X||_2 of shape `(d_in,)`.

    Returns:
        torch.Tensor: Importance score matrix S of shape `(d_out, d_in)`.
    """
    if weight.dim() != 2:
        raise ValueError(f"Expected 2D weight tensor (d_out, d_in), got shape {weight.shape}")
    if activation_norms.numel() != weight.shape[1]:
        raise ValueError(
            f"Activation norm dimension ({activation_norms.numel()}) does not match "
            f"weight in_features ({weight.shape[1]})."
        )
    norms = activation_norms.to(device=weight.device, dtype=weight.dtype).view(1, -1)
    return torch.abs(weight) * norms


def execute_wanda_pruning(
    model: nn.Module,
    calibration_inputs: List[torch.Tensor],
    sparsity_ratio: float = 0.5,
    sparsity_type: SparsityType = "unstructured",
) -> nn.Module:
    """
    Executes the Wanda one-shot pruning algorithm across all `nn.Linear` layers.

    MATHEMATICAL FORMULATION (Sun et al., ICLR 2024):
        1. Activation Norm Accumulation:
           For an input activation tensor X of shape (N * L, d_in) entering a linear
           layer, compute the L2 norm of each input feature j across all calibration
           tokens:
               ||X_j||_2 = sqrt( sum_{t=1}^{N*L} X_{t, j}^2 )

        2. Per-Output Row Importance Comparison:
           Compute S_{i, j} = |W_{i, j}| * ||X_j||_2.
           Crucially, Wanda compares and prunes weights per-output row i (along dim=1),
           NOT globally across the entire matrix. This ensures every output neuron
           retains a balanced fan-in of active synapses.

        3. Semi-Structured 2:4 Mode:
           When `sparsity_type == "2:4"`, for every contiguous group of 4 input weights
           along each output row, the 2 weights with the lowest S_{i, j} are zeroed out.

    Args:
        model: Causal language model or PyTorch `nn.Module` to prune.
        calibration_inputs: List of calibration token batches or input tensors.
        sparsity_ratio: Fraction of weights to zero out per output row in `[0.0, 1.0)`.
        sparsity_type: `"unstructured"` (per-output row) or `"2:4"` (semi-structured).

    Returns:
        nn.Module: Pruned model (updated in-place and returned).
    """
    if not (0.0 <= sparsity_ratio < 1.0):
        raise ValueError(f"sparsity_ratio must be in [0.0, 1.0). Got: {sparsity_ratio}")

    logger.info(
        f"Starting Wanda Pruning (mode={sparsity_type}, target_sparsity={sparsity_ratio * 100:.1f}%)"
    )
    device = next(model.parameters()).device

    # Accumulate sum of squared activations per channel so total L2 norm is exact across batches
    sum_sq_activations: Dict[str, torch.Tensor] = {}

    def get_activation_hook(name: str):
        def hook(module: nn.Module, inp: tuple, out: torch.Tensor) -> None:
            x = inp[0].detach().float()
            if x.dim() >= 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.dim() == 1:
                x = x.unsqueeze(0)
            sq_sum = torch.sum(x ** 2, dim=0)
            if name not in sum_sq_activations:
                sum_sq_activations[name] = sq_sum
            else:
                sum_sq_activations[name] += sq_sum

        return hook

    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(get_activation_hook(name)))

    logger.info("Running calibration forward passes to measure activation L2 norms ||X_j||_2...")
    model.eval()
    with torch.no_grad():
        for batch in calibration_inputs:
            model(batch.to(device))

    for handle in handles:
        handle.remove()

    logger.info("Computing Wanda scores S_ij = |W_ij| * ||X_j||_2 and applying row-wise masks...")

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in sum_sq_activations:
            W = module.weight.data
            x_l2_norm = torch.sqrt(sum_sq_activations[name])
            S = compute_wanda_scores(W, x_l2_norm)

            if sparsity_type == "2:4":
                in_features = S.shape[1]
                if in_features % 4 != 0:
                    raise ValueError(
                        f"Layer '{name}' in_features ({in_features}) is not divisible by 4 for 2:4 sparsity."
                    )
                grouped_S = S.view(-1, 4)
                _, prune_idx = torch.topk(grouped_S, k=2, dim=1, largest=False)
                mask_grouped = torch.ones_like(grouped_S, dtype=W.dtype)
                mask_grouped.scatter_(1, prune_idx, 0.0)
                mask = mask_grouped.view_as(W)
                module.weight.data.mul_(mask)
            elif sparsity_type == "unstructured":
                num_prune_per_row = int(sparsity_ratio * S.shape[1])
                if num_prune_per_row <= 0:
                    continue
                _, prune_indices = torch.topk(S, k=num_prune_per_row, dim=1, largest=False)
                mask = torch.ones_like(W)
                mask.scatter_(1, prune_indices, 0.0)
                module.weight.data.mul_(mask)
            else:
                raise ValueError(f"Unsupported sparsity_type: '{sparsity_type}'.")

    logger.info("Wanda pruning executed successfully.")
    return model


def main() -> None:
    config = load_config()
    pruning_cfg = config.get("pruning", {})
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")

    logger.info("=" * 60)
    logger.info("MODULE 04: WANDA PRUNING RUN")
    logger.info("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    loader = ZipfCalibrationLoader(
        tokenizer,
        num_samples=pruning_cfg.get("num_calibration_samples", 32),
    )
    calib_batches = loader.get_batches()

    calculate_sparsity(model)
    execute_wanda_pruning(
        model,
        calib_batches,
        sparsity_ratio=pruning_cfg.get("sparsity_ratio", 0.5),
        sparsity_type=pruning_cfg.get("sparsity_type", "unstructured"),
    )
    calculate_sparsity(model)


if __name__ == "__main__":
    main()
