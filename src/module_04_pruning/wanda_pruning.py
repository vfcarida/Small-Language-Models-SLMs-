"""
==============================================================================
MODULE 04 — WANDA PRUNING (Pruning by Weights and Activations)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 45 minutes (reading) + execution

OBJECTIVE:
    Implement the Wanda algorithm, a state-of-the-art technique for parameter
    pruning in SLMs. It outperforms static magnitude pruning by accounting for
    neural activations, and runs much faster than SparseGPT (One-Shot execution).

CONCEPTS COVERED:
    1. PyTorch Forward Hooks configuration.
    2. Wanda Math (Score = |W| * ||X||_2).
    3. Zipf Calibration Sampling.
    4. Sparsity Mask generation.

REFERENCES:
    - Sun et al., "A Simple and Effective Pruning Approach for LLMs (Wanda)", arXiv:2306.11695, 2024
    - Frantar et al., "SparseGPT: Massive Language Models Can be Pruned in One-Shot", 2023

RUN COMMAND:
    python src/module_04_pruning/wanda_pruning.py
==============================================================================
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.utils.config import load_config
from src.utils.metrics import calculate_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZipfCalibrationLoader:
    """
    Dataloader implementing Zipf-sampling.
    Calibration text distribution ensures semantic frequency matching.
    """
    def __init__(self, tokenizer, dataset_name="wikitext", split="train", num_samples=128, seq_len=2048):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        self.num_samples = num_samples
        self.seq_len = seq_len

    def get_batches(self) -> list:
        logger.info(f"Generating {self.num_samples} calibration batches...")
        
        # Concatenate text segments
        text = "\n\n".join(self.dataset["text"][:1000])
        encodings = self.tokenizer(text, return_tensors="pt")
        
        inps = []
        for i in range(self.num_samples):
            start_idx = i * (self.seq_len // 2)
            if start_idx + self.seq_len > encodings.input_ids.shape[1]:
                break
            inp = encodings.input_ids[:, start_idx:start_idx + self.seq_len]
            inps.append(inp)
            
        return inps


def execute_wanda_pruning(model: nn.Module, calibration_inputs: list, sparsity_ratio: float = 0.5) -> nn.Module:
    """
    Executes the Wanda Pruning algorithm.
    
    MATHEMATICAL INTUITION:
        Standard Magnitude Pruning uses: Score_ij = |W_ij|
        Wanda evaluates: Score_ij = |W_ij| * ||X_j||_2
        Where:
            - W_ij is the parameter weight.
            - ||X_j||_2 is the Euclidean L2 norm of activations received by weight W_ij.
            
    Args:
        model: Causal language model to prune.
        calibration_inputs: Batches used to measure activation magnitudes.
        sparsity_ratio: The target sparsity ratio (e.g. 0.5 to zero out 50% of weights).
        
    Returns:
        nn.Module: Pruned model.
    """
    logger.info(f"\nStarting WANDA Pruning (Sparsity Target: {sparsity_ratio*100}%)")
    device = next(model.parameters()).device
    
    activation_norms = {}
    
    def get_activation_hook(name):
        def hook(module, input, output):
            x = input[0].detach() 
            if len(x.shape) == 3:
                x = x.view(-1, x.shape[-1])
            norm = torch.norm(x, p=2, dim=0)
            if name not in activation_norms:
                activation_norms[name] = norm
            else:
                activation_norms[name] += norm
        return hook

    # Register hooks on PyTorch Linear layers
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(get_activation_hook(name)))
            
    logger.info("Running calibration forward passes to measure activation norms ||X_j||_2...")
    model.eval()
    with torch.no_grad():
        for batch in calibration_inputs:
            model(batch.to(device))
            
    # Cleanup forward hooks to free memory
    for handle in handles:
        handle.remove()
        
    logger.info("Computing Wanda scores and applying pruning masks...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in activation_norms:
            W = module.weight.data
            X_norm = activation_norms[name].to(W.device)
            
            # WANDA CORE EQUATION: S_ij = |W_ij| * ||X_j||_2
            S = torch.abs(W) * X_norm
            
            flat_S = torch.flatten(S)
            k = int((1 - sparsity_ratio) * flat_S.numel())
            if k == 0:
                continue
                
            threshold, _ = torch.kthvalue(flat_S, flat_S.numel() - k + 1)
            
            # Formulate structural binary mask
            mask = (S >= threshold).float()
            module.weight.data.mul_(mask)
            
    logger.info("Wanda pruning executed successfully.")
    return model


def main():
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
        num_samples=pruning_cfg.get("num_calibration_samples", 32)
    )
    calib_batches = loader.get_batches()
    
    calculate_sparsity(model)
    execute_wanda_pruning(
        model, 
        calib_batches, 
        sparsity_ratio=pruning_cfg.get("sparsity_ratio", 0.5)
    )
    calculate_sparsity(model)


if __name__ == "__main__":
    main()
