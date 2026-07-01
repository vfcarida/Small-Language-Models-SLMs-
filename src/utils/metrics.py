"""
==============================================================================
METRICS MODULE — Reusable Evaluation and Instrumentation Functions
==============================================================================
This module contains standard metric evaluation formulas used across the codebase,
fully documented with mathematical explanations in International English.

EDUCATIONAL CONCERN:
    Metrics serve as the core language to measure model performance and quality.
    In the context of SLMs, the most critical dimensions are:
    
    1. PERPLEXITY (PPL): Measures how "surprised" a model is by new text.
       Lower is better. A theoretically perfect model achieves PPL = 1.0.
       Formula: PPL = exp(-1/N * Σ log P(token_i | context))
       
    2. ACCURACY: Percentage of correct outputs in structured tasks/benchmarks.
    
    3. LATENCY (TTFT): Time-to-first-token generated (crucial for user experience).
    
    4. THROUGHPUT: Average tokens generated per second.
    
    5. VRAM USAGE: Peak GPU memory allocated by the model weights and cache.

REFERENCES:
    - Jurafsky & Martin, "Speech and Language Processing", Chapter 3 (Perplexity)
    - EleutherAI, "Language Model Evaluation Harness", 2024
==============================================================================
"""

import time
import torch
import gc
import functools
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class GPUOOMHandler:
    """
    Context manager to execute PyTorch operations with dynamic memory recovery.
    Catches CUDA Out of Memory (OOM) exceptions, triggers garbage collection,
    and empties the PyTorch CUDA cache to free up VRAM resources.
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and "out of memory" in str(exc_val).lower():
            logger.warning("CUDA Out Of Memory (OOM) detected! Triggering GPU memory reclamation...")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GPU cache cleared successfully.")
            raise RuntimeError(
                "Execution halted due to CUDA Out of Memory (OOM). "
                "Consider reducing batch size, sequence length, or enabling gradient checkpointing."
            ) from exc_val
        return False


def oom_protected(func):
    """
    Decorator to wrap functions running large tensor operations, providing OOM protection
    by reclaiming GPU memory resources upon failure.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"Function {func.__name__} crashed due to CUDA Out Of Memory.")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    f"OOM error in {func.__name__}. Try decreasing the input batch size or model context length."
                ) from e
            raise
    return wrapper


@oom_protected
def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Calculates the perplexity of a language model on a given text sequence.
    
    MATHEMATICAL THEORY:
        Perplexity is defined as the exponentiated negative average log-likelihood
        of the token sequences predicted by the model:
        
            PPL = exp( -1/N * Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁) )
        
        Where:
        - N is the total number of tokens in the sequence.
        - P(xᵢ | context) is the probability assigned to token xᵢ given historical context.
        
        INTERPRETATION:
        - PPL = 1.0: Perfect predictions (impossible in practical scenarios).
        - PPL = 10.0: The model is as "confused" as if it had to choose uniformly 
                      from 10 different candidates per token.
        - PPL = 1000.0: Extremely high confusion (poor model calibration or text out-of-domain).
    
    IMPLEMENTATION:
        For long inputs, a sliding window with overlap (determined by stride) is used
        to ensure proper contextual history is preserved without exceeding memory boundaries.
    
    Args:
        model: Causal language model (e.g., Qwen2.5-1.5B).
        tokenizer: Tokenizer corresponding to the model architecture.
        text: Test string to calculate perplexity over.
        max_length: Maximum sequence context length window.
        stride: Slide step size (overlap sequence = max_length - stride).
    
    Returns:
        float: Computed perplexity score. Lower values indicate better alignment.
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    
    device = next(model.parameters()).device
    nlls = []
    
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        target_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_chunk.clone()
        target_ids[:, :-target_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
            
    if not nlls:
        return float("inf")
        
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


@oom_protected
def measure_inference_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    Measures latency and throughput metrics during inference runs.
    
    METRICS MEASURED:
        1. TTFT (Time-To-First-Token): Latency until generation of the initial token (ms).
        2. Throughput: Token production speed (tokens/sec).
        3. Total Time: Combined duration of generation block (ms).
    
    BEST PRACTICES:
        - Warming up GPU before starting timing evaluations (compiles kernels and warms cache).
        - Repeating runs and using median stats to shield against outliers.
        - Synchronizing CUDA threads before taking timing ticks for accurate GPU logs.
    
    Args:
        model: Loaded language model.
        tokenizer: Tokenizer.
        prompt: Raw string prompt input.
        max_new_tokens: Token count limit to generate.
        num_warmup: Run rounds ignored from computation results.
        num_runs: Execution iterations to evaluate median values.
    
    Returns:
        Dict: Values for 'ttft_ms', 'throughput_tokens_per_sec', and 'total_time_ms'.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 1. GPU WARMUP
    logger.info(f"Running {num_warmup} warm-up cycles...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # 2. TIME-TO-FIRST-TOKEN (TTFT) MEASUREMENT
    ttft_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1)
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_times.append((time.perf_counter() - start) * 1000)
        
    # 3. SUSTAINED THROUGHPUT MEASUREMENT
    total_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
            )
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_times.append((time.perf_counter() - start) * 1000)
        
    median_ttft = float(np.median(ttft_times))
    median_total = float(np.median(total_times))
    throughput = max_new_tokens / (median_total / 1000.0) if median_total > 0 else 0.0
    
    results = {
        "ttft_ms": round(median_ttft, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "total_time_ms": round(median_total, 2),
    }
    
    logger.info(f"Latency assessment finished: {results}")
    return results


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Computes total memory size of model parameters in megabytes.
    
    THEORY:
        Memory size is directly calculated by: Parameters * bytes per parameter.
        - FP32 = 4 bytes/param, FP16/BF16 = 2 bytes/param, INT8 = 1 byte/param, INT4 = 0.5 bytes/param.
    
    Args:
        model: The target model.
        
    Returns:
        float: Computed size in MB.
    """
    total_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    size_mb = total_bytes / (1024 ** 2)
    return round(size_mb, 2)


def get_vram_usage_gb() -> float:
    """
    Retrieves current active VRAM allocation size in gigabytes.
    
    Returns:
        float: Active memory usage in GB (0.0 if CUDA is not available).
    """
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
    return 0.0


def get_peak_vram_usage_gb() -> float:
    """
    Retrieves peak CUDA VRAM memory allocation size in gigabytes.
    
    Returns:
        float: Peak memory usage in GB (0.0 if CUDA is not available).
    """
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    return 0.0


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Aggregates overall, trainable, and frozen parameter counts in the network.
    
    Args:
        model: Loaded neural network model.
        
    Returns:
        Dict: Breakdown values with keys 'total', 'trainable', 'frozen', and 'trainable_percent'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    trainable_pct = (trainable / total * 100) if total > 0 else 0.0
    
    result = {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": round(trainable_pct, 4),
    }
    
    logger.info(
        f"Parameter Counts — Total: {total:,} | Trainable: {trainable:,} "
        f"({trainable_pct:.2f}%) | Frozen: {frozen:,}"
    )
    return result


def calculate_sparsity(model: PreTrainedModel) -> Dict[str, float]:
    """
    Calculates model weight sparsity percentage (ratio of weights that are exactly zero).
    
    Args:
        model: Target model layers.
        
    Returns:
        Dict: Results with keys 'total_params', 'zero_params', and 'sparsity_percent'.
    """
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
        
    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0.0
    
    result = {
        "total_params": total_params,
        "zero_params": zero_params,
        "sparsity_percent": round(sparsity, 2),
    }
    
    logger.info(f"Model global weight sparsity: {sparsity:.2f}%")
    return result
