"""
==============================================================================
MODULE 01 — BASIC INFERENCE WITH SMALL LANGUAGE MODELS
==============================================================================
LEVEL: 🟢 Beginner
ESTIMATED TIME: 15 minutes

OBJECTIVE:
    Learn how to load an SLM from the Hugging Face Hub and generate text
    using different decoding strategies (greedy, sampling, top-k, top-p).

CONCEPTS COVERED:
    1. Loading pre-trained models with AutoModelForCausalLM.
    2. Decoding strategies and their operational trade-offs.
    3. Hardware usage instrumentation (VRAM/RAM).
    4. Text generation variations using temperatures.

PREREQUISITES:
    pip install transformers torch accelerate

REFERENCES:
    - Holtzman et al., "The Curious Case of Neural Text Degeneration", 2020
      (Nucleus sampling paper)
    - HuggingFace, "How to generate text", 2024

RUN COMMAND:
    python src/module_01_inference/basic_inference.py
==============================================================================
"""

import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.metrics import (
    count_parameters,
    get_model_size_mb,
    get_vram_usage_gb,
    measure_inference_latency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    model_id: str = "Qwen/Qwen2.5-1.5B",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
    compile_model: bool = False,
) -> tuple:
    """
    Loads a Small Language Model from the Hugging Face Hub.
    
    PARAMETERS AND PRECISION:
        torch_dtype:
            - float32 (FP32): Maximum precision, 4 bytes/param. 1.5B model ~ 6 GB VRAM.
            - float16 (FP16): Standard GPU float, 2 bytes/param. 1.5B model ~ 3 GB VRAM.
              ⚠️ May cause numerical instability (overflow in exponential layers).
            - bfloat16 (BF16): 2 bytes/param, matching exponent dynamic range of FP32.
              ✅ Recommended for modern architectures (prevents NaN values).
              
    Args:
        model_id: Model registry identifier.
        torch_dtype: Data type string representation.
        device_map: Automatic layer distribution schema.
        compile_model: Flag indicating whether to execute PyTorch graph compilation.
        
    Returns:
        Tuple: Model and tokenizer instances.
    """
    logger.info(f"{'='*60}")
    logger.info(f"LOADING MODEL ARCHITECTURE: {model_id}")
    logger.info(f"{'='*60}")
    
    # Verify Flash Attention 2 package presence
    try:
        import flash_attn
        logger.info("Flash Attention 2 library detected. Supported layer mappings active.")
    except ImportError:
        logger.warning(
            "Flash Attention 2 package is not installed. "
            "Falling back to PyTorch native Scaled Dot-Product Attention (SDPA)."
        )
        
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    vram_before = get_vram_usage_gb()
    
    logger.info("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Default pad_token unset. Resolved pad = eos: '{tokenizer.eos_token}'")
        
    logger.info(f"Loading weights in precision type: {torch_dtype}...")
    start_time = time.perf_counter()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    # Support PyTorch Model compilation
    if compile_model:
        if hasattr(torch, "compile"):
            logger.info("Compiling model graph via torch.compile...")
            model = torch.compile(model)
            logger.info("PyTorch model compiled successfully.")
        else:
            logger.warning("torch.compile is not available in the current PyTorch setup.")
            
    load_time = time.perf_counter() - start_time
    vram_after = get_vram_usage_gb()
    
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    logger.info(f"\n📊 MODEL RUNTIME STATISTICS:")
    logger.info(f"   Total Parameters: {params['total']:,}")
    logger.info(f"   Memory Profile: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
    logger.info(f"   VRAM Allocation Change: +{vram_after - vram_before:.3f} GB")
    logger.info(f"   Loading Duration: {load_time:.2f}s")
    
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    strategy: str = "greedy",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Generates text outputs using specific decoding algorithms.
    
    DECODING STRATEGIES:
        1. GREEDY: Selects the absolute highest probability token next.
           - Highly consistent and reproducible.
           - Can lead to repetitive loops.
        2. SAMPLING: Introduces probabilistic variance based on temperature scaling.
        3. TOP-K: Truncates vocab choices to the top-K highest scoring tokens.
        4. TOP-P (Nucleus): Restricts selection to the smallest set of tokens 
           whose cumulative probability meets the threshold P.
           
    Args:
        model: Loaded language model.
        tokenizer: Tokenizer.
        prompt: Raw query input.
        strategy: Algorithm identifier ('greedy', 'sample', 'top_k', 'top_p').
        max_new_tokens: Token count limit to generate.
        temperature: Probabilistic randomness scaling.
        top_k: Top K choices cutoff.
        top_p: Nucleus scale threshold.
        
    Returns:
        str: Generated text output.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if strategy == "greedy":
        gen_kwargs["do_sample"] = False
    elif strategy == "sample":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    elif strategy == "top_k":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
    elif strategy == "top_p":
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = 0
    else:
        raise ValueError(f"Unknown generation decoding strategy: {strategy}")
        
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
        
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return generated_text


def demonstrate_decoding_strategies(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
):
    """
    Compares outputs generated using distinct decoding options side-by-side.
    """
    prompt = "The future of artificial intelligence in healthcare will"
    
    logger.info(f"\n{'='*60}")
    logger.info("DECODING STRATEGY COMPARISON")
    logger.info(f"Input Prompt: \"{prompt}\"")
    logger.info(f"{'='*60}")
    
    strategies = [
        ("greedy", {"temperature": 1.0}),
        ("sample", {"temperature": 0.3}),
        ("sample", {"temperature": 0.7}),
        ("sample", {"temperature": 1.5}),
        ("top_k", {"temperature": 0.7, "top_k": 50}),
        ("top_p", {"temperature": 0.7, "top_p": 0.9}),
    ]
    
    for strategy, kwargs in strategies:
        label = f"{strategy} (T={kwargs.get('temperature', 1.0)}"
        if "top_k" in kwargs:
            label += f", K={kwargs['top_k']}"
        if "top_p" in kwargs:
            label += f", P={kwargs['top_p']}"
        label += ")"
        
        text = generate_text(
            model, tokenizer, prompt,
            strategy=strategy,
            max_new_tokens=50,
            **kwargs,
        )
        logger.info(f"\n   📝 Strategy: {label}:")
        logger.info(f"      {text[:200]}...")


def main():
    logger.info("🚀 MODULE 01: BASIC INFERENCE PIPELINE RUN")
    logger.info("=" * 60)
    
    model, tokenizer = load_model(
        model_id="Qwen/Qwen2.5-1.5B",
        torch_dtype="bfloat16",
        compile_model=False,
    )
    
    prompt = "Explain what a Small Language Model is in simple terms:"
    text = generate_text(model, tokenizer, prompt, strategy="top_p", max_new_tokens=100)
    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Response: {text}")
    
    demonstrate_decoding_strategies(model, tokenizer)
    
    logger.info(f"\n{'='*60}")
    logger.info("LATENCY MEASUREMENT RUNS")
    logger.info(f"{'='*60}")
    
    latency = measure_inference_latency(
        model, tokenizer,
        prompt="Solve this: 2 + 2 = ",
        max_new_tokens=50,
        num_warmup=1,
        num_runs=3,
    )
    
    logger.info(f"\n📊 MEASURED INFERENCE PROFILE:")
    logger.info(f"   TTFT: {latency['ttft_ms']:.2f} ms")
    logger.info(f"   Throughput: {latency['throughput_tokens_per_sec']:.2f} tokens/s")
    logger.info(f"   Total Duration: {latency['total_time_ms']:.2f} ms")
    
    # Cleanup memory allocations
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.info(f"\n{'='*60}")
    logger.info("✅ MODULE 01 RUN COMPLETED!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
