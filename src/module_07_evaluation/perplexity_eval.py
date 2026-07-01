"""
==============================================================================
MODULE 07 — PERPLEXITY AND LATENCY EVALUATION RUNS
==============================================================================
LEVEL: 🟡 Intermediate
ESTIMATED TIME: 15 minutes

OBJECTIVE:
    Learn how to calculate and observe model perplexity scores on target text segments
    and evaluate processing speed metrics (TTFT, throughput) under controlled settings.

CONCEPTS COVERED:
    1. Cross-Entropy Loss over sequences.
    2. Sliding Window perplexity algorithms.
    3. TTFT (Time-To-First-Token) calculations.

REFERENCES:
    - Jurafsky & Martin, "Speech and Language Processing", 2024
    - HuggingFace Evaluate Library

RUN COMMAND:
    python src/module_07_evaluation/perplexity_eval.py
==============================================================================
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.metrics import calculate_perplexity, measure_inference_latency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("MODULE 07: PERPLEXITY AND LATENCY ASSESSMENT RUN")
    logger.info("=" * 60)
    
    model_id = "Qwen/Qwen2.5-1.5B"
    
    logger.info(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Text validation sequences
    simple_text = "The sky is blue and the grass is green. The sun shines bright during the day."
    dense_text = "Quantum mechanics describes the superposition of eigenstates under Hamiltonian systems in Hilbert spaces."
    gibberish_text = "Shoes fly because pineapple does not understand mathematical quantum tires."
    
    logger.info("\n📊 ASSESSING PERPLEXITY (Lower = better; 1.0 represents theoretical perfection):")
    
    test_cases = [
        ("Simple", simple_text),
        ("Dense", dense_text),
        ("Gibberish", gibberish_text)
    ]
    
    for label, text in test_cases:
        ppl = calculate_perplexity(model, tokenizer, text)
        logger.info(f"   [{label} Case]: PPL = {ppl:.2f} | Context excerpt: '{text[:40]}...'")
        
    logger.info("\n   Observation: Standard structured language sequences result in lower perplexity scores.")
    logger.info("   Out-of-distribution or random token assemblies result in high scores.")
    
    logger.info("\n⏱️ MEASURING LATENCY AND THROUGHPUT PERFORMANCE:")
    prompt = "Explain the core concepts of general relativity in three paragraphs:"
    
    metrics = measure_inference_latency(
        model, tokenizer, prompt, max_new_tokens=128, num_warmup=1, num_runs=3
    )
    
    logger.info(f"   Prompt: {prompt}")
    logger.info(f"   Time To First Token (TTFT): {metrics['ttft_ms']:.1f} ms")
    logger.info(f"   Throughput Speed: {metrics['throughput_tokens_per_sec']:.1f} tokens/s")
    
    if metrics['throughput_tokens_per_sec'] < 20:
        logger.info("   💡 Optimization notice: Throughput speeds below 20 tokens/sec can impact interactive chatbots.")
        logger.info("      Consider deploying INT4 models (AWQ) or using accelerated serving packages (vLLM).")


if __name__ == "__main__":
    main()
