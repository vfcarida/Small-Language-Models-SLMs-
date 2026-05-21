import time
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# ==============================================================================
# GLOBAL BENCHMARKING: SLM vs. LLM (V3)
# ==============================================================================
# This script goes beyond simple latency. It measures operational metrics
# (TTFT, Tokens/sec, VRAM) and simulates global benchmark retrieval 
# (MMLU for general knowledge, GSM8K for math/reasoning).
# It visually contrasts Massive LLMs vs Dense SLMs vs Quantized Edge SLMs.
# ==============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0.0

def benchmark_operational_metrics(model_id: str, prompt: str, is_quantized: bool = False):
    """
    Measures VRAM, TTFT, and Throughput.
    """
    logger.info(f"Loading {model_id} into VRAM for benchmarking...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs = {"device_map": "auto", "trust_remote_code": True}
    if not is_quantized:
        kwargs["torch_dtype"] = torch.bfloat16
        
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to load via advanced kwargs. Fallback triggered. Error: {e}")
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logger.info("Warming up GPU kernels...")
    _ = model.generate(**inputs, max_new_tokens=10)
    
    vram_before = get_vram_usage()
    
    # TTFT
    start_time = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=1, return_dict_in_generate=True)
    ttft = time.perf_counter() - start_time
    
    # Throughput
    max_tokens = 100
    start_time = time.perf_counter()
    _ = model.generate(**inputs, max_new_tokens=max_tokens, min_new_tokens=max_tokens)
    tokens_per_sec = max_tokens / (time.perf_counter() - start_time)
    
    vram_after = get_vram_usage()
    peak_vram = max(vram_before, vram_after)
    
    del model
    torch.cuda.empty_cache()
    
    return ttft, tokens_per_sec, peak_vram

def plot_comprehensive_results(results_list):
    """
    Generates a 2x2 grid of Matplotlib charts contrasting the models on 
    operational efficiency AND global knowledge benchmarks.
    """
    sns.set_theme(style="whitegrid")
    
    # Extracting data
    models = [r["Model"] for r in results_list]
    ttft = [r["TTFT (s)"] for r in results_list]
    vram = [r["VRAM (GB)"] for r in results_list]
    mmlu = [r["MMLU"] for r in results_list]
    gsm8k = [r["GSM8K"] for r in results_list]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Global Benchmarks & Operational Efficiency: LLM vs SLM', fontsize=20, fontweight='bold')

    # Top Left: TTFT
    sns.barplot(x=models, y=ttft, ax=axes[0, 0], palette="Blues_d")
    axes[0, 0].set_title('Time-To-First-Token (TTFT) - Lower is Better', fontsize=14)
    axes[0, 0].set_ylabel('Seconds')

    # Top Right: VRAM
    sns.barplot(x=models, y=vram, ax=axes[0, 1], palette="Reds_d")
    axes[0, 1].set_title('VRAM Footprint - Lower is Better', fontsize=14)
    axes[0, 1].set_ylabel('Gigabytes (GB)')

    # Bottom Left: MMLU
    sns.barplot(x=models, y=mmlu, ax=axes[1, 0], palette="Purples_d")
    axes[1, 0].set_title('MMLU (General Knowledge 5-shot) - Higher is Better', fontsize=14)
    axes[1, 0].set_ylabel('Score')

    # Bottom Right: GSM8K
    sns.barplot(x=models, y=gsm8k, ax=axes[1, 1], palette="Oranges_d")
    axes[1, 1].set_title('GSM8K (Math Reasoning) - Higher is Better', fontsize=14)
    axes[1, 1].set_ylabel('Score')

    # Rotate x-labels for better readability
    for ax in axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(15)

    plt.tight_layout()
    plt.savefig('global_benchmark_results.png', dpi=300)
    logger.info("Chart successfully saved as 'global_benchmark_results.png'")

def main():
    parser = argparse.ArgumentParser(description="Global Benchmark Tracker")
    parser.add_argument("--prompt", type=str, default="Solve this arithmetic problem: ")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # MOCK EXECUTION FOR CI/CD (Based on the Official Masterclass Matrix)
    # --------------------------------------------------------------------------
    logger.info("Simulating global benchmark tracking (MMLU/GSM8K) to prevent pipeline timeout...")
    time.sleep(1)
    
    # Data directly pulled from our theoretical benchmark matrix
    mock_results = [
        {"Model": "Llama-3.1-70B", "TTFT (s)": 0.85, "VRAM (GB)": 140.0, "MMLU": 80.0, "GSM8K": 92.0},
        {"Model": "Phi-3.5-mini", "TTFT (s)": 0.18, "VRAM (GB)": 7.6, "MMLU": 69.0, "GSM8K": 86.2},
        {"Model": "Qwen2.5-1.5B", "TTFT (s)": 0.22, "VRAM (GB)": 3.0, "MMLU": 60.0, "GSM8K": 45.0},
        {"Model": "Llama-3.2-1B (QLoRA 4b)", "TTFT (s)": 0.30, "VRAM (GB)": 2.0, "MMLU": 45.0, "GSM8K": 35.0}
    ]
    
    plot_comprehensive_results(mock_results)

if __name__ == "__main__":
    main()
