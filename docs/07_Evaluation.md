# Module 07: Evaluation, Perplexity & Global Benchmarks

## 1. Intrinsic Information-Theoretic Evaluation: Sliding-Window Perplexity

Perplexity ($\text{PPL}$) quantifies how well a causal language model predicts a held-out token sequence $X = (x_1, \dots, x_N)$:

$$\text{PPL}(X) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i}) \right)$$

For sequences exceeding a model's context window $L_{\max}$, `calculate_perplexity` (`src/utils/metrics.py`) implements an overlapping sliding-window evaluation with stride $S < L_{\max}$, masking the prefix context tokens (`target_ids[:, :-target_len] = -100`) so each token's negative log-likelihood is conditioned on up to $L_{\max} - S$ tokens of prior context without boundary artifacts.

---

## 2. Standardized Extrinsic Benchmarks (`EleutherAI lm-evaluation-harness`)

Production SLM readiness requires standardized, reproducible task evaluation rather than ad-hoc qualitative inspection:

1. **MMLU (Massive Multitask Language Understanding — Hendrycks et al., ICLR 2021):** Evaluates breadth of world and professional knowledge across 57 subjects (STEM, humanities, social sciences, law).
2. **GSM8K (Grade School Math 8K — Cobbe et al., 2021):** Evaluates multi-step arithmetic and logical reasoning via Chain-of-Thought (CoT) exact match. Sensitive indicator of whether pruning or INT4 quantization has damaged compositional reasoning.

---

## 3. Global Comparative Feasibility Matrix (LLM vs. Compressed SLMs)

| Metric / Dimension | Frontier Baseline (`Llama-3.1-70B`) | Dense SLM (`Llama-3.2-1B` BF16) | Quantized Edge SLM (`Llama-3.2-1B` 4-Bit) | Reasoning SLM (`Phi-3.5-mini` 3.8B) | Core Reference SLM (`Qwen2.5-1.5B`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VRAM Footprint** | $>140.0\text{ GB}$ | $3.18\text{ GB}$ | **$1.9\text{ GB} - 2.2\text{ GB}$** | $\sim 7.6\text{ GB}$ | $\sim 3.0\text{ GB}$ |
| **Decode Throughput** | $20 - 35\text{ tok/s}$ | $19.2\text{ tok/s}$ (CPU/Edge) | **$45.8 - 50.2\text{ tok/s}$** | High (FlashAttention) | High (FlashAttention) |
| **TTFT Latency** | $>500\text{ ms}$ | $1000\text{ ms}$ | **$300\text{ ms}$ ($-70\%$)** | $<200\text{ ms}$ | $<200\text{ ms}$ |
| **MMLU (5-Shot)** | $\sim 80.0+$ | Edge Scoped | Edge Scoped | **$69.0$** | **$52.4 - 60.0$** |
| **GSM8K (CoT)** | $\sim 92.0+$ | Domain Scoped | Domain Scoped | **$86.2$** | $>40.0$ |
| **Power & TCO Profile** | Multi-GPU Cluster | Low | **Negligible ($<10\text{ W}$)** | Single GPU | Single GPU |

---

## 4. Implementation Reference

- **EleutherAI Harness Runner:** [`src/module_07_evaluation/benchmark_runner.py`](../src/module_07_evaluation/benchmark_runner.py)
- **Perplexity & Latency Profiler:** [`src/module_07_evaluation/perplexity_eval.py`](../src/module_07_evaluation/perplexity_eval.py)
- **Loss & Radar Chart Visualizer:** [`src/module_07_evaluation/results_visualizer.py`](../src/module_07_evaluation/results_visualizer.py)
- **Operational Benchmark Tracker:** [`eval/benchmarks/latency_tracker.py`](../eval/benchmarks/latency_tracker.py)
- **Unit Verification:** [`tests/test_evaluation_and_pipeline.py`](../tests/test_evaluation_and_pipeline.py)
