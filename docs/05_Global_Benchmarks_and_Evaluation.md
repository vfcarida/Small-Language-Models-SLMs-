# 5. Global Benchmarks and Evaluation (SLM vs LLM)

The definitive analysis of corporate value depends on comparing structured data across architectures. The table below systematizes tests focused on crossing Large Models (Llama 3.1 70B baseline) with previously indicated compact SLMs, evaluated in restricted computational environments with base BF16 precisions against AWQ/QLoRA INT4.

## Comparative Benchmark Matrix

| Parameter / Metric | LLM Generic (Baseline: Llama-3.1-70B Instruct) | SLM 1: Llama-3.2-1B (BF16 Baseline) | SLM 1.1: Llama-3.2-1B (SpinQuant / QLoRA 4-bit) | SLM 2: Phi-3.5-mini-instruct (3.8B) | SLM 3: Qwen2.5-1.5B |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VRAM Footprint** | > 140 GB | 3.18 GB | 1.9 GB - 2.2 GB | ~ 7.6 GB | ~ 3 GB |
| **Decode Rate (Tokens/sec)** | ~ 20 to 35 t/s (Heavy, multicluster) | 19.2 t/s (Restricted CPU/Edge env) | 45.8 to 50.2 t/s (Accelerated, ~2.5x boost) | High (Optimized Flash-Attn vLLM) | High (Supports Flash-Attn) |
| **Time to First Token (TTFT)** | Variation (> 500ms usual) | 1.0 s | 0.3 s (-76% real-time response gain) | Ultra-low (< 200ms) | Ultra-low |
| **MMLU (General Knowledge - 5 shot)** | ~ 80.0+ | N/A (Edge Focus) | N/A | 69.0 | 52.4 to 69.0 |
| **Math (GSM8K CoT)** | ~ 92.0+ | N/A | N/A | 86.2 | > 40.0 (Can surpass depending on split) |
| **Context Length (Max)** | 128k Tokens | 128k Tokens | 8k Tokens (Technical limitation of quantized compression) | 128k Tokens | 32k to 128k |
| **Analytical Operation Cost** | Very High | Extremely Low | Almost negligible (Sub 10W Power Draw) | Low | Low |

## Interpretation of Results and Feasibility Discussion

The consolidated results conclusively prove that the structural conversion of LLMs to SLMs does not lethally affect operational proficiency in delimited tasks. 

When Llama-3.2-1B undergoes compression quantization (SpinQuant and QLoRA), the memory footprint is decimated to **less than 2 gigabytes**, enabling invisible offline installation on practically every active global smartphone. This comes at the virtually imperceptible practical cost of reducing the context window from 128k to 8k tokens. Simultaneously, TTFT indices collapse by a staggering **76%**, enabling interfaces with the ideal perceptive instantaneity required for predictive daily automatic functionalities (Smart Replies, live categorization in POS terminals).

On the other hand, in corporate scopes, Phi-3.5-mini shocks traditional analytical expectations by reaching a score of **86.2** in the dense GSM8K challenge (mathematical performance with chain of thought), frighteningly approaching the artificial intelligence of trillion-parameter LLMs while spending less than one-hundredth of the energetic processing cost.

### LGPD and Deterministic Scopes in Banking
The primary negative point found in comparative tests is restricted to generalist and cross-domain tasks (where SLMs suffer from vocabulary width limitations and semantic "overfitting" outside the trained scope). 

However, in financial contexts where predictability (**deterministic scope**) is required for LGPD (General Data Protection Law) and banking security, the systemic deficiency in generic capability is actually a highly valuable safeguard inherent to the specialized SLM. It prevents the model from answering out-of-domain questions or hallucinating general advice, keeping it strictly locked to its financial orchestration tasks.
