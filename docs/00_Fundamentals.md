# Module 00: Transformer Fundamentals, Tokenization & Grouped Query Attention

## 1. The Open-Source Small Language Model (SLM) Landscape

Small Language Models (SLMs)—typically spanning **1B to 4B parameters**—are purpose-built Transformer and hybrid architectures engineered to eliminate the quadratic memory and latency bottlenecks of frontier 70B–1T+ parameter Large Language Models (LLMs). By combining high-density synthetic pre-training data, Grouped Query Attention (GQA), and sub-4-bit post-training compression, modern SLMs achieve frontier-competitive accuracy on scoped enterprise tasks while running within **2 GB to 8 GB of VRAM**.

### Reference SLM Architecture Matrix

| Model Family | Parameters | Architectural Differentiators & Operational Profile | Primary Reference |
| :--- | :---: | :--- | :--- |
| **Qwen2.5-1.5B** | 1.54B | Grouped Query Attention (GQA) with 12 Query heads and 2 KV heads per layer; 151,936 BPE vocabulary; 32k–128k context length. Strong mathematical and coding inductive bias. | [Qwen Team (2024)](https://huggingface.co/Qwen) |
| **Llama-3.2-1B** | 1.23B | Edge-optimized architecture with tied input-output embeddings, 128k context window, and on-device pruning/distillation lineage from Llama-3.1-8B/70B. | [Meta AI (2024)](https://huggingface.co/meta-llama) |
| **Phi-3.5-mini** | 3.82B | Trained on textbook-grade synthetic reasoning corpora; supports 128k context and achieves 86.2% on GSM8K Chain-of-Thought benchmarks. | [Microsoft Research (2024)](https://huggingface.co/microsoft) |
| **Gemma-3-4B** | 4.00B | Interleaved local sliding-window and global attention layers for bounded KV-cache growth across long-document workloads. | [Google DeepMind (2025)](https://huggingface.co/google) |
| **DeepSeek-R1-Distill** | 1.5B | Distilled from DeepSeek-R1 reasoning trajectories (`arXiv:2501.12948`), emitting explicit `<think>` chains of thought for verifiable logical deduction. | [DeepSeek-AI (2025)](https://arxiv.org/abs/2501.12948) |

---

## 2. Subword Tokenization & Byte-Pair Encoding (BPE)

Before processing natural language or source code, a Transformer maps raw UTF-8 strings into discrete integer token IDs $x = (x_1, \dots, x_n) \in \{1, \dots, |V|\}^n$.

1. **Byte-Pair Encoding (Sennrich et al., ACL 2016):** Iteratively merges the most frequent adjacent byte or character pairs until reaching target vocabulary capacity $|V|$.
2. **Compression Efficiency & Sequence Cost:** Because self-attention scales as $\mathcal{O}(n^2)$ in sequence length $n$, a tokenizer with a higher character-per-token compression ratio directly reduces both Time-To-First-Token (TTFT) latency and KV-cache memory consumption.
3. **Control & Alignment Tokens:** Special tokens (`<|im_start|>`, `<|im_end|>`, `<|eot_id|>`) delineate system instructions, user turns, and assistant responses. Setting `pad_token = eos_token` when `pad_token` is undefined prevents batching failures during SFT and evaluation.

---

## 3. Scaled Dot-Product Attention & Grouped Query Attention (GQA)

Given input representations $X \in \mathbb{R}^{n \times d_{\text{model}}}$, Scaled Dot-Product Attention (Vaswani et al., NeurIPS 2017) projects inputs into Queries $Q = X W_Q$, Keys $K = X W_K$, and Values $V = X W_V$:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

### KV-Cache Memory Reduction via GQA (Ainslie et al., EMNLP 2023)

During autoregressive decoding, storing historical Key and Value tensors in the **KV-Cache** dominates VRAM bandwidth. For batch size $B$, sequence length $n$, $L$ transformer layers, $h_{kv}$ Key-Value heads, head dimension $d_h$, and $b$ bytes per element (2 bytes for FP16/BF16):

$$\text{Memory}_{\text{KV}} = 2 \cdot B \cdot L \cdot h_{kv} \cdot n \cdot d_h \cdot b$$

- **Multi-Head Attention (MHA, $h_{kv} = h_q = 32$):** Requires **128.00 MB** of KV-Cache for a 24-layer model at $n = 2048$.
- **Grouped Query Attention (GQA-2, $h_{kv} = 2$, e.g., Qwen2.5-1.5B):** Shares 1 Key/Value head across every 16 Query heads, compressing the KV-Cache by **16x** to **8.00 MB** with negligible perplexity degradation.

---

## 4. Implementation Reference

- **Tokenizer Explorer:** [`src/module_00_fundamentals/tokenizer_explorer.py`](../src/module_00_fundamentals/tokenizer_explorer.py)
- **Attention & GQA Visualizer:** [`src/module_00_fundamentals/attention_visualizer.py`](../src/module_00_fundamentals/attention_visualizer.py)
- **Unit Verification:** [`tests/test_fundamentals.py`](../tests/test_fundamentals.py)
