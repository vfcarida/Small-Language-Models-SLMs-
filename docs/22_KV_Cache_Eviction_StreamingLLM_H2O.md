# Module 22: Long-Context KV-Cache Eviction (`StreamingLLM` Attention Sinks & `H2O` Oracle)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__22__kv__cache__eviction-green?style=for-the-badge)](../src/module_22_kv_cache_eviction/)

---

## 1. Executive & Architectural Motivation

When a Small Language Model processes multi-hour streaming logs, voice transcripts, or autonomous agent trajectories ($L > 100\text{k}$ tokens) on fixed-SRAM edge hardware, the KV-cache cannot grow unboundedly. However, naive **Sliding-Window Eviction** (discarding all tokens older than the last $W$ positions) causes catastrophic perplexity collapse the moment the very first token $t=0$ is evicted.

Module 22 unifies two landmark discoveries in long-context inference:
1. **Attention Sinks (`StreamingLLM`, Xiao et al., MIT Han Lab / ICLR 2024, `arXiv:2309.17453`)**: Because Softmax requires attention weights across all visible tokens to sum to $1.0$, autoregressive models learn to dump surplus attention mass onto the initial $N_{\text{sink}} \approx 4$ tokens even when those tokens are semantically irrelevant. Retaining these initial sink tokens stabilizes sliding-window attention over millions of tokens.
2. **Heavy-Hitter Oracle (`H2O`, Zhang et al., NeurIPS 2023, `arXiv:2306.14048`)**: Attention matrices in language models are $>95\%$ sparse; a small subset of past tokens ("Heavy Hitters") receives the vast majority of cumulative attention mass. Retaining the top $N_{\text{hh}}$ heavy-hitters alongside $N_{\text{sink}}$ sinks and $N_{\text{recent}}$ local tokens preserves long-range retrieval under a strict $O(1)$ cache budget.

---

## 2. Mathematical Foundations

### 2.1 Why Softmax Creates Attention Sinks

At decoding step $t$, attention weights over retained indices $\mathcal{C}_t$ are:

$$
A_{t, j} = \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_j / \sqrt{d_k})}{\sum_{u \in \mathcal{C}_t} \exp(\mathbf{q}_t^\top \mathbf{k}_u / \sqrt{d_k})}
$$

When token $t$ does not strongly match any token in the recent window, a model trained with causal masking relies on the always-visible initial tokens $u \in \{0, \dots, N_{\text{sink}}-1\}$ having large positive $\mathbf{q}_t^\top \mathbf{k}_u$ logits in the denominator. Evicting $\{0, \dots, N_{\text{sink}}-1\}$ shrinks the denominator and redistributes uniform noise across recent values $\mathbf{v}_j$.

### 2.2 Cumulative Heavy-Hitter Eviction Policy

For every token $j \in \{0, \dots, t\}$, define the cumulative attention score:

$$
s_j^{(t)} = \sum_{\tau = j}^t A_{\tau, j}
$$

At any step $t > B_{\max} = N_{\text{sink}} + N_{\text{hh}} + N_{\text{recent}}$, the retained index set $\mathcal{C}_t$ is partitioned into three disjoint regions:

$$
\mathcal{C}_t = \underbrace{\{0, \dots, N_{\text{sink}}-1\}}_{\text{Attention Sinks}} \cup \underbrace{\text{Top-}N_{\text{hh}}\!\left(\{s_j^{(t)}\}_{j=N_{\text{sink}}}^{t - N_{\text{recent}} - 1}\right)}_{\text{Heavy-Hitter Anchors}} \cup \underbrace{\{t - N_{\text{recent}}, \dots, t - 1\}}_{\text{Local Sliding Window}}
$$

---

## 3. Reference Implementation

- **[`src/module_22_kv_cache_eviction/kv_eviction_policy.py`](../src/module_22_kv_cache_eviction/kv_eviction_policy.py)**:
  - `H2OStreamingKVCache`: Computes the exact retained index set $\mathcal{C}_t$ of bounded cardinality $|\mathcal{C}_t| \le B_{\max}$.
  - `compare_kv_eviction_fidelity`: Demonstrates that `Sink + H2O + Recent` reduces attention output $L_2$ reconstruction error from `9.4959` (Naive Sliding Window) down to `< 1e-6` at a **12.8x KV-cache compression ratio**.

```bash
python src/module_22_kv_cache_eviction/kv_eviction_policy.py
```

---

## 4. Academic References

1. **Xiao, G., Tian, Y., Chen, B., Han, S., & Lewis, M.** (2024). *Efficient Streaming Language Models with Attention Sinks (StreamingLLM).* MIT Han Lab / ICLR 2024 (`arXiv:2309.17453`).
2. **Zhang, Z., Sheng, Y., Zhou, T., Chen, T., et al.** (2023). *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models.* NeurIPS 2023 (`arXiv:2306.14048`).
