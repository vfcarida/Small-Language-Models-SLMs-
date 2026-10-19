# Module 10 — Lossless Speculative Decoding & Medusa Multi-Token Draft Heads

> **Source Implementation:** [`src/module_10_speculative/speculative_decoding.py`](../src/module_10_speculative/speculative_decoding.py)  
> **Verification Suite:** [`tests/test_speculative_decoding.py`](../tests/test_speculative_decoding.py)

---

## 1. Scientific Foundation: SLMs as Universal Lossless Accelerators

Autoregressive generation in Large Language Models is fundamentally **memory-bandwidth bound**: generating $K$ tokens sequentially requires reading all model weights from HBM $K$ times. **Speculative Decoding** ([Leviathan et al., ICML 2023](https://arxiv.org/abs/2211.17192); [Chen et al., DeepMind 2023](https://arxiv.org/abs/2302.01318)) uses a compact, quantized Small Language Model (`0.5B–1.5B` parameters) as a fast **Draft Model** $q(x)$ to propose $K$ candidate tokens $\tilde{x}_1, \dots, \tilde{x}_K$, which are then verified in **a single parallel forward pass** of the target model $p(x)$.

### Exact Speculative Rejection Sampling & Residual Identity

For each lookahead position $i \in \{1, \dots, K\}$:
1. **Acceptance Rule:** Accept draft token $\tilde{x}_i \sim q_i(x)$ with probability:
   $$\alpha_i(\tilde{x}_i) = \min\!\left(1, \frac{p_i(\tilde{x}_i)}{q_i(\tilde{x}_i)}\right)$$
2. **Residual Resampling on Rejection:** At the first rejected position $i$, sample a replacement token $x'_i$ from the normalized residual distribution:
   $$p^{\text{res}}_i(x) = \frac{\max\big(0, p_i(x) - q_i(x)\big)}{\sum_{v=1}^{|V|} \max\big(0, p_i(v) - q_i(v)\big)}$$
3. **Marginal Distribution Identity Theorem:**
   $$P(X = x) = q(x)\min\!\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_{v} \min(q(v), p(v))\right) p^{\text{res}}(x) = \min(q(x), p(x)) + \max(0, p(x) - q(x)) = p(x)$$
   Verified numerically in [`verify_marginal_distribution_identity`](../src/module_10_speculative/speculative_decoding.py) (`max_abs_error < 1e-6`).

---

## 2. Wall-Clock Speedup Law & Medusa Multi-Horizon Heads

Given mean acceptance rate $\alpha \in [0, 1]$, lookahead $K$, and draft-to-target latency ratio $c = T_{\text{draft}} / T_{\text{target}}$:

$$\mathbb{E}[\text{tokens per step}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}, \quad \text{Speedup}(K, \alpha, c) = \frac{1 - \alpha^{K+1}}{(1 - \alpha)(1 + K \cdot c)}$$

When a separate draft model is undesirable, [`MedusaDraftHead`](../src/module_10_speculative/speculative_decoding.py) ([Cai et al., ICML 2024](https://arxiv.org/abs/2401.10774)) attaches $K$ lightweight residual SiLU blocks ($h^{(k)} = h + W_k \text{SiLU}(h)$) directly to the SLM's final hidden state to predict $K$ future positions in parallel.

---

## 3. Quickstart Usage

```bash
python src/module_10_speculative/speculative_decoding.py
```
