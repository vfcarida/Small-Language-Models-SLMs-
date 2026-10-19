# Module 16: Latent-Space Continuous Reasoning (`Coconut` & Recurrent-Depth SLMs)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__16__latent__reasoning-green?style=for-the-badge)](../src/module_16_latent_reasoning/)

---

## 1. Executive & Architectural Motivation

While Reinforcement Learning with Verifiable Rewards (RLVR / GRPO, Module 11) equips Small Language Models with test-time reasoning via explicit `<think>...</think>` Chain-of-Thought (CoT) tokens, discrete token-level reasoning suffers from two fundamental bottlenecks on edge hardware:
1. **The Vocabulary Information Bottleneck**: Every intermediate step is forced through a discrete one-hot token sample $y_t \in \{1, \dots, |V|\}$ ($\sim 17\text{ bits}$), collapsing the rich superposition of alternative reasoning hypotheses encoded in the continuous hidden state $\mathbf{h}_t \in \mathbb{R}^{d_{\text{model}}}$ ($\sim 24,576\text{ bits}$ in `FP16`).
2. **Autoregressive KV-Cache & Bandwidth Inflation**: Generating $256\text{–}1024$ intermediate `<think>` tokens requires $256\text{–}1024$ sequential memory-bandwidth-bound decoding steps and expands the KV-cache linearly.

**Coconut (Chain of Continuous Thought)** (**Hao et al., Meta FAIR 2024**, `arXiv:2412.06769`) and **Recurrent-Depth Latent Reasoning** (**Geiping et al., 2025**, `arXiv:2502.05171`) bypass discrete token emission during the reasoning phase by feeding the continuous hidden state $\mathbf{h}^{(k)}$ back into the recurrent block for $K \ll L_{\text{CoT}}$ steps before decoding the final answer.

---

## 2. Mathematical Foundations

### 2.1 Continuous Thought Recurrence Conditioned on the Input Anchor

Given the final prompt representation $\mathbf{h}^{(0)} \in \mathbb{R}^{d_{\text{model}}}$, step $k \in \{1, \dots, K_{\max}\}$ updates the latent thought vector via damped residual refinement:

$$
\mathbf{h}^{(k)} = \text{LayerNorm}\!\left(\mathbf{h}^{(k-1)} + \alpha_k \cdot \Phi_{\text{thought}}\!\left([\mathbf{h}^{(k-1)} \mathbin{\Vert} \mathbf{h}^{(0)}]\right)\right), \qquad \alpha_k = k^{-3/2}
$$

Maintaining direct concatenation with the input anchor $\mathbf{h}^{(0)}$ prevents latent drift across deep recurrences (`Geiping et al., 2025`).

### 2.2 Adaptive Pondering Gate via Step-to-Step Symmetric KL Convergence

Rather than executing a fixed number of latent steps $K_{\max}$ for trivial queries, `AdaptivePonderingGate` monitors the predictive distribution $\mathbf{p}^{(k)} = \text{softmax}(\mathbf{W}_{\text{head}} \mathbf{h}^{(k)})$ across consecutive latent iterations using the **Symmetric Kullback-Leibler Divergence** (Jeffreys Divergence):

$$
D_{\text{sym}}\!\left(\mathbf{p}^{(k)} \parallel \mathbf{p}^{(k-1)}\right) = \frac{1}{2} \left[ D_{\text{KL}}\!\left(\mathbf{p}^{(k)} \parallel \mathbf{p}^{(k-1)}\right) + D_{\text{KL}}\!\left(\mathbf{p}^{(k-1)} \parallel \mathbf{p}^{(k)}\right) \right]
$$

Recurrence halts immediately at step $k^*$ once $D_{\text{sym}}(\mathbf{p}^{(k^*)} \parallel \mathbf{p}^{(k^*-1)}) < \tau$.

---

## 3. Reference Implementation

- **[`src/module_16_latent_reasoning/coconut_reasoning.py`](../src/module_16_latent_reasoning/coconut_reasoning.py)**:
  - `AdaptivePonderingGate`: Computes step-to-step symmetric KL divergence for dynamic early halting.
  - `LatentThoughtReasoner`: Continuous latent recurrence module returning `(final_logits, final_thought_state, LatentReasoningTrace)`.
  - `compare_discrete_cot_vs_latent_coconut`: Analytical KV-cache and latency profiler demonstrating **6.0x KV-cache reduction** and **7.27x decoding speedup** when replacing 256 discrete CoT tokens with 16 latent Coconut steps.

```bash
python src/module_16_latent_reasoning/coconut_reasoning.py
```

---

## 4. Academic References

1. **Hao, S., Sukhbaatar, S., Su, D., Li, X., Hu, Z., Weston, J., & Yu, T.** (2024). *Training Large Language Models to Reason in a Continuous Latent Space (Coconut).* Meta FAIR (`arXiv:2412.06769`).
2. **Geiping, J., McLeish, S., Jain, N., Kirchenbauer, J., et al.** (2025). *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.* (`arXiv:2502.05171`).
