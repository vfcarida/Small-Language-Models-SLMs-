# Module 20: Mixture-of-Depths (`MoD`) & Dynamic Token Routing for SLMs

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__20__mixture__of__depths-green?style=for-the-badge)](../src/module_20_mixture_of_depths/)

---

## 1. Executive & Architectural Motivation

Standard Transformer decoders apply identical per-token floating-point computation (FLOPs) across every layer $\ell \in \{1, \dots, L\}$, regardless of whether a token represents trivial boilerplate syntax (`"the"`, `","`, whitespace) or a pivotal mathematical deduction. **Mixture-of-Depths (`MoD`)** (**Raposo et al., Google DeepMind 2024**, `arXiv:2404.02258`) introduces **static-graph dynamic token routing**: each MoD layer enforces a hard capacity factor $\gamma \in (0, 1]$ (e.g., $\gamma = 0.25$) so that **only the top $K = \lfloor \gamma L \rfloor$ tokens** scored by a lightweight linear router execute the heavy Attention/FFN sub-block, while the remaining $(1 - \gamma)L$ tokens bypass the layer along the residual identity stream.

Because $K$ is known statically prior to tensor allocation, MoD achieves **deterministic memory shapes and up to 80% FLOPs reduction** without dynamic graph recompilation or load-balancing auxiliary penalties.

---

## 2. Mathematical Foundations

### 2.1 Capacity-Bounded Top-$K$ Token Gather & Residual Scatter

Given sequence representations $\mathbf{X} \in \mathbb{R}^{B \times L \times d_{\text{model}}}$ at layer $\ell$, a scalar routing projection $\mathbf{w}_r \in \mathbb{R}^{d_{\text{model}}}$ computes token importance weights:

$$
r_{b, t} = \sigma\!\left(\mathbf{w}_r^\top \mathbf{x}_{b, t}\right) \in (0, 1)
$$

Let $P_\gamma(\mathbf{r}_b)$ denote the $(1 - \gamma)$-percentile threshold across sequence positions $t \in \{1, \dots, L\}$ (selecting exactly $K = \max(1, \lfloor \gamma L \rfloor)$ indices $\mathcal{I}_b$). The output at position $t$ is:

$$
\mathbf{y}_{b, t} = \begin{cases} \mathbf{x}_{b, t} + r_{b, t} \cdot f_{\text{block}}\!\left(\tilde{\mathbf{X}}_b\right)_t & \text{if } t \in \mathcal{I}_b \\ \mathbf{x}_{b, t} & \text{if } t \notin \mathcal{I}_b \end{cases}
$$

Multiplying $f_{\text{block}}(\cdot)$ by $r_{b, t}$ places the router weight on the differentiable gradient path $\frac{\partial \mathbf{y}_{b, t}}{\partial \mathbf{w}_r}$, training $\mathbf{w}_r$ end-to-end via standard language modeling backpropagation.

### 2.2 Causal Auxiliary Predictor for $O(1)$ Autoregressive Sampling

During training, $\mathcal{I}_b$ is determined via sequence-wide `Top-K`, which depends on future tokens $t' > t$. To preserve strict causality during token-by-token autoregressive inference, a lightweight auxiliary binary classifier $a_{b, t} = \mathbf{w}_{\text{aux}}^\top \text{sg}(\mathbf{x}_{b, t}) + b_{\text{aux}}$ is trained via Binary Cross-Entropy (BCE) to predict Top-$K$ membership $m_{b, t} = \mathbb{I}[t \in \mathcal{I}_b]$:

$$
\mathcal{L}_{\text{aux}} = -\frac{1}{B L} \sum_{b=1}^B \sum_{t=1}^L \left[ m_{b, t} \log \sigma(a_{b, t}) + (1 - m_{b, t}) \log(1 - \sigma(a_{b, t})) \right]
$$

---

## 3. Reference Implementation

- **[`src/module_20_mixture_of_depths/mod_routing.py`](../src/module_20_mixture_of_depths/mod_routing.py)**:
  - `MixtureOfDepthsBlock`: Implements Top-$K$ gather, heavy sub-block execution, router-weighted residual scatter-add, and stop-gradient causal auxiliary logits.
  - `compute_mod_auxiliary_loss`: Binary cross-entropy objective for causal threshold routing.
  - `profile_mod_compute_savings`: Analytical FLOPs profiler demonstrating **80.8% FLOPs reduction (5.20x speedup)** at $\gamma = 0.25$ on a 28-layer 1.5B SLM at $4\text{k}$ context length.

```bash
python src/module_20_mixture_of_depths/mod_routing.py
```

---

## 4. Academic References

1. **Raposo, D., Ritter, S., Richards, B., Lillicrap, T., Humphreys, P. C., & Santoro, A.** (2024). *Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.* Google DeepMind (`arXiv:2404.02258`).
2. **Liu, Z., Wang, J., Dao, T., et al.** (2023). *DejaVu: Contextual Sparsity for Efficient LLMs at Inference Time.* ICML 2023 (`arXiv:2310.17157`).
