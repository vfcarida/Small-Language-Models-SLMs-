# Module 30: Differential Transformer (`DiffAttn`) for Attention-Noise Cancellation

> **Module Path:** `src/module_30_differential_attention/diff_attention.py`  
> **Primary Reference:**  
> - Ye et al. (Microsoft Research / ICLR 2025), *"Differential Transformer"*, `arXiv:2410.05258`.

---

## 1. Cancelling Common-Mode Context Noise in SLMs

Standard softmax attention assigns non-zero probability mass to irrelevant distractor tokens across long contexts, diluting needle retrieval and inducing hallucinations in Small Language Models. Inspired by noise-cancelling differential amplifiers, **Differential Attention (`DifferentialAttention`)** computes the difference between two separate softmax attention maps:

$$\text{DiffAttn}(X) = \left(\text{softmax}\!\left(\frac{Q_1 K_1^\top}{\sqrt{d_h}}\right) - \lambda\,\text{softmax}\!\left(\frac{Q_2 K_2^\top}{\sqrt{d_h}}\right)\right) V$$

where the learnable scalar $\lambda$ is reparameterized for training stability around a depth-scheduled initializer $\lambda_{\text{init}}(\ell) = 0.8 - 0.6 \exp(-0.3 \ell)$:

$$\lambda = \exp(\boldsymbol{\lambda}_{q_1} \cdot \boldsymbol{\lambda}_{k_1}) - \exp(\boldsymbol{\lambda}_{q_2} \cdot \boldsymbol{\lambda}_{k_2}) + \lambda_{\text{init}}(\ell)$$
