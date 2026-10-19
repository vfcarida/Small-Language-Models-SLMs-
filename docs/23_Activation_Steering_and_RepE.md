# Module 23: Activation Steering & Representation Engineering (`RepE` & `CAA`)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__23__representation__steering-green?style=for-the-badge)](../src/module_23_representation_steering/)

---

## 1. Executive & Architectural Motivation

Traditional alignment (`Module 06` DPO/SimPO) permanently modifies model weights, whereas system-prompt instructions consume precious context-window tokens and can be overridden by adversarial prompt injections. **Representation Engineering (`RepE`)** (**Zou et al., Center for AI Safety / CMU, 2023**, `arXiv:2310.01405`), **Contrastive Activation Addition (`CAA`)** (**Rimsky et al., ACL 2024**, `arXiv:2312.06681`), and **Directional Subspace Ablation** (**Arditi et al., NeurIPS 2024**, `arXiv:2406.11717`) operate directly on the linear geometry of the residual stream $\mathbf{h}^{(\ell)} \in \mathbb{R}^{d_{\text{model}}}$ at inference time—enabling instantaneous, zero-token, zero-finetuning behavioral control.

---

## 2. Mathematical Foundations

### 2.1 Extracting Linear Concept Directions (`CAA` & `RepE` Reading Vectors)

Given $N$ contrastive prompt pairs $\{(x_i^+, x_i^-)\}_{i=1}^N$ exemplifying a target attribute (e.g., *Truthful* vs. *Hallucinated*, or *Concise* vs. *Verbose*), let $\mathbf{h}_i^+, \mathbf{h}_i^- \in \mathbb{R}^{d_{\text{model}}}$ be the residual stream activations at layer $\ell$, and define the difference vectors $\mathbf{d}_i = \mathbf{h}_i^+ - \mathbf{h}_i^-$.

1. **Contrastive Activation Addition (`mean_diff`, Rimsky et al., ACL 2024)**:

$$
\mathbf{v}_{\text{CAA}} = \frac{1}{N} \sum_{i=1}^N (\mathbf{h}_i^+ - \mathbf{h}_i^-), \qquad \hat{\mathbf{v}} = \frac{\mathbf{v}_{\text{CAA}}}{\|\mathbf{v}_{\text{CAA}}\|_2}
$$

2. **Linear Artificial Tomography (`pca`, Zou et al., 2023)**:
   Computes the leading right singular vector $\hat{\mathbf{v}}$ of the difference matrix $\mathbf{D} = [\mathbf{d}_1^\top; \dots; \mathbf{d}_N^\top] \in \mathbb{R}^{N \times d_{\text{model}}}$:

$$
\hat{\mathbf{v}} = \arg\max_{\|\mathbf{v}\|_2 = 1} \sum_{i=1}^N (\mathbf{d}_i^\top \mathbf{v})^2
$$

### 2.2 Norm-Preserving Activation Steering & Orthogonal Subspace Ablation

- **Norm-Preserving Additive Steering (`apply_activation_steering`)**:
  Adds $\alpha \hat{\mathbf{v}}$ and rescales back to the original token norm $\|\mathbf{h}_t\|_2$ so downstream `RMSNorm` layers remain calibrated:

$$
\tilde{\mathbf{h}}_t = (\mathbf{h}_t + \alpha \hat{\mathbf{v}}) \cdot \frac{\|\mathbf{h}_t\|_2}{\|\mathbf{h}_t + \alpha \hat{\mathbf{v}}\|_2}
$$

- **Orthogonal Rank-1 Subspace Ablation (`ablate_concept_subspace`, Arditi et al., NeurIPS 2024)**:
  Projects out all variance along an undesirable concept direction $\hat{\mathbf{v}}$:

$$
\mathbf{h}_t^{\perp} = \left(\mathbf{I} - \hat{\mathbf{v}}\hat{\mathbf{v}}^\top\right) \mathbf{h}_t = \mathbf{h}_t - (\mathbf{h}_t^\top \hat{\mathbf{v}})\hat{\mathbf{v}} \implies \langle \mathbf{h}_t^{\perp}, \hat{\mathbf{v}} \rangle = 0
$$

---

## 3. Reference Implementation

- **[`src/module_23_representation_steering/activation_steering.py`](../src/module_23_representation_steering/activation_steering.py)**:
  - `extract_steering_vector`: Supports `"pca"` (RepE SVD reading vector) and `"mean_diff"` (CAA), recovering synthetic ground-truth directions with **>0.996 cosine similarity**.
  - `apply_activation_steering`: Norm-preserving residual stream steering.
  - `ablate_concept_subspace`: Exact rank-1 orthogonal projection with residual dot product $< 10^{-6}$.

```bash
python src/module_23_representation_steering/activation_steering.py
```

---

## 4. Academic References

1. **Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., et al.** (2023). *Representation Engineering: A Top-Down Approach to AI Transparency (RepE).* (`arXiv:2310.01405`).
2. **Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M.** (2024). *Steering Llama 2 via Contrastive Activation Addition (CAA).* ACL 2024 (`arXiv:2312.06681`).
3. **Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N.** (2024). *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024 (`arXiv:2406.11717`).
