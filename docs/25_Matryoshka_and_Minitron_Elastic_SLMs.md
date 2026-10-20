# Module 25: Matryoshka Representation Learning (MRL) & Elastic Sub-Network Slicing

> **Module Path:** `src/module_25_matryoshka_minitron/elastic_matryoshka.py`  
> **Primary References:**  
> - Kusupati et al. (NeurIPS 2022), *"Matryoshka Representation Learning"*, `arXiv:2205.13147`.  
> - Muralidharan et al. (NeurIPS 2024 / NVIDIA), *"Compact Language Models via Pruning and Knowledge Distillation (Minitron)"*, `arXiv:2407.14679`.

---

## 1. Multi-Granularity Matryoshka Representation Learning (MRL)

Standard Transformer embeddings diffuse semantic information uniformly across all $d$ coordinates. If an edge runtime truncates the hidden state to $d' < d$ to save memory bandwidth or FLOPs, accuracy collapses.

**Matryoshka Representation Learning (`compute_matryoshka_loss`)** explicitly optimizes a nested hierarchy of prefix dimensions $\mathcal{M} = \{d_1, d_2, \dots, d_k\}$ (e.g., $\{64, 128, 256, 512\}$) simultaneously during training:

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in \mathcal{M}} c_m \cdot \mathcal{L}_{\text{CE}}\!\left(\mathbf{h}_{:, :m} \mathbf{W}_{:, :m}^\top + \mathbf{b},\; \mathbf{y}\right)$$

Because the coarse prefix $\mathbf{h}_{:, :d_1}$ participates in every nested loss term, the optimizer packs the highest-variance principal semantic features into the earliest coordinates.

---

## 2. NVIDIA Minitron Activation & Depth Importance Ranking

When compressing an existing pre-trained SLM without training from scratch, **Minitron (`rank_minitron_structural_importance`)** estimates structural importance from a small calibration pass:

1. **Width / Channel Importance:**
   $$\mathcal{I}_{\text{channel}}(j) = \frac{1}{B \cdot T} \sum_{b=1}^{B} \sum_{t=1}^{T} |X_{b, t, j}|$$
   Reordering weight columns by descending $\mathcal{I}_{\text{channel}}(j)$ turns any pre-trained linear layer into an ordered Matryoshka layer.
2. **Depth / Block Influence ($\text{BI}_\ell$):**
   $$\text{BI}_\ell = 1 - \mathbb{E}_{b, t}\!\left[\frac{\langle \mathbf{X}_{\ell, b, t}^{\text{in}}, \mathbf{X}_{\ell, b, t}^{\text{out}} \rangle}{\|\mathbf{X}_{\ell, b, t}^{\text{in}}\|_2 \|\mathbf{X}_{\ell, b, t}^{\text{out}}\|_2}\right]$$
   Layers whose output is nearly collinear with their input ($\text{BI}_\ell \approx 0$) contribute minimal representation transformation and can be bypassed first under latency constraints.

---

## 3. Dynamic Runtime Slicing (`ElasticMatryoshkaLinear`)

`ElasticMatryoshkaLinear` permutes its weight columns once via `reorder_by_importance(channel_ranking)` and then supports instantaneous execution at any `active_in_features` $m \le d_{\text{in}}$ with zero duplicate checkpoint overhead.
