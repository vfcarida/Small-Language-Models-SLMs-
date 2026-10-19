# Module 14: Federated & Differential-Privacy LoRA Aggregation (DP-SGD + FFA-LoRA)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__14__federated__dp__lora-green?style=for-the-badge)](../src/module_14_federated_dp_lora/)

---

## 1. Executive & Architectural Motivation

Regulated enterprise silos (healthcare, banking, defense, telecommunications) frequently prohibit centralizing raw domain logs on a single training cluster. While Small Language Models (SLMs) can be fine-tuned locally on each edge node using LoRA (`Module 06`), two critical challenges arise when aggregating client adapters across $K$ decentralized silos:
1. **Training-Data Memorization & Privacy Leakage**: Standard gradient updates can leak verbatim PII or confidential records via membership inference and gradient inversion attacks.
2. **LoRA Cross-Term Aggregation Interference**: Naive Federated Averaging (`FedAvg`) averages $\bar{\mathbf{B}} = \sum_{k=1}^K p_k \mathbf{B}_k$ and $\bar{\mathbf{A}} = \sum_{k=1}^K p_k \mathbf{A}_k$ independently. Because matrix multiplication is bilinear, the product of averages does **not** equal the average of products:

$$
\bar{\mathbf{B}} \bar{\mathbf{A}} = \left(\sum_{k=1}^K p_k \mathbf{B}_k\right)\left(\sum_{j=1}^K p_j \mathbf{A}_j\right) = \underbrace{\sum_{k=1}^K p_k^2 \mathbf{B}_k \mathbf{A}_k}_{\text{Attenuated Direct Terms}} + \underbrace{\sum_{k \neq j} p_k p_j \mathbf{B}_k \mathbf{A}_j}_{\text{Destructive Cross-Term Noise } \neq \mathbf{0}} \neq \sum_{k=1}^K p_k \mathbf{B}_k \mathbf{A}_k
$$

Module 14 solves both problems simultaneously by combining **$(\epsilon, \delta)$-Differentially Private SGD (DP-SGD)** per-sample gradient clipping and Gaussian noise calibration with **Federated Freeze-A LoRA (FFA-LoRA)** (`Sun et al., ICLR 2024`) and **SVD-Exact LoRA Aggregation**.

---

## 2. Mathematical Foundations

### 2.1 $(\epsilon, \delta)$-Differential Privacy via Per-Sample Clipping & Gaussian Mechanism

Given a minibatch of $B$ examples with per-sample LoRA gradients $\{\mathbf{g}_i\}_{i=1}^B$, **DP-SGD** (`Abadi et al., CCS 2016`, `arXiv:1607.00133`; `Yu et al., ICLR 2022`, `arXiv:2110.06500`) bounds the maximum $L_2$ sensitivity of any single training record to $C > 0$:

$$
\bar{\mathbf{g}}_i = \mathbf{g}_i \cdot \min\left(1, \frac{C}{\|\mathbf{g}_i\|_2}\right)
$$

Isotropic Gaussian noise calibrated to the noise multiplier $\sigma$ is then injected prior to averaging:

$$
\tilde{\mathbf{g}} = \frac{1}{B} \left( \sum_{i=1}^B \bar{\mathbf{g}}_i + \mathcal{N}\left(\mathbf{0}, \sigma^2 C^2 \mathbf{I}\right) \right)
$$

Under Poisson subsampling with rate $q = B / N$ over $T$ composition steps, $\sigma$ is calibrated analytically to satisfy $(\epsilon, \delta)$-DP:

$$
\sigma \ge \frac{q \sqrt{2 T \ln(1.25 / \delta)}}{\epsilon}
$$

### 2.2 Eliminating Federated LoRA Cross-Term Interference (`FFA-LoRA` & `SVD-Exact`)

1. **Federated Freeze-A LoRA (`ffa_lora`, Sun et al., ICLR 2024 `arXiv:2403.12313`)**:
   All $K$ clients share an identical frozen projection $\mathbf{A}_0$ ($\mathbf{A}_1 = \dots = \mathbf{A}_K = \mathbf{A}_0$) and only optimize and transmit $\mathbf{B}_k$ under DP-SGD. By distributive linearity:

$$
\left(\sum_{k=1}^K p_k \mathbf{B}_k\right) \mathbf{A}_0 = \sum_{k=1}^K p_k (\mathbf{B}_k \mathbf{A}_0)
$$

   This eliminates cross-term error **exactly ($\|\Delta \mathbf{W}_{\text{ideal}} - \Delta \mathbf{W}_{\text{FFA}}\|_F = 0$)** while halving uplink communication bandwidth and halving the number of privatized gradient coordinates.
2. **SVD-Exact LoRA Aggregation (`svd_exact`)**:
   When clients train heterogeneous $(\mathbf{A}_k, \mathbf{B}_k)$ pairs, the central coordinator reconstructs $\Delta \mathbf{W}_{\text{ideal}} = \frac{\alpha}{r} \sum_{k=1}^K p_k \mathbf{B}_k \mathbf{A}_k$ and computes a truncated rank-$r$ Singular Value Decomposition $\mathbf{U}_r \boldsymbol{\Sigma}_r \mathbf{V}_r^\top$, setting $\mathbf{B}_{\text{server}} = \mathbf{U}_r \boldsymbol{\Sigma}_r^{1/2}$ and $\mathbf{A}_{\text{server}} = \boldsymbol{\Sigma}_r^{1/2} \mathbf{V}_r^\top$.

---

## 3. Reference Implementation

- **[`src/module_14_federated_dp_lora/dp_federated_lora.py`](../src/module_14_federated_dp_lora/dp_federated_lora.py)**:
  - `compute_gaussian_dp_sigma`: Analytic Gaussian mechanism noise multiplier calibration for target $(\epsilon, \delta, q, T)$.
  - `clip_and_noise_gradients`: Vectorized per-sample $L_2$ norm clipping and deterministic seeded Gaussian noise injection.
  - `aggregate_federated_lora`: Supports `"ffa_lora"`, `"svd_exact"`, and `"naive_fedavg"`, returning server adapters $(\mathbf{A}_{\text{server}}, \mathbf{B}_{\text{server}})$ and exact Frobenius reconstruction error $\|\Delta \mathbf{W}_{\text{ideal}} - \Delta \mathbf{W}_{\text{reconstructed}}\|_F$.

```bash
python src/module_14_federated_dp_lora/dp_federated_lora.py
```

---

## 4. Academic References

1. **Abadi, M., Chu, A., Goodfellow, I., McMahan, H. B., Mironov, I., Talwar, K., & Zhang, L.** (2016). *Deep Learning with Differential Privacy.* ACM CCS 2016 (`arXiv:1607.00133`).
2. **Yu, D., Naik, S., Backurs, A., et al.** (2022). *Differentially Private Fine-tuning of Language Models.* ICLR 2022 (`arXiv:2110.06500`).
3. **Sun, Y., Li, Z., Li, Y., & Ding, B.** (2024). *Improving LoRA in Privacy-preserving Federated Learning (FFA-LoRA).* ICLR 2024 (`arXiv:2403.12313`).
