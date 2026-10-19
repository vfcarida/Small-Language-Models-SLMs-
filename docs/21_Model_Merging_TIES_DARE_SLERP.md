# Module 21: Task-Arithmetic & Evolutionary Model Merging (`TIES`, `DARE` & `SLERP`)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__21__model__merging-green?style=for-the-badge)](../src/module_21_model_merging/)

---

## 1. Executive & Architectural Motivation

Organizations frequently fine-tune separate Small Language Model checkpoints ($\boldsymbol{\theta}_1, \dots, \boldsymbol{\theta}_K$) from a common pretrained foundation model $\boldsymbol{\theta}_{\text{base}}$ for distinct domains (e.g., SQL generation, mathematical reasoning, and legal extraction). Serving $K$ separate full-weight checkpoints or running joint multi-task retraining is operationally costly. **Weight-Space Model Merging** (**Task Arithmetic**, `Ilharco et al., ICLR 2023`; **TIES-Merging**, `Yadav et al., NeurIPS 2023`; **DARE**, `Yu et al., ICML 2024`) composes multiple specialist SLMs into a single unified checkpoint in seconds on CPU with **zero gradient updates**.

---

## 2. Mathematical Foundations

### 2.1 Drop And REscale (`DARE`, Yu et al., ICML 2024)

Given a fine-tuned task vector $\boldsymbol{\tau}_k = \boldsymbol{\theta}_k - \boldsymbol{\theta}_{\text{base}}$, **DARE** (`arXiv:2311.03099`) observes that up to $90\%$ of parameter deltas can be randomly zeroed without degrading task accuracy provided the surviving coordinates are rescaled by $\frac{1}{1-p}$ to preserve the unbiased expectation:

$$
\mathbf{m}_k \sim \text{Bernoulli}(1 - p), \qquad \tilde{\boldsymbol{\tau}}_k = \frac{\mathbf{m}_k \odot \boldsymbol{\tau}_k}{1 - p} \implies \mathbb{E}\!\left[\tilde{\boldsymbol{\tau}}_k\right] = \boldsymbol{\tau}_k
$$

### 2.2 `TIES-Merging`: Trim, Elect Sign & Disjoint Merge (`Yadav et al., NeurIPS 2023`)

Naive averaging $\frac{1}{K}\sum_k \boldsymbol{\tau}_k$ degrades when redundant small-magnitude parameters dilute salient updates or when two specialists push the same weight in opposite signs ($\tau_{1,i} > 0, \tau_{2,i} < 0$). **TIES-Merging** (`arXiv:2306.01708`) resolves both failure modes in three steps:
1. **Trim**: Retain only the top-$\rho$ fraction of parameters by magnitude $|\tau_{k,i}|$ per task vector, setting the remaining $(1-\rho)$ entries to $0$.
2. **Elect Sign**: Compute the aggregate consensus sign per parameter coordinate $i$:

$$
\gamma_i = \text{sgn}\!\left(\sum_{k=1}^K \hat{\tau}_{k, i}\right) \in \{-1, 0, +1\}
$$

3. **Disjoint Merge**: Average only the task vectors $\mathcal{A}_i = \{k : \text{sgn}(\hat{\tau}_{k, i}) = \gamma_i \land \hat{\tau}_{k, i} \neq 0\}$ that agree with the elected sign:

$$
\tau_{\text{TIES}, i} = \frac{1}{|\mathcal{A}_i|} \sum_{k \in \mathcal{A}_i} \hat{\tau}_{k, i}, \qquad \boldsymbol{\theta}_{\text{merged}} = \boldsymbol{\theta}_{\text{base}} + \lambda \boldsymbol{\tau}_{\text{TIES}}
$$

### 2.3 Spherical Linear Interpolation (`SLERP`)

When merging two homologous models $\mathbf{w}_0, \mathbf{w}_1$, linear interpolation $(1-t)\mathbf{w}_0 + t\mathbf{w}_1$ shrinks the layer $L_2$ norm at $t=0.5$. **SLERP** follows the constant-curvature geodesic arc on the hypersphere:

$$
\Omega = \arccos\!\left(\frac{\mathbf{w}_0 \cdot \mathbf{w}_1}{\|\mathbf{w}_0\|_2 \|\mathbf{w}_1\|_2}\right), \qquad \text{SLERP}(t; \mathbf{w}_0, \mathbf{w}_1) = \frac{\sin((1-t)\Omega)}{\sin \Omega}\mathbf{w}_0 + \frac{\sin(t\Omega)}{\sin \Omega}\mathbf{w}_1
$$

---

## 3. Reference Implementation

- **[`src/module_21_model_merging/task_arithmetic_merging.py`](../src/module_21_model_merging/task_arithmetic_merging.py)**:
  - `dare_sparsify_delta`: Unbiased Bernoulli delta sparsification.
  - `ties_merge_task_vectors`: Trim, Elect Sign, and Disjoint Merge with sign-conflict telemetry.
  - `slerp_weights`: Geodesic hyperspherical interpolation with collinear fallback.

```bash
python src/module_21_model_merging/task_arithmetic_merging.py
```

---

## 4. Academic References

1. **Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M.** (2023). *TIES-Merging: Resolving Interference When Merging Models.* NeurIPS 2023 (`arXiv:2306.01708`).
2. **Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y.** (2024). *Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE).* ICML 2024 (`arXiv:2311.03099`).
3. **Ilharco, G., Ribeiro, M. T., Wortsman, M., et al.** (2023). *Editing Models with Task Arithmetic.* ICLR 2023 (`arXiv:2212.04089`).
