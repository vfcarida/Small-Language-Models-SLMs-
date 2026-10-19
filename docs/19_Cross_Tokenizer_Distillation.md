# Module 19: Cross-Tokenizer Knowledge Distillation via Optimal Transport (`ULD` & `MinED`)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__19__cross__tokenizer__kd-green?style=for-the-badge)](../src/module_19_cross_tokenizer_kd/)

---

## 1. Executive & Architectural Motivation

Standard logit-based Knowledge Distillation (`Module 03`) requires the Teacher LLM and Student SLM to share an **identical vocabulary** ($|V_T| = |V_S|$ and $V_T = V_S$) so token-wise Kullback-Leibler or Jensen-Shannon divergences can be evaluated coordinate by coordinate. In real-world enterprise deployments, however, the strongest teacher models (`DeepSeek-V3`, `Qwen2.5-72B`, `Llama-3.1-70B`) and the most hardware-efficient edge students (`Llama-3.2-1B`, `SmolLM2-1.7B`, `Phi-3.5-mini`) frequently use **incompatible tokenizers** with different vocabulary sizes ($|V_T| \neq |V_S|$) and subword conventions (`Ġ` Byte-Level BPE vs. `▁` SentencePiece).

Module 19 enables **Cross-Tokenizer Knowledge Distillation** by combining two complementary peer-reviewed formulations:
1. **Universal Logit Distillation (`ULD`, Boizard et al., ICML/TMLR 2024, `arXiv:2402.12030`)**: Aligns the structural shape (entropy and tail decay) of teacher and student probability distributions via 1D Optimal Transport (Wasserstein-1 distance on sorted probabilities) regardless of vocabulary size.
2. **Minimum Edit Distance Vocabulary Alignment (`MinED`, Wan et al., ACL 2024, `arXiv:2401.14691`)**: Maps canonical teacher subwords onto their nearest student tokens via Levenshtein edit distance to construct a probability-conserving projection matrix $\mathbf{M} \in \mathbb{R}^{|V_T| \times |V_S|}$.

---

## 2. Mathematical Foundations

### 2.1 Universal Logit Distillation (`ULD`) via 1D Wasserstein Optimal Transport

Let $\mathbf{p}_S = \text{softmax}(\mathbf{z}_S / T) \in \Delta^{|V_S|-1}$ and $\mathbf{p}_T = \text{softmax}(\mathbf{z}_T / T) \in \Delta^{|V_T|-1}$. Sorting both distributions in descending order yields rank-ordered probability vectors $\mathbf{p}_S^\downarrow$ and $\mathbf{p}_T^\downarrow$. Zero-padding the smaller vector to $V_{\max} = \max(|V_S|, |V_T|)$ yields the closed-form 1D Wasserstein-1 distance:

$$
\mathcal{L}_{\text{ULD}}(\mathbf{z}_S, \mathbf{z}_T) = T \sum_{r=1}^{V_{\max}} \left| p_{S, r}^\downarrow - p_{T, r}^\downarrow \right|
$$

Because sorting is piecewise linear with unit Jacobian along the permutation indices, $\nabla_{\mathbf{z}_S} \mathcal{L}_{\text{ULD}}$ backpropagates cleanly into the Student SLM.

### 2.2 Minimum Edit Distance (`MinED`) Probability Projection

For each teacher token $t \in V_T$, let $\tilde{t} = \text{canon}(t)$ denote its normalized string after stripping tokenizer-specific whitespace markers (`Ġ`, `▁`, `##`). Each teacher token is assigned to the closest student token $s^*(t) \in V_S$:

$$
s^*(t) = \arg\min_{s \in V_S} \text{Levenshtein}\!\left(\text{canon}(t), \text{canon}(s)\right)
$$

Defining $M_{t, s} = \mathbb{I}[s = s^*(t)]$ produces a row-stochastic matrix $\mathbf{M} \in \{0, 1\}^{|V_T| \times |V_S|}$ ($\sum_{s \in V_S} M_{t, s} = 1$) that projects $\mathbf{p}_T$ onto $\hat{\mathbf{p}}_T = \mathbf{p}_T \mathbf{M} \in \Delta^{|V_S|-1}$ without losing probability mass.

---

## 3. Reference Implementation

- **[`src/module_19_cross_tokenizer_kd/optimal_transport_kd.py`](../src/module_19_cross_tokenizer_kd/optimal_transport_kd.py)**:
  - `compute_uld_wasserstein_loss`: Differentiable 1D Optimal Transport rank-quantile loss across arbitrary $|V_S| \neq |V_T|$.
  - `align_cross_tokenizer_vocab`: Canonical subword normalization and Levenshtein projection matrix builder.
  - `CrossTokenizerDistiller`: Hybrid objective $\mathcal{L} = \lambda \mathcal{L}_{\text{ULD}} + (1 - \lambda) T^2 D_{\text{KL}}(\mathbf{p}_T \mathbf{M} \parallel \mathbf{p}_S)$.

```bash
python src/module_19_cross_tokenizer_kd/optimal_transport_kd.py
```

---

## 4. Academic References

1. **Boizard, N., El Hanchi, A., Hudelot, C., & Colombo, P.** (2024). *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs (ULD).* ICML / TMLR 2024 (`arXiv:2402.12030`).
2. **Wan, F., Huang, X., Cai, D., Quan, X., Bi, W., & Shi, S.** (2024). *Knowledge Fusion of Large Language Models (MinED).* ICLR / ACL 2024 (`arXiv:2401.14691`).
