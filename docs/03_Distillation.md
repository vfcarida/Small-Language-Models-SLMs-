# Module 03: On-Policy Knowledge Distillation (GKD & MiniLLM)

## 1. From Classical Soft Labels to Autoregressive Exposure Bias

In classical Knowledge Distillation (Hinton et al., 2015, `arXiv:1503.02531`), a compact Student model ($p_S$) is trained to match the temperature-softened probability distribution of a high-capacity Teacher model ($p_T$):

$$p_T(v \mid x_{<t}) = \frac{\exp(z_{T,v} / T)}{\sum_w \exp(z_{T,w} / T)}, \quad \mathcal{L}_{\text{FKL}} = T^2 \, D_{\text{KL}}(p_T \parallel p_S)$$

However, in autoregressive sequence generation, standard supervised distillation suffers from two fundamental failure modes:
1. **Exposure Bias (Distribution Mismatch):** During static distillation, the student only observes prefixes $x_{<t}$ drawn from the ground-truth dataset or the teacher. During production inference, the student conditions on its *own* imperfectly generated tokens, compounding errors across long horizons.
2. **Mode-Averaging Overestimation:** Minimizing Forward KL $D_{\text{KL}}(p_T \parallel p_S)$ forces a low-capacity student to cover *every* mode of the teacher distribution, spreading probability mass across implausible low-density voids between modes.

---

## 2. Mode-Seeking Reverse KL (MiniLLM) & Generalized JSD (GKD)

Our implementation in `src/module_03_distillation/gkd_pipeline.py` provides three mathematically verified divergence objectives via `compute_distillation_loss`:

### 1. Reverse KL Divergence (`mode="reverse_kl"` — MiniLLM, Gu et al., ICLR 2024, `arXiv:2306.08543`)
$$\mathcal{L}_{\text{RKL}} = T^2 \, D_{\text{KL}}(p_S \parallel p_T) = T^2 \sum_{v \in V} p_S(v) \log \frac{p_S(v)}{p_T(v)}$$
Reverse KL is **mode-seeking**: it heavily penalizes the student whenever $p_S(v) > 0$ in regions where $p_T(v) \approx 0$, locking the compact SLM onto the highest-confidence modes of the teacher and suppressing hallucinations.

### 2. Generalized Jensen-Shannon Divergence (`mode="jsd"` — GKD, Agarwal et al., ICLR 2024, `arXiv:2306.13649`)
Let $M = \beta p_T + (1 - \beta) p_S$ for interpolation parameter $\beta \in [0, 1]$:
$$\mathcal{L}_{\text{JSD}}^{(\beta)} = T^2 \left[ \beta \, D_{\text{KL}}(p_T \parallel M) + (1 - \beta) \, D_{\text{KL}}(p_S \parallel M) \right]$$
Combined with **on-policy student rollouts** (fraction $\lambda \in [0, 1]$ of trajectories sampled directly from the student and scored by the teacher), GKD eliminates exposure bias while maintaining bounded gradients.

---

## 3. Decoupled Client-Server Teacher Oracle (`VLLMTeacherOracle`)

Co-locating a 70B teacher (~140 GB VRAM) and a 1.5B student (~3 GB VRAM) on a single edge training node is physically impossible. `VLLMTeacherOracle` decouples the architecture:
1. The Teacher LLM is served on a remote GPU cluster via an OpenAI-compatible `vLLM` endpoint.
2. The local Student node generates on-policy trajectories and queries `/v1/completions` (`echo=True, logprobs=5`) with exponential backoff retry resilience.
3. Divergence gradients are computed locally on the student GPU.

---

## 4. Implementation Reference

- **Distillation Losses & Oracle Client:** [`src/module_03_distillation/gkd_pipeline.py`](../src/module_03_distillation/gkd_pipeline.py)
- **Unit Verification:** [`tests/test_distillation.py`](../tests/test_distillation.py)
