# Module 06: Preference Alignment — DPO & Reference-Free SimPO

## 1. From RLHF Instability to Closed-Form Preference Optimization

Classical Reinforcement Learning from Human Feedback (RLHF) requires fitting a separate Reward Model $r_\phi(x, y)$ on pairwise human preferences $(x, y_w \succ y_l)$ and optimizing the policy $\pi_\theta$ via Proximal Policy Optimization (PPO) under a KL divergence penalty against a frozen reference policy $\pi_{\text{ref}}$. Keeping four models in VRAM simultaneously (Policy, Reference, Reward, Value Critic) is prohibitive on single-GPU SLM infrastructure.

---

## 2. Direct Preference Optimization (DPO — Rafailov et al., NeurIPS 2023)

Rafailov et al. (`arXiv:2305.18290`) proved that the KL-constrained reward maximization objective admits a closed-form analytical mapping between the optimal policy $\pi^*$ and the ground-truth reward function:

$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Substituting this parameterization into the Bradley-Terry preference model cancels the intractable partition function $Z(x)$ and yields the **DPO Loss** (`compute_dpo_loss`):

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

---

## 3. Reference-Free Simple Preference Optimization (SimPO — Meng et al., NeurIPS 2024)

While DPO eliminates the Reward Model, it still requires evaluating $\pi_{\text{ref}}(y \mid x)$, and its implicit reward $\beta \log \frac{\pi_\theta}{\pi_{\text{ref}}}$ can diverge from the length-normalized log-likelihood metric actually maximized during beam/greedy decoding (causing length exploitation).

**SimPO** (Meng et al., NeurIPS 2024, `arXiv:2405.14734`) eliminates the reference model completely (**0 GB reference VRAM overhead**) by defining the reward directly as the length-normalized average log-probability of the policy alongside a strictly positive target reward margin $\gamma > 0$ (`compute_simpo_loss`):

$$r_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})$$

$$\mathcal{L}_{\text{SimPO}}(\pi_\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$

### Alignment Paradigm Comparison

| Method | Reference Model Required? | Length Normalization? | Key Hyperparameters | Primary Reference |
| :--- | :---: | :---: | :--- | :--- |
| **PPO (RLHF)** | Yes (+ Reward & Critic) | No | $\epsilon_{\text{clip}}, \beta_{\text{KL}}$ | Schulman et al. (2017) |
| **DPO** | Yes (or PEFT adapter toggle) | No | $\beta \in [0.05, 0.5]$ | Rafailov et al., NeurIPS 2023 (`arXiv:2305.18290`) |
| **ORPO** | No (Monolithic SFT + Odds Ratio) | Yes | $\lambda_{\text{OR}}$ | Hong et al., EMNLP 2024 (`arXiv:2403.07691`) |
| **SimPO** | **No (100% Reference-Free)** | **Yes ($1/|y|$)** | $\beta \in [2.0, 2.5], \gamma \in [0.5, 1.5]$ | Meng et al., NeurIPS 2024 (`arXiv:2405.14734`) |

---

## 4. Implementation Reference

- **DPO & SimPO Losses + Alignment Trainer:** [`src/module_06_alignment/dpo_trainer.py`](../src/module_06_alignment/dpo_trainer.py)
- **Preference Dataset Loader & Synthetic Fallback:** [`src/utils/data_utils.py`](../src/utils/data_utils.py)
- **Unit Verification:** [`tests/test_alignment.py`](../tests/test_alignment.py)
