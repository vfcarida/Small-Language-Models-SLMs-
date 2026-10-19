# Module 11 — Reinforcement Learning with Verifiable Rewards (`RLVR / GRPO`)

> **Source Implementation:** [`src/module_11_rlvr_grpo/grpo_trainer.py`](../src/module_11_rlvr_grpo/grpo_trainer.py)  
> **Verification Suite:** [`tests/test_rlvr_grpo.py`](../tests/test_rlvr_grpo.py)

---

## 1. Scientific Foundation: Critic-Free Reasoning Alignment for SLMs

Standard Proximal Policy Optimization (PPO) requires maintaining a trainable **Value Critic network** of the same size as the policy model in GPU memory, doubling activation and optimizer states. **Group Relative Policy Optimization (GRPO)**—introduced in **DeepSeekMath** ([Shao et al., 2024, `arXiv:2402.03300`](https://arxiv.org/abs/2402.03300)) and scaled in **DeepSeek-R1** ([DeepSeek-AI, 2025, `arXiv:2501.12948`](https://arxiv.org/abs/2501.12948))—eliminates the Critic entirely by sampling a group of $G$ completions $\{o_1, \dots, o_G\}$ for each prompt $q$ and normalizing their verifiable rewards $\{r_1, \dots, r_G\}$ within the group:

$$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$$

---

## 2. Mathematical Formulation of the GRPO Objective

Given token probability ratios $\rho_{i, t} = \frac{\pi_\theta(o_{i, t} \mid q, o_{i, <t})}{\pi_{\theta_{\text{old}}}(o_{i, t} \mid q, o_{i, <t})}$ and **Schulman's unbiased non-negative KL estimator** (`compute_schulman_kl`):

$$D_{\text{KL}}\big(\pi_\theta \parallel \pi_{\text{ref}}\big)_{i, t} = \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} - \log \frac{\pi_{\text{ref}}(o_{i, t} \mid q, o_{i, <t})}{\pi_\theta(o_{i, t} \mid q, o_{i, <t})} - 1 \ge 0$$

The GRPO loss (`compute_grpo_loss`) minimizes:

$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left[ \min\!\Big(\rho_{i, t} A_i, \text{clip}(\rho_{i, t}, 1 - \epsilon, 1 + \epsilon) A_i\Big) - \beta D_{\text{KL}}\big(\pi_\theta \parallel \pi_{\text{ref}}\big)_{i, t} \right]$$

---

## 3. Deterministic Verifiable Reward Oracles (`VerifiableRewardOracle`)

Instead of relying on a neural reward model susceptible to reward hacking, RLVR scores SLM rollouts with deterministic verifiers:
1. **Structural Reasoning Format (`verify_reasoning_format`):** Enforces explicit `<think>...</think><answer>...</answer>` separation before emitting the final answer.
2. **Exact Mathematical Equivalence (`verify_math_answer`):** Extracts answers from `<answer>...</answer>` or `\boxed{...}` and verifies numerical tolerance $|y - y^*| \le 10^{-6}$.
3. **Agentic JSON Schema Compliance (`verify_json_schema`):** Verifies valid JSON syntax and presence of mandatory schema keys for tool-calling SLMs.

---

## 4. Quickstart Usage

```bash
python src/module_11_rlvr_grpo/grpo_trainer.py
```
