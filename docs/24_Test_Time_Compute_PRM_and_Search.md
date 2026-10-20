# Module 24: Test-Time Compute Scaling via Process Reward Models (PRM) & Tree Search

> **Module Path:** `src/module_24_test_time_prm/prm_tree_search.py`  
> **Primary References:**  
> - Lightman et al. (ICLR 2024 / OpenAI), *"Let's Verify Step by Step"*, `arXiv:2305.20050`.  
> - Snell et al. (UC Berkeley, 2024), *"Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters"*, `arXiv:2408.03314`.

---

## 1. Outcome Reward Models (ORM) vs. Process Reward Models (PRM)

When a Small Language Model (1B–3B parameters) tackles multi-step mathematical or logical reasoning, a single arithmetic slip at step $t=2$ invalidates the entire downstream chain $s_{3:T}$—or worse, produces a lucky false-positive final answer.

An **Outcome Reward Model (ORM)** scores only the completed trajectory $\tau = (s_1, \dots, s_T)$, providing sparse feedback. A **Process Reward Model (PRM)** evaluates every intermediate reasoning step $s_t$:

$$r_t = P(\text{step } t \text{ is valid} \mid q, s_{1:t}) \in (0, 1)$$

To rank full or partial trajectories, `ProcessRewardScorer.aggregate_step_scores` supports three canonical reduction rules:
1. **Bottleneck Minimum (`mode="min"`, Lightman et al., 2024):**
   $$\text{PRM}_{\min}(\tau) = \min_{1 \le t \le T} r_t$$
   A reasoning chain is only as valid as its weakest deduction step.
2. **Joint Product (`mode="prod"`):**
   $$\text{PRM}_{\text{prod}}(\tau) = \prod_{t=1}^T r_t$$
3. **Geometric Mean (`mode="mean_log"`):**
   $$\text{PRM}_{\text{geom}}(\tau) = \exp\!\left(\frac{1}{T}\sum_{t=1}^T \log r_t\right)$$

---

## 2. Best-of-$N$ vs. PRM-Weighted Self-Consistency Voting

Given $N$ independently sampled trajectories $\{\tau_1, \dots, \tau_N\}$ with final answers $a_i = \text{ans}(\tau_i)$:
- **Best-of-$N$ (`best_of_n_answer`):** Picks $a_{i^*}$ from the single highest-scoring trajectory $i^* = \arg\max_i \text{PRM}(\tau_i)$.
- **PRM-Weighted Voting (`prm_weighted_vote_answer`, Snell et al., 2024):** Marginalizes over distinct trajectories that reach the same conclusion $a$, weighting each vote by its step-verified reliability:

$$a_{\text{PRM-Vote}} = \arg\max_{a \in \mathcal{A}} \sum_{i : a_i = a} \text{PRM}(\tau_i)$$

This prevents a single high-frequency hallucinated answer from winning an unweighted majority vote when its intermediate steps fail verification.

---

## 3. Step-Level Beam Search (`step_level_beam_search`)

Instead of generating $N$ complete trajectories before filtering, `step_level_beam_search` prunes flawed reasoning branches at every step $t \in \{1, \dots, T_{\max}\}$:
1. Maintain $B$ (`beam_width`) active partial trajectories.
2. Expand each beam into $M$ (`branching_factor`) candidate next steps.
3. Score all $B \times M$ candidates with the PRM and retain only the top-$B$ beams for step $t+1$.
