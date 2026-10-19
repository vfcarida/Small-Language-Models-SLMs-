# Module 13: Multi-Agent Collaborative SLM Swarms & Debate Verification

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__13__agentic__swarms-green?style=for-the-badge)](../src/module_13_agentic_swarms/)

---

## 1. Executive & Architectural Motivation

A foundational shift in enterprise AI systems is the transition from monolithic frontier LLMs to **compound agentic systems composed of specialized Small Language Models** (**Belcák et al., NVIDIA Research 2025**, *"Small Language Models are the Future of Agentic AI"*, `arXiv:2506.02153`; **Gartner Top Strategic Technology Trends 2025/2026**).

However, deploying $M$ independent fine-tuned SLMs concurrently on edge or on-premise GPUs would multiply VRAM consumption by $M \times$. Module 13 resolves this via **Shared-Backbone Multi-LoRA Hot-Swapping** and **Multi-Round Generator-Critic-Verifier Debate**:
1. **Shared Frozen Base + Role-Specific LoRA Adapters**: A single quantized base model $\mathbf{W}_0$ stays resident in GPU memory while lightweight low-rank adapters $\{(\mathbf{A}_m, \mathbf{B}_m)\}_{m=1}^M$ specialize each micro-agent role (`planner`, `solver_a`, `solver_b`, `critic`, `trism_auditor`), cutting multi-agent VRAM overhead by **>70%**.
2. **Confidence-Weighted Debate & Deterministic Verification**: Multiple SLM solvers propose candidate solutions in Round 1, which are audited by a deterministic `VerifiableRewardOracle` (Module 11) and `PIIGuardrail` (Module 08) before a Critic agent repairs reasoning traces and executes confidence-weighted consensus voting in Round 2.

---

## 2. Mathematical Foundations

### 2.1 Shared-Backbone Multi-LoRA Forward Pass

Given a shared frozen base projection $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and $M$ role-specific LoRA adapters $(\mathbf{A}_m \in \mathbb{R}^{r \times d_{\text{in}}}, \mathbf{B}_m \in \mathbb{R}^{d_{\text{out}} \times r})$ with rank $r \ll \min(d_{\text{in}}, d_{\text{out}})$, the activation for agent role $m$ is:

$$
\mathbf{y}^{(m)} = \mathbf{x} \mathbf{W}_0^\top + \frac{\alpha}{r} \left(\mathbf{x} \mathbf{A}_m^\top\right) \mathbf{B}_m^\top
$$

For $M = 4$ concurrent 1.54B-parameter `INT4` SLMs ($d_{\text{model}} = 1536, r = 16$), sharing $\mathbf{W}_0$ reduces total VRAM from **3,112.9 MB** (4 independent models) to **824.5 MB** (**73.5% VRAM savings**).

### 2.2 Confidence & Reliability Weighted Consensus Voting

Given candidate proposals $\{(a_m, c_m, w_m)\}_{m=1}^M$ where $a_m$ is the extracted answer, $c_m \in [0, 1]$ is the agent's self-consistency confidence, and $w_m > 0$ is the agent's historical reliability weight (**Du et al., ICML 2024** `arXiv:2305.14325`; **Wang et al., ICLR 2023** `arXiv:2203.11171`), the aggregated consensus score for candidate answer $y$ is:

$$
\text{Score}(y) = \frac{\sum_{m : a_m = y} w_m \cdot c_m}{\sum_{j=1}^M w_j \cdot c_j}
$$

Normalized Shannon disagreement entropy across distinct candidate answers $\mathcal{Y}$ quantifies swarm epistemic uncertainty:

$$
H_{\text{norm}}(\mathcal{Y}) = -\frac{1}{\ln |\mathcal{Y}|} \sum_{y \in \mathcal{Y}} \text{Score}(y) \ln \text{Score}(y) \in [0, 1]
$$

---

## 3. Reference Implementation

- **[`src/module_13_agentic_swarms/slm_swarm.py`](../src/module_13_agentic_swarms/slm_swarm.py)**:
  - `MultiLoRASwarmBackbone`: Implements shared-base role-specific LoRA forward dispatch (`forward_role`) and analytical fleet VRAM profiling (`estimate_vram_savings`).
  - `confidence_weighted_vote`: Computes weighted consensus winner, consensus probability, and normalized disagreement entropy.
  - `SLMAgentSwarm`: Orchestrates 2-round Generator-Critic-Verifier debate integrated with `VerifiableRewardOracle` (from Module 11) and automatic `PIIGuardrail` redaction (from Module 08).

```bash
python src/module_13_agentic_swarms/slm_swarm.py
```

---

## 4. Academic References

1. **Belcák, P., Heinrich, G., Diao, S., et al.** (2025). *Small Language Models are the Future of Agentic AI.* NVIDIA Research (`arXiv:2506.02153`).
2. **Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I.** (2024). *Improving Factuality and Reasoning in Language Models through Multiagent Debate.* ICML 2024 (`arXiv:2305.14325`).
3. **Wang, X., Wei, J., Schuurmans, D., et al.** (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR 2023 (`arXiv:2203.11171`).
