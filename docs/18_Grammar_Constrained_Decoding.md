# Module 18: Grammar-Constrained & Finite-State Machine (FSM) Structured Decoding

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__18__constrained__decoding-green?style=for-the-badge)](../src/module_18_constrained_decoding/)

---

## 1. Executive & Architectural Motivation

In enterprise agentic workflows and Model Context Protocol (MCP) tool-calling gateways, Small Language Models (`<2B` parameters) must emit machine-parseable JSON objects, SQL queries, or DSL expressions. Even after Supervised Fine-Tuning (Module 02) and RLVR/GRPO alignment (Module 11), aggressive low-bit quantization (`INT3`/`INT2` or ternary `1.58-bit`) can introduce nonzero probability mass on syntax-breaking tokens (e.g., missing quotes, trailing commas, or hallucinated keys), triggering expensive retry loops.

**Grammar-Constrained Decoding** (**Outlines**, `Willard & Louf, 2023`, `arXiv:2307.09702`; **XGrammar**, `Dong et al., CMU / OctoAI, 2024`, `arXiv:2411.15100`) eliminates syntactic failures at the root by compiling the target JSON schema or regular grammar into a **Deterministic Finite Automaton (DFA / FSM)** over the model's vocabulary $V$ ahead of time and masking invalid vocabulary transitions to $-\infty$ in $O(1)$ lookup time per decoding step.

---

## 2. Mathematical Foundations

### 2.1 Vocabulary-Indexed Deterministic Finite Automaton (DFA)

Let a regular grammar or bounded-depth JSON schema be represented by a 5-tuple DFA $\mathcal{M} = (Q, V, \delta, q_0, F)$, where:
- $Q$ is the finite set of automaton states,
- $V = \{1, \dots, |V|\}$ is the subword token vocabulary,
- $\delta : Q \times V \to Q \cup \{\bot\}$ is the state transition function,
- $q_0 \in Q$ is the initial state, and $F \subseteq Q$ is the set of accepting states.

During offline compilation, for every state $q \in Q$ we precompute the allowed vocabulary subset:

$$
\mathcal{A}(q) = \{ v \in V : \delta(q, v) \neq \bot \}
$$

### 2.2 Zero-Overhead Conditional Probability Projection

At autoregressive step $t$ in state $q_t \in Q$, raw SLM logits $\mathbf{z}_t \in \mathbb{R}^{|V|}$ are transformed via additive masking:

$$
\tilde{z}_{t, v} = \begin{cases} z_{t, v} & \text{if } v \in \mathcal{A}(q_t) \\ -\infty & \text{if } v \notin \mathcal{A}(q_t) \end{cases}
$$

Applying the softmax operator yields the exact conditional probability distribution conditioned on grammatical validity $v \in \mathcal{A}(q_t)$, preserving relative probability ratios between all valid tokens:

$$
\tilde{P}(v \mid \mathbf{x}, q_t) = \frac{\exp(z_{t, v}) \cdot \mathbb{I}[v \in \mathcal{A}(q_t)]}{\sum_{u \in \mathcal{A}(q_t)} \exp(z_{t, u})} \implies \frac{\tilde{P}(v_1 \mid \mathbf{x}, q_t)}{\tilde{P}(v_2 \mid \mathbf{x}, q_t)} = \frac{P(v_1 \mid \mathbf{x})}{P(v_2 \mid \mathbf{x})} \quad \forall v_1, v_2 \in \mathcal{A}(q_t)
$$

---

## 3. Reference Implementation

- **[`src/module_18_constrained_decoding/fsm_guided_decoding.py`](../src/module_18_constrained_decoding/fsm_guided_decoding.py)**:
  - `apply_fsm_logit_mask`: Exact conditional logit mask preserving relative probabilities over valid tokens.
  - `FiniteStateGrammarIndex`: Pre-indexed $O(1)$ DFA transition and allowed-vocabulary lookup table.
  - `JSONSchemaFSMDecoder`: Compiles typed JSON schemas (`string`, `number`, `boolean`) into a DFA and guarantees **100% valid JSON and 100% schema compliance on the first pass**, even under adversarial logits heavily biased toward garbage tokens.

```bash
python src/module_18_constrained_decoding/fsm_guided_decoding.py
```

---

## 4. Academic References

1. **Willard, B. T., & Louf, R.** (2023). *Efficient Guided Generation for Large Language Models (Outlines).* (`arXiv:2307.09702`).
2. **Dong, Y., Ruan, C. F., Cai, Y., Lai, R., Xu, Z., et al.** (2024). *XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models.* CMU / OctoAI (`arXiv:2411.15100`).
