# Module 08: Calibrated Hybrid SLM-LLM Routing & AI TRiSM Guardrails

## 1. Strategic Context: The "Small-First" Enterprise Architecture

Enterprise AI adoption has shifted from routing every prompt to a monolithic frontier LLM toward **heterogeneous, right-sized SLM-LLM orchestration**:
- **Agentic AI Economics (Belcák et al., NVIDIA Research 2025, `arXiv:2506.02153`):** Agentic workflows invoke models dozens of times per user request for scoped subtasks (intent classification, JSON schema extraction, tool selection, log triage). Serving these high-frequency steps on a $<4\text{B}$ SLM cuts inference TCO by **$10\times$ to $30\times$**.
- **On-Device & Cloud Collaboration (Narayan et al., Stanford Hazy Research 2025, `arXiv:2502.15964`):** The *Minions* protocol demonstrates that executing privacy-sensitive data extraction locally on compact SLMs while escalating only abstract reasoning synthesis to cloud LLMs preserves frontier accuracy at a fraction of cloud token cost.
- **Preference-Calibrated Routing (Ong et al., ICLR 2025, `arXiv:2406.18665`):** *RouteLLM* formalizes dynamic query routing via calibrated complexity and confidence thresholds $\tau$, optimizing the Pareto frontier between latency/cost and response quality.

---

## 2. Mathematical Formulation of Confidence & Routing Gates

In `src/module_08_routing/hybrid_router.py`, every incoming request passes through a three-stage decision pipeline:

### Stage 1: AI TRiSM Data-Sovereignty & PII Guardrail (`PIIGuardrail`)
Operationalizing **Gartner AI TRiSM (Trust, Risk, and Security Management)** and **GDPR / LGPD** data-residency mandates, `PIIGuardrail` inspects prompts for sensitive entities (`EMAIL`, `SSN_OR_CPF`, `CREDIT_CARD`, `SECRET_KEY`, `CONFIDENTIAL_MARKER`). When `enforce_local_on_pii=True`, requests containing regulated data are strictly prohibited from leaving the on-premises boundary (`CLOUD_LLM` escalation is blocked).

### Stage 2: Token-Level Predictive Confidence (`compute_logit_confidence`)
Given SLM vocabulary logits $z \in \mathbb{R}^{|V|}$ and softmax probabilities $p = \text{softmax}(z)$, we combine **Normalized Shannon Entropy Confidence** and the **Top-2 Probability Margin**:

$$C_{\text{entropy}}(p) = 1 - \frac{-\sum_{v=1}^{|V|} p_v \log p_v}{\log |V|} \in [0, 1], \quad C_{\text{margin}}(p) = p_{(1)} - p_{(2)} \in [0, 1]$$

$$C_{\text{composite}}(p) = \frac{1}{2} C_{\text{entropy}}(p) + \frac{1}{2} C_{\text{margin}}(p)$$

### Stage 3: Calibrated Escalation & Epistemic Abstention (`HybridSLMRouter.route`)
Let $c(q) \in [0, 1]$ denote the structural complexity score (`score_query_complexity`) and $C \in [0, 1]$ denote the SLM confidence score:

$$\text{Target}(q) = \begin{cases}
\texttt{ABSTAIN\_HUMAN\_REVIEW}, & \text{if } \text{PII}(q) = \text{True} \land C < \tau_{\text{abstain}} \\
\texttt{LOCAL\_SLM}, & \text{if } \text{PII}(q) = \text{True} \land C \ge \tau_{\text{abstain}} \\
\texttt{LOCAL\_SLM}, & \text{if } \text{PII}(q) = \text{False} \land c(q) < \tau_{\text{complex}} \land C \ge \tau_{\text{conf}} \\
\texttt{CLOUD\_LLM}, & \text{otherwise.}
\end{cases}$$

### Stage 4: Trained Bradley-Terry & Matrix Factorization Preference Calibration (`CalibratedPreferenceRouter`)
Implemented in `src/module_08_routing/router_calibration.py` following **RouteLLM (Ong et al., ICLR 2025, `arXiv:2406.18665`)**, `CalibratedPreferenceRouter` maps an 8-dimensional query feature vector $\phi(q) \in \mathbb{R}^8$ into a calibrated win probability:

$$P(\text{SLM suffices} \mid q) = \sigma\!\left( \frac{\mathbf{w}^\top \phi(q) + b + (\mathbf{u}_{\text{slm}} - \mathbf{u}_{\text{llm}})^\top V \phi(q)}{T} \right)$$

It evaluates **Expected Calibration Error (ECE)**, **Brier Score**, and **Percentage of Performance Gap Recovered (PGR)**, solving for the exact threshold $\tau^*$ that satisfies a target SLM offload budget (`calibrate_threshold_for_budget`).

---

## 3. Top 10 Enterprise Workloads Where SLMs Outperform LLMs on ROI

1. **Intent Classification & Ticket Routing**
2. **Strict JSON / Entity Extraction from Unstructured Documents**
3. **PII-Sensitive On-Premises Financial & Healthcare Triage (GDPR/LGPD)**
4. **RAG Query Rewriting & Hypothetical Document Expansion**
5. **High-Throughput Log Anomaly & Telemetry Tagging**
6. **Tool Selection & Function-Calling Argument Formatting**
7. **Deterministic FAQ & Narrow-Domain Policy QA**
8. **Inline Safety & Toxicity Guardrail Filtering**
9. **Candidate Response Scoring & Reranking**
10. **Offline Edge / Mobile Personalization & Code Completion**

---

## 4. Implementation Reference

- **Hybrid Router, PII Guardrail & ROI Calculator:** [`src/module_08_routing/hybrid_router.py`](../src/module_08_routing/hybrid_router.py)
- **Trained Preference Router Calibration (RouteLLM):** [`src/module_08_routing/router_calibration.py`](../src/module_08_routing/router_calibration.py)
- **Unit Verification:** [`tests/test_routing.py`](../tests/test_routing.py), [`tests/test_router_calibration.py`](../tests/test_router_calibration.py)

