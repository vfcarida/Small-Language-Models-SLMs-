# Security & AI TRiSM Governance Policy (`SECURITY.md`)

> **Maintainer & Principal Architect:** [Vinicius Caridá (`vfcarida`)](https://github.com/vfcarida)

---

## 1. Supported Versions

| Version | Status | Python Support |
| :---: | :---: | :--- |
| **`2.0.x`** | **Active / Supported (`Stable`)** | Python `3.10`, `3.11`, `3.12` |
| `< 2.0.0` | Superseded by `2.0.0` | Upgrade recommended |

---

## 2. Enterprise AI TRiSM & Data Sovereignty Controls

`slm-masterclass` embeds defense-in-depth security and regulatory compliance controls directly into the runtime architecture:

1. **Zero Untrusted Deserialization (`safetensors` & Explicit Manifests):**
   All checkpoint loading paths and edge artifact exporters ([`src/module_09_serving/edge_exporter.py`](src/module_09_serving/edge_exporter.py)) compute and verify **SHA-256 cryptographic digests** and favor `safetensors` over arbitrary Python `pickle` execution.
2. **Inline PII Redaction & Local SLM Pinning (`GDPR` / `LGPD` / `HIPAA`):**
   [`PIIGuardrail`](src/module_08_routing/hybrid_router.py) inspects incoming prompts before routing. Any payload containing SSN, CPF, credit card, email, or phone identifiers is strictly pinned to the on-premises SLM (`RouteTarget.LOCAL_SLM`) or routed to epistemic abstention (`RouteTarget.ABSTAIN_HUMAN_REVIEW`), preventing sensitive data exfiltration to external cloud APIs.
3. **Differential Privacy & Federated LoRA (`(ε, δ)`-DP-SGD):**
   [`src/module_14_federated_dp_lora/dp_federated_lora.py`](src/module_14_federated_dp_lora/dp_federated_lora.py) enforces per-sample $L_2$ gradient clipping and analytically calibrated Gaussian noise prior to cross-silo LoRA aggregation.
4. **Targeted Knowledge Erasure & Right to be Forgotten (`NPO` & `RMU`):**
   [`src/module_29_machine_unlearning/unlearning_objectives.py`](src/module_29_machine_unlearning/unlearning_objectives.py) provides bounded-gradient Negative Preference Optimization (`NPO`) and Representation Misdirection for Unlearning (`RMU`) to erase revoked or hazardous knowledge without model collapse.
5. **Cryptographic Provenance Watermarking (`EU AI Act Art. 50`):**
   [`src/module_32_statistical_watermarking/kgw_watermark.py`](src/module_32_statistical_watermarking/kgw_watermark.py) embeds SHA-256 keyed green-list logit watermarks verifiable via zero-knowledge binomial $z$-tests.

---

## 3. Reporting a Vulnerability

If you discover a security vulnerability, serialization flaw, or guardrail bypass, please report it privately via [GitHub Security Advisories](https://github.com/vfcarida/Small-Language-Models-SLMs-/security/advisories) rather than opening a public issue.
