"""
==============================================================================
MODULE 08 — CALIBRATED HYBRID SLM-LLM ROUTING & AI TRiSM GUARDRAILS
==============================================================================
LEVEL: 🔴 Production / Enterprise Architecture
ESTIMATED TIME: 30 minutes

OBJECTIVE:
    Implement the "Small-First" Corporate Routing Pattern combining:
    1. Data-Sovereignty & PII Guardrails (Gartner AI TRiSM / GDPR / LGPD).
    2. Structural Query Complexity Scoring & Calibrated Threshold Routing
       (RouteLLM, Ong et al., ICLR 2025; Minions, Narayan et al., 2025).
    3. Token-Level Epistemic Confidence & Abstention Gates (Entropy + Top-2 Margin).
    4. Enterprise TCO / ROI & Latency Telemetry Calculator.

REFERENCES:
    - Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data",
      ICLR 2025 (arXiv:2406.18665)
    - Narayan et al., "Minions: Cost-efficient Collaboration Between On-device
      and Cloud Language Models", 2025 (arXiv:2502.15964)
    - Belcák et al., "Small Language Models are the Future of Agentic AI",
      NVIDIA Research, 2025 (arXiv:2506.02153)
    - Gartner, "AI TRiSM (Trust, Risk, and Security Management) & Domain-Specific
      Language Models (DSLMs)", 2025-2026

RUN COMMAND:
    python src/module_08_routing/hybrid_router.py
==============================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RouteTarget(str, Enum):
    """Target execution tier selected by the hybrid router."""

    LOCAL_SLM = "local_slm"
    CLOUD_LLM = "cloud_llm"
    ABSTAIN_HUMAN_REVIEW = "abstain_human_review"


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable audit record for a single routed request (AI TRiSM telemetry)."""

    query: str
    target: RouteTarget
    complexity_score: float
    confidence_score: float
    contains_pii: bool
    pii_categories: List[str] = field(default_factory=list)
    rationale: str = ""


class PIIGuardrail:
    """
    Deterministic regex-based Data Sovereignty & PII detector enforcing on-premises
    execution for regulated enterprise workloads (GDPR / LGPD / HIPAA / AI TRiSM).
    """

    PATTERNS: Dict[str, re.Pattern] = {
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        "SSN_OR_CPF": re.compile(
            r"\b(?:\d{3}-\d{2}-\d{4}|\d{3}\.\d{3}\.\d{3}-\d{2})\b"
        ),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "SECRET_KEY": re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b"),
        "CONFIDENTIAL_MARKER": re.compile(
            r"\b(?:CONFIDENTIAL|INTERNAL_ONLY|RESTRICTED_PII|PATIENT_ID|ACCOUNT_NUMBER)\b",
            re.IGNORECASE,
        ),
    }

    @classmethod
    def inspect(cls, text: str) -> List[str]:
        """Returns a sorted list of sensitive PII/compliance categories detected in `text`."""
        detected = []
        for category, pattern in cls.PATTERNS.items():
            if pattern.search(text):
                detected.append(category)
        return sorted(detected)

    @classmethod
    def redact(cls, text: str) -> str:
        """Replaces detected PII tokens with `[REDACTED_<CATEGORY>]` placeholders."""
        redacted = text
        for category, pattern in cls.PATTERNS.items():
            redacted = pattern.sub(f"[REDACTED_{category}]", redacted)
        return redacted


def compute_logit_confidence(logits: torch.Tensor) -> Dict[str, float]:
    """
    Computes token-level predictive confidence from SLM output logits using both
    normalized Shannon entropy and top-2 probability margin.

    MATHEMATICAL FORMULATION:
        Given logit vector z in R^V and probability distribution p = softmax(z):
        1. Normalized Entropy Confidence:
           H(p) = -sum_{v=1}^V p_v log(p_v)
           C_entropy = 1.0 - (H(p) / log(V)) in [0, 1]
        2. Top-2 Probability Margin:
           C_margin = p_(1) - p_(2) in [0, 1]
        3. Composite Confidence:
           C = 0.5 * C_entropy + 0.5 * C_margin

    Args:
        logits: Tensor of shape `(vocab_size,)` or `(..., vocab_size)`.

    Returns:
        Dict[str, float]: `'entropy_confidence'`, `'margin_confidence'`, and `'composite_confidence'`.
    """
    if logits.numel() == 0:
        raise ValueError("Logits tensor cannot be empty.")

    flat_logits = logits.reshape(-1, logits.shape[-1]).float()
    vocab_size = flat_logits.shape[-1]
    if vocab_size < 2:
        return {
            "entropy_confidence": 1.0,
            "margin_confidence": 1.0,
            "composite_confidence": 1.0,
        }

    probs = F.softmax(flat_logits, dim=-1)
    log_probs = F.log_softmax(flat_logits, dim=-1)

    entropy = -torch.sum(probs * log_probs, dim=-1)
    max_entropy = math.log(vocab_size)
    entropy_conf = (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0).mean().item()

    top2 = torch.topk(probs, k=2, dim=-1).values
    margin_conf = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0).mean().item()

    composite = 0.5 * entropy_conf + 0.5 * margin_conf
    return {
        "entropy_confidence": round(entropy_conf, 4),
        "margin_confidence": round(margin_conf, 4),
        "composite_confidence": round(composite, 4),
    }


def score_query_complexity(query: str) -> float:
    """
    Estimates structural and reasoning complexity of an incoming prompt in `[0.0, 1.0]`
    following RouteLLM / NVIDIA Agentic SLM task taxonomy heuristics:
    - Low complexity (`< 0.5`): intent classification, JSON extraction, tagging,
      sentiment, short FAQ, template rewriting.
    - High complexity (`>= 0.6`): multi-step mathematical proofs, theorem derivation,
      multi-hop synthesis, complex system architecture design.
    """
    q_lower = query.strip().lower()
    if not q_lower:
        return 0.0

    words = q_lower.split()
    length_factor = min(len(words) / 120.0, 0.35)

    reasoning_markers = [
        "prove",
        "derive",
        "step-by-step",
        "step by step",
        "theorem",
        "multi-step",
        "differential equation",
        "synthesize",
        "counterfactual",
        "formal verification",
        "asymptotic",
        "quantum",
        "optimality proof",
    ]
    simple_task_markers = [
        "classify",
        "extract json",
        "sentiment",
        "intent",
        "summarize in one sentence",
        "tag",
        "route",
        "translate",
        "yes or no",
        "format as",
    ]

    reasoning_hits = sum(1 for m in reasoning_markers if m in q_lower)
    simple_hits = sum(1 for m in simple_task_markers if m in q_lower)

    score = 0.25 + length_factor + (0.22 * reasoning_hits) - (0.18 * simple_hits)
    return round(max(0.0, min(1.0, score)), 4)


class HybridSLMRouter:
    """
    Calibrated "Small-First" Enterprise Router & AI TRiSM Policy Engine.

    DECISION LOGIC:
        1. Inspect query for sensitive PII/Compliance markers via `PIIGuardrail`.
        2. Score structural complexity `c in [0, 1]` and optional SLM logit confidence `conf in [0, 1]`.
        3. If PII is present and `enforce_local_on_pii=True`:
           - If SLM confidence >= `abstain_confidence_floor`, route strictly to `LOCAL_SLM`
             (preserving 100% on-premises data sovereignty).
           - Otherwise, trigger `ABSTAIN_HUMAN_REVIEW` so private data is never leaked
             to an external cloud API nor hallucinated locally.
        4. If no PII is present:
           - Route to `LOCAL_SLM` when `complexity_score < complexity_threshold` AND
             `confidence_score >= confidence_threshold`.
           - Otherwise escalate to `CLOUD_LLM`.
    """

    def __init__(
        self,
        complexity_threshold: float = 0.60,
        confidence_threshold: float = 0.65,
        abstain_confidence_floor: float = 0.35,
        enforce_local_on_pii: bool = True,
        calibrated_router: Optional[object] = None,
    ) -> None:
        self.complexity_threshold = complexity_threshold
        self.confidence_threshold = confidence_threshold
        self.abstain_confidence_floor = abstain_confidence_floor
        self.enforce_local_on_pii = enforce_local_on_pii
        self.calibrated_router = calibrated_router

    def route(
        self,
        query: str,
        slm_logits: Optional[torch.Tensor] = None,
    ) -> RoutingDecision:
        pii_categories = PIIGuardrail.inspect(query)
        contains_pii = len(pii_categories) > 0
        complexity = score_query_complexity(query)

        if slm_logits is not None:
            conf = compute_logit_confidence(slm_logits)["composite_confidence"]
        elif self.calibrated_router is not None and hasattr(
            self.calibrated_router, "predict_win_probability"
        ):
            conf = float(self.calibrated_router.predict_win_probability(query))
        else:
            # Estimate expected SLM confidence inversely from task complexity when pre-routing
            conf = round(max(0.0, min(1.0, 1.0 - 0.65 * complexity)), 4)

        if contains_pii and self.enforce_local_on_pii:
            if conf < self.abstain_confidence_floor:
                return RoutingDecision(
                    query=query,
                    target=RouteTarget.ABSTAIN_HUMAN_REVIEW,
                    complexity_score=complexity,
                    confidence_score=conf,
                    contains_pii=True,
                    pii_categories=pii_categories,
                    rationale=(
                        f"PII detected ({', '.join(pii_categories)}) blocks cloud escalation, "
                        f"and SLM confidence ({conf:.2f}) is below abstention floor "
                        f"({self.abstain_confidence_floor:.2f})."
                    ),
                )
            return RoutingDecision(
                query=query,
                target=RouteTarget.LOCAL_SLM,
                complexity_score=complexity,
                confidence_score=conf,
                contains_pii=True,
                pii_categories=pii_categories,
                rationale=(
                    f"Pinned to on-premises SLM to enforce data sovereignty for detected PII "
                    f"({', '.join(pii_categories)})."
                ),
            )

        if complexity < self.complexity_threshold and conf >= self.confidence_threshold:
            return RoutingDecision(
                query=query,
                target=RouteTarget.LOCAL_SLM,
                complexity_score=complexity,
                confidence_score=conf,
                contains_pii=contains_pii,
                pii_categories=pii_categories,
                rationale=(
                    f"Complexity ({complexity:.2f} < {self.complexity_threshold:.2f}) and "
                    f"confidence ({conf:.2f} >= {self.confidence_threshold:.2f}) qualify for edge SLM."
                ),
            )

        return RoutingDecision(
            query=query,
            target=RouteTarget.CLOUD_LLM,
            complexity_score=complexity,
            confidence_score=conf,
            contains_pii=contains_pii,
            pii_categories=pii_categories,
            rationale=(
                f"Escalated to frontier LLM due to high complexity ({complexity:.2f}) "
                f"or low SLM confidence ({conf:.2f})."
            ),
        )


def evaluate_routing_economics(
    decisions: Sequence[RoutingDecision],
    avg_tokens_per_query: int = 500,
    slm_cost_per_1m_usd: float = 0.08,
    llm_cost_per_1m_usd: float = 2.40,
    slm_ttft_ms: float = 180.0,
    llm_ttft_ms: float = 850.0,
) -> Dict[str, float]:
    """
    Computes enterprise unit-economics and latency telemetry comparing the Hybrid
    SLM-LLM Router against a 100% Cloud Frontier LLM baseline.
    """
    total = len(decisions)
    if total == 0:
        raise ValueError("decisions sequence cannot be empty.")

    slm_count = sum(1 for d in decisions if d.target == RouteTarget.LOCAL_SLM)
    llm_count = sum(1 for d in decisions if d.target == RouteTarget.CLOUD_LLM)
    abstain_count = sum(1 for d in decisions if d.target == RouteTarget.ABSTAIN_HUMAN_REVIEW)

    tokens_m = (avg_tokens_per_query / 1_000_000.0)
    baseline_cloud_cost = total * tokens_m * llm_cost_per_1m_usd
    hybrid_cost = (
        (slm_count * tokens_m * slm_cost_per_1m_usd)
        + (llm_count * tokens_m * llm_cost_per_1m_usd)
    )

    cost_savings_pct = (
        ((baseline_cloud_cost - hybrid_cost) / baseline_cloud_cost * 100.0)
        if baseline_cloud_cost > 0
        else 0.0
    )

    active_queries = max(slm_count + llm_count, 1)
    hybrid_avg_ttft = (
        (slm_count * slm_ttft_ms + llm_count * llm_ttft_ms) / active_queries
    )
    latency_reduction_pct = ((llm_ttft_ms - hybrid_avg_ttft) / llm_ttft_ms) * 100.0

    return {
        "total_queries": float(total),
        "slm_offload_ratio": round(slm_count / total, 4),
        "llm_escalation_ratio": round(llm_count / total, 4),
        "abstention_ratio": round(abstain_count / total, 4),
        "baseline_cloud_cost_usd": round(baseline_cloud_cost, 6),
        "hybrid_cost_usd": round(hybrid_cost, 6),
        "cost_savings_percent": round(cost_savings_pct, 2),
        "hybrid_avg_ttft_ms": round(hybrid_avg_ttft, 2),
        "latency_reduction_percent": round(latency_reduction_pct, 2),
    }


def main() -> None:
    config = load_config()
    routing_cfg = config.get("routing", {})

    router = HybridSLMRouter(
        complexity_threshold=routing_cfg.get("complexity_threshold", 0.60),
        confidence_threshold=routing_cfg.get("confidence_threshold", 0.65),
        enforce_local_on_pii=routing_cfg.get("enforce_local_on_pii", True),
    )

    sample_workload = [
        "Classify customer intent: 'I want to check my checking account balance.'",
        "Extract JSON entities from invoice #8492: Vendor=Acme Corp, Total=$420.00.",
        "Summarize in one sentence: The quarterly server uptime reached 99.98%.",
        "Route ticket to billing or technical support: 'My password reset link expired.'",
        "CONFIDENTIAL: Audit wire transfer for customer john.doe@enterprise.com (SSN 123-45-6789).",
        "Tag log severity: 'Connection pool exhausted on worker node 04.'",
        "Format as bullet points: Qwen2.5-1.5B uses GQA and fits in under 2GB VRAM when quantized.",
        "Translate to English: 'The edge model runs locally with zero external API latency.'",
        "Derive a step-by-step formal optimality proof for non-convex Cayley manifold SGD under rank constraints.",
    ]

    logger.info("=" * 72)
    logger.info("MODULE 08: HYBRID SLM-LLM ROUTING & AI TRiSM TELEMETRY")
    logger.info("=" * 72)

    decisions = [router.route(q) for q in sample_workload]
    for idx, dec in enumerate(decisions, 1):
        logger.info(
            f"[{idx:02d}] Target={dec.target.value:<20} | Complexity={dec.complexity_score:.2f} "
            f"| Conf={dec.confidence_score:.2f} | PII={dec.contains_pii} | {dec.rationale}"
        )

    econ = evaluate_routing_economics(decisions)
    logger.info("\n" + "-" * 72)
    logger.info(
        f"Summary Telemetry: SLM Offload={econ['slm_offload_ratio']*100:.1f}% | "
        f"Cost Savings={econ['cost_savings_percent']:.1f}% | "
        f"Avg TTFT={econ['hybrid_avg_ttft_ms']:.1f}ms (-{econ['latency_reduction_percent']:.1f}%)"
    )
    logger.info("-" * 72)


if __name__ == "__main__":
    main()
