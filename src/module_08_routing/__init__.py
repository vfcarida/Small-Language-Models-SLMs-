"""Module 08: Calibrated Hybrid SLM-LLM Routing, AI TRiSM PII Guardrails, and TCO/ROI Telemetry."""

from src.module_08_routing.hybrid_router import (
    HybridSLMRouter,
    PIIGuardrail,
    RouteTarget,
    RoutingDecision,
    compute_logit_confidence,
    evaluate_routing_economics,
    score_query_complexity,
)
from src.module_08_routing.router_calibration import (
    CalibratedPreferenceRouter,
    PreferenceSample,
    compute_expected_calibration_error,
    extract_query_features,
    generate_synthetic_preference_dataset,
)

__all__ = [
    "CalibratedPreferenceRouter",
    "HybridSLMRouter",
    "PIIGuardrail",
    "PreferenceSample",
    "RouteTarget",
    "RoutingDecision",
    "compute_expected_calibration_error",
    "compute_logit_confidence",
    "evaluate_routing_economics",
    "extract_query_features",
    "generate_synthetic_preference_dataset",
    "score_query_complexity",
]
