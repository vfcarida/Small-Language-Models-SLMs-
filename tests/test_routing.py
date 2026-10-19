import torch
from src.module_08_routing.hybrid_router import (
    HybridSLMRouter,
    PIIGuardrail,
    RouteTarget,
    compute_logit_confidence,
    evaluate_routing_economics,
    score_query_complexity,
)


def test_pii_guardrail_inspection_and_redaction():
    text = "Contact alice@company.com regarding SSN 123-45-6789 (CONFIDENTIAL)."
    categories = PIIGuardrail.inspect(text)
    assert "EMAIL" in categories
    assert "SSN_OR_CPF" in categories
    assert "CONFIDENTIAL_MARKER" in categories

    redacted = PIIGuardrail.redact(text)
    assert "alice@company.com" not in redacted
    assert "123-45-6789" not in redacted
    assert "[REDACTED_EMAIL]" in redacted
    assert "[REDACTED_SSN_OR_CPF]" in redacted


def test_compute_logit_confidence_peaked_vs_uniform():
    peaked_logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])
    uniform_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    peaked_conf = compute_logit_confidence(peaked_logits)
    uniform_conf = compute_logit_confidence(uniform_logits)

    assert peaked_conf["composite_confidence"] > 0.99
    assert uniform_conf["composite_confidence"] < 0.01


def test_score_query_complexity_orders_simple_below_complex():
    simple_q = "Classify customer intent and extract JSON fields: 'Check order status'"
    complex_q = (
        "Derive a step-by-step formal optimality proof and asymptotic convergence theorem "
        "for non-convex differential equation solvers under counterfactual assumptions."
    )
    assert score_query_complexity(simple_q) < 0.35
    assert score_query_complexity(complex_q) >= 0.65


def test_hybrid_slm_router_decisions():
    router = HybridSLMRouter(
        complexity_threshold=0.60,
        confidence_threshold=0.65,
        abstain_confidence_floor=0.35,
        enforce_local_on_pii=True,
    )

    # 1. Simple query -> LOCAL_SLM
    d_simple = router.route("Classify intent: 'Reset my password'")
    assert d_simple.target == RouteTarget.LOCAL_SLM

    # 2. Complex math/reasoning query without PII -> CLOUD_LLM
    d_complex = router.route(
        "Prove step-by-step the asymptotic optimality theorem of quantum differential equations."
    )
    assert d_complex.target == RouteTarget.CLOUD_LLM

    # 3. Query with PII and high SLM confidence -> Pinned to LOCAL_SLM
    peaked_logits = torch.tensor([[50.0, 0.0, 0.0, 0.0]])
    d_pii_ok = router.route(
        "Extract JSON for customer john@bank.com with SSN 123-45-6789",
        slm_logits=peaked_logits,
    )
    assert d_pii_ok.target == RouteTarget.LOCAL_SLM
    assert d_pii_ok.contains_pii is True

    # 4. Query with PII and flat/uncertain SLM logits -> ABSTAIN_HUMAN_REVIEW
    flat_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    d_pii_abstain = router.route(
        "Synthesize counterfactual legal liability for SSN 123-45-6789",
        slm_logits=flat_logits,
    )
    assert d_pii_abstain.target == RouteTarget.ABSTAIN_HUMAN_REVIEW

    # 5. Evaluate routing economics
    econ = evaluate_routing_economics([d_simple, d_complex, d_pii_ok, d_pii_abstain])
    assert econ["total_queries"] == 4.0
    assert econ["slm_offload_ratio"] == 0.5
    assert econ["cost_savings_percent"] > 50.0
    assert econ["latency_reduction_percent"] > 40.0
