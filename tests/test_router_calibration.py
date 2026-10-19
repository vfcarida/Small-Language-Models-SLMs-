"""Unit tests for Module 08B Trained Router Calibration (src/module_08_routing/router_calibration.py)."""

from pathlib import Path
import pytest

from src.module_08_routing.hybrid_router import HybridSLMRouter, RouteTarget
from src.module_08_routing.router_calibration import (
    CalibratedPreferenceRouter,
    extract_query_features,
    generate_synthetic_preference_dataset,
)


def test_extract_query_features_shape_and_bounds() -> None:
    feat_simple = extract_query_features(
        "Extract JSON fields from invoice #102: Vendor=Acme, Amount=$45."
    )
    feat_complex = extract_query_features(
        "Prove the asymptotic convergence rate and formal optimality theorem on Stiefel manifolds."
    )
    assert feat_simple.shape == (8,)
    assert feat_complex.shape == (8,)
    assert float(feat_simple.min().item()) >= 0.0
    assert float(feat_simple.max().item()) <= 1.0
    # Routine task affinity (index 4) higher for invoice extraction; reasoning density (index 2) higher for theorem proof
    assert feat_simple[4] > feat_complex[4]
    assert feat_complex[2] > feat_simple[2]


def test_calibrated_preference_router_fit_and_pgr() -> None:
    dataset = generate_synthetic_preference_dataset(num_samples=60, seed=42)
    router = CalibratedPreferenceRouter()

    metrics = router.fit(dataset, epochs=120, lr=0.08, seed=42)
    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["accuracy"] >= 0.85
    assert 0.0 <= metrics["ece"] <= 0.5

    p_routine = router.predict_win_probability(
        "Classify customer intent for message: 'Check my order status.'"
    )
    p_proof = router.predict_win_probability(
        "Prove formal non-convex convergence theorem on Cayley manifolds step-by-step."
    )
    assert p_routine > p_proof

    tau = router.calibrate_threshold_for_budget(
        [s.query for s in dataset], target_slm_offload_ratio=0.65
    )
    assert 0.0 <= tau <= 1.0

    pgr_stats = router.evaluate_pgr(dataset, threshold=tau)
    assert pgr_stats["pgr"] > 0.70
    assert pgr_stats["avg_routed_quality"] > pgr_stats["avg_slm_quality"]


def test_router_checkpoint_save_and_load_parity(tmp_path: Path) -> None:
    dataset = generate_synthetic_preference_dataset(num_samples=30, seed=7)
    router = CalibratedPreferenceRouter()
    router.fit(dataset, epochs=50, lr=0.08, seed=7)
    router.calibrate_threshold_for_budget([s.query for s in dataset], 0.60)

    ckpt_file = router.save_checkpoint(tmp_path / "router_ckpt.json")
    assert ckpt_file.exists()

    loaded = CalibratedPreferenceRouter.load_checkpoint(ckpt_file)
    assert loaded.calibrated_threshold == router.calibrated_threshold

    test_q = "Summarize in one sentence: Server cluster 02 maintained 99.99% uptime."
    assert loaded.predict_win_probability(test_q) == pytest.approx(
        router.predict_win_probability(test_q), abs=1e-5
    )


def test_hybrid_slm_router_with_calibrated_preference_router() -> None:
    dataset = generate_synthetic_preference_dataset(num_samples=45, seed=42)
    cal_router = CalibratedPreferenceRouter()
    cal_router.fit(dataset, epochs=100, lr=0.08, seed=42)

    hybrid = HybridSLMRouter(
        complexity_threshold=0.60,
        confidence_threshold=0.60,
        calibrated_router=cal_router,
    )
    dec_local = hybrid.route("Extract JSON fields from invoice #901 total $88.00.")
    dec_cloud = hybrid.route(
        "Derive a formal step-by-step optimality proof and counterexample on Stiefel manifolds."
    )
    assert dec_local.target == RouteTarget.LOCAL_SLM
    assert dec_cloud.target == RouteTarget.CLOUD_LLM
