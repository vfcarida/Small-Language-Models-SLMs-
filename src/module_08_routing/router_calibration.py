"""
==============================================================================
MODULE 08B — TRAINED ROUTER CALIBRATION (BRADLEY-TERRY & MATRIX FACTORIZATION)
==============================================================================
OBJECTIVE:
    Implement a trainable preference-calibrated routing model grounded in
    RouteLLM (Ong et al., ICLR 2025, arXiv:2406.18665):
    1. Structural 8D query feature representation phi(q).
    2. Bradley-Terry / Bilinear Matrix Factorization preference head predicting
       P(SLM suffices | q) = sigma(w^T phi(q) + b + (u_slm - u_llm)^T V phi(q)).
    3. Calibration metrics: Expected Calibration Error (ECE), Brier Score, and
       Performance Gap Recovered (PGR).
    4. Budget-constrained threshold solver and safe JSON checkpoint persistence.

RUN COMMAND:
    python src/module_08_routing/router_calibration.py
==============================================================================
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REASONING_MARKERS = (
    "prove",
    "derive",
    "theorem",
    "optimality",
    "convergence",
    "non-convex",
    "asymptotic",
    "formal",
    "manifold",
    "differential",
    "counterexample",
    "eigenvalue",
    "stiefel",
    "cayley",
)

_ROUTINE_MARKERS = (
    "classify",
    "extract",
    "summarize",
    "tag",
    "route",
    "translate",
    "format",
    "json",
    "intent",
    "severity",
    "bullet",
    "invoice",
    "ticket",
    "sentiment",
)

_MULTI_STEP_MARKERS = (
    "step-by-step",
    "first",
    "second",
    "finally",
    "compare and contrast",
    "architecture",
    "synthesize",
)


@dataclass(frozen=True)
class PreferenceSample:
    """Pairwise preference observation comparing local SLM vs. frontier LLM."""

    query: str
    slm_win_label: float  # 1.0 = SLM matches/wins, 0.5 = tie, 0.0 = LLM strictly required
    slm_quality: float = 0.85
    llm_quality: float = 0.95


def extract_query_features(query: str) -> torch.Tensor:
    """
    Extract an 8-dimensional normalized feature vector phi(q) in [0, 1]^8:
        [0] length_norm: normalized token count
        [1] ttr: lexical type-token diversity ratio
        [2] reasoning_density: formal math/proof/theory markers
        [3] symbolic_density: code/equation punctuation density
        [4] routine_task_affinity: structured extraction/classification/summary markers
        [5] numeric_density: proportion of numeric tokens
        [6] multi_step_density: multi-clause and multi-step reasoning markers
        [7] bias_term: constant 1.0
    """
    text = query.strip()
    if not text:
        return torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=torch.float32)

    lowered = text.lower()
    words = re.findall(r"\w+", lowered)
    n_words = max(len(words), 1)

    length_norm = min(n_words / 80.0, 1.0)
    ttr = len(set(words)) / n_words

    reasoning_hits = sum(1 for kw in _REASONING_MARKERS if kw in lowered)
    reasoning_density = min(reasoning_hits / 2.0, 1.0)

    symbolic_chars = sum(1 for ch in text if ch in "{}[]()=<>+*/\\^_")
    symbolic_density = min(symbolic_chars / max(len(text) * 0.15, 1.0), 1.0)

    routine_hits = sum(1 for kw in _ROUTINE_MARKERS if kw in lowered)
    routine_affinity = min(routine_hits / 2.0, 1.0)

    numeric_tokens = sum(1 for w in words if any(ch.isdigit() for ch in w))
    numeric_density = min(numeric_tokens / max(n_words * 0.4, 1.0), 1.0)

    multi_step_hits = sum(1 for kw in _MULTI_STEP_MARKERS if kw in lowered)
    multi_step_density = min(multi_step_hits / 2.0, 1.0)

    return torch.tensor(
        [
            float(length_norm),
            float(ttr),
            float(reasoning_density),
            float(symbolic_density),
            float(routine_affinity),
            float(numeric_density),
            float(multi_step_density),
            1.0,
        ],
        dtype=torch.float32,
    )


class CalibratedPreferenceRouter(nn.Module):
    """
    Bradley-Terry & Low-Rank Bilinear Matrix Factorization preference router
    (inspired by RouteLLM, Ong et al., ICLR 2025, arXiv:2406.18665).

    Predicts:
        P(SLM suffices | q) = sigma( (w^T phi(q) + (u_slm - u_llm)^T V phi(q)) / T )
    """

    def __init__(
        self,
        feature_dim: int = 8,
        embed_dim: int = 4,
        default_threshold: float = 0.55,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        self.calibrated_threshold = default_threshold

        self.linear = nn.Linear(feature_dim, 1, bias=True)
        self.query_proj = nn.Linear(feature_dim, embed_dim, bias=False)
        self.slm_embedding = nn.Parameter(torch.randn(embed_dim) * 0.1)
        self.llm_embedding = nn.Parameter(torch.randn(embed_dim) * 0.1)
        self.log_temperature = nn.Parameter(torch.zeros(1))

        self._init_default_weights()

    def _init_default_weights(self) -> None:
        """Initialize interpretable prior weights before fine-tuning on preference data."""
        with torch.no_grad():
            # Features: [len, ttr, reasoning, symbolic, routine, numeric, multi_step, bias]
            prior = torch.tensor(
                [[-1.2, -0.3, -3.5, -0.8, 2.6, 0.4, -2.2, 0.8]],
                dtype=torch.float32,
            )
            if self.linear.weight.shape == prior.shape:
                self.linear.weight.copy_(prior)
            self.linear.bias.fill_(0.2)
            nn.init.xavier_uniform_(self.query_proj.weight, gain=0.2)

    def forward_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Compute calibrated preference logits for a batch of query feature vectors."""
        if features.dim() == 1:
            features = features.unsqueeze(0)
        linear_term = self.linear(features).squeeze(-1)
        q_latent = self.query_proj(features)
        model_diff = self.slm_embedding - self.llm_embedding
        mf_term = torch.matmul(q_latent, model_diff)
        temperature = torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0)
        return (linear_term + mf_term) / temperature

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return calibrated probability P(SLM suffices | q) in (0, 1)."""
        return torch.sigmoid(self.forward_logits(features))

    def predict_win_probability(self, query: str) -> float:
        """Predict P(SLM suffices | query) for a single natural-language prompt."""
        self.eval()
        with torch.no_grad():
            feat = extract_query_features(query)
            prob = self.forward(feat).item()
        return round(float(prob), 4)

    def fit(
        self,
        samples: Sequence[PreferenceSample],
        epochs: int = 120,
        lr: float = 0.08,
        weight_decay: float = 1e-3,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Train the preference router via L2-regularized binary cross-entropy
        and return training & calibration metrics (loss, accuracy, Brier, ECE).
        """
        if not samples:
            raise ValueError("PreferenceSample sequence cannot be empty.")

        torch.manual_seed(seed)
        self.train()

        x = torch.stack([extract_query_features(s.query) for s in samples])
        y = torch.tensor([s.slm_win_label for s in samples], dtype=torch.float32)

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )

        with torch.no_grad():
            initial_probs = self.forward(x)
            initial_loss = F.binary_cross_entropy(initial_probs, y).item()

        for _ in range(epochs):
            optimizer.zero_grad()
            probs = self.forward(x)
            loss = F.binary_cross_entropy(probs, y)
            loss.backward()
            optimizer.step()

        self.eval()
        with torch.no_grad():
            final_probs = self.forward(x)
            final_loss = F.binary_cross_entropy(final_probs, y).item()
            preds = (final_probs >= self.calibrated_threshold).float()
            binary_targets = (y >= 0.5).float()
            accuracy = (preds == binary_targets).float().mean().item()
            brier_score = torch.mean((final_probs - y) ** 2).item()
            ece = compute_expected_calibration_error(final_probs, binary_targets)

        return {
            "initial_loss": round(initial_loss, 6),
            "final_loss": round(final_loss, 6),
            "accuracy": round(accuracy, 4),
            "brier_score": round(brier_score, 6),
            "ece": round(ece, 6),
        }

    def calibrate_threshold_for_budget(
        self,
        queries: Sequence[str],
        target_slm_offload_ratio: float = 0.75,
    ) -> float:
        """
        Solve for the threshold tau* in [0, 1] such that the fraction of queries
        with P(SLM suffices | q) >= tau* matches `target_slm_offload_ratio`.
        """
        if not queries:
            raise ValueError("queries cannot be empty.")
        if not (0.0 <= target_slm_offload_ratio <= 1.0):
            raise ValueError("target_slm_offload_ratio must lie in [0.0, 1.0].")

        probs_desc = sorted(
            (self.predict_win_probability(q) for q in queries), reverse=True
        )
        if target_slm_offload_ratio == 0.0:
            tau = 1.0
        elif target_slm_offload_ratio == 1.0:
            tau = 0.0
        elif len(probs_desc) == 1:
            tau = probs_desc[0]
        else:
            k = int(round(target_slm_offload_ratio * len(probs_desc)))
            k = max(1, min(k, len(probs_desc) - 1))
            tau = 0.5 * (probs_desc[k - 1] + probs_desc[k])
            tau = max(1e-4, min(1.0 - 1e-4, tau))

        self.calibrated_threshold = round(float(tau), 4)
        return self.calibrated_threshold

    def evaluate_pgr(
        self,
        samples: Sequence[PreferenceSample],
        threshold: float | None = None,
    ) -> Dict[str, float]:
        """
        Compute Percentage of Performance Gap Recovered (PGR, Ong et al., ICLR 2025):
            PGR = (Quality(Routed) - Quality(All-SLM)) / (Quality(All-LLM) - Quality(All-SLM))
        """
        if not samples:
            raise ValueError("samples cannot be empty.")

        tau = self.calibrated_threshold if threshold is None else threshold
        slm_scores = [s.slm_quality for s in samples]
        llm_scores = [s.llm_quality for s in samples]

        routed_scores: List[float] = []
        slm_count = 0
        for s in samples:
            win_prob = self.predict_win_probability(s.query)
            if win_prob >= tau:
                routed_scores.append(s.slm_quality)
                slm_count += 1
            else:
                routed_scores.append(s.llm_quality)

        avg_slm = sum(slm_scores) / len(samples)
        avg_llm = sum(llm_scores) / len(samples)
        avg_routed = sum(routed_scores) / len(samples)

        gap = max(avg_llm - avg_slm, 1e-8)
        pgr = (avg_routed - avg_slm) / gap

        return {
            "threshold": round(float(tau), 4),
            "slm_offload_ratio": round(slm_count / len(samples), 4),
            "avg_slm_quality": round(avg_slm, 4),
            "avg_llm_quality": round(avg_llm, 4),
            "avg_routed_quality": round(avg_routed, 4),
            "pgr": round(float(pgr), 4),
        }

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save router parameters and calibrated threshold safely as JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "format_version": "1.0",
            "architecture": "CalibratedPreferenceRouter-RouteLLM",
            "feature_dim": self.feature_dim,
            "embed_dim": self.embed_dim,
            "calibrated_threshold": self.calibrated_threshold,
            "state_dict": {
                k: v.detach().cpu().tolist() for k, v in self.state_dict().items()
            },
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out_path

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "CalibratedPreferenceRouter":
        """Load a CalibratedPreferenceRouter from a safe JSON checkpoint."""
        ckpt_path = Path(path)
        payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
        router = cls(
            feature_dim=int(payload["feature_dim"]),
            embed_dim=int(payload["embed_dim"]),
            default_threshold=float(payload["calibrated_threshold"]),
        )
        state_dict = {
            k: torch.tensor(v, dtype=torch.float32)
            for k, v in payload["state_dict"].items()
        }
        router.load_state_dict(state_dict, strict=True)
        router.eval()
        return router


def compute_expected_calibration_error(
    probs: torch.Tensor, labels: torch.Tensor, num_bins: int = 5
) -> float:
    """Compute Expected Calibration Error (ECE) across uniform probability bins."""
    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, device=probs.device)
    ece = torch.zeros(1, device=probs.device)
    n = probs.numel()
    for i in range(num_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == num_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = mask.sum().item()
        if count > 0:
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += (count / n) * torch.abs(avg_conf - avg_acc)
    return float(ece.item())


def generate_synthetic_preference_dataset(
    num_samples: int = 80, seed: int = 42
) -> List[PreferenceSample]:
    """
    Generate a reproducible preference calibration dataset combining routine
    SLM-amenable queries and complex frontier-LLM queries.
    """
    rng = random.Random(seed)
    routine_templates = [
        "Classify customer intent for message: 'Order #{id} delivery status check.'",
        "Extract JSON fields (vendor, amount, date) from invoice #{id} total ${amt}.",
        "Summarize in one sentence: Server cluster {id} maintained 99.95% uptime.",
        "Tag support ticket #{id} severity: 'User cannot reset account password.'",
        "Format as bullet points: Qwen2.5-1.5B fits in 1.2GB VRAM with AWQ INT4.",
        "Translate to English: 'Local edge inference eliminates external network latency.'",
        "Route inquiry #{id} to billing or technical support: 'Duplicate charge on card.'",
    ]
    frontier_templates = [
        "Prove the asymptotic convergence rate of non-convex Cayley manifold SGD for rank-{id} constraints.",
        "Derive a formal step-by-step optimality proof and counterexample for theorem #{id} on Stiefel manifolds.",
        "Synthesize a multi-region Byzantine-fault-tolerant consensus proof under asynchronous network partitions #{id}.",
        "Derive the closed-form eigenvalue spectrum of the Hessian for deep linear networks of depth {id}.",
    ]

    samples: List[PreferenceSample] = []
    for i in range(num_samples):
        if i % 3 != 0:
            tmpl = rng.choice(routine_templates)
            q = tmpl.format(id=1000 + i, amt=round(rng.uniform(25.0, 950.0), 2))
            samples.append(
                PreferenceSample(
                    query=q,
                    slm_win_label=1.0,
                    slm_quality=0.93,
                    llm_quality=0.95,
                )
            )
        else:
            tmpl = rng.choice(frontier_templates)
            q = tmpl.format(id=10 + i)
            samples.append(
                PreferenceSample(
                    query=q,
                    slm_win_label=0.0,
                    slm_quality=0.42,
                    llm_quality=0.94,
                )
            )
    return samples


def main() -> None:
    dataset = generate_synthetic_preference_dataset(num_samples=90, seed=42)
    router = CalibratedPreferenceRouter()
    metrics = router.fit(dataset, epochs=150, lr=0.08, seed=42)
    tau = router.calibrate_threshold_for_budget(
        [s.query for s in dataset], target_slm_offload_ratio=0.67
    )
    pgr_stats = router.evaluate_pgr(dataset, threshold=tau)
    ckpt_path = router.save_checkpoint("./results/router_checkpoint.json")

    logger.info("=" * 72)
    logger.info("MODULE 08B: ROUTE-LLM PREFERENCE CALIBRATION REPORT")
    logger.info("=" * 72)
    logger.info(
        f"Training: Initial Loss={metrics['initial_loss']:.4f} -> Final Loss={metrics['final_loss']:.4f} "
        f"| Accuracy={metrics['accuracy']*100:.1f}% | ECE={metrics['ece']:.4f}"
    )
    logger.info(
        f"Budget Calibration: tau*={tau:.4f} | SLM Offload={pgr_stats['slm_offload_ratio']*100:.1f}% "
        f"| PGR={pgr_stats['pgr']*100:.1f}% | Routed Quality={pgr_stats['avg_routed_quality']:.4f}"
    )
    logger.info(f"Saved calibrated router checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
