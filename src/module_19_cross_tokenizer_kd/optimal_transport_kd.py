"""
==============================================================================
MODULE 19 — CROSS-TOKENIZER KNOWLEDGE DISTILLATION VIA OPTIMAL TRANSPORT (ULD / MinED)
==============================================================================
OBJECTIVE:
    Distill knowledge from a frontier Teacher LLM with vocabulary V_T into a
    compact Student SLM with an incompatible vocabulary V_S (|V_T| != |V_S|):
    1. `compute_uld_wasserstein_loss`: Universal Logit Distillation (ULD,
       Boizard et al., ICML/TMLR 2024, arXiv:2402.12030) via sorted rank-quantile
       Optimal Transport L1 distance with automatic zero-padding across
       mismatched vocabulary dimensions.
    2. `align_cross_tokenizer_vocab`: Minimum Edit Distance (MinED, Wan et al.,
       ACL 2024, arXiv:2401.14691) subword string alignment producing a
       probability-conserving projection matrix M in R^(|V_T| x |V_S|).
    3. `CrossTokenizerDistiller`: Hybrid differentiable loss combining ULD
       Wasserstein rank matching + MinED projected KL divergence.

RUN COMMAND:
    python src/module_19_cross_tokenizer_kd/optimal_transport_kd.py
==============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _canonicalize_token(token_str: str) -> str:
    """Strip BPE/SentencePiece/WordPiece leading markers ('Ġ', '▁', '##') and lowercase."""
    cleaned = token_str.lstrip("Ġ▁").removeprefix("##").strip().lower()
    return cleaned if cleaned else token_str.lower()


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute exact Levenshtein character edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_uld_wasserstein_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.5,
) -> torch.Tensor:
    """
    Compute the Universal Logit Distillation (ULD) Optimal Transport loss
    (Boizard et al., 2024, arXiv:2402.12030) between student logits of shape
    (..., |V_S|) and teacher logits of shape (..., |V_T|) where |V_S| != |V_T|:
        1. p_S = softmax(z_S / T),  p_T = softmax(z_T / T)
        2. Sort p_S and p_T in descending order along the vocabulary axis.
        3. Zero-pad the smaller vocabulary to V_max = max(|V_S|, |V_T|).
        4. Return T * mean(sum_{r=1}^{V_max} |p_{S, r}^down - p_{T, r}^down|).
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be strictly positive.")
    if student_logits.shape[:-1] != teacher_logits.shape[:-1]:
        raise ValueError("Batch/sequence dimensions of student_logits and teacher_logits must match.")

    p_s = F.softmax(student_logits.float() / temperature, dim=-1)
    p_t = F.softmax(teacher_logits.detach().float() / temperature, dim=-1)

    p_s_sorted, _ = torch.sort(p_s, dim=-1, descending=True)
    p_t_sorted, _ = torch.sort(p_t, dim=-1, descending=True)

    vs = p_s_sorted.shape[-1]
    vt = p_t_sorted.shape[-1]
    if vs < vt:
        p_s_sorted = F.pad(p_s_sorted, (0, vt - vs), value=0.0)
    elif vt < vs:
        p_t_sorted = F.pad(p_t_sorted, (0, vs - vt), value=0.0)

    l1_ot = torch.sum(torch.abs(p_s_sorted - p_t_sorted), dim=-1)
    return (temperature * l1_ot.mean()).to(student_logits.dtype)


def align_cross_tokenizer_vocab(
    student_tokens: Sequence[str],
    teacher_tokens: Sequence[str],
) -> Tuple[torch.Tensor, float]:
    """
    Build a probability-conserving cross-tokenizer projection matrix
    M in R^(|V_T| x |V_S|) via Minimum Edit Distance (MinED, Wan et al., ACL 2024):
    Each teacher token t in V_T is mapped to its closest canonical student token
    s*(t) = argmin_{s in V_S} Levenshtein(canon(t), canon(s)).

    Multiplying a teacher distribution p_T in R^(..., |V_T|) by M produces a valid
    projected probability distribution p_T_proj = p_T @ M in R^(..., |V_S|)
    whose elements are non-negative and sum to 1.0.

    Returns:
        Tuple of (projection_matrix_T_to_S, exact_match_fraction).
    """
    if not student_tokens or not teacher_tokens:
        raise ValueError("student_tokens and teacher_tokens must be non-empty.")

    canon_s = [_canonicalize_token(s) for s in student_tokens]
    canon_t = [_canonicalize_token(t) for t in teacher_tokens]

    # Fast exact-match lookup table from canonical string -> student index
    exact_s_map: Dict[str, int] = {}
    for s_idx, cs in enumerate(canon_s):
        if cs not in exact_s_map:
            exact_s_map[cs] = s_idx

    vt = len(teacher_tokens)
    vs = len(student_tokens)
    proj = torch.zeros((vt, vs), dtype=torch.float32)
    exact_matches = 0

    for t_idx, ct in enumerate(canon_t):
        if ct in exact_s_map:
            best_s = exact_s_map[ct]
            exact_matches += 1
        else:
            best_s = min(
                range(vs),
                key=lambda s_idx: levenshtein_distance(ct, canon_s[s_idx]),
            )
        proj[t_idx, best_s] = 1.0

    exact_fraction = exact_matches / float(vt)
    return proj, round(exact_fraction, 4)


class CrossTokenizerDistiller:
    """
    Hybrid Cross-Tokenizer Knowledge Distillation combining:
        - ULD Wasserstein-1 sorted rank loss (handles unaligned vocabulary tails)
        - MinED projected KL divergence (preserves exact token identity on shared stems)
    """

    def __init__(
        self,
        student_tokens: Sequence[str],
        teacher_tokens: Sequence[str],
        uld_weight: float = 0.5,
        temperature: float = 1.5,
    ) -> None:
        if not (0.0 <= uld_weight <= 1.0):
            raise ValueError("uld_weight must be in [0, 1].")
        self.uld_weight = uld_weight
        self.temperature = temperature
        self.proj_t_to_s, self.exact_match_rate = align_cross_tokenizer_vocab(
            student_tokens, teacher_tokens
        )

    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid cross-tokenizer distillation loss for student_logits (..., |V_S|)
        and teacher_logits (..., |V_T|).
        """
        loss_uld = compute_uld_wasserstein_loss(
            student_logits, teacher_logits, temperature=self.temperature
        )

        # Project teacher distribution onto student vocabulary space via MinED matrix M
        proj = self.proj_t_to_s.to(device=student_logits.device, dtype=torch.float32)
        p_t = F.softmax(teacher_logits.detach().float() / self.temperature, dim=-1)
        p_t_projected = torch.matmul(p_t, proj).clamp(min=1e-8)

        log_p_s = F.log_softmax(student_logits.float() / self.temperature, dim=-1)
        kl_mined = (
            (self.temperature**2)
            * torch.sum(p_t_projected * (torch.log(p_t_projected) - log_p_s), dim=-1).mean()
        )

        total_loss = (
            self.uld_weight * loss_uld.float() + (1.0 - self.uld_weight) * kl_mined
        ).to(student_logits.dtype)

        metrics = {
            "total_loss": round(float(total_loss.item()), 4),
            "uld_wasserstein_loss": round(float(loss_uld.item()), 4),
            "mined_projected_kl_loss": round(float(kl_mined.item()), 4),
            "vocab_exact_match_rate": self.exact_match_rate,
        }
        return total_loss, metrics


def main() -> None:
    torch.manual_seed(42)
    # Simulate SentencePiece Student (|V_S|=8) vs Byte-Level BPE Teacher (|V_T|=12)
    student_vocab = ["▁the", "▁model", "▁quant", "ization", "▁fast", "▁edge", "▁slm", "<eos>"]
    teacher_vocab = [
        "Ġthe",
        "Ġmodel",
        "Ġquantization",
        "Ġquant",
        "##ization",
        "Ġfast",
        "Ġfaster",
        "Ġedge",
        "Ġcloud",
        "Ġslm",
        "Ġllm",
        "<|endoftext|>",
    ]

    distiller = CrossTokenizerDistiller(
        student_tokens=student_vocab,
        teacher_tokens=teacher_vocab,
        uld_weight=0.5,
        temperature=1.5,
    )

    s_logits = torch.randn(2, 6, len(student_vocab), requires_grad=True)
    t_logits = torch.randn(2, 6, len(teacher_vocab))

    loss, stats = distiller.compute_loss(s_logits, t_logits)
    loss.backward()

    logger.info("=" * 72)
    logger.info("MODULE 19: CROSS-TOKENIZER KNOWLEDGE DISTILLATION (ULD + MinED)")
    logger.info("=" * 72)
    logger.info(
        f"[Vocab Alignment] Teacher |V_T|={len(teacher_vocab)} -> Student |V_S|={len(student_vocab)} "
        f"| Canonical Exact Match={stats['vocab_exact_match_rate']:.1%}"
    )
    logger.info(
        f"[Hybrid OT-KD] Total Loss={stats['total_loss']:.4f} "
        f"(ULD Wasserstein={stats['uld_wasserstein_loss']:.4f}, MinED KL={stats['mined_projected_kl_loss']:.4f}) "
        f"| Student Grad Norm={torch.linalg.norm(s_logits.grad):.4f}"
    )


if __name__ == "__main__":
    main()
