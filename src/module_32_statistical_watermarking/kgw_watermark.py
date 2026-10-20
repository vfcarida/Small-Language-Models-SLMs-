"""
Module 32: Statistical Watermarking (KGW Green-List Logit Bias) for AI TRiSM
============================================================================

Implements the Kirchenbauer-Geiping-Wen (KGW) cryptographic green-list statistical
watermarking framework (ICML 2023) for Small Language Model (<4B) output provenance
and EU AI Act Article 50 / Gartner AI TRiSM regulatory compliance:
1. ``partition_green_red_vocab``: Deterministically hashes the preceding token
   $x_{t-1}$ (and secret key $K$) via SHA-256 to partition the vocabulary $V$ into
   a "green list" $G_t$ of size $\\lfloor \\gamma |V| \\rfloor$ and a "red list" $R_t$.
2. ``apply_kgw_watermark_logits``: Adds a soft watermark bias $\\delta > 0$ to green-list
   token logits prior to softmax sampling: $\\tilde{\\ell}_{t, v} = \\ell_{t, v} + \\delta \\cdot \\mathbb{I}[v \\in G_t]$.
3. ``detect_kgw_watermark_zscore``: Computes the one-sided binomial $z$-statistic
   and Gaussian tail $p$-value without requiring access to the model weights:
   $$z = \\frac{|S|_G - \\gamma T}{\\sqrt{T \\gamma (1 - \\gamma)}}$$

References:
    - Kirchenbauer et al. (ICML 2023). "A Watermark for Large Language Models."
      arXiv:2301.10226.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Sequence

import torch


def partition_green_red_vocab(
    prev_token_id: int,
    vocab_size: int,
    gamma: float = 0.25,
    secret_salt: int = 15485863,
) -> torch.Tensor:
    """Deterministically select green-list vocabulary indices $G_t \\subset \\{0, \\dots, |V|-1\\}$.

    Uses SHA-256 over ``(secret_salt, prev_token_id)`` to seed a local PyTorch
    generator, ensuring cross-platform reproducibility for zero-knowledge detection.

    Args:
        prev_token_id: Preceding token ID $x_{t-1}$.
        vocab_size: Vocabulary cardinality $|V|$.
        gamma: Green-list fraction $\\gamma \\in (0, 1)$.
        secret_salt: Secret integer key shared between generator and provenance verifier.

    Returns:
        1D ``BoolTensor`` of length ``vocab_size`` where ``True`` marks green-list tokens.
    """
    if vocab_size < 2:
        raise ValueError(f"vocab_size must be >= 2, got {vocab_size}.")
    if not (0.0 < gamma < 1.0):
        raise ValueError(f"gamma must be in (0, 1), got {gamma}.")

    digest = hashlib.sha256(f"{int(secret_salt)}:{int(prev_token_id)}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], byteorder="little") & 0x7FFFFFFFFFFFFFFF
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)

    green_count = max(1, min(vocab_size - 1, int(round(gamma * vocab_size))))
    perm = torch.randperm(vocab_size, generator=gen)
    green_mask = torch.zeros(vocab_size, dtype=torch.bool)
    green_mask[perm[:green_count]] = True
    return green_mask


def apply_kgw_watermark_logits(
    logits: torch.Tensor,
    prev_token_ids: Sequence[int],
    gamma: float = 0.25,
    delta: float = 2.0,
    secret_salt: int = 15485863,
) -> torch.Tensor:
    """Apply KGW soft green-list logit bias $\\delta > 0$ to next-token logits.

    $$\\tilde{\\ell}_{b, v} = \\ell_{b, v} + \\delta \\cdot \\mathbb{I}[v \\in G(x_{b, t-1})]$$

    Args:
        logits: Next-token logits of shape ``(B, vocab_size)`` or ``(vocab_size,)``.
        prev_token_ids: Sequence of $B$ preceding token IDs (or length 1 for 1D logits).
        gamma: Green-list fraction $\\gamma \\in (0, 1)$.
        delta: Soft watermark bias $\\delta \\ge 0$ added to green-list logits.
        secret_salt: Secret integer key.

    Returns:
        Watermarked logit tensor of the same shape as ``logits``.
    """
    if delta < 0.0:
        raise ValueError(f"delta must be non-negative, got {delta}.")

    squeeze_out = False
    if logits.ndim == 1:
        logits_2d = logits.unsqueeze(0)
        squeeze_out = True
    elif logits.ndim == 2:
        logits_2d = logits
    else:
        raise ValueError("logits must be 1D (vocab_size,) or 2D (B, vocab_size).")

    bsz, vocab_size = logits_2d.shape
    if len(prev_token_ids) != bsz:
        raise ValueError(
            f"Length of prev_token_ids ({len(prev_token_ids)}) must match batch size ({bsz})."
        )

    out = logits_2d.clone()
    for b, prev_tok in enumerate(prev_token_ids):
        mask = partition_green_red_vocab(
            prev_token_id=int(prev_tok),
            vocab_size=vocab_size,
            gamma=gamma,
            secret_salt=secret_salt,
        ).to(device=out.device)
        out[b, mask] = out[b, mask] + float(delta)

    return out.squeeze(0) if squeeze_out else out


def detect_kgw_watermark_zscore(
    token_ids: Sequence[int],
    vocab_size: int,
    gamma: float = 0.25,
    z_threshold: float = 4.0,
    secret_salt: int = 15485863,
) -> Dict[str, object]:
    """Compute the KGW statistical watermark $z$-score and $p$-value on a token sequence.

    Under the null hypothesis $H_0$ (human or unwatermarked text), each token
    $x_t$ ($t \\ge 1$) falls in $G(x_{t-1})$ with independent probability $\\gamma$:

    $$z = \\frac{|S|_G - \\gamma T}{\\sqrt{T \\gamma (1 - \\gamma)}}, \\quad p = \\frac{1}{2} \\text{erfc}\\!\\left(\\frac{z}{\\sqrt{2}}\\right)$$

    Args:
        token_ids: Sequence of token IDs of length $L \\ge 2$ ($T = L - 1$ scored transitions).
        vocab_size: Vocabulary size $|V|$.
        gamma: Green-list fraction $\\gamma \\in (0, 1)$ used during generation.
        z_threshold: Detection threshold (default $4.0$ corresponds to $p \\approx 3.17 \\times 10^{-5}$).
        secret_salt: Secret integer key.

    Returns:
        Dictionary containing:
            - ``"z_score"``: One-sided normal approximation $z$-statistic.
            - ``"p_value"``: One-sided false-positive probability under $H_0$.
            - ``"is_watermarked"``: Boolean flag ``z_score >= z_threshold``.
            - ``"green_token_count"``: Number of observed green tokens $|S|_G$.
            - ``"num_tokens_scored"``: Number of transitions $T = L - 1$.
            - ``"green_fraction"``: Empirical ratio $|S|_G / T$.
    """
    if len(token_ids) < 2:
        raise ValueError("token_ids must contain at least 2 tokens to score transitions.")

    green_hits = 0
    total_scored = len(token_ids) - 1
    for t in range(1, len(token_ids)):
        prev_tok = int(token_ids[t - 1])
        curr_tok = int(token_ids[t])
        if curr_tok < 0 or curr_tok >= vocab_size:
            raise ValueError(f"Token ID {curr_tok} out of range [0, {vocab_size - 1}].")
        green_mask = partition_green_red_vocab(
            prev_token_id=prev_tok,
            vocab_size=vocab_size,
            gamma=gamma,
            secret_salt=secret_salt,
        )
        if bool(green_mask[curr_tok].item()):
            green_hits += 1

    expected_green = gamma * total_scored
    std_dev = math.sqrt(total_scored * gamma * (1.0 - gamma))
    z_score = (green_hits - expected_green) / max(std_dev, 1e-12)
    p_value = 0.5 * math.erfc(z_score / math.sqrt(2.0))

    return {
        "z_score": float(z_score),
        "p_value": float(p_value),
        "is_watermarked": bool(z_score >= z_threshold),
        "green_token_count": green_hits,
        "num_tokens_scored": total_scored,
        "green_fraction": float(green_hits) / float(total_scored),
    }


def main() -> None:
    """Run a deterministic demonstration of KGW watermarking and z-score provenance detection."""
    torch.manual_seed(42)
    vocab_size = 128
    seq: List[int] = [0]
    for _ in range(60):
        raw_logits = torch.zeros(vocab_size)
        wm_logits = apply_kgw_watermark_logits(raw_logits, [seq[-1]], gamma=0.25, delta=3.5)
        probs = torch.softmax(wm_logits, dim=-1)
        next_tok = int(torch.multinomial(probs, num_samples=1).item())
        seq.append(next_tok)
    det = detect_kgw_watermark_zscore(seq, vocab_size=vocab_size, gamma=0.25)
    print(
        f"[Module 32] KGW Watermark z_score={det['z_score']:.2f} | "
        f"p_value={det['p_value']:.2e} | detected={det['is_watermarked']}"
    )


if __name__ == "__main__":
    main()
