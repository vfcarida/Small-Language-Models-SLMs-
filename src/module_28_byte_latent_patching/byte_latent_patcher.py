"""
Module 28: Byte-Latent & Dynamic Patch Tokenization (BLT) for Tokenizer-Free SLMs
=================================================================================

Implements entropy-driven dynamic byte patching (Pagnoni et al., Meta FAIR 2024)
to allocate Transformer compute proportionally to information density:
1. ``compute_byte_stream_entropy``: Computes next-byte Shannon entropy
   $H(B_t \\mid b_{<t}) = -\\sum_{v=0}^{255} p_t(v) \\log_2 p_t(v) \\in [0, 8]$ bits.
2. ``segment_bytes_into_dynamic_patches``: Partitions raw UTF-8 byte streams into
   variable-length patches whenever next-byte entropy exceeds a global threshold
   $H_t > \\theta_{\\text{global}}$ or exhibits a monotonic spike
   $H_t - H_{t-1} > \\theta_{\\text{delta}}$.
3. ``ByteLatentPatchPooling``: Cross-attention / gated pooling encoder mapping
   variable-length byte spans into fixed-dimension latent patch representations.

References:
    - Pagnoni et al. (Meta FAIR, 2024). "Byte Latent Transformer: Patches Scale
      Better Than Tokens." arXiv:2412.09871.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_byte_stream_entropy(byte_logits: torch.Tensor) -> torch.Tensor:
    """Compute base-2 Shannon entropy (in bits) from next-byte logits.

    $$H(p_t) = -\\sum_{v=0}^{V-1} p_t(v) \\log_2 p_t(v)$$

    Args:
        byte_logits: Tensor of shape ``(..., vocab_size)`` (typically ``vocab_size=256``).

    Returns:
        Entropy tensor in bits of shape ``(...)`` with values in $[0, \\log_2 V]$.
    """
    if byte_logits.ndim < 1 or byte_logits.shape[-1] < 2:
        raise ValueError("byte_logits must have last dimension >= 2.")
    probs = F.softmax(byte_logits, dim=-1)
    log2_probs = F.log_softmax(byte_logits, dim=-1) / math.log(2.0)
    entropy = -(probs * log2_probs).sum(dim=-1)
    return entropy.clamp(min=0.0)


def segment_bytes_into_dynamic_patches(
    byte_ids: Sequence[int],
    byte_entropies: Sequence[float],
    global_entropy_threshold: float = 2.5,
    monotonic_delta_threshold: float = 1.2,
    max_patch_bytes: int = 16,
) -> Dict[str, object]:
    """Segment a byte sequence into dynamic BLT patches via entropy boundaries.

    A new patch boundary is started at byte position $t > 0$ if any of the
    following holds (Pagnoni et al., Meta FAIR 2024):
    1. **Global Entropy Threshold:** $H_t > \\theta_{\\text{global}}$ (high uncertainty at
       morpheme or word starts).
    2. **Monotonic Entropy Jump:** $H_t - H_{t-1} > \\theta_{\\text{delta}}$ (sudden transition
       from predictable continuation to unpredictable branch).
    3. **Maximum Patch Length Cap:** Current patch reaches ``max_patch_bytes``.

    Args:
        byte_ids: Sequence of integer byte values in $[0, 255]$ of length $L$.
        byte_entropies: Sequence of next-byte entropies $H_t$ of length $L$.
        global_entropy_threshold: Global threshold $\\theta_{\\text{global}}$ in bits.
        monotonic_delta_threshold: Relative jump threshold $\\theta_{\\text{delta}}$ in bits.
        max_patch_bytes: Maximum number of bytes allowed in a single patch.

    Returns:
        Dictionary containing:
            - ``"patches"``: List of byte-ID lists representing variable-length patches.
            - ``"patch_spans"``: List of ``(start_idx, end_idx)`` half-open intervals.
            - ``"num_patches"``: Total number of patches $P$.
            - ``"compression_ratio"``: Ratio $L / P$ (average bytes per patch).
    """
    if len(byte_ids) == 0:
        raise ValueError("byte_ids must be non-empty.")
    if len(byte_ids) != len(byte_entropies):
        raise ValueError("byte_ids and byte_entropies must have equal length.")
    if max_patch_bytes <= 0:
        raise ValueError("max_patch_bytes must be positive.")

    patches: List[List[int]] = []
    patch_spans: List[ tuple[int, int] ] = []

    current_start = 0
    current_patch: List[int] = [int(byte_ids[0])]

    for t in range(1, len(byte_ids)):
        h_curr = float(byte_entropies[t])
        h_prev = float(byte_entropies[t - 1])
        is_global_spike = h_curr > global_entropy_threshold
        is_monotonic_jump = (h_curr - h_prev) > monotonic_delta_threshold
        is_max_len = len(current_patch) >= max_patch_bytes

        if is_global_spike or is_monotonic_jump or is_max_len:
            patches.append(current_patch)
            patch_spans.append((current_start, t))
            current_start = t
            current_patch = [int(byte_ids[t])]
        else:
            current_patch.append(int(byte_ids[t]))

    patches.append(current_patch)
    patch_spans.append((current_start, len(byte_ids)))

    num_patches = len(patches)
    return {
        "patches": patches,
        "patch_spans": patch_spans,
        "num_patches": num_patches,
        "compression_ratio": float(len(byte_ids)) / float(num_patches),
    }


class ByteLatentPatchPooling(nn.Module):
    """Cross-attention pooling encoder mapping variable-length byte patches to SLM latents.

    Embeds raw bytes $b \\in \\{0, \\dots, 255\\}$ into $\\mathbb{R}^{d_{\\text{model}}}$ and
    applies a learned query cross-attention pooling head over each variable-length
    span $[s_k, e_k)$ to produce patch latent vectors $\\mathbf{P} \\in \\mathbb{R}^{P \\times d_{\\text{model}}}$.
    """

    def __init__(self, d_model: int, byte_vocab_size: int = 256) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        self.d_model = d_model
        self.byte_embedding = nn.Embedding(byte_vocab_size, d_model)
        self.patch_query = nn.Parameter(torch.randn(1, d_model) / math.sqrt(d_model))
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.val_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        byte_ids: torch.Tensor,
        patch_spans: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """Pool 1D ``byte_ids`` tensor of length $L$ into ``(num_patches, d_model)``."""
        if byte_ids.ndim != 1:
            raise ValueError("byte_ids must be a 1D tensor of byte indices.")
        if len(patch_spans) == 0:
            raise ValueError("patch_spans must contain at least one span.")

        byte_embs = self.byte_embedding(byte_ids)  # (L, d_model)
        keys = self.key_proj(byte_embs)
        vals = self.val_proj(byte_embs)
        scale = 1.0 / math.sqrt(self.d_model)

        pooled_patches: List[torch.Tensor] = []
        for start, end in patch_spans:
            k_span = keys[start:end]  # (span_len, d_model)
            v_span = vals[start:end]
            attn_logits = torch.matmul(self.patch_query, k_span.t()) * scale  # (1, span_len)
            attn_weights = F.softmax(attn_logits, dim=-1)
            patch_vec = torch.matmul(attn_weights, v_span).squeeze(0) + byte_embs[start:end].mean(dim=0)
            pooled_patches.append(self.out_norm(patch_vec))

        return torch.stack(pooled_patches, dim=0)


def main() -> None:
    """Run a deterministic demonstration of BLT entropy patching and pooling."""
    text = b"Small Language Models scale with dynamic byte patches!"
    byte_ids = list(text)
    entropies = [3.2 if chr(b) == " " else 0.4 for b in byte_ids]
    seg = segment_bytes_into_dynamic_patches(byte_ids, entropies)
    pooler = ByteLatentPatchPooling(d_model=32)
    latents = pooler(torch.tensor(byte_ids, dtype=torch.long), seg["patch_spans"])
    print(
        f"[Module 28] Bytes={len(byte_ids)} -> Patches={seg['num_patches']} "
        f"({seg['compression_ratio']:.2f} bytes/patch) | Latent shape={tuple(latents.shape)}"
    )


if __name__ == "__main__":
    main()
