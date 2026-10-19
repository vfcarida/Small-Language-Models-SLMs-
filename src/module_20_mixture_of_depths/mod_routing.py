"""
==============================================================================
MODULE 20 — MIXTURE-OF-DEPTHS (MoD) & DYNAMIC TOKEN ROUTING FOR SLMs
==============================================================================
OBJECTIVE:
    Dynamically allocate Transformer/SSM layer FLOPs only to high-complexity
    context tokens while routing redundant tokens around the layer via the
    residual stream (Raposo et al., Google DeepMind 2024, arXiv:2404.02258):
    1. `MixtureOfDepthsBlock`: Capacity-bounded Top-K token router that gathers
       exactly K = max(1, floor(capacity_factor * seq_len)) tokens per sequence,
       executes the heavy sub-block f(x) exclusively on the gathered subset,
       and scatters weighted outputs back into the residual tensor.
    2. `compute_mod_auxiliary_loss`: Binary cross-entropy (BCE) causal predictor
       loss training an O(1) autoregressive threshold router so inference does
       not require non-causal sequence-wide Top-K lookahead.
    3. `profile_mod_compute_savings`: Analytical FLOPs and prefill/decode
       speedup profiler across capacity factors gamma in (0, 1].

RUN COMMAND:
    python src/module_20_mixture_of_depths/mod_routing.py
==============================================================================
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MixtureOfDepthsBlock(nn.Module):
    """
    Mixture-of-Depths (MoD) Dynamic Token Routing Layer
    (Raposo et al., Google DeepMind 2024, arXiv:2404.02258):

    Given input sequence X in R^(B x L x d_model) and capacity ratio gamma in (0, 1]:
        1. Router scores each token: r_{b, t} = sigmoid(w_r^T x_{b, t}) in (0, 1).
        2. Select top K = max(1, int(gamma * L)) token indices I_b per batch item.
        3. Gather X_selected in R^(B x K x d_model) and compute heavy transformation
           Y_selected = f_block(X_selected) * r_{b, I_b}.
        4. Scatter-add Y_selected back into X (the remaining L - K tokens bypass
           f_block with zero FLOPs expended).
    """

    def __init__(
        self,
        d_model: int = 64,
        capacity_factor: float = 0.5,
    ) -> None:
        super().__init__()
        if not (0.0 < capacity_factor <= 1.0):
            raise ValueError("capacity_factor must be in (0, 1].")

        self.d_model = d_model
        self.capacity_factor = capacity_factor

        # Primary token router and causal auxiliary predictor (for autoregressive inference)
        self.router = nn.Linear(d_model, 1, bias=False)
        self.causal_aux_predictor = nn.Linear(d_model, 1, bias=True)

        # Heavy Transformer FFN / SwiGLU sub-block executed only on Top-K tokens
        self.norm = nn.LayerNorm(d_model)
        self.heavy_block = nn.Sequential(
            nn.Linear(d_model, d_model * 4, bias=False),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model, bias=False),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Execute capacity-bounded Top-K token routing.

        Returns:
            Tuple of (output_tensor, routing_telemetry_dict).
        """
        if x.ndim != 3:
            raise ValueError("Input x must have shape (batch, seq_len, d_model).")

        batch_size, seq_len, d = x.shape
        k_cap = max(1, int(self.capacity_factor * seq_len))

        # 1. Compute router weights r in (0, 1) of shape (B, L)
        router_logits = self.router(x).squeeze(-1)
        router_weights = torch.sigmoid(router_logits)

        # 2. Select Top-K token positions per sequence
        top_vals, top_indices = torch.topk(router_weights, k=k_cap, dim=-1, sorted=False)
        # Sort indices to preserve causal relative token order
        top_indices_sorted, perm = torch.sort(top_indices, dim=-1)
        top_vals_sorted = torch.gather(top_vals, dim=-1, index=perm)

        # 3. Gather selected tokens: shape (B, K, d_model)
        gather_idx = top_indices_sorted.unsqueeze(-1).expand(-1, -1, d)
        x_selected = torch.gather(x, dim=1, index=gather_idx)

        # 4. Run heavy block ONLY on the K selected tokens and weight by router score
        block_out = self.heavy_block(self.norm(x_selected))
        weighted_out = block_out * top_vals_sorted.unsqueeze(-1)

        # 5. Residual scatter-add back into full sequence tensor
        output = x.clone()
        output.scatter_add_(dim=1, index=gather_idx, src=weighted_out.to(x.dtype))

        # Build binary target mask (1 if token was selected in Top-K, 0 otherwise)
        top_k_mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=x.device)
        top_k_mask.scatter_(dim=1, index=top_indices_sorted, value=1.0)

        # Auxiliary causal predictor logits (stop-gradient on x so aux head doesn't corrupt backbone)
        aux_logits = self.causal_aux_predictor(x.detach()).squeeze(-1)

        telemetry = {
            "top_k_mask": top_k_mask,
            "aux_logits": aux_logits,
            "router_weights": router_weights,
            "routed_token_fraction": torch.tensor(float(k_cap) / float(seq_len), device=x.device),
        }
        return output, telemetry


def compute_mod_auxiliary_loss(
    aux_logits: torch.Tensor,
    top_k_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Binary Cross-Entropy (BCE) loss training the causal auxiliary predictor
    to predict whether a token will be included in the sequence Top-K set
    (Raposo et al., 2024, Section 2.2).
    """
    return F.binary_cross_entropy_with_logits(aux_logits.float(), top_k_mask.float())


def profile_mod_compute_savings(
    seq_len: int = 4096,
    d_model: int = 1536,
    num_layers: int = 28,
    capacity_factor: float = 0.25,
    mod_interleave_step: int = 2,  # Apply MoD on every 2nd layer (14 MoD + 14 Dense layers)
) -> Dict[str, float]:
    """
    Calculate per-pass FLOPs (in GFLOPs) for a dense Transformer SLM vs. an
    interleaved Mixture-of-Depths SLM (Raposo et al., 2024).
    """
    if seq_len < 1 or not (0.0 < capacity_factor <= 1.0):
        raise ValueError("Invalid seq_len or capacity_factor.")

    # Per-layer FLOPs (Attention + SwiGLU FFN): ~24 * L * d^2 + 4 * L^2 * d
    def _layer_flops(n_tokens: int) -> float:
        return 24.0 * n_tokens * (d_model**2) + 4.0 * (n_tokens**2) * d_model

    dense_layer_f = _layer_flops(seq_len)
    k_tokens = max(1, int(capacity_factor * seq_len))
    # Router overhead is 2 * L * d_model (negligible compared to 24 * L * d^2)
    mod_layer_f = _layer_flops(k_tokens) + 2.0 * seq_len * d_model

    num_mod_layers = num_layers // max(1, mod_interleave_step)
    num_dense_layers = num_layers - num_mod_layers

    total_dense_gflops = (num_layers * dense_layer_f) / 1e9
    total_mod_gflops = (
        num_dense_layers * dense_layer_f + num_mod_layers * mod_layer_f
    ) / 1e9

    savings_pct = ((total_dense_gflops - total_mod_gflops) / total_dense_gflops) * 100.0
    return {
        "dense_gflops": round(total_dense_gflops, 2),
        "mod_gflops": round(total_mod_gflops, 2),
        "flops_reduction_percent": round(savings_pct, 2),
        "speedup_multiplier_x": round(total_dense_gflops / max(total_mod_gflops, 1e-6), 2),
    }


def main() -> None:
    torch.manual_seed(42)
    mod_block = MixtureOfDepthsBlock(d_model=64, capacity_factor=0.25)
    x = torch.randn(2, 16, 64, requires_grad=True)

    out, telemetry = mod_block(x)
    aux_loss = compute_mod_auxiliary_loss(telemetry["aux_logits"], telemetry["top_k_mask"])
    total_loss = out.pow(2).mean() + 0.1 * aux_loss
    total_loss.backward()

    prof = profile_mod_compute_savings(
        seq_len=4096, d_model=1536, num_layers=28, capacity_factor=0.25, mod_interleave_step=1
    )

    logger.info("=" * 72)
    logger.info("MODULE 20: MIXTURE-OF-DEPTHS (MoD) DYNAMIC TOKEN ROUTING")
    logger.info("=" * 72)
    logger.info(
        f"[MoD Forward/Backward] Routed Fraction={telemetry['routed_token_fraction'].item():.1%} "
        f"(75.0% tokens bypassed FFN) | Aux BCE Loss={aux_loss.item():.4f}"
    )
    logger.info(
        f"[4k Context Compute] Dense SLM={prof['dense_gflops']:.1f} GFLOPs "
        f"vs. MoD SLM (gamma=0.25)={prof['mod_gflops']:.1f} GFLOPs "
        f"(FLOPs Reduction={prof['flops_reduction_percent']:.1f}%, Speedup={prof['speedup_multiplier_x']:.2f}x)"
    )


if __name__ == "__main__":
    main()
