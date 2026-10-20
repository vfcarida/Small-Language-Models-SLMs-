"""
Module 25: Matryoshka Representation Learning (MRL) & Elastic Sub-Network Slicing
=================================================================================

Implements nested multi-granularity representations and activation-based structural
importance ranking for elastic Small Language Models (<4B parameters):
1. ``compute_matryoshka_loss``: Multi-granularity cross-entropy objective over nested
   embedding dimensions $\\mathcal{M} = \\{d_1, d_2, \\dots, d_k\\}$ (Kusupati et al., NeurIPS 2022).
2. ``rank_minitron_structural_importance``: Activation-norm width importance and
   cosine Block Influence (BI) depth importance ranking (Muralidharan et al.,
   NVIDIA Minitron, NeurIPS 2024).
3. ``ElasticMatryoshkaLinear``: Dynamic width-sliceable linear projection supporting
   runtime execution at any active prefix dimension $m \\le d_{\\text{in}}$ after
   channel importance reordering.

References:
    - Kusupati et al. (NeurIPS 2022). "Matryoshka Representation Learning."
      arXiv:2205.13147.
    - Muralidharan et al. (NeurIPS 2024 / NVIDIA). "Compact Language Models via
      Pruning and Knowledge Distillation (Minitron)." arXiv:2407.14679.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_matryoshka_loss(
    hidden_states: torch.Tensor,
    classifier_weight: torch.Tensor,
    targets: torch.Tensor,
    nesting_dims: Sequence[int],
    dim_weights: Optional[Sequence[float]] = None,
    classifier_bias: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Compute multi-granularity Matryoshka Representation Learning (MRL) loss.

    Given hidden representations $\\mathbf{h} \\in \\mathbb{R}^{N \\times d}$, tied or
    shared readout matrix $\\mathbf{W} \\in \\mathbb{R}^{V \\times d}$, and nested prefix
    dimensions $\\mathcal{M} = \\{d_1, d_2, \\dots, d_k\\}$ with $1 \\le d_1 < \\dots < d_k \\le d$:

    $$\\mathcal{L}_{\\text{MRL}} = \\sum_{i=1}^{k} c_i \\cdot \\mathcal{L}_{\\text{CE}}\\!\\left(\\frac{1}{\\sqrt{d_i}} \\mathbf{h}_{:, :d_i} \\mathbf{W}_{:, :d_i}^\\top + \\mathbf{b},\\; \\mathbf{y}\\right)$$

    Args:
        hidden_states: Tensor of shape ``(N, d)`` or ``(B, T, d)``.
        classifier_weight: Readout matrix of shape ``(num_classes, d)``.
        targets: Ground-truth class indices of shape ``(N,)`` or ``(B, T)``.
        nesting_dims: Strictly positive prefix dimensions $d_i \\le d$.
        dim_weights: Optional positive scalar weights $c_i$ per nesting dimension.
        classifier_bias: Optional bias vector of shape ``(num_classes,)``.

    Returns:
        Dictionary containing:
            - ``"total_loss"``: Scalar differentiable Matryoshka loss tensor.
            - ``"per_dim_losses"``: Dict mapping each $d_i$ to its detached scalar loss.
    """
    if len(nesting_dims) == 0:
        raise ValueError("nesting_dims must contain at least one dimension.")
    full_dim = hidden_states.shape[-1]
    if classifier_weight.shape[1] != full_dim:
        raise ValueError(
            f"classifier_weight second dim ({classifier_weight.shape[1]}) must match "
            f"hidden_states last dim ({full_dim})."
        )

    for d_i in nesting_dims:
        if d_i <= 0 or d_i > full_dim:
            raise ValueError(
                f"Each nesting dimension must be in [1, {full_dim}], got {d_i}."
            )

    if dim_weights is None:
        weights = [1.0 / len(nesting_dims)] * len(nesting_dims)
    else:
        if len(dim_weights) != len(nesting_dims):
            raise ValueError("dim_weights must have the same length as nesting_dims.")
        weights = [float(w) for w in dim_weights]

    flat_h = hidden_states.reshape(-1, full_dim)
    flat_targets = targets.reshape(-1)

    total_loss = torch.zeros((), device=hidden_states.device, dtype=hidden_states.dtype)
    per_dim_losses: Dict[int, float] = {}

    for d_i, w_i in zip(nesting_dims, weights):
        h_slice = flat_h[:, :d_i]
        w_slice = classifier_weight[:, :d_i]
        logits = F.linear(h_slice, w_slice, classifier_bias)
        loss_i = F.cross_entropy(logits, flat_targets)
        total_loss = total_loss + w_i * loss_i
        per_dim_losses[int(d_i)] = float(loss_i.detach().item())

    return {
        "total_loss": total_loss,
        "per_dim_losses": per_dim_losses,
    }


def rank_minitron_structural_importance(
    calibration_activations: torch.Tensor,
    layer_inputs: Optional[Sequence[torch.Tensor]] = None,
    layer_outputs: Optional[Sequence[torch.Tensor]] = None,
) -> Dict[str, object]:
    """Rank channel width and layer depth importance via NVIDIA Minitron criteria.

    1. **Channel / Hidden Width Importance (Muralidharan et al., 2024):**
       Aggregates L2-over-batch and mean-absolute-over-sequence activation magnitude:
       $$\\mathcal{I}_{\\text{channel}}(j) = \\frac{1}{B \\cdot T} \\sum_{b=1}^{B} \\sum_{t=1}^{T} |X_{b, t, j}|$$
       Channels are sorted in descending order of $\\mathcal{I}_{\\text{channel}}(j)$.

    2. **Layer Depth Block Influence (BI):**
       Measures how much layer $\\ell$ transforms the residual stream:
       $$\\text{BI}_\\ell = 1 - \\mathbb{E}_{b, t}\\!\\left[\\frac{\\langle \\mathbf{X}_{\\ell, b, t}^{\\text{in}}, \\mathbf{X}_{\\ell, b, t}^{\\text{out}} \\rangle}{\\|\\mathbf{X}_{\\ell, b, t}^{\\text{in}}\\|_2 \\|\\mathbf{X}_{\\ell, b, t}^{\\text{out}}\\|_2}\\right]$$
       Layers with higher $\\text{BI}_\\ell$ are more critical; layers with $\\text{BI}_\\ell \\approx 0$
       act as near-identities and can be pruned first.

    Args:
        calibration_activations: Calibration tensor of shape ``(B, T, d)`` or ``(N, d)``.
        layer_inputs: Optional list of layer input tensors ``X_in`` of shape ``(..., d)``.
        layer_outputs: Optional list of layer output tensors ``X_out`` of shape ``(..., d)``.

    Returns:
        Dictionary with:
            - ``"channel_importance"``: 1D tensor of length $d$ with importance scores.
            - ``"channel_ranking"``: 1D ``LongTensor`` of channel indices sorted most-to-least important.
            - ``"layer_block_influence"``: List of $\\text{BI}_\\ell$ scores (if layer tensors supplied).
            - ``"layer_ranking"``: List of layer indices sorted most-to-least important.
    """
    if calibration_activations.ndim < 2:
        raise ValueError("calibration_activations must have at least 2 dimensions.")
    flat_act = calibration_activations.reshape(-1, calibration_activations.shape[-1])
    channel_importance = flat_act.abs().mean(dim=0)
    channel_ranking = torch.argsort(channel_importance, descending=True)

    layer_bi_scores: List[float] = []
    layer_ranking: List[int] = []

    if layer_inputs is not None or layer_outputs is not None:
        if layer_inputs is None or layer_outputs is None or len(layer_inputs) != len(layer_outputs):
            raise ValueError("layer_inputs and layer_outputs must both be provided with equal length.")
        for x_in, x_out in zip(layer_inputs, layer_outputs):
            f_in = x_in.reshape(-1, x_in.shape[-1]).float()
            f_out = x_out.reshape(-1, x_out.shape[-1]).float()
            cos_sim = F.cosine_similarity(f_in, f_out, dim=-1, eps=1e-8)
            bi = float((1.0 - cos_sim.mean()).item())
            layer_bi_scores.append(bi)
        layer_ranking = sorted(
            range(len(layer_bi_scores)), key=lambda idx: layer_bi_scores[idx], reverse=True
        )

    return {
        "channel_importance": channel_importance,
        "channel_ranking": channel_ranking,
        "layer_block_influence": layer_bi_scores,
        "layer_ranking": layer_ranking,
    }


class ElasticMatryoshkaLinear(nn.Module):
    """Dynamic width-sliceable linear projection for elastic SLM inference.

    Stores a full weight matrix $\\mathbf{W} \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$
    whose input columns can be canonically reordered by Minitron importance so that
    any prefix ``active_in_features`` $m \\in [1, d_{\\text{in}}]$ executes a dense
    sub-matrix multiply using only the top-$m$ most salient input channels.

    Args:
        in_features: Maximum input dimension $d_{\\text{in}}$.
        out_features: Output dimension $d_{\\text{out}}$.
        bias: Whether to include an additive bias vector.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive.")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.register_buffer(
            "channel_order",
            torch.arange(in_features, dtype=torch.long),
        )

    @torch.no_grad()
    def reorder_by_importance(self, channel_ranking: torch.Tensor) -> None:
        """Permute weight input columns in-place so prefix slices capture top channels."""
        if channel_ranking.numel() != self.in_features:
            raise ValueError(
                f"Expected channel_ranking of length {self.in_features}, got {channel_ranking.numel()}."
            )
        order = channel_ranking.to(device=self.weight.device, dtype=torch.long)
        self.weight.copy_(self.weight[:, order])
        self.channel_order.copy_(order)

    def forward(
        self,
        x: torch.Tensor,
        active_in_features: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply elastic linear projection using the top ``active_in_features`` channels.

        Args:
            x: Input tensor of shape ``(..., in_features)`` or already sliced ``(..., m)``.
            active_in_features: Optional active input width $m \\in [1, d_{\\text{in}}]$.

        Returns:
            Output tensor of shape ``(..., out_features)``.
        """
        m = self.in_features if active_in_features is None else int(active_in_features)
        if m <= 0 or m > self.in_features:
            raise ValueError(
                f"active_in_features must be in [1, {self.in_features}], got {m}."
            )
        if x.shape[-1] == self.in_features:
            x_ordered = x.index_select(dim=-1, index=self.channel_order[:m])
        elif x.shape[-1] == m:
            x_ordered = x
        else:
            raise ValueError(
                f"Input last dimension ({x.shape[-1]}) must match in_features "
                f"({self.in_features}) or active_in_features ({m})."
            )
        w_slice = self.weight[:, :m]
        return F.linear(x_ordered, w_slice, self.bias)


def main() -> None:
    """Run a deterministic demonstration of MRL loss and Minitron elastic slicing."""
    torch.manual_seed(42)
    layer = ElasticMatryoshkaLinear(in_features=32, out_features=16, bias=False)
    x = torch.randn(8, 32) * 0.02
    x[:, [2, 7, 15, 21]] *= 50.0
    ranking = rank_minitron_structural_importance(x)
    layer.reorder_by_importance(ranking["channel_ranking"])
    y_full = layer(x, active_in_features=32)
    y_slice = layer(x, active_in_features=4)
    rel_err = torch.linalg.norm(y_full - y_slice) / torch.linalg.norm(y_full)
    print(f"[Module 25] Top-4 channels: {ranking['channel_ranking'][:4].tolist()} | Relative error: {rel_err.item():.4f}")


if __name__ == "__main__":
    main()

