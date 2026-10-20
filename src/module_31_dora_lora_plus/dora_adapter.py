"""
Module 31: Weight-Decomposed Low-Rank Adaptation (DoRA) & Asymmetric LoRA+
==========================================================================

Implements two high-efficiency Parameter-Efficient Fine-Tuning (PEFT) upgrades
that close the accuracy and convergence gap between standard LoRA and full
fine-tuning on Small Language Models (<4B parameters):
1. ``DoRALinear``: Weight-Decomposed Low-Rank Adaptation (Liu et al., NVIDIA /
   ICML 2024 Oral) decoupling pre-trained weights into a trainable magnitude
   vector $\\mathbf{m} \\in \\mathbb{R}^{d_{\\text{out}}}$ and a low-rank directional
   update:
   $$\\mathbf{W}' = \\text{diag}(\\mathbf{m}) \\frac{\\mathbf{W}_0 + \\frac{\\alpha}{r}\\mathbf{B}\\mathbf{A}}{\\|\\mathbf{W}_0 + \\frac{\\alpha}{r}\\mathbf{B}\\mathbf{A}\\|_c}$$
   with zero-overhead inference weight folding (``merge_into_linear``).
2. ``build_lora_plus_param_groups``: Asymmetric learning-rate parameter grouping
   (Hayou et al., ICML 2024, LoRA+) setting $\\eta_B = \\lambda_{\\text{ratio}} \\cdot \\eta_A$
   ($\\lambda_{\\text{ratio}} \\in [8, 16]$) to achieve optimal feature learning width scaling.

References:
    - Liu et al. (NVIDIA Research / ICML 2024 Oral). "DoRA: Weight-Decomposed
      Low-Rank Adaptation." arXiv:2402.09353.
    - Hayou et al. (ICML 2024). "LoRA+: Efficient Low Rank Adaptation of Large
      Models." arXiv:2402.12354.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation (DoRA) Linear Layer (Liu et al., ICML 2024).

    Decomposes a frozen base weight $\\mathbf{W}_0 \\in \\mathbb{R}^{d_{\\text{out}} \\times d_{\\text{in}}}$
    into row-wise $L_2$ magnitudes $\\mathbf{m} \\in \\mathbb{R}^{d_{\\text{out}}}$ and a unit-norm
    directional matrix updated by low-rank factors $\\mathbf{B} \\in \\mathbb{R}^{d_{\\text{out}} \\times r}$
    and $\\mathbf{A} \\in \\mathbb{R}^{r \\times d_{\\text{in}}}$:

    $$\\mathbf{V} = \\mathbf{W}_0 + \\frac{\\alpha}{r} \\mathbf{B}\\mathbf{A}, \\quad \\mathbf{W}_{\\text{DoRA}} = \\frac{\\mathbf{m}}{\\|\\mathbf{V}\\|_{\\text{row}}} \\odot \\mathbf{V}$$

    At initialization ($\\mathbf{B} = \\mathbf{0}$ and $\\mathbf{m} = \\|\\mathbf{W}_0\\|_{\\text{row}}$),
    $\\mathbf{W}_{\\text{DoRA}} \\equiv \\mathbf{W}_0$ holds identically.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        lora_alpha: float = 16.0,
        bias: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("in_features, out_features, and rank must be positive.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(rank)
        self.eps = eps

        # Frozen base weight and optional bias
        w0 = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(w0, a=math.sqrt(5))
        self.register_buffer("weight_base", w0)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Low-rank directional factors A and B
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Trainable row-wise magnitude vector m initialized to ||W_0||_2
        init_mag = torch.linalg.norm(w0, dim=1)
        self.magnitude = nn.Parameter(init_mag)

    def effective_weight(self) -> torch.Tensor:
        """Compute the full weight-decomposed matrix $\\mathbf{W}_{\\text{DoRA}}$."""
        delta_w = self.scaling * torch.matmul(self.lora_B, self.lora_A)
        v_mat = self.weight_base + delta_w
        row_norms = torch.linalg.norm(v_mat, dim=1, keepdim=True).clamp(min=self.eps)
        scale_col = self.magnitude.unsqueeze(1) / row_norms
        return scale_col * v_mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DoRA linear transformation to ``(..., in_features)``."""
        return F.linear(x, self.effective_weight(), self.bias)

    @torch.no_grad()
    def merge_into_linear(self) -> nn.Linear:
        """Fold DoRA magnitude and directional updates into a standard ``nn.Linear`` layer."""
        merged = nn.Linear(
            self.in_features,
            self.out_features,
            bias=(self.bias is not None),
            device=self.weight_base.device,
            dtype=self.weight_base.dtype,
        )
        merged.weight.copy_(self.effective_weight())
        if self.bias is not None:
            merged.bias.copy_(self.bias)
        return merged


def build_lora_plus_param_groups(
    model: nn.Module,
    base_lr: float = 1e-4,
    lora_plus_lambda: float = 16.0,
    weight_decay: float = 0.0,
) -> List[Dict[str, object]]:
    """Build asymmetric LoRA+ optimizer parameter groups (Hayou et al., ICML 2024).

    In large-width neural networks ($d \\gg 1$), standard LoRA ($\\eta_A = \\eta_B$) is
    suboptimal because $\\mathbf{A}$ projects $O(d)$-dimensional inputs while $\\mathbf{B}$
    maps $O(r)$-dimensional latents. LoRA+ sets:

    $$\\eta_A = \\eta_{\\text{base}}, \\quad \\eta_B = \\lambda_{\\text{ratio}} \\cdot \\eta_{\\text{base}} \\quad (\\lambda_{\\text{ratio}} \\in [8, 16])$$

    Args:
        model: PyTorch module containing ``lora_A``, ``lora_B``, and optional ``magnitude`` parameters.
        base_lr: Base learning rate $\\eta_A > 0$.
        lora_plus_lambda: Asymmetric multiplier $\\lambda_{\\text{ratio}} \\ge 1.0$.
        weight_decay: Weight decay coefficient.

    Returns:
        List of PyTorch optimizer parameter group dicts with keys ``"params"``, ``"lr"``,
        ``"weight_decay"``, and ``"group_name"``.
    """
    if base_lr <= 0.0 or lora_plus_lambda < 1.0:
        raise ValueError("base_lr must be > 0 and lora_plus_lambda must be >= 1.0.")

    params_a: List[nn.Parameter] = []
    params_b: List[nn.Parameter] = []
    params_other: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_B" in name:
            params_b.append(param)
        elif "lora_A" in name:
            params_a.append(param)
        else:
            params_other.append(param)

    groups: List[Dict[str, object]] = []
    if params_a:
        groups.append(
            {
                "group_name": "lora_A",
                "params": params_a,
                "lr": float(base_lr),
                "weight_decay": float(weight_decay),
            }
        )
    if params_b:
        groups.append(
            {
                "group_name": "lora_B_fast",
                "params": params_b,
                "lr": float(base_lr * lora_plus_lambda),
                "weight_decay": float(weight_decay),
            }
        )
    if params_other:
        groups.append(
            {
                "group_name": "other_trainable",
                "params": params_other,
                "lr": float(base_lr),
                "weight_decay": float(weight_decay),
            }
        )
    return groups


def main() -> None:
    """Run a deterministic demonstration of DoRA merging and LoRA+ parameter groups."""
    torch.manual_seed(42)
    dora = DoRALinear(in_features=32, out_features=16, rank=4)
    with torch.no_grad():
        dora.lora_B.normal_(0.0, 0.05)
    x = torch.randn(4, 32)
    y_dora = dora(x)
    merged = dora.merge_into_linear()
    y_merged = merged(x)
    groups = build_lora_plus_param_groups(dora, base_lr=1e-4, lora_plus_lambda=16.0)
    print(
        f"[Module 31] DoRA merge max_diff={(y_dora - y_merged).abs().max().item():.2e} | "
        f"LoRA+ groups={[(g['group_name'], g['lr']) for g in groups]}"
    )


if __name__ == "__main__":
    main()
