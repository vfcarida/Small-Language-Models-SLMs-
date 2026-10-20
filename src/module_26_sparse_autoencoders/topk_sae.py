"""
Module 26: Mechanistic Interpretability via Top-K Sparse Autoencoders (SAE)
===========================================================================

Implements monosemantic feature decomposition for Small Language Model (<4B)
residual streams to support AI TRiSM auditing and causal circuit discovery:
1. ``TopKSparseAutoencoder``: Overcomplete dictionary autoencoder ($M \\gg d$) with
   pre-encoder centering $\\bar{\\mathbf{x}} = \\mathbf{x} - \\mathbf{b}_{\\text{pre}}$,
   exact $\\text{TopK}(\\text{ReLU}(\\cdot), k)$ activation sparsity (eliminating $L_1$
   feature shrinkage), and unit-norm decoder columns $\\|\\mathbf{w}_{\\text{dec}, i}\\|_2 = 1$
   (Gao et al., OpenAI 2024).
2. ``compute_sae_loss_with_auxk``: Combined reconstruction MSE and dead-latent
   ``AuxK`` auxiliary residual reconstruction loss preventing dictionary collapse.
3. ``clamp_sae_feature``: Surgical causal intervention clamping a single monosemantic
   latent feature $z_i \\leftarrow \\alpha$ and projecting back to the residual stream.

References:
    - Gao et al. (OpenAI, 2024). "Scaling and evaluating sparse autoencoders."
      arXiv:2406.04093.
    - Templeton et al. (Anthropic, 2024). "Scaling Monosemanticity: Extracting
      Interpretable Features from Claude 3 Sonnet." Transformer Circuits Thread.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSparseAutoencoder(nn.Module):
    """Overcomplete Top-$K$ Sparse Autoencoder (Gao et al., OpenAI 2024).

    Decomposes a residual stream activation $\\mathbf{x} \\in \\mathbb{R}^d$ into an
    overcomplete sparse feature code $\\mathbf{z} \\in \\mathbb{R}_{\\ge 0}^M$ ($M \\gg d$)
    with at most $k$ non-zero active features per token:

    $$\\bar{\\mathbf{x}} = \\mathbf{x} - \\mathbf{b}_{\\text{pre}}$$
    $$\\mathbf{a} = \\text{ReLU}(\\mathbf{W}_{\\text{enc}} \\bar{\\mathbf{x}} + \\mathbf{b}_{\\text{enc}})$$
    $$\\mathbf{z} = \\text{TopK}(\\mathbf{a}, k)$$
    $$\\hat{\\mathbf{x}} = \\mathbf{W}_{\\text{dec}} \\mathbf{z} + \\mathbf{b}_{\\text{pre}}$$

    Args:
        d_model: Residual stream dimension $d$.
        dict_size: Overcomplete dictionary size $M$ (typically $4d$ to $64d$).
        k_active: Number of active sparse features $k$ retained per token.
    """

    def __init__(self, d_model: int, dict_size: int, k_active: int) -> None:
        super().__init__()
        if d_model <= 0 or dict_size <= 0:
            raise ValueError("d_model and dict_size must be positive.")
        if k_active <= 0 or k_active > dict_size:
            raise ValueError(f"k_active must be in [1, {dict_size}], got {k_active}.")

        self.d_model = d_model
        self.dict_size = dict_size
        self.k_active = k_active

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))

        # Initialize decoder columns on the unit sphere and tie initial encoder transpose
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec)
            self.normalize_decoder_columns()
            self.W_enc.copy_(self.W_dec.t())

        self.register_buffer(
            "steps_since_fired",
            torch.zeros(dict_size, dtype=torch.long),
        )

    @torch.no_grad()
    def normalize_decoder_columns(self) -> None:
        """Project every decoder dictionary atom $\\mathbf{w}_{\\text{dec}, i}$ onto the unit $L_2$ sphere."""
        norms = torch.linalg.norm(self.W_dec, dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode residual activations into sparse Top-$K$ latents and pre-TopK ReLU activations.

        Args:
            x: Activation tensor of shape ``(..., d_model)``.

        Returns:
            Tuple ``(z_sparse, pre_activations)`` each of shape ``(..., dict_size)``.
        """
        x_centered = x - self.b_pre
        pre_act = F.relu(F.linear(x_centered, self.W_enc, self.b_enc))
        topk_vals, topk_idx = torch.topk(pre_act, k=self.k_active, dim=-1)
        z_sparse = torch.zeros_like(pre_act)
        z_sparse.scatter_(-1, topk_idx, topk_vals)
        return z_sparse, pre_act

    def decode(self, z_sparse: torch.Tensor) -> torch.Tensor:
        """Decode sparse feature activations back to the residual stream $\\mathbb{R}^d$."""
        return F.linear(z_sparse, self.W_dec, self.b_pre)

    def forward(
        self, x: torch.Tensor, update_firing_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Run full Top-$K$ SAE pass and optionally update dead-feature tracking counters."""
        z_sparse, pre_act = self.encode(x)
        x_hat = self.decode(z_sparse)

        if update_firing_stats and self.training:
            with torch.no_grad():
                flat_z = z_sparse.reshape(-1, self.dict_size)
                fired_mask = (flat_z > 0).any(dim=0)
                self.steps_since_fired.add_(1)
                self.steps_since_fired[fired_mask] = 0

        return {
            "x_hat": x_hat,
            "z_sparse": z_sparse,
            "pre_activations": pre_act,
        }


def compute_sae_loss_with_auxk(
    sae: TopKSparseAutoencoder,
    x: torch.Tensor,
    sae_out: Dict[str, torch.Tensor],
    k_aux: int = 8,
    alpha_aux: float = 1.0 / 32.0,
    dead_threshold_steps: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute Top-$K$ SAE reconstruction MSE plus dead-latent AuxK loss (Gao et al., 2024).

    To prevent dead dictionary atoms from never receiving gradient updates, the
    residual reconstruction error $\\mathbf{e} = \\mathbf{x} - \\hat{\\mathbf{x}}$ is modeled
    by the top-$k_{\\text{aux}}$ dead features ($\\text{steps\\_since\\_fired} \\ge \\tau_{\\text{dead}}$):

    $$\\mathcal{L} = \\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|_2^2 + \\alpha_{\\text{aux}} \\|\\mathbf{e} - \\hat{\\mathbf{e}}_{\\text{dead}}\\|_2^2$$

    Args:
        sae: ``TopKSparseAutoencoder`` instance.
        x: Input activations of shape ``(..., d_model)``.
        sae_out: Output dict returned by ``sae(x)``.
        k_aux: Number of dead latents selected in the AuxK auxiliary pass.
        alpha_aux: Scalar coefficient weighting the AuxK auxiliary loss.
        dead_threshold_steps: Minimum non-firing steps for a latent to be considered dead.

    Returns:
        Dictionary with ``"total_loss"``, ``"recon_loss"``, ``"auxk_loss"``, and ``"fvu"``
        (Fraction of Variance Unexplained).
    """
    x_hat = sae_out["x_hat"]
    pre_act = sae_out["pre_activations"]

    residual = x - x_hat
    recon_loss = F.mse_loss(x_hat, x)

    # Fraction of Variance Unexplained (FVU)
    x_var = ((x - x.mean(dim=0, keepdim=True)) ** 2).mean().clamp(min=1e-8)
    fvu = recon_loss / x_var

    dead_mask = sae.steps_since_fired >= dead_threshold_steps
    num_dead = int(dead_mask.sum().item())

    if num_dead > 0 and k_aux > 0:
        effective_k_aux = min(k_aux, num_dead)
        masked_pre_act = pre_act.masked_fill(~dead_mask, 0.0)
        aux_vals, aux_idx = torch.topk(masked_pre_act, k=effective_k_aux, dim=-1)
        z_aux = torch.zeros_like(pre_act)
        z_aux.scatter_(-1, aux_idx, aux_vals)
        # Reconstruct detached residual error using dead decoder columns
        e_hat = F.linear(z_aux, sae.W_dec)
        auxk_loss = F.mse_loss(e_hat, residual.detach())
    else:
        auxk_loss = torch.zeros((), device=x.device, dtype=x.dtype)

    total_loss = recon_loss + alpha_aux * auxk_loss
    return {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "auxk_loss": auxk_loss,
        "fvu": fvu,
    }


def clamp_sae_feature(
    sae: TopKSparseAutoencoder,
    x: torch.Tensor,
    feature_idx: int,
    clamp_value: float,
) -> torch.Tensor:
    """Perform a surgical causal intervention on a single monosemantic SAE feature.

    Preserves the exact SAE reconstruction error $\\mathbf{e} = \\mathbf{x} - \\hat{\\mathbf{x}}$
    while clamping latent feature $i$ to $\\alpha$ (Templeton et al., Anthropic 2024):

    $$\\tilde{\\mathbf{x}} = \\mathbf{x} + (\\alpha - z_i) \\mathbf{w}_{\\text{dec}, i}$$

    Args:
        sae: Trained ``TopKSparseAutoencoder``.
        x: Residual stream activation tensor of shape ``(..., d_model)``.
        feature_idx: Dictionary index $i \\in [0, M - 1]$ of the target monosemantic feature.
        clamp_value: Target activation value $\\alpha$ (e.g., $0.0$ for ablation or $>0$ for steering).

    Returns:
        Intervened residual stream tensor $\\tilde{\\mathbf{x}}$ of the same shape as ``x``.
    """
    if feature_idx < 0 or feature_idx >= sae.dict_size:
        raise ValueError(
            f"feature_idx must be in [0, {sae.dict_size - 1}], got {feature_idx}."
        )
    z_sparse, _ = sae.encode(x)
    current_z_i = z_sparse[..., feature_idx : feature_idx + 1]
    delta_z = float(clamp_value) - current_z_i
    decoder_atom = sae.W_dec[:, feature_idx]
    return x + delta_z * decoder_atom


def main() -> None:
    """Run a deterministic demonstration of Top-K SAE encoding, AuxK loss, and feature clamping."""
    torch.manual_seed(42)
    sae = TopKSparseAutoencoder(d_model=16, dict_size=64, k_active=4)
    x = torch.randn(8, 16)
    out = sae(x)
    losses = compute_sae_loss_with_auxk(sae, x, out)
    print(
        f"[Module 26] Top-K SAE recon_loss={losses['recon_loss'].item():.4f} | "
        f"fvu={losses['fvu'].item():.4f}"
    )


if __name__ == "__main__":
    main()

