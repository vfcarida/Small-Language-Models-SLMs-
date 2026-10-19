"""Module 20: Mixture-of-Depths (MoD) & Dynamic Token Routing for SLMs."""

from src.module_20_mixture_of_depths.mod_routing import (
    MixtureOfDepthsBlock,
    compute_mod_auxiliary_loss,
    profile_mod_compute_savings,
)

__all__ = [
    "MixtureOfDepthsBlock",
    "compute_mod_auxiliary_loss",
    "profile_mod_compute_savings",
]
