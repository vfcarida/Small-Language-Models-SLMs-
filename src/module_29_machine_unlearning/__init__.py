"""Module 29: Machine Unlearning & Targeted Knowledge Erasure (NPO & RMU)."""

from src.module_29_machine_unlearning.unlearning_objectives import (
    compute_npo_unlearning_loss,
    compute_rmu_representation_loss,
)

__all__ = [
    "compute_npo_unlearning_loss",
    "compute_rmu_representation_loss",
]
