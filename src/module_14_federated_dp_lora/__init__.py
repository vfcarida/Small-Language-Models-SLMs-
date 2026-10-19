"""Module 14: Federated & Differential-Privacy LoRA Aggregation (DP-SGD + FFA-LoRA)."""

from src.module_14_federated_dp_lora.dp_federated_lora import (
    DPLoRAReport,
    aggregate_federated_lora,
    clip_and_noise_gradients,
    compute_gaussian_dp_sigma,
)

__all__ = [
    "DPLoRAReport",
    "aggregate_federated_lora",
    "clip_and_noise_gradients",
    "compute_gaussian_dp_sigma",
]
