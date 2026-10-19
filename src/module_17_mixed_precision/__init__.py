"""Module 17: Layer-Adaptive Mixed-Precision (LAMP) & Hessian Sensitivity Bit Allocator."""

from src.module_17_mixed_precision.lamp_allocator import (
    LAMPAllocationReport,
    allocate_mixed_precision_bits,
    apply_lamp_mixed_precision,
    compute_layer_hessian_sensitivity,
)

__all__ = [
    "LAMPAllocationReport",
    "allocate_mixed_precision_bits",
    "apply_lamp_mixed_precision",
    "compute_layer_hessian_sensitivity",
]
