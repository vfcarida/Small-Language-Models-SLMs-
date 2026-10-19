"""Module 15: Ternary 1.58-Bit Native SLMs (BitNet b1.58 & Add-Only Kernels)."""

from src.module_15_bitnet_ternary.bitnet_linear import (
    BitLinear,
    profile_bitnet_hardware_savings,
    quantize_activations_absmax_int8,
    quantize_weights_absmean_ternary,
)

__all__ = [
    "BitLinear",
    "profile_bitnet_hardware_savings",
    "quantize_activations_absmax_int8",
    "quantize_weights_absmean_ternary",
]
