"""Module 05: Post-Training Quantization — AWQ Salient Scaling, SpinQuant/Cayley Stiefel Rotation, KVQuant INT4, and BitsAndBytes NF4."""

from src.module_05_quantization.awq_quantization import (
    apply_hadamard_rotation_smoothing,
    compute_awq_scales,
    pseudo_quantize_tensor,
)
from src.module_05_quantization.bitsandbytes_quant import estimate_quantized_memory_mb
from src.module_05_quantization.cayley_kv_quant import (
    build_normalized_hadamard,
    cayley_orthogonal_step,
    optimize_cayley_rotation,
    quantize_kv_cache_int4,
    ste_pseudo_quantize,
)

__all__ = [
    "apply_hadamard_rotation_smoothing",
    "build_normalized_hadamard",
    "cayley_orthogonal_step",
    "compute_awq_scales",
    "estimate_quantized_memory_mb",
    "optimize_cayley_rotation",
    "pseudo_quantize_tensor",
    "quantize_kv_cache_int4",
    "ste_pseudo_quantize",
]
