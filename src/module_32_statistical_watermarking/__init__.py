"""Module 32: Statistical Watermarking (KGW Green-List Logit Bias) for AI TRiSM."""

from src.module_32_statistical_watermarking.kgw_watermark import (
    apply_kgw_watermark_logits,
    detect_kgw_watermark_zscore,
    partition_green_red_vocab,
)

__all__ = [
    "apply_kgw_watermark_logits",
    "detect_kgw_watermark_zscore",
    "partition_green_red_vocab",
]
