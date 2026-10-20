"""Module 25: Matryoshka Representation Learning (MRL) & Elastic Sub-Network Slicing."""

from src.module_25_matryoshka_minitron.elastic_matryoshka import (
    ElasticMatryoshkaLinear,
    compute_matryoshka_loss,
    rank_minitron_structural_importance,
)

__all__ = [
    "ElasticMatryoshkaLinear",
    "compute_matryoshka_loss",
    "rank_minitron_structural_importance",
]
