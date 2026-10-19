"""
Unit tests for Module 22: Long-Context KV-Cache Eviction (StreamingLLM Sinks & H2O Oracle).
"""

import pytest
import torch

from src.module_22_kv_cache_eviction.kv_eviction_policy import (
    H2OStreamingKVCache,
    compare_kv_eviction_fidelity,
)


def test_h2o_streaming_kv_cache_retains_sinks_hh_and_recent():
    cache = H2OStreamingKVCache(num_sink_tokens=2, num_heavy_hitters=2, num_recent_tokens=3)
    scores = torch.zeros(20)
    # Plant two heavy-hitters at indices 7 and 11
    scores[7] = 50.0
    scores[11] = 40.0

    retained = cache.select_retained_indices(scores).tolist()
    assert len(retained) == 7
    # Must contain sinks [0, 1], heavy hitters [7, 11], and recent [17, 18, 19]
    assert retained == [0, 1, 7, 11, 17, 18, 19]

    with pytest.raises(ValueError, match="Invalid sink"):
        H2OStreamingKVCache(num_sink_tokens=0)


def test_compare_kv_eviction_fidelity_beats_sliding_window():
    report = compare_kv_eviction_fidelity(
        seq_len=128, head_dim=32, num_sink=4, num_hh=8, num_recent=8, seed=42
    )
    assert report.cache_budget_tokens == 20
    assert report.h2o_sink_l2_error < report.sliding_window_l2_error
    assert report.error_reduction_vs_window_x > 2.0
