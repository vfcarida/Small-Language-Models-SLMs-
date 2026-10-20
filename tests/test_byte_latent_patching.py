"""Unit tests for Module 28: Byte-Latent & Dynamic Patch Tokenization (BLT)."""

import torch

from src.module_28_byte_latent_patching import (
    ByteLatentPatchPooling,
    compute_byte_stream_entropy,
    segment_bytes_into_dynamic_patches,
)


def test_compute_byte_stream_entropy_bounds_uniform_and_peaked() -> None:
    # Uniform over 256 bytes -> 8.0 bits of entropy
    uniform_logits = torch.zeros(2, 256)
    h_uniform = compute_byte_stream_entropy(uniform_logits)
    assert torch.allclose(h_uniform, torch.tensor([8.0, 8.0]), atol=1e-4)

    # Peaked distribution -> ~0.0 bits of entropy
    peaked_logits = torch.full((2, 256), -100.0)
    peaked_logits[:, 65] = 100.0
    h_peaked = compute_byte_stream_entropy(peaked_logits)
    assert torch.all(h_peaked < 1e-4)


def test_segment_bytes_into_dynamic_patches_splits_on_entropy_spikes() -> None:
    byte_ids = list(b"hello world")
    # High entropy at index 5 (' ') and index 6 ('w')
    entropies = [0.5, 0.4, 0.3, 0.2, 0.2, 3.5, 3.1, 0.4, 0.3, 0.2, 0.1]
    res = segment_bytes_into_dynamic_patches(
        byte_ids=byte_ids,
        byte_entropies=entropies,
        global_entropy_threshold=2.5,
        monotonic_delta_threshold=1.5,
        max_patch_bytes=8,
    )
    assert res["num_patches"] == 3
    # Concatenating patches recovers exact byte stream losslessly
    reconstructed = [b for patch in res["patches"] for b in patch]
    assert reconstructed == byte_ids
    assert res["compression_ratio"] > 3.0


def test_byte_latent_patch_pooling_outputs_patch_sequence() -> None:
    torch.manual_seed(13)
    byte_ids = torch.tensor(list(b"dynamic byte patching"), dtype=torch.long)
    spans = [(0, 7), (7, 12), (12, 21)]
    pooler = ByteLatentPatchPooling(d_model=24)
    patch_latents = pooler(byte_ids, spans)
    assert patch_latents.shape == (3, 24)
