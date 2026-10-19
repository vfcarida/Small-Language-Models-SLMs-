import pytest
import torch
from src.module_05_quantization.awq_quantization import (
    apply_hadamard_rotation_smoothing,
    compute_awq_scales,
    pseudo_quantize_tensor,
)
from src.module_05_quantization.bitsandbytes_quant import estimate_quantized_memory_mb


def test_pseudo_quantize_tensor_asymmetric_and_symmetric():
    torch.manual_seed(42)
    W = torch.randn(16, 64)
    deq_asym, scales_a, zeros_a = pseudo_quantize_tensor(W, w_bit=4, q_group_size=32, zero_point=True)
    deq_sym, scales_s, zeros_s = pseudo_quantize_tensor(W, w_bit=4, q_group_size=32, zero_point=False)

    assert deq_asym.shape == W.shape
    assert scales_a.shape == (16, 2)
    assert zeros_a.shape == (16, 2)
    assert deq_sym.shape == W.shape
    assert torch.all(zeros_s == 0)
    # 4-bit quantization reconstruction error should be bounded
    assert torch.mean((W - deq_asym) ** 2).item() < 0.05


def test_compute_awq_scales_reduces_outlier_mse():
    torch.manual_seed(42)
    W = torch.randn(32, 64)
    X = torch.randn(16, 64)
    # Inject a massive activation outlier on channel 0 (simulating LLM emergent outlier feature)
    X[:, 0] *= 50.0

    scales, best_alpha, metrics = compute_awq_scales(W, X, w_bit=4, q_group_size=32, n_grid=20)
    assert scales.shape == (64,)
    assert 0.0 <= best_alpha <= 1.0
    # AWQ optimal alpha search must achieve strictly lower or equal output MSE than naive RTN (alpha=0)
    assert metrics["awq_mse"] <= metrics["rtn_mse"]
    assert metrics["awq_mse"] < metrics["rtn_mse"] * 0.8


def test_hadamard_rotation_smoothing_preserves_linear_output():
    torch.manual_seed(42)
    W = torch.randn(16, 32)
    X = torch.randn(8, 32)
    X[:, 0] *= 40.0  # Extreme outlier in channel 0

    rot_W, rot_X = apply_hadamard_rotation_smoothing(W, X)
    # Verify exact mathematical invariance: (X H)(W H)^T == X W^T
    orig_out = X @ W.t()
    rot_out = rot_X @ rot_W.t()
    assert torch.allclose(orig_out, rot_out, atol=1e-4)
    # Verify that the extreme maximum activation magnitude is smoothed down
    assert rot_X.abs().max().item() < X.abs().max().item()


def test_estimate_quantized_memory_mb():
    num_params = 1_500_000_000
    fp16 = estimate_quantized_memory_mb(num_params, bit_width=16)
    int8 = estimate_quantized_memory_mb(num_params, bit_width=8)
    nf4_single = estimate_quantized_memory_mb(num_params, bit_width=4, double_quant=False)
    nf4_double = estimate_quantized_memory_mb(num_params, bit_width=4, double_quant=True)

    assert fp16["total_memory_mb"] > int8["total_memory_mb"] > nf4_single["total_memory_mb"]
    # Double quantization must reduce scale overhead vs. single quantization
    assert nf4_double["scale_overhead_mb"] < nf4_single["scale_overhead_mb"]
    assert nf4_double["effective_bits_per_param"] == pytest.approx(4.127, abs=1e-3)
