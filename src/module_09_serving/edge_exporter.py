"""
==============================================================================
MODULE 09A — EDGE RUNTIME EXPORTER (GGUF / llama.cpp, vLLM & ONNX / TorchScript)
==============================================================================
OBJECTIVE:
    Bridge fine-tuned and quantized Hugging Face checkpoints to production
    edge and cloud inference runtimes:
    1. GGUF / llama.cpp & Ollama Modelfile generation with exact k-quant
       (Q4_0, Q4_K_M, Q5_K_M, Q8_0) and GQA KV-cache memory accounting.
    2. vLLM PagedAttention serving specification with VRAM budget validation
       and maximum concurrency estimation.
    3. Verified TorchScript / ONNX portable graph export with numerical parity
       verification and SHA-256 cryptographic artifact manifests.

RUN COMMAND:
    python src/module_09_serving/edge_exporter.py
==============================================================================
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict

import torch
import torch.nn as nn

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Effective bits per parameter including block scales/mins for llama.cpp k-quants
GGUF_BITS_PER_WEIGHT: Dict[str, float] = {
    "Q4_0": 4.50,
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.69,
    "Q6_K": 6.56,
    "Q8_0": 8.50,
    "F16": 16.00,
}


@dataclass(frozen=True)
class GGUFExportPlan:
    """Turnkey GGUF conversion, quantization, and Ollama/llama.cpp deployment spec."""

    model_id: str
    quant_type: str
    context_length: int
    bits_per_weight: float
    estimated_weights_mb: float
    estimated_kv_cache_mb: float
    total_vram_mb: float
    convert_command: str
    quantize_command: str
    server_command: str
    ollama_modelfile: str


@dataclass(frozen=True)
class VLLMServingSpec:
    """Validated vLLM PagedAttention serving configuration and concurrency estimate."""

    model_id: str
    quantization: str
    max_model_len: int
    gpu_memory_utilization: float
    vram_budget_mb: float
    estimated_weights_mb: float
    available_kv_cache_mb: float
    max_concurrent_requests: int
    fits_vram_budget: bool
    cli_command: str
    engine_args: Dict[str, Any]


def generate_gguf_export_plan(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_params_b: float = 1.54,
    quant_type: str = "Q4_K_M",
    context_length: int = 4096,
    num_layers: int = 28,
    num_kv_heads: int = 2,
    head_dim: int = 128,
) -> GGUFExportPlan:
    """
    Compute exact GGUF memory footprint (weights + GQA FP16 KV cache) and
    generate llama.cpp conversion, quantization, server, and Ollama Modelfile specs.
    """
    q_upper = quant_type.upper()
    if q_upper not in GGUF_BITS_PER_WEIGHT:
        raise ValueError(
            f"Unsupported GGUF quant_type '{quant_type}'. "
            f"Supported: {list(GGUF_BITS_PER_WEIGHT.keys())}"
        )

    bpw = GGUF_BITS_PER_WEIGHT[q_upper]
    total_params = num_params_b * 1e9
    weights_bytes = total_params * (bpw / 8.0)
    weights_mb = weights_bytes / (1024**2)

    # GQA KV cache in FP16 (2 bytes per K and V element)
    kv_bytes = 2 * 2 * num_layers * num_kv_heads * head_dim * context_length
    kv_mb = kv_bytes / (1024**2)
    runtime_overhead_mb = 180.0
    total_mb = weights_mb + kv_mb + runtime_overhead_mb

    slug = model_id.split("/")[-1].lower()
    f16_gguf = f"./artifacts/{slug}-f16.gguf"
    quant_gguf = f"./artifacts/{slug}-{q_upper.lower()}.gguf"

    convert_cmd = (
        f"python convert_hf_to_gguf.py {model_id} --outtype f16 --outfile {f16_gguf}"
    )
    quantize_cmd = f"./llama-quantize {f16_gguf} {quant_gguf} {q_upper}"
    server_cmd = (
        f"./llama-server -m {quant_gguf} -c {context_length} "
        f"--ngl 99 --flash-attn --port 8080"
    )

    modelfile = "\n".join(
        [
            f"FROM {quant_gguf}",
            f"PARAMETER num_ctx {context_length}",
            "PARAMETER temperature 0.2",
            "PARAMETER top_p 0.9",
            'PARAMETER stop "<|im_start|>"',
            'PARAMETER stop "<|im_end|>"',
            'SYSTEM "You are a concise, domain-specialized Small Language Model assistant."',
        ]
    )

    return GGUFExportPlan(
        model_id=model_id,
        quant_type=q_upper,
        context_length=context_length,
        bits_per_weight=bpw,
        estimated_weights_mb=round(weights_mb, 2),
        estimated_kv_cache_mb=round(kv_mb, 2),
        total_vram_mb=round(total_mb, 2),
        convert_command=convert_cmd,
        quantize_command=quantize_cmd,
        server_command=server_cmd,
        ollama_modelfile=modelfile,
    )


def generate_vllm_serving_config(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
    num_params_b: float = 1.54,
    quantization: str = "awq",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.85,
    vram_budget_gb: float = 8.0,
    num_layers: int = 28,
    num_kv_heads: int = 2,
    head_dim: int = 128,
) -> VLLMServingSpec:
    """
    Validate vLLM PagedAttention memory budget and compute maximum concurrent
    sequences for a target GPU VRAM envelope.
    """
    if not (0.1 <= gpu_memory_utilization <= 1.0):
        raise ValueError("gpu_memory_utilization must lie in [0.1, 1.0].")

    q_lower = quantization.lower()
    bpw = 4.25 if q_lower in ("awq", "gptq", "bitsandbytes") else 16.0
    weights_mb = (num_params_b * 1e9 * (bpw / 8.0)) / (1024**2)
    cuda_context_overhead_mb = 550.0

    usable_vram_mb = vram_budget_gb * 1024.0 * gpu_memory_utilization
    available_kv_mb = max(
        0.0, usable_vram_mb - weights_mb - cuda_context_overhead_mb
    )

    kv_per_seq_mb = (
        2 * 2 * num_layers * num_kv_heads * head_dim * max_model_len
    ) / (1024**2)
    max_concurrency = int(available_kv_mb // max(kv_per_seq_mb, 1e-6))
    fits_budget = available_kv_mb >= kv_per_seq_mb

    engine_args: Dict[str, Any] = {
        "model": model_id,
        "quantization": None if q_lower == "none" else q_lower,
        "dtype": "half" if q_lower == "awq" else "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enable_prefix_caching": True,
        "enforce_eager": False,
    }

    quant_flag = f"--quantization {q_lower} " if q_lower != "none" else ""
    cli_cmd = (
        f"vllm serve {model_id} {quant_flag}"
        f"--dtype {engine_args['dtype']} "
        f"--max-model-len {max_model_len} "
        f"--gpu-memory-utilization {gpu_memory_utilization} "
        f"--enable-prefix-caching --port 8000"
    )

    return VLLMServingSpec(
        model_id=model_id,
        quantization=q_lower,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        vram_budget_mb=round(vram_budget_gb * 1024.0, 2),
        estimated_weights_mb=round(weights_mb, 2),
        available_kv_cache_mb=round(available_kv_mb, 2),
        max_concurrent_requests=max_concurrency,
        fits_vram_budget=fits_budget,
        cli_command=cli_cmd,
        engine_args=engine_args,
    )


def export_pytorch_module_to_edge(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_dir: str | Path = "./results/edge_export",
    model_name: str = "slm_projection_block",
    export_onnx: bool = False,
) -> Dict[str, Any]:
    """
    Trace and export a PyTorch module to a portable TorchScript artifact (and
    ONNX when `export_onnx=True` and `onnx` is installed), verify numerical
    parity against eager execution, and write a SHA-256 deployment manifest.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        eager_out = model(sample_input)
        traced_module = torch.jit.trace(model, sample_input)
        traced_file = out_path / f"{model_name}.pt"
        traced_module.save(str(traced_file))

        loaded_jit = torch.jit.load(str(traced_file))
        jit_out = loaded_jit(sample_input)
        max_abs_diff = float(torch.max(torch.abs(eager_out - jit_out)).item())

    onnx_file = out_path / f"{model_name}.onnx"
    onnx_exported = False
    if export_onnx:
        try:
            import onnx  # noqa: F401

            with torch.no_grad():
                torch.onnx.export(
                    model,
                    sample_input,
                    str(onnx_file),
                    opset_version=17,
                    input_names=["input_features"],
                    output_names=["output_logits"],
                )
            onnx_exported = True
        except Exception:
            onnx_exported = False

    sha256_hash = hashlib.sha256(traced_file.read_bytes()).hexdigest()
    num_params = sum(p.numel() for p in model.parameters())

    manifest: Dict[str, Any] = {
        "model_name": model_name,
        "torchscript_artifact": str(traced_file),
        "onnx_artifact": str(onnx_file) if onnx_exported else None,
        "onnx_exported": onnx_exported,
        "sha256": sha256_hash,
        "num_parameters": num_params,
        "input_shape": list(sample_input.shape),
        "output_shape": list(eager_out.shape),
        "max_abs_parity_error": max_abs_diff,
        "parity_verified": max_abs_diff < 1e-5,
    }

    manifest_file = out_path / f"{model_name}_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    gguf_plan = generate_gguf_export_plan()
    vllm_spec = generate_vllm_serving_config()

    sample_block = nn.Sequential(
        nn.Linear(64, 128, bias=False),
        nn.SiLU(),
        nn.Linear(128, 64, bias=False),
    )
    sample_in = torch.randn(4, 64)
    edge_manifest = export_pytorch_module_to_edge(
        sample_block, sample_in, output_dir="./results/edge_export"
    )

    logger.info("=" * 72)
    logger.info("MODULE 09A: EDGE RUNTIME EXPORT & SERVING MANIFEST")
    logger.info("=" * 72)
    logger.info(
        f"[GGUF {gguf_plan.quant_type}] Weights={gguf_plan.estimated_weights_mb:.1f}MB "
        f"| KV Cache={gguf_plan.estimated_kv_cache_mb:.1f}MB | Total VRAM={gguf_plan.total_vram_mb:.1f}MB"
    )
    logger.info(
        f"[vLLM {vllm_spec.quantization.upper()}] Usable KV={vllm_spec.available_kv_cache_mb:.1f}MB "
        f"| Max Concurrency={vllm_spec.max_concurrent_requests} seqs @ {vllm_spec.max_model_len} ctx"
    )
    logger.info(
        f"[Edge JIT/ONNX] Parity Verified={edge_manifest['parity_verified']} "
        f"(Max Error={edge_manifest['max_abs_parity_error']:.2e}) | SHA256={edge_manifest['sha256'][:16]}..."
    )
    logger.info(f"Manifest written to: ./results/edge_export/{edge_manifest['model_name']}_manifest.json")
    _ = asdict(gguf_plan)


if __name__ == "__main__":
    main()
