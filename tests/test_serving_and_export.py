"""Unit and live HTTP integration tests for Module 09 Edge Export & OpenAI/MCP Serving."""

import json
from pathlib import Path
import threading
import urllib.request

import pytest
import torch
import torch.nn as nn

from src.module_09_serving.edge_exporter import (
    export_pytorch_module_to_edge,
    generate_gguf_export_plan,
    generate_vllm_serving_config,
)
from src.module_09_serving.openai_server import (
    SLMInferenceGateway,
    create_http_server,
)


def test_generate_gguf_export_plan() -> None:
    plan = generate_gguf_export_plan(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        num_params_b=1.54,
        quant_type="Q4_K_M",
        context_length=4096,
    )
    assert plan.quant_type == "Q4_K_M"
    assert plan.bits_per_weight == 4.85
    assert 800.0 <= plan.estimated_weights_mb <= 1100.0
    assert plan.total_vram_mb < 2048.0
    assert "llama-quantize" in plan.quantize_command
    assert "FROM " in plan.ollama_modelfile

    with pytest.raises(ValueError, match="Unsupported GGUF quant_type"):
        generate_gguf_export_plan(quant_type="INVALID_QUANT")


def test_generate_vllm_serving_config() -> None:
    spec = generate_vllm_serving_config(
        model_id="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        num_params_b=1.54,
        quantization="awq",
        max_model_len=4096,
        gpu_memory_utilization=0.85,
        vram_budget_gb=8.0,
    )
    assert spec.fits_vram_budget is True
    assert spec.max_concurrent_requests >= 20
    assert "--quantization awq" in spec.cli_command


def test_export_pytorch_module_to_edge_parity(tmp_path: Path) -> None:
    block = nn.Sequential(
        nn.Linear(32, 64, bias=False),
        nn.SiLU(),
        nn.Linear(64, 16, bias=False),
    )
    sample_x = torch.randn(2, 32)
    manifest = export_pytorch_module_to_edge(
        block, sample_x, output_dir=tmp_path, model_name="test_slm_head"
    )
    assert manifest["parity_verified"] is True
    assert manifest["max_abs_parity_error"] < 1e-5
    assert len(manifest["sha256"]) == 64
    assert (tmp_path / "test_slm_head.pt").exists()
    assert (tmp_path / "test_slm_head_manifest.json").exists()


def test_openai_gateway_and_mcp_in_memory() -> None:
    gateway = SLMInferenceGateway()

    models_resp = gateway.list_models()
    assert models_resp["object"] == "list"
    assert len(models_resp["data"]) == 3

    slm_resp = gateway.create_chat_completion(
        {
            "model": "slm-hybrid-router",
            "messages": [
                {
                    "role": "user",
                    "content": "Classify customer intent: 'Check my account balance.'",
                }
            ],
        }
    )
    assert slm_resp["object"] == "chat.completion"
    assert slm_resp["x_slm_routing"]["target_tier"] == "local_slm"

    cloud_resp = gateway.create_chat_completion(
        {
            "model": "slm-hybrid-router",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Derive a formal step-by-step optimality proof and counterexample "
                        "for non-convex Cayley manifold optimization."
                    ),
                }
            ],
        }
    )
    assert cloud_resp["x_slm_routing"]["target_tier"] == "cloud_llm"

    mcp_list = gateway.handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    tool_names = {t["name"] for t in mcp_list["result"]["tools"]}
    assert tool_names == {"route_query", "redact_pii", "estimate_edge_vram"}

    mcp_pii = gateway.handle_mcp_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "redact_pii",
                "arguments": {"text": "Contact alice@company.com or SSN 123-45-6789."},
            },
        }
    )
    structured = mcp_pii["result"]["structuredContent"]
    assert structured["contains_pii"] is True
    assert "[REDACTED_EMAIL]" in structured["redacted_text"]
    assert "[REDACTED_SSN_OR_CPF]" in structured["redacted_text"]


def test_live_http_server_endpoints() -> None:
    gateway = SLMInferenceGateway()
    server = create_http_server(gateway, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    try:
        # 1. GET /health
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
            assert resp.status == 200
            health_data = json.loads(resp.read().decode("utf-8"))
            assert health_data["status"] == "healthy"

        # 2. POST /v1/chat/completions
        chat_req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": "slm-hybrid-router",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Summarize in one sentence: Server uptime reached 99.98%.",
                        }
                    ],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(chat_req, timeout=5) as resp:
            assert resp.status == 200
            chat_data = json.loads(resp.read().decode("utf-8"))
            assert chat_data["x_slm_routing"]["target_tier"] == "local_slm"

        # 3. POST /mcp
        mcp_req = urllib.request.Request(
            f"{base_url}/mcp",
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 99,
                    "method": "tools/call",
                    "params": {
                        "name": "estimate_edge_vram",
                        "arguments": {"num_params_b": 1.54, "quant_type": "Q4_K_M"},
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(mcp_req, timeout=5) as resp:
            assert resp.status == 200
            mcp_data = json.loads(resp.read().decode("utf-8"))
            assert mcp_data["result"]["structuredContent"]["fits_8gb_consumer_gpu"] is True

        # 4. GET /v1/routing/telemetry
        with urllib.request.urlopen(f"{base_url}/v1/routing/telemetry", timeout=5) as resp:
            assert resp.status == 200
            telemetry_data = json.loads(resp.read().decode("utf-8"))
            assert telemetry_data["recorded_decisions"] == 1
    finally:
        server.shutdown()
        server.server_close()
