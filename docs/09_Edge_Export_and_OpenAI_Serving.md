# Module 09 — Edge Runtime Export (`GGUF`, `vLLM`, `ONNX`/JIT) & OpenAI-Compatible + MCP Serving Gateway

> **Source Implementations:**
> - [`src/module_09_serving/edge_exporter.py`](../src/module_09_serving/edge_exporter.py)
> - [`src/module_09_serving/openai_server.py`](../src/module_09_serving/openai_server.py)
>
> **Verification Suite:** [`tests/test_serving_and_export.py`](../tests/test_serving_and_export.py)

---

## 1. Architectural Purpose

Once a Small Language Model (`1B–4B` parameters) has been distilled (Module 03), pruned (Module 04), quantized (Module 05), and preference-aligned (Module 06), enterprise adoption depends on **zero-friction edge export and standard protocol interoperability**. Module 09 provides two production sub-systems:

1. **Edge Runtime Exporter (`edge_exporter.py`):**
   - **GGUF / `llama.cpp` & `Ollama` (`generate_gguf_export_plan`):** Computes exact k-quant weight footprints (`Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `F16`) and Grouped-Query Attention (GQA) FP16 KV-cache budgets, outputting deterministic `convert_hf_to_gguf.py`, `llama-quantize`, `llama-server`, and `Ollama` `Modelfile` artifacts.
   - **`vLLM` PagedAttention Validator (`generate_vllm_serving_config`):** Verifies that model weights plus the CUDA execution context fit within the target GPU VRAM budget (`--gpu-memory-utilization`) and computes maximum concurrent request capacity at the configured context length.
   - **Portable TorchScript / ONNX Graph Exporter (`export_pytorch_module_to_edge`):** Traces and exports PyTorch modules, verifies numerical parity (`max_abs_parity_error < 1e-5`) against eager execution, and writes a cryptographic SHA-256 artifact manifest.

2. **OpenAI-Compatible & Model Context Protocol (MCP) Gateway (`openai_server.py`):**
   - Exposes standard OpenAI endpoints (`POST /v1/chat/completions`, `GET /v1/models`, `GET /v1/routing/telemetry`, `GET /health`) enriched with an `x_slm_routing` **Gartner AI TRiSM** telemetry object (`target_tier`, `complexity_score`, `confidence_score`, `calibrated_win_probability`, `contains_pii`, `pii_categories`, `rationale`).
   - Exposes a **Model Context Protocol (MCP) JSON-RPC 2.0** endpoint (`POST /mcp`) providing three agentic tools: `route_query`, `redact_pii`, and `estimate_edge_vram`.

---

## 2. Quickstart Usage

```bash
# 1. Generate GGUF, vLLM, and TorchScript/ONNX edge export manifests
python src/module_09_serving/edge_exporter.py

# 2. Run an in-memory smoke verification of the OpenAI & MCP Gateway
python src/module_09_serving/openai_server.py --smoke-test

# 3. Launch the live HTTP server on port 8000
python src/module_09_serving/openai_server.py --host 127.0.0.1 --port 8000
```
