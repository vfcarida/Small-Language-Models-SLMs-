"""
==============================================================================
MODULE 09B — OPENAI-COMPATIBLE & MCP SERVING GATEWAY FOR HYBRID SLMs
==============================================================================
OBJECTIVE:
    Provide a zero-friction, OpenAI-compatible REST endpoint and Model Context
    Protocol (MCP) JSON-RPC 2.0 server wrapping:
    1. Calibrated Hybrid SLM-LLM Routing (`HybridSLMRouter` + `CalibratedPreferenceRouter`)
    2. Enterprise AI TRiSM PII Guardrails (`PIIGuardrail`)
    3. Edge VRAM & GGUF/vLLM Sizing Tools (`generate_gguf_export_plan`)

ENDPOINTS:
    - GET  /health               -> Readiness & router configuration probe
    - GET  /v1/models            -> OpenAI-compatible model catalog
    - GET  /v1/routing/telemetry -> Fleet SLM offload, TCO savings & latency metrics
    - POST /v1/chat/completions  -> OpenAI chat completion + `x_slm_routing` audit
    - POST /mcp                  -> Model Context Protocol (MCP) JSON-RPC 2.0 handler

RUN COMMAND:
    python src/module_09_serving/openai_server.py --smoke-test
==============================================================================
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.module_05_quantization.bitsandbytes_quant import estimate_quantized_memory_mb
from src.module_08_routing.hybrid_router import (
    HybridSLMRouter,
    PIIGuardrail,
    RouteTarget,
    RoutingDecision,
    evaluate_routing_economics,
)
from src.module_08_routing.router_calibration import CalibratedPreferenceRouter
from src.module_09_serving.edge_exporter import generate_gguf_export_plan
from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SLMInferenceGateway:
    """
    Stateful enterprise serving gateway uniting OpenAI `/v1/chat/completions`
    and Model Context Protocol (MCP) JSON-RPC 2.0 tool execution with
    calibrated "Small-First" SLM-LLM routing and AI TRiSM telemetry.
    """

    def __init__(
        self,
        slm_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        llm_model_id: str = "meta-llama/Llama-3.1-70B-Instruct",
        router: Optional[HybridSLMRouter] = None,
        calibrated_router: Optional[CalibratedPreferenceRouter] = None,
        redact_pii_before_cloud: bool = True,
    ) -> None:
        self.slm_model_id = slm_model_id
        self.llm_model_id = llm_model_id
        self.calibrated_router = calibrated_router or CalibratedPreferenceRouter()
        self.router = router or HybridSLMRouter(
            calibrated_router=self.calibrated_router
        )
        self.redact_pii_before_cloud = redact_pii_before_cloud
        self.decision_history: List[RoutingDecision] = []

    def list_models(self) -> Dict[str, Any]:
        """Return an OpenAI-compatible `/v1/models` response."""
        created_ts = 1720000000
        return {
            "object": "list",
            "data": [
                {
                    "id": "slm-hybrid-router",
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "slm-masterclass",
                },
                {
                    "id": self.slm_model_id,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "local-edge-slm",
                },
                {
                    "id": self.llm_model_id,
                    "object": "model",
                    "created": created_ts,
                    "owned_by": "frontier-cloud-llm",
                },
            ],
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Return cumulative fleet routing economics for `/v1/routing/telemetry`."""
        econ = evaluate_routing_economics(self.decision_history)
        return {
            "status": "ok",
            "slm_model_id": self.slm_model_id,
            "llm_model_id": self.llm_model_id,
            "recorded_decisions": len(self.decision_history),
            "economics": econ,
        }

    def create_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an OpenAI-compatible `/v1/chat/completions` request through the
        calibrated hybrid router and AI TRiSM guardrail pipeline.
        """
        t0 = time.perf_counter()
        messages = payload.get("messages", [])
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Request payload must contain a non-empty 'messages' list.")

        user_messages = [
            str(m.get("content", ""))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ]
        query = user_messages[-1] if user_messages else str(messages[-1].get("content", ""))

        decision = self.router.route(query)
        self.decision_history.append(decision)
        win_prob = self.calibrated_router.predict_win_probability(query)

        if decision.target == RouteTarget.ABSTAIN_HUMAN_REVIEW:
            executed_model = "ai-trism-abstention-gate"
            assistant_text = (
                "[AI TRiSM ABSTENTION] Sensitive PII detected with insufficient local SLM "
                "confidence. Request held for authorized human review to preserve data sovereignty."
            )
        elif decision.target == RouteTarget.LOCAL_SLM:
            executed_model = self.slm_model_id
            sanitized_preview = PIIGuardrail.redact(query) if decision.contains_pii else query
            assistant_text = (
                f"[Executed on Local Edge SLM ({self.slm_model_id})] "
                f"Processed request locally with zero external API egress: {sanitized_preview[:120]}"
            )
        else:
            executed_model = self.llm_model_id
            outbound_query = (
                PIIGuardrail.redact(query)
                if self.redact_pii_before_cloud and decision.contains_pii
                else query
            )
            assistant_text = (
                f"[Escalated to Frontier Cloud LLM ({self.llm_model_id})] "
                f"Synthesized high-complexity response for: {outbound_query[:120]}"
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        prompt_tokens = max(1, len(query.split()))
        completion_tokens = max(1, len(assistant_text.split()))

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": executed_model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_text,
                    },
                    "finish_reason": (
                        "content_filter"
                        if decision.target == RouteTarget.ABSTAIN_HUMAN_REVIEW
                        else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "x_slm_routing": {
                "target_tier": decision.target.value,
                "executed_model": executed_model,
                "complexity_score": decision.complexity_score,
                "confidence_score": decision.confidence_score,
                "calibrated_win_probability": win_prob,
                "contains_pii": decision.contains_pii,
                "pii_categories": list(decision.pii_categories),
                "rationale": decision.rationale,
                "routing_latency_ms": elapsed_ms,
            },
        }

    def handle_mcp_request(self, rpc_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a Model Context Protocol (MCP) JSON-RPC 2.0 request supporting
        `tools/list` and `tools/call` (`route_query`, `redact_pii`, `estimate_edge_vram`).
        """
        req_id = rpc_request.get("id", 1)
        method = rpc_request.get("method", "")
        params = rpc_request.get("params", {}) or {}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "route_query",
                            "description": "Evaluate query complexity, PII sovereignty guardrails, and calibrated SLM win probability.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                                "required": ["query"],
                            },
                        },
                        {
                            "name": "redact_pii",
                            "description": "Inspect and redact PII tokens (EMAIL, SSN/CPF, CREDIT_CARD, PHONE, IPv4) under AI TRiSM.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                            },
                        },
                        {
                            "name": "estimate_edge_vram",
                            "description": "Calculate GGUF and NF4/AWQ VRAM budgets for a given parameter count and context length.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "num_params_b": {"type": "number"},
                                    "quant_type": {"type": "string"},
                                    "context_length": {"type": "integer"},
                                },
                                "required": ["num_params_b"],
                            },
                        },
                    ]
                },
            }

        if method == "tools/call":
            tool_name = params.get("name")
            args = params.get("arguments", {}) or {}

            if tool_name == "route_query":
                q = str(args.get("query", ""))
                dec = self.router.route(q)
                win_prob = self.calibrated_router.predict_win_probability(q)
                result_data = {
                    "target_tier": dec.target.value,
                    "complexity_score": dec.complexity_score,
                    "confidence_score": dec.confidence_score,
                    "calibrated_win_probability": win_prob,
                    "contains_pii": dec.contains_pii,
                    "pii_categories": list(dec.pii_categories),
                    "rationale": dec.rationale,
                }
            elif tool_name == "redact_pii":
                text = str(args.get("text", ""))
                cats = PIIGuardrail.inspect(text)
                redacted = PIIGuardrail.redact(text)
                result_data = {
                    "contains_pii": len(cats) > 0,
                    "pii_categories": cats,
                    "redacted_text": redacted,
                }
            elif tool_name == "estimate_edge_vram":
                num_params_b = float(args.get("num_params_b", 1.54))
                quant_type = str(args.get("quant_type", "Q4_K_M"))
                context_length = int(args.get("context_length", 4096))
                gguf_plan = generate_gguf_export_plan(
                    num_params_b=num_params_b,
                    quant_type=quant_type,
                    context_length=context_length,
                )
                nf4_mem = estimate_quantized_memory_mb(
                    int(num_params_b * 1e9), bit_width=4, double_quant=True
                )
                result_data = {
                    "num_params_b": num_params_b,
                    "gguf_quant_type": gguf_plan.quant_type,
                    "gguf_total_vram_mb": gguf_plan.total_vram_mb,
                    "nf4_double_quant_weights_mb": nf4_mem["total_memory_mb"],
                    "fits_8gb_consumer_gpu": gguf_plan.total_vram_mb <= 8192.0,
                }
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown MCP tool: '{tool_name}'",
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result_data),
                        }
                    ],
                    "structuredContent": result_data,
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32601,
                "message": f"Unsupported JSON-RPC method: '{method}'",
            },
        }


def create_http_server(
    gateway: Optional[SLMInferenceGateway] = None,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    """Create a thread-safe HTTP server exposing OpenAI and MCP endpoints."""
    active_gateway = gateway or SLMInferenceGateway()

    class GatewayHTTPRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, body: Dict[str, Any]) -> None:
            encoded = json.dumps(body).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "healthy",
                        "slm_model_id": active_gateway.slm_model_id,
                        "llm_model_id": active_gateway.llm_model_id,
                    },
                )
            elif self.path == "/v1/models":
                self._send_json(200, active_gateway.list_models())
            elif self.path == "/v1/routing/telemetry":
                self._send_json(200, active_gateway.get_telemetry())
            else:
                self._send_json(404, {"error": f"Unknown GET endpoint: {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            content_len = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_len).decode("utf-8") if content_len > 0 else "{}"
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                self._send_json(400, {"error": "Malformed JSON request body."})
                return

            if self.path == "/v1/chat/completions":
                try:
                    resp = active_gateway.create_chat_completion(payload)
                    self._send_json(200, resp)
                except ValueError as exc:
                    self._send_json(400, {"error": str(exc)})
            elif self.path == "/mcp":
                resp = active_gateway.handle_mcp_request(payload)
                self._send_json(200, resp)
            else:
                self._send_json(404, {"error": f"Unknown POST endpoint: {self.path}"})

    return ThreadingHTTPServer((host, port), GatewayHTTPRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI-Compatible & MCP Serving Gateway for Hybrid SLMs"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run an in-memory smoke test of OpenAI and MCP endpoints and exit.",
    )
    args = parser.parse_args()

    config = load_config()
    routing_cfg = config.get("routing", {})
    gateway = SLMInferenceGateway(
        slm_model_id=routing_cfg.get("slm_model_id", "Qwen/Qwen2.5-1.5B-Instruct-AWQ"),
        llm_model_id=routing_cfg.get("llm_model_id", "meta-llama/Llama-3.1-70B-Instruct"),
    )

    if args.smoke_test:
        logger.info("=" * 72)
        logger.info("MODULE 09B: OPENAI + MCP GATEWAY SMOKE VERIFICATION")
        logger.info("=" * 72)
        resp_slm = gateway.create_chat_completion(
            {
                "model": "slm-hybrid-router",
                "messages": [
                    {
                        "role": "user",
                        "content": "Extract JSON fields from invoice #402: Total=$199.00",
                    }
                ],
            }
        )
        logger.info(
            f"[OpenAI /v1/chat/completions] Routed to={resp_slm['x_slm_routing']['target_tier']} "
            f"| Model={resp_slm['model']} | WinProb={resp_slm['x_slm_routing']['calibrated_win_probability']}"
        )
        mcp_resp = gateway.handle_mcp_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "estimate_edge_vram",
                    "arguments": {"num_params_b": 1.54, "quant_type": "Q4_K_M"},
                },
            }
        )
        logger.info(
            f"[MCP tools/call estimate_edge_vram] Result={mcp_resp['result']['structuredContent']}"
        )
        return

    server = create_http_server(gateway, host=args.host, port=args.port)
    logger.info(f"Serving OpenAI (/v1/chat/completions) & MCP (/mcp) on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
