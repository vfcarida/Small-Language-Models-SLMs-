# Product & Engineering Roadmap (`slm-masterclass`)

> **Principal Architect:** [Vinicius Caridá (`vfcarida`)](https://github.com/vfcarida)

---

## 1. Vision & Architectural North Star

`slm-masterclass` is designed to be the definitive open-source engineering playbook and glass-box PyTorch reference toolkit for **Small Language Models (<4B parameters)**. Grounded in peer-reviewed research from **ICLR, NeurIPS, ICML, EMNLP, and MLSys** alongside enterprise governance frameworks (**Gartner Domain-Specific Language Models & AI TRiSM**, **BCG AI at Scale**, and **McKinsey State of AI**), this project bridges the gap between theoretical compression mathematics and reproducible edge/on-premises deployment.

---

## 2. Maturity Tiers & Verification Governance

To ensure engineering transparency for adopters and contributors, every module in this repository is governed by an explicit four-tier maturity taxonomy:

| Maturity Tier | Engineering & Verification Requirements |
| :---: | :--- |
| **`Stable`** | Complete pure-PyTorch mathematical implementation + production ecosystem integration, 100% passing deterministic unit tests in `tests/`, full international English docstrings and `docs/` guide, and zero unguarded optional imports. |
| **`Beta`** | Functional implementation with unit-tested core logic; full multi-hour GPU training or external cluster integration (e.g., remote 70B vLLM server) requires user-provisioned GPU hardware. |
| **`Alpha`** | Experimental prototype or early algorithmic module undergoing API stabilization. |
| **`Planned`** | Scoped architectural milestone scheduled for an upcoming evolution round. |

---

## 3. Capability & Milestone Matrix

| Track / Module | Architectural Scope | Primary Research / Industry Anchor | Current Status | Verification Gate |
| :--- | :--- | :--- | :---: | :--- |
| **Core Configuration & Telemetry (`src/utils/`)** | Schema-validated YAML config (`config.py`), Alpaca/Chat/Preference dataset formatting (`data_utils.py`), sliding-window perplexity, and `GPUOOMHandler` (`metrics.py`). | Jurafsky & Martin (2024); OpenAI Fine-Tuning Best Practices | **`Stable`** | `tests/test_config.py`, `tests/test_data_utils.py`, `tests/test_metrics.py` |
| **Module 00: Transformer Fundamentals (`src/module_00_fundamentals/`)** | BPE subword compression profiling (`tokenizer_explorer.py`), step-by-step Scaled Dot-Product Attention math, heatmaps, and MHA vs. GQA KV-cache profiler (`attention_visualizer.py`). | Vaswani et al. (NeurIPS 2017); Ainslie et al. (EMNLP 2023, `arXiv:2305.13245`) | **`Stable`** | `tests/test_fundamentals.py` |
| **Module 01: Autoregressive Inference (`src/module_01_inference/`)** | Greedy, temperature, Top-K, and Nucleus (Top-P) decoding (`basic_inference.py`), `torch.compile`, SDPA/FlashAttention-2 detection, and TTFT/throughput profiling. | Holtzman et al. (ICLR 2020, `arXiv:1904.09751`) | **`Stable`** | `tests/test_inference.py` |
| **Module 02: SFT with LoRA & QLoRA (`src/module_02_sft/`)** | All-linear LoRA (`sft_lora.py`), 4-bit NormalFloat (`NF4`) + Double Quantization (`sft_qlora.py`), and analytical adapter weight merging (`merge_lora_adapter`). | Hu et al. (ICLR 2022, `arXiv:2106.09685`); Dettmers et al. (NeurIPS 2023, `arXiv:2305.14314`) | **`Stable`** | `tests/test_sft.py` |
| **Module 03: Knowledge Distillation (`src/module_03_distillation/`)** | Temperature-scaled Forward KL, Mode-Seeking Reverse KL (MiniLLM), Generalized JSD (GKD), and decoupled `VLLMTeacherOracle` client (`gkd_pipeline.py`). | Hinton et al. (`arXiv:1503.02531`); Agarwal et al. (ICLR 2024, `arXiv:2306.13649`); Gu et al. (ICLR 2024, `arXiv:2306.08543`) | **`Stable`** | `tests/test_distillation.py` |
| **Module 04: Model Pruning (`src/module_04_pruning/`)** | Unstructured & NVIDIA `2:4` semi-structured Magnitude Pruning (`magnitude_pruning.py`) and per-output-row Wanda Pruning + Zipf lexical entropy calibration (`wanda_pruning.py`). | Sun et al. (ICLR 2024, `arXiv:2306.11695`); Frantar & Alistarh (ICML 2023, `arXiv:2301.00774`) | **`Stable`** | `tests/test_pruning.py` |
| **Module 05: Post-Training Quantization (`src/module_05_quantization/`)** | Pure-PyTorch AWQ salient-channel scaling grid search ($s = s_X^\alpha$), group-wise INT4 pseudo-quantization, SpinQuant Hadamard rotation smoothing, AutoAWQ CLI (`awq_quantization.py`), and NF4 memory calculator (`bitsandbytes_quant.py`). | Lin et al. (MLSys 2024 Best Paper, `arXiv:2306.00978`); Liu et al. (ICLR 2025, `arXiv:2405.16406`) | **`Stable`** | `tests/test_quantization.py` |
| **Module 06: Preference Alignment (`src/module_06_alignment/`)** | Bradley-Terry Direct Preference Optimization (`compute_dpo_loss`), Reference-Free Simple Preference Optimization (`compute_simpo_loss`), and `DPOTrainer` orchestration (`dpo_trainer.py`). | Rafailov et al. (NeurIPS 2023, `arXiv:2305.18290`); Meng et al. (NeurIPS 2024, `arXiv:2405.14734`) | **`Stable`** | `tests/test_alignment.py` |
| **Module 07: Evaluation, Pareto Frontier & Reporting (`src/module_07_evaluation/`, `eval/benchmarks/`, `pipeline/`)** | EleutherAI `lm-evaluation-harness` runner (`benchmark_runner.py`), Cross-Architecture Empirical Pareto Suite & live PyTorch compression ablation (`pareto_suite.py`), loss/radar visualizers (`results_visualizer.py`), and Markdown report generator (`generate_report.py`). | Gao et al. (EleutherAI, 2023); Hendrycks et al. (ICLR 2021) | **`Stable`** | `tests/test_evaluation_and_pipeline.py`, `tests/test_pareto_suite.py` |
| **Module 08: Hybrid SLM-LLM Routing & Calibration (`src/module_08_routing/`)** | Calibrated "Small-First" query router (`HybridSLMRouter`), `PIIGuardrail` (AI TRiSM / GDPR / LGPD), token-logit entropy/margin gate, TCO/ROI calculator (`hybrid_router.py`), and Bradley-Terry / Matrix Factorization preference calibration (`router_calibration.py`). | Ong et al. (ICLR 2025, `arXiv:2406.18665`); Narayan et al. (`arXiv:2502.15964`); Belcák et al. (`arXiv:2506.02153`); Gartner AI TRiSM | **`Stable`** | `tests/test_routing.py`, `tests/test_router_calibration.py` |
| **Module 09: Edge Runtime Export & OpenAI/MCP Serving (`src/module_09_serving/`)** | GGUF / `llama.cpp` + Ollama `Modelfile` planner, `vLLM` PagedAttention validator, TorchScript/ONNX graph exporter (`edge_exporter.py`), and OpenAI `/v1/chat/completions` + Model Context Protocol (MCP) JSON-RPC 2.0 gateway (`openai_server.py`). | Kwon et al. (SOSP 2023, PagedAttention/vLLM); Anthropic MCP Specification | **`Stable`** | `tests/test_serving_and_export.py` |

---

## 4. Next-Horizon Frontier Tracks (`Planned`)

1. **Track H1 — Speculative Decoding & Medusa Draft Heads (`Planned`):**
   Use a quantized 0.5B–1.5B SLM as an exact speculative draft model verifying parallel token trees against a 7B–70B target model with lossless distribution preservation.
2. **Track H2 — Learned Cayley Manifold Rotation Optimization (`Planned`):**
   Extend Module 05's analytical Hadamard rotation (`apply_hadamard_rotation_smoothing`) with Cayley SGD optimization over the Stiefel manifold $R^\top R = I$ for sub-4-bit (`INT3`/`INT2`) weight and KV-cache quantization.
3. **Track H3 — Reinforcement Learning with Verifiable Rewards (RLVR / GRPO) (`Planned`):**
   Implement Group Relative Policy Optimization (GRPO) with deterministic code/math verifiers inspired by `DeepSeek-R1` (`arXiv:2501.12948`) to incentivize multi-step reasoning in 1.5B SLMs without a value critic network.

