"""
SLM Masterclass — Core Package
==============================

Production-grade reference toolkit and engineering playbook for training,
distilling, pruning, quantizing, aligning, evaluating, and routing Small
Language Models (SLMs).

Progressive Architecture Modules:
    - module_00_fundamentals: Tokenization (BPE) and Self-Attention / GQA KV-cache math
    - module_01_inference: Autoregressive decoding strategies and TTFT/throughput profiling
    - module_02_sft: Supervised Fine-Tuning with LoRA and 4-bit NormalFloat (QLoRA)
    - module_03_distillation: On-Policy Generalized Knowledge Distillation (GKD & MiniLLM)
    - module_04_pruning: Magnitude and Wanda Pruning (Per-Output Unstructured & 2:4 Semi-Structured)
    - module_05_quantization: Activation-Aware Weight Quantization (AWQ) and BitsAndBytes NF4
    - module_06_alignment: Preference Alignment via DPO and Reference-Free SimPO
    - module_07_evaluation: Perplexity, Latency Telemetry, and EleutherAI LM-Eval Harness
    - module_08_routing: Calibrated Hybrid SLM-LLM Routing, AI TRiSM PII Guardrails, and TCO/ROI
"""

__version__ = "1.1.0"
__author__ = "Vinicius Caridá"
__all__ = ["__version__", "__author__"]
