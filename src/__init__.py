"""
SLM Masterclass — Core Package
==============================

Production-grade reference toolkit and engineering playbook for training,
distilling, pruning, quantizing, aligning, evaluating, routing, and auditing
Small Language Models (<4B parameters).

Progressive Architecture Modules (00 through 32):
    - module_00_fundamentals: Tokenization (BPE) and Self-Attention / GQA KV-cache math
    - module_01_inference: Autoregressive decoding strategies and TTFT/throughput profiling
    - module_02_sft: Supervised Fine-Tuning with LoRA and 4-bit NormalFloat (QLoRA)
    - module_03_distillation: On-Policy Generalized Knowledge Distillation (GKD & MiniLLM)
    - module_04_pruning: Magnitude and Wanda Pruning (Per-Output Unstructured & 2:4 Semi-Structured)
    - module_05_quantization: AWQ, SpinQuant Hadamard, Cayley Stiefel (INT3/INT2), and INT4 KV-Cache
    - module_06_alignment: Preference Alignment via DPO and Reference-Free SimPO
    - module_07_evaluation: Perplexity, Latency Telemetry, and EleutherAI LM-Eval Harness
    - module_08_routing: Calibrated Hybrid SLM-LLM Routing, RouteLLM, and AI TRiSM PII Guardrails
    - module_09_serving: GGUF / vLLM / TorchScript Edge Exporter and OpenAI + MCP Gateway
    - module_10_speculative: Lossless Speculative Decoding and Medusa Multi-Horizon Draft Heads
    - module_11_rlvr_grpo: Reinforcement Learning with Verifiable Rewards (RLVR) and Critic-Free GRPO
    - module_12_hybrid_ssm: State-Space and Hybrid Mamba-Transformer SLMs with O(1) Recurrence
    - module_13_agentic_swarms: Multi-Agent Collaborative SLM Swarms and Debate Verification
    - module_14_federated_dp_lora: Federated and Differential-Privacy LoRA (DP-SGD, FFA-LoRA, SVD-Exact)
    - module_15_bitnet_ternary: Ternary 1.58-Bit Native SLMs (BitNet b1.58) and Multiplier-Free Kernels
    - module_16_latent_reasoning: Latent-Space Continuous Reasoning (Coconut) and Adaptive Pondering
    - module_17_mixed_precision: Layer-Adaptive Mixed-Precision (LAMP) and Hessian Bit Allocator
    - module_18_constrained_decoding: Grammar-Constrained and DFA/FSM Structured JSON Decoding
    - module_19_cross_tokenizer_kd: Cross-Tokenizer Knowledge Distillation (ULD Optimal Transport & MinED)
    - module_20_mixture_of_depths: Mixture-of-Depths (MoD) Dynamic Token Routing
    - module_21_model_merging: Task-Arithmetic Model Merging (TIES-Merging, DARE, and SLERP)
    - module_22_kv_cache_eviction: Long-Context KV-Cache Eviction (StreamingLLM Sinks & H2O Oracle)
    - module_23_representation_steering: Activation Steering and Representation Engineering (RepE & CAA)
    - module_24_test_time_prm: Test-Time Compute Scaling via Process Reward Models (PRM) and Beam Search
    - module_25_matryoshka_minitron: Matryoshka Representation Learning (MRL) and Minitron Elastic Slicing
    - module_26_sparse_autoencoders: Mechanistic Interpretability via Top-K Sparse Autoencoders (SAE)
    - module_27_mla_latent_attention: Multi-Head Latent Attention (MLA) and Weight-Absorbed KV Compression
    - module_28_byte_latent_patching: Byte-Latent Dynamic Patch Tokenization (BLT)
    - module_29_machine_unlearning: Machine Unlearning via NPO and Representation Misdirection (RMU)
    - module_30_differential_attention: Differential Transformer (DiffAttn) Noise Cancellation
    - module_31_dora_lora_plus: Weight-Decomposed Low-Rank Adaptation (DoRA) and Asymmetric LoRA+
    - module_32_statistical_watermarking: Statistical Watermarking (KGW Green-List Logit Bias) for AI TRiSM
"""

__version__ = "2.0.0"
__author__ = "Vinicius Caridá"
__all__ = ["__version__", "__author__"]
