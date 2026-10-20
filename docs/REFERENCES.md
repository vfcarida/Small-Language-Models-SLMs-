# Primary Scientific, Algorithmic & Enterprise Governance Bibliography

> **Principal Architect:** [Vinicius Caridá (`vfcarida`)](https://github.com/vfcarida)

This document catalogs the peer-reviewed academic papers (*NeurIPS, ICML, ICLR, ACL, EMNLP, COLM, MLSys, SOSP*) and enterprise strategy frameworks (*Gartner AI TRiSM, BCG, McKinsey*) implemented across **Modules 00 through 32** of `slm-masterclass`.

---

## 1. Foundational Attention, Tokenization & Inference (`Modules 00–01, 10, 18`)

1. **Vaswani et al. (NeurIPS 2017).** *"Attention Is All You Need."* [`arXiv:1706.03762`](https://arxiv.org/abs/1706.03762) — Scaled Dot-Product Attention (`src/module_00_fundamentals/`).
2. **Ainslie et al. (Google Research, EMNLP 2023).** *"GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints."* [`arXiv:2305.13245`](https://arxiv.org/abs/2305.13245) — Grouped-Query Attention KV-cache memory profiling (`src/module_00_fundamentals/`).
3. **Holtzman et al. (ICLR 2020).** *"The Curious Case of Neural Text Degeneration."* [`arXiv:1904.09751`](https://arxiv.org/abs/1904.09751) — Nucleus (`top_p`) and `top_k` sampling (`src/module_01_inference/`).
4. **Leviathan et al. (Google Research, ICML 2023).** *"Fast Inference from Transformers via Speculative Decoding."* [`arXiv:2211.17192`](https://arxiv.org/abs/2211.17192) — Lossless speculative rejection sampling (`src/module_10_speculative/`).
5. **Cai et al. ( Princeton / Together AI, ICML 2024).** *"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads."* [`arXiv:2401.10774`](https://arxiv.org/abs/2401.10774) — Multi-horizon draft heads (`src/module_10_speculative/`).
6. **Willard & Louf (2023).** *"Efficient Guided Generation for Large Language Models (Outlines)."* [`arXiv:2307.09702`](https://arxiv.org/abs/2307.09702) — DFA/FSM vocabulary masking (`src/module_18_constrained_decoding/`).

---

## 2. Parameter-Efficient Fine-Tuning & Model Merging (`Modules 02, 14, 21, 31`)

7. **Hu et al. (Microsoft, ICLR 2022).** *"LoRA: Low-Rank Adaptation of Large Language Models."* [`arXiv:2106.09685`](https://arxiv.org/abs/2106.09685) — Low-rank adapter math (`src/module_02_sft/`).
8. **Dettmers et al. (UW, NeurIPS 2023).** *"QLoRA: Efficient Finetuning of Quantized LLMs."* [`arXiv:2305.14314`](https://arxiv.org/abs/2305.14314) — 4-bit NormalFloat (`NF4`) and Double Quantization (`src/module_02_sft/`).
9. **Sun et al. (ICLR 2024).** *"Improving LoRA in Privacy-preserving Federated Learning (FFA-LoRA)."* [`arXiv:2403.12313`](https://arxiv.org/abs/2403.12313) — Cross-term-free DP-Federated LoRA (`src/module_14_federated_dp_lora/`).
10. **Abadi et al. (Google, ACM CCS 2016).** *"Deep Learning with Differential Privacy (DP-SGD)."* [`arXiv:1607.00133`](https://arxiv.org/abs/1607.00133) — Per-sample $L_2$ clipping and Gaussian mechanism (`src/module_14_federated_dp_lora/`).
11. **Yadav et al. (NeurIPS 2023).** *"TIES-Merging: Resolving Interference When Merging Models."* [`arXiv:2306.01708`](https://arxiv.org/abs/2306.01708) — Trim, Elect Sign & Merge (`src/module_21_model_merging/`).
12. **Yu et al. (ICML 2024).** *"Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch (DARE)."* [`arXiv:2311.03099`](https://arxiv.org/abs/2311.03099) — Drop And REscale task vectors (`src/module_21_model_merging/`).
13. **Liu et al. (NVIDIA Research, ICML 2024 Oral).** *"DoRA: Weight-Decomposed Low-Rank Adaptation."* [`arXiv:2402.09353`](https://arxiv.org/abs/2402.09353) — Magnitude-direction weight decomposition (`src/module_31_dora_lora_plus/`).
14. **Hayou et al. (ICML 2024).** *"LoRA+: Efficient Low Rank Adaptation of Large Models."* [`arXiv:2402.12354`](https://arxiv.org/abs/2402.12354) — Asymmetric learning rates $\eta_B = \lambda \eta_A$ (`src/module_31_dora_lora_plus/`).

---

## 3. Knowledge Distillation, Pruning & Quantization (`Modules 03–05, 15, 17, 19, 25`)

15. **Agarwal et al. (Google DeepMind, ICLR 2024).** *"On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)."* [`arXiv:2306.13649`](https://arxiv.org/abs/2306.13649) — Generalized JSD distillation (`src/module_03_distillation/`).
16. **Gu et al. (Microsoft Research / Tsinghua, ICLR 2024).** *"MiniLLM: On-Policy Distillation of Large Language Models."* [`arXiv:2306.08543`](https://arxiv.org/abs/2306.08543) — Mode-seeking Reverse KL (`src/module_03_distillation/`).
17. **Boizard et al. (ICML / TMLR 2024).** *"Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs (ULD)."* [`arXiv:2402.12030`](https://arxiv.org/abs/2402.12030) — 1D Wasserstein Optimal Transport KD (`src/module_19_cross_tokenizer_kd/`).
18. **Sun et al. (CMU & Princeton, ICLR 2024).** *"A Simple and Effective Pruning Approach for Large Language Models (Wanda)."* [`arXiv:2306.11695`](https://arxiv.org/abs/2306.11695) — Weight-Activation pruning (`src/module_04_pruning/`).
19. **Lin et al. (MIT Han Lab, MLSys 2024 Best Paper Award).** *"AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration."* [`arXiv:2306.00978`](https://arxiv.org/abs/2306.00978) — Salient-channel scaling (`src/module_05_quantization/`).
20. **Liu et al. (Meta AI, ICLR 2025).** *"SpinQuant: LLM quantization with learned rotations."* [`arXiv:2405.16406`](https://arxiv.org/abs/2405.16406) — Hadamard and Cayley Stiefel rotations (`src/module_05_quantization/`).
21. **Ma et al. (Microsoft Research, 2024).** *"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58)."* [`arXiv:2402.17764`](https://arxiv.org/abs/2402.17764) — Ternary $\{-1,0,+1\}$ weights (`src/module_15_bitnet_ternary/`).
22. **Kim et al. (UC Berkeley, ICML 2024).** *"SqueezeLLM: Dense-and-Sparse Quantization."* [`arXiv:2306.07629`](https://arxiv.org/abs/2306.07629) — Hessian sensitivity mixed-precision (`src/module_17_mixed_precision/`).
23. **Kusupati et al. (NeurIPS 2022).** *"Matryoshka Representation Learning."* [`arXiv:2205.13147`](https://arxiv.org/abs/2205.13147) — Multi-granularity nested prefixes (`src/module_25_matryoshka_minitron/`).
24. **Muralidharan et al. (NVIDIA Research, NeurIPS 2024).** *"Compact Language Models via Pruning and Knowledge Distillation (Minitron)."* [`arXiv:2407.14679`](https://arxiv.org/abs/2407.14679) — Width/depth importance slicing (`src/module_25_matryoshka_minitron/`).

---

## 4. Alignment, Reasoning, Next-Gen Architectures & AI TRiSM (`Modules 06, 08–09, 11–13, 16, 20, 22–24, 26–30, 32`)

25. **Rafailov et al. (Stanford, NeurIPS 2023).** *"Direct Preference Optimization (DPO)."* [`arXiv:2305.18290`](https://arxiv.org/abs/2305.18290) (`src/module_06_alignment/`).
26. **Meng et al. (Princeton & UVA, NeurIPS 2024).** *"SimPO: Simple Preference Optimization with a Reference-Free Reward."* [`arXiv:2405.14734`](https://arxiv.org/abs/2405.14734) (`src/module_06_alignment/`).
27. **DeepSeek-AI (2025).** *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (GRPO)."* [`arXiv:2501.12948`](https://arxiv.org/abs/2501.12948) (`src/module_11_rlvr_grpo/`).
28. **Lightman et al. (OpenAI, ICLR 2024).** *"Let's Verify Step by Step (Process Reward Models)."* [`arXiv:2305.20050`](https://arxiv.org/abs/2305.20050) (`src/module_24_test_time_prm/`).
29. **Snell et al. (UC Berkeley, 2024).** *"Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters."* [`arXiv:2408.03314`](https://arxiv.org/abs/2408.03314) (`src/module_24_test_time_prm/`).
30. **Hao et al. (Meta FAIR, 2024).** *"Training Large Language Models to Reason in a Continuous Latent Space (Coconut)."* [`arXiv:2412.06769`](https://arxiv.org/abs/2412.06769) (`src/module_16_latent_reasoning/`).
31. **Gu & Dao (CMU & Princeton, COLM 2024).** *"Mamba: Linear-Time Sequence Modeling with Selective State Spaces."* [`arXiv:2312.00752`](https://arxiv.org/abs/2312.00752) (`src/module_12_hybrid_ssm/`).
32. **Dong et al. (NVIDIA Research, ICLR 2025).** *"Hymba: A Hybrid-head Architecture for Small Language Models."* [`arXiv:2411.13676`](https://arxiv.org/abs/2411.13676) (`src/module_12_hybrid_ssm/`).
33. **Raposo et al. (Google DeepMind, 2024).** *"Mixture-of-Depths: Dynamically allocating compute in transformer-based language models."* [`arXiv:2404.02258`](https://arxiv.org/abs/2404.02258) (`src/module_20_mixture_of_depths/`).
34. **Xiao et al. (MIT Han Lab, ICLR 2024).** *"Efficient Streaming Language Models with Attention Sinks (StreamingLLM)."* [`arXiv:2309.17453`](https://arxiv.org/abs/2309.17453) (`src/module_22_kv_cache_eviction/`).
35. **Zhang et al. (NeurIPS 2023).** *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models."* [`arXiv:2306.14048`](https://arxiv.org/abs/2306.14048) (`src/module_22_kv_cache_eviction/`).
36. **Liu et al. (DeepSeek-AI, 2024).** *"DeepSeek-V2: Multi-Head Latent Attention (MLA)."* [`arXiv:2405.04434`](https://arxiv.org/abs/2405.04434) (`src/module_27_mla_latent_attention/`).
37. **Pagnoni et al. (Meta FAIR, 2024).** *"Byte Latent Transformer: Patches Scale Better Than Tokens (BLT)."* [`arXiv:2412.09871`](https://arxiv.org/abs/2412.09871) (`src/module_28_byte_latent_patching/`).
38. **Ye et al. (Microsoft Research, ICLR 2025).** *"Differential Transformer (DiffAttn)."* [`arXiv:2410.05258`](https://arxiv.org/abs/2410.05258) (`src/module_30_differential_attention/`).
39. **Zou et al. (Center for AI Safety / CMU, 2023).** *"Representation Engineering: A Top-Down Approach to AI Transparency (RepE)."* [`arXiv:2310.01405`](https://arxiv.org/abs/2310.01405) (`src/module_23_representation_steering/`).
40. **Gao et al. (OpenAI, 2024).** *"Scaling and evaluating sparse autoencoders (Top-K SAE)."* [`arXiv:2406.04093`](https://arxiv.org/abs/2406.04093) (`src/module_26_sparse_autoencoders/`).
41. **Zhang et al. (COLM 2024).** *"Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning (NPO)."* [`arXiv:2404.05868`](https://arxiv.org/abs/2404.05868) (`src/module_29_machine_unlearning/`).
42. **Li et al. (ICML 2024).** *"The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning (RMU)."* [`arXiv:2403.03218`](https://arxiv.org/abs/2403.03218) (`src/module_29_machine_unlearning/`).
43. **Kirchenbauer et al. (ICML 2023).** *"A Watermark for Large Language Models (KGW)."* [`arXiv:2301.10226`](https://arxiv.org/abs/2301.10226) (`src/module_32_statistical_watermarking/`).
44. **Belcák et al. (NVIDIA Research, 2025).** *"Small Language Models are the Future of Agentic AI."* [`arXiv:2506.02153`](https://arxiv.org/abs/2506.02153) (`src/module_08_routing/`, `src/module_13_agentic_swarms/`).
45. **Ong et al. (UC Berkeley / LMSYS, ICLR 2025).** *"RouteLLM: Learning to Route LLMs with Preference Data."* [`arXiv:2406.18665`](https://arxiv.org/abs/2406.18665) (`src/module_08_routing/`).
46. **Narayan et al. (Stanford Hazy Research, 2025).** *"Minions: Cost-efficient Collaboration Between On-device and Cloud Language Models."* [`arXiv:2502.15964`](https://arxiv.org/abs/2502.15964) (`src/module_08_routing/`).
