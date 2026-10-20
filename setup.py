# ==============================================================================
# SETUP.PY — Package Installation Manifest for SLM Masterclass
# ==============================================================================
# Enables installing the repository as an editable Python package:
#   pip install -e .
#
# Synchronized with pyproject.toml (PEP 621) to ensure all internal imports
# (from src.utils import ...) and CLI entry points resolve across environments.
# ==============================================================================

from setuptools import find_packages, setup

setup(
    name="slm-masterclass",
    version="2.0.0",
    author="Vinicius Caridá",
    description=(
        "Production-grade reference toolkit and engineering playbook for training, "
        "distilling, pruning, quantizing, aligning, and routing Small Language Models (SLMs)."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vfcarida/Small-Language-Models-SLMs-",
    packages=find_packages(include=["src*", "pipeline*", "eval*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "accelerate>=0.32.0",
        "peft>=0.12.0",
        "trl>=0.9.6",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "safetensors>=0.4.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "quantization": [
            "bitsandbytes>=0.43.0",
            "autoawq>=0.2.6",
            "auto-gptq>=0.7.0",
        ],
        "serving": [
            "vllm>=0.5.0",
        ],
        "eval": [
            "lm-eval>=0.4.3",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "ruff>=0.5.0",
        ],
        "all": [
            "bitsandbytes>=0.43.0",
            "autoawq>=0.2.6",
            "auto-gptq>=0.7.0",
            "vllm>=0.5.0",
            "lm-eval>=0.4.3",
            "pytest>=8.0",
            "pytest-cov>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "slm-pipeline=pipeline.run_full_pipeline:main",
            "slm-eval=src.module_07_evaluation.benchmark_runner:main",
            "slm-pareto=eval.benchmarks.pareto_suite:main",
            "slm-report=pipeline.generate_report:main",
            "slm-route=src.module_08_routing.hybrid_router:main",
            "slm-calibrate=src.module_08_routing.router_calibration:main",
            "slm-export=src.module_09_serving.edge_exporter:main",
            "slm-serve=src.module_09_serving.openai_server:main",
            "slm-cayley=src.module_05_quantization.cayley_kv_quant:main",
            "slm-speculative=src.module_10_speculative.speculative_decoding:main",
            "slm-grpo=src.module_11_rlvr_grpo.grpo_trainer:main",
            "slm-hybrid-ssm=src.module_12_hybrid_ssm.hybrid_ssm_block:main",
            "slm-swarm=src.module_13_agentic_swarms.slm_swarm:main",
            "slm-dp-lora=src.module_14_federated_dp_lora.dp_federated_lora:main",
            "slm-bitnet=src.module_15_bitnet_ternary.bitnet_linear:main",
            "slm-coconut=src.module_16_latent_reasoning.coconut_reasoning:main",
            "slm-lamp=src.module_17_mixed_precision.lamp_allocator:main",
            "slm-fsm-decode=src.module_18_constrained_decoding.fsm_guided_decoding:main",
            "slm-cross-kd=src.module_19_cross_tokenizer_kd.optimal_transport_kd:main",
            "slm-mod=src.module_20_mixture_of_depths.mod_routing:main",
            "slm-merge=src.module_21_model_merging.task_arithmetic_merging:main",
            "slm-kv-evict=src.module_22_kv_cache_eviction.kv_eviction_policy:main",
            "slm-steer=src.module_23_representation_steering.activation_steering:main",
            "slm-prm=src.module_24_test_time_prm.prm_tree_search:main",
            "slm-matryoshka=src.module_25_matryoshka_minitron.elastic_matryoshka:main",
            "slm-sae=src.module_26_sparse_autoencoders.topk_sae:main",
            "slm-mla=src.module_27_mla_latent_attention.mla_attention:main",
            "slm-blt=src.module_28_byte_latent_patching.byte_latent_patcher:main",
            "slm-unlearn=src.module_29_machine_unlearning.unlearning_objectives:main",
            "slm-diff-attn=src.module_30_differential_attention.diff_attention:main",
            "slm-dora=src.module_31_dora_lora_plus.dora_adapter:main",
            "slm-watermark=src.module_32_statistical_watermarking.kgw_watermark:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
)
