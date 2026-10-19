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
    version="1.1.0",
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
            "slm-report=pipeline.generate_report:main",
            "slm-route=src.module_08_routing.hybrid_router:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
)
