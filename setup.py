# ==============================================================================
# SETUP.PY — Instalação do Pacote SLM Masterclass
# ==============================================================================
# Este arquivo permite instalar o projeto como um pacote Python editável:
#   pip install -e .
#
# Isso garante que todos os imports internos (from src.utils import ...) funcionem
# corretamente, independentemente do diretório de execução.
# ==============================================================================

from setuptools import setup, find_packages

setup(
    name="slm-masterclass",
    version="1.0.0",
    author="Vinicius",
    description=(
        "Production-grade masterclass for training, compressing, and deploying "
        "Small Language Models (SLMs) — from fundamentals to edge deployment."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/vinicius/Small-Language-Models-SLMs-",
    
    # find_packages() descobre automaticamente todos os diretórios com __init__.py
    packages=find_packages(),
    
    python_requires=">=3.10",
    
    # Dependências mínimas para execução local (sem GPU)
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.43.0",
        "datasets>=2.20.0",
        "accelerate>=0.32.0",
        "peft>=0.12.0",
        "trl>=0.9.6",
        "bitsandbytes>=0.43.0",
        "matplotlib>=3.9.0",
        "seaborn>=0.13.2",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scipy>=1.13.0",
    ],
    
    # Dependências extras para funcionalidades avançadas
    extras_require={
        # pip install -e ".[quantization]"
        "quantization": [
            "autoawq>=0.2.6",
            "auto-gptq>=0.7.0",
        ],
        # pip install -e ".[serving]"
        "serving": [
            "vllm>=0.5.0",
        ],
        # pip install -e ".[eval]"
        "eval": [
            "lm-eval>=0.4.3",
        ],
        # pip install -e ".[dev]"
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
            "black>=24.0",
            "isort>=5.13",
            "flake8>=7.0",
        ],
        # pip install -e ".[all]"
        "all": [
            "autoawq>=0.2.6",
            "auto-gptq>=0.7.0",
            "vllm>=0.5.0",
            "lm-eval>=0.4.3",
            "flash-attn>=2.5.8",
            "wandb>=0.17.0",
            "pytest>=8.0",
        ],
    },
    
    # Entry points para scripts de linha de comando
    entry_points={
        "console_scripts": [
            # Permite rodar diretamente do terminal:
            #   slm-pipeline       → executa o pipeline completo
            #   slm-eval           → executa avaliação de benchmarks
            #   slm-report         → gera o relatório de resultados
            "slm-pipeline=pipeline.run_full_pipeline:main",
            "slm-eval=src.module_07_evaluation.benchmark_runner:main",
            "slm-report=pipeline.generate_report:main",
        ],
    },
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
    ],
)
