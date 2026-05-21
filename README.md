# 🚀 The Definitive Small Language Model (SLM) Masterclass

Welcome to the **Ultimate Step-by-Step Guide** for Engineering, Training, Pruning, and Deploying Small Language Models (SLMs). 

While Large Language Models (LLMs) like GPT-4 or Claude 3 serve as "generalist consultants", their immense size makes them costly, slow, and hard to deploy in edge or privacy-strict environments. **SLMs (sub-3B parameters)** such as Qwen2.5-1.5B, Llama-3.2-1B, and Phi-3.5-mini are emerging as the "fast specialists". They beat LLMs on cost-per-query, latency, and operational control for specific, high-volume tasks.

This repository is a deeply researched, **hands-on masterclass** providing theoretical foundations and line-by-line code implementations to build state-of-the-art SLM pipelines.

---

## 📚 Masterclass Index

This guide is split into **Theoretical Tutorials** and **Practical Implementations**.

### Part 1: Theoretical Tutorials (`/docs`)
1. [**Introduction & Top 10 Use Cases**](docs/01_Introduction_and_Use_Cases.md): Why SLMs beat LLMs on cost, and where to use them (Intent Classification, Routing, Edge Personalization).
2. [**Generalized Knowledge Distillation (GKD)**](docs/02_Knowledge_Distillation_Deep_Dive.md): The math behind KL vs. Jensen-Shannon divergences, and how to use massive LLMs (via vLLM) as teachers without blowing up your VRAM.
3. [**Parametric Pruning (Wanda)**](docs/03_Parametric_Pruning_Wanda.md): Unstructured pruning using Activation norms and Zipf sampling to preserve syntax outliers.
4. [**Activation-Aware Weight Quantization (AWQ)**](docs/04_AWQ_Quantization_and_Edge_Deployment.md): Why INT4 quantization using AWQ preserves 99% of accuracy for Edge deployment compared to SmoothQuant.

### Part 2: Hands-On Implementations (`/src` & `/eval`)
All code in this repository is annotated **line-by-line** to explain the "why" alongside the "how".
- 🔬 **Distillation**: [`src/distillation/gkd_pipeline.py`](src/distillation/gkd_pipeline.py)
- ✂️ **Pruning**: [`src/pruning/wanda_heuristics.py`](src/pruning/wanda_heuristics.py)
- 🗜️ **Quantization**: [`src/quantization/autoawq_cli.py`](src/quantization/autoawq_cli.py)
- 📊 **Benchmarking vs LLMs**: [`eval/benchmarks/latency_tracker.py`](eval/benchmarks/latency_tracker.py)

---

## 🛠️ Quick Start (Dockerized Environment)

We provide a production-ready `Dockerfile` to ensure `flash-attention`, `vllm`, and `autoawq` compile flawlessly on CUDA devices.

```bash
# 1. Build the environment
docker build -t slm-masterclass .

# 2. Run the interactive container with GPU support
docker run --gpus all -it --rm -v $(pwd):/workspace slm-masterclass

# 3. Inside the container, run a benchmark comparing an SLM vs LLM
python src/eval/benchmarks/latency_tracker.py
```

---

## 📖 Literature Review & References

This masterclass was built upon a deep survey of the current state-of-the-art in Small Language Models. We highly recommend reviewing the foundational texts:

1. **"10 SLM Use Cases That Beat LLMs on Cost" (Modexa, 2025):** Advocates for a "Small First" architecture. SLMs are perfect for Intent Classification, FAQ micro-QA, structured data extraction, and tool orchestration. [Medium](https://medium.com/@Modexa/10-slm-use-cases-that-beat-llms-on-cost-7e2fa0acd361)
2. **Hugging Face SLM Overview:** A comprehensive look at the ecosystem, highlighting how models like FLAN-T5, Llama-3.2-1B, and DistilGPT2 fit into modern NLP architectures. [HuggingFace Blog](https://huggingface.co/blog/jjokah/small-language-model)
3. **Awesome Small Language Models:** A curated list of tools (Peft, bitsandbytes) and frameworks vital for SLM optimization. [GitHub/slashml](https://github.com/slashml/awesome-small-language-models)
4. **SLMs Survey:** Extensive surveys on parameter-efficient fine-tuning (PEFT), pruning heuristics (like Wanda), and quantization limits. [GitHub/FairyFali/SLMs-Survey](https://github.com/FairyFali/SLMs-Survey)

*Authored by Antigravity under the SLM Systems Engineering Initiative.*
