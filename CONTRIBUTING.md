# Contributing to Small Language Models (SLMs) Masterclass

Thank you for your interest in contributing to the **Small Language Models (SLMs) Masterclass** repository, designed and architected by **Vinicius Caridá (`vfcarida`)**.

---

## 1. Engineering Philosophy & Design Invariants

Every module in this repository adheres to four non-negotiable engineering standards:

1. **Glass-Box Mathematical Rigor:** Core algorithms (Attention, GKD/MiniLLM Distillation, Wanda Unstructured & 2:4 Pruning, AWQ Salient Scaling, DPO/SimPO Preference Optimization, and Calibrated Hybrid Routing) must expose pure-PyTorch mathematical functions that can be inspected, understood, and unit-tested deterministically on CPU in milliseconds.
2. **Core vs. Optional Dependency Isolation:** Hardware-bound or platform-specific packages (`autoawq`, `flash-attn`, `vllm`, `lm-eval`, `bitsandbytes`) must be guarded behind lazy or conditional imports so the core library and test suite run cleanly on any CPU or GPU machine.
3. **Deterministic Reproducibility:** All dataset splits, calibration loaders, and stochastic transformations must accept an explicit `seed: int = 42` parameter.
4. **100% International Technical English:** All source code, docstrings, inline comments, configuration files, and documentation must be written in clear, rigorous international English.

---

## 2. Local Development Setup

```bash
git clone https://github.com/vfcarida/Small-Language-Models-SLMs-.git
cd Small-Language-Models-SLMs-
pip install -e ".[dev]"
```

Run the automated test suite from the repository root:

```bash
pytest -v
```

---

## 3. Maturity Gate Requirements

Before proposing a feature graduation in `ROADMAP.md` (e.g., from `Beta` to `Stable`):
- Ensure 100% of unit and mathematical property tests pass (`pytest -v`).
- Provide docstring mathematical formulations and primary peer-reviewed citations (arXiv ID or venue).
- Verify that all internal links in `README.md`, `ROADMAP.md`, `ARCHITECTURE.md`, and `docs/` resolve cleanly.
- Use Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
