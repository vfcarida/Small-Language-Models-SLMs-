# Module 32: Statistical Watermarking (`KGW` Green-List Logit Bias) for AI TRiSM

> **Module Path:** `src/module_32_statistical_watermarking/kgw_watermark.py`  
> **Primary Reference:**  
> - Kirchenbauer et al. (ICML 2023), *"A Watermark for Large Language Models"*, `arXiv:2301.10226`.

---

## 1. Cryptographic Green-List Partitioning & Soft Logit Bias

To comply with **EU AI Act Article 50** and **Gartner AI TRiSM** provenance requirements, `apply_kgw_watermark_logits` hashes the preceding token $x_{t-1}$ with a secret salt via SHA-256 to partition the vocabulary $V$ into a **Green List** $G_t$ of size $\lfloor \gamma |V| \rfloor$ and adds a soft bias $\delta > 0$:

$$\tilde{\ell}_{t, v} = \ell_{t, v} + \delta \cdot \mathbb{I}[v \in G(x_{t-1})]$$

---

## 2. Zero-Knowledge Binomial $z$-Score Verification (`detect_kgw_watermark_zscore`)

Without accessing the SLM weights or original prompt, the verifier counts green-list hits $|S|_G$ across $T$ token transitions and evaluates the one-sided normal $z$-statistic:

$$z = \frac{|S|_G - \gamma T}{\sqrt{T \gamma (1 - \gamma)}}, \quad p = \frac{1}{2}\text{erfc}\!\left(\frac{z}{\sqrt{2}}\right)$$

At $z \ge 4.0$, the false-positive rate on human text is below $3.2 \times 10^{-5}$.
