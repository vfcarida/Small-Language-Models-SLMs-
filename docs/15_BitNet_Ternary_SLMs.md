# Module 15: Ternary `1.58-Bit` Native SLMs (`BitNet b1.58` & Add-Only Kernels)

[![Back to Hub](https://img.shields.io/badge/⬅️_Back_to-Documentation_Hub-blue?style=for-the-badge)](../README.md)
[![Code Module](https://img.shields.io/badge/💻_Source_Code-src/module__15__bitnet__ternary-green?style=for-the-badge)](../src/module_15_bitnet_ternary/)

---

## 1. Executive & Architectural Motivation

Post-Training Quantization (PTQ, Modules 05 & 05B) compresses pretrained `FP16`/`BF16` models down to `INT4` or `INT3`, but still requires dequantization and floating-point Multiply-Accumulate (MAC) instructions during execution unless specialized lookup kernels are used. **BitNet b1.58** (**Ma et al., Microsoft Research 2024**, `arXiv:2402.17764`; **BitNet b1.58 2B4T, 2025**, `arXiv:2504.12285`) introduces a paradigm shift for Small Language Models: training native ternary weights $\widetilde{\mathbf{W}} \in \{-1, 0, +1\}^{d_{\text{out}} \times d_{\text{in}}}$ from scratch or via Quantization-Aware Training (QAT).

Because each weight takes one of 3 discrete states ($\log_2(3) \approx 1.585\text{ bits}$), matrix multiplication $\mathbf{Y} = \mathbf{X} \widetilde{\mathbf{W}}^\top$ **eliminates floating-point multiplications entirely**, replacing them with pure integer additions ($+x_j$), subtractions ($-x_j$), and zero-skips ($0$). On 7nm/5nm silicon, an `INT8` addition consumes **~0.03 pJ** compared to **~1.50 pJ** for an `FP16` MAC (**50x arithmetic energy reduction**) while shrinking a 1.54B SLM to **291 MB**.

---

## 2. Mathematical Foundations

### 2.1 `absmean` Ternary Weight Quantization ($\{-1, 0, +1\}$)

Given a full-precision master weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, the scaling factor $\gamma$ is the global mean absolute value:

$$
\gamma = \frac{1}{d_{\text{out}} d_{\text{in}}} \sum_{i=1}^{d_{\text{out}}} \sum_{j=1}^{d_{\text{in}}} |W_{i,j}|
$$

Each weight is rounded to the nearest integer and clipped to $\{-1, 0, +1\}$:

$$
\widetilde{\mathbf{W}} = \text{RoundClip}\!\left(\frac{\mathbf{W}}{\gamma + \epsilon}, -1, +1\right), \qquad \text{RoundClip}(z, a, b) = \max(a, \min(b, \lfloor z \rceil))
$$

### 2.2 Per-Token `absmax` `INT8` Activation Quantization & Straight-Through Estimator (STE)

After applying parameter-free `RMSNorm` to input activations $\mathbf{x}_t \in \mathbb{R}^{d_{\text{in}}}$, each token is scaled dynamically into $[-Q_b, +Q_b]$ where $Q_b = 2^{b-1} - 1 = 127$ for $b = 8$:

$$
\eta_t = \max_{j} |x_{t, j}|, \qquad \widetilde{\mathbf{x}}_t = \text{RoundClip}\!\left(\frac{Q_b}{\eta_t + \epsilon} \mathbf{x}_t, -Q_b, +Q_b\right)
$$

Because $\lfloor \cdot \rceil$ has zero derivative almost everywhere, **Quantization-Aware Training (QAT)** routes gradients through the identity Jacobian via the **Straight-Through Estimator (STE)**:

$$
\mathbf{W}_{\text{STE}} = \mathbf{W} + \text{sg}\!\left(\gamma \widetilde{\mathbf{W}} - \mathbf{W}\right) \implies \frac{\partial \mathbf{W}_{\text{STE}}}{\partial \mathbf{W}} = \mathbf{I}
$$

### 2.3 Multiplier-Free Add-Only Inner Product Equivalence

For any output channel $i$, the inner product decomposes into two disjoint index sets $\mathcal{P}_i = \{j : \widetilde{W}_{i,j} = +1\}$ and $\mathcal{N}_i = \{j : \widetilde{W}_{i,j} = -1\}$:

$$
y_{t, i} = \left(\frac{\gamma \eta_t}{Q_b}\right) \left( \sum_{j \in \mathcal{P}_i} \widetilde{x}_{t, j} - \sum_{j \in \mathcal{N}_i} \widetilde{x}_{t, j} \right)
$$

---

## 3. Reference Implementation

- **[`src/module_15_bitnet_ternary/bitnet_linear.py`](../src/module_15_bitnet_ternary/bitnet_linear.py)**:
  - `quantize_weights_absmean_ternary` & `quantize_activations_absmax_int8`.
  - `BitLinear`: Differentiable QAT layer with `RMSNorm`, STE backward pass, and `forward_add_only` verifying exact numerical equivalence with floating-point execution.
  - `profile_bitnet_hardware_savings`: Analytical memory and 7nm ALU energy profiler.

```bash
python src/module_15_bitnet_ternary/bitnet_linear.py
```

---

## 4. Academic References

1. **Ma, S., Wang, H., Ma, L., et al.** (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits (BitNet b1.58).* Microsoft Research (`arXiv:2402.17764`).
2. **Ma, S., Wang, H., et al.** (2025). *BitNet b1.58 2B4T Technical Report.* Microsoft Research (`arXiv:2504.12285`).
