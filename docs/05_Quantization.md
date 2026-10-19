# Module 05: Post-Training Quantization — AWQ, SpinQuant & BitsAndBytes NF4

## 1. Weight-Only vs. Weight-Activation Quantization

Quantization maps high-precision floating-point tensors (`FP32`/`BF16`) into compact integer or quantile representations (`INT8`/`INT4`/`NF4`), slashing memory footprint by up to $4\times$ and relieving memory-bandwidth bottlenecks during autoregressive decoding.

### Analytical Memory Footprint with Double Quantization

In `src/module_05_quantization/bitsandbytes_quant.py` (`estimate_quantized_memory_mb`), a 4-bit blockwise quantized tensor with group size $G = 64$ incurs:
- **Single Quantization (FP32 scales):** $4 + \frac{32}{64} = 4.50\text{ bits/parameter}$.
- **Double Quantization (FP8 scales with second-level group size 256, Dettmers et al., 2023):**
  $$b_{\text{eff}} = 4 + \frac{8}{64} + \frac{32}{64 \times 256} \approx 4.127\text{ bits/parameter}$$

---

## 2. Activation-Aware Weight Quantization (AWQ — Lin et al., MLSys 2024 Best Paper)

Naive Round-To-Nearest (RTN) INT4 quantization degrades SLM reasoning because ~1% of weight channels process large-magnitude activation outliers (`arXiv:2306.00978`).

### Mathematical Formulation (`compute_awq_scales`)

Instead of mixed-precision hardware branching, AWQ protects salient channels via an **equivalent per-channel scaling transformation**. For weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, input activation $X$, and positive per-channel scale vector $s \in \mathbb{R}_{>0}^{d_{\text{in}}}$:

$$W X = \left(W \cdot \text{diag}(s)\right) \left(\text{diag}(s)^{-1} X\right)$$

Multiplying salient weight columns by $s_j > 1$ prior to group-wise INT4 quantization $Q(\cdot)$ reduces their relative quantization error $\Delta_j / s_j$. AWQ parameterizes $s$ solely by the average activation magnitude $s_X = \text{mean}_t(|X_{t,:}|)$ via a single hyperparameter $\alpha \in [0, 1]$:

$$s(\alpha) = s_X^\alpha, \quad \alpha^* = \arg\min_{\alpha \in [0, 1]} \left\| Q\left(W \cdot \text{diag}(s(\alpha))\right) \text{diag}(s(\alpha))^{-1} X - W X \right\|_F^2$$

A fast 20-step grid search over $\alpha \in [0, 1]$ finds the optimal trade-off between protecting salient channels and preserving non-salient channels—requiring **zero gradient backpropagation** and never overfitting the calibration set.

---

## 3. Orthogonal Rotation Smoothing (SpinQuant — Liu et al., ICLR 2025)

For extreme low-bit quantization of both weights and activations/KV-cache (`arXiv:2405.16406`), `apply_hadamard_rotation_smoothing` demonstrates orthogonal rotation invariance using a normalized Hadamard matrix $H \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ satisfying $H^\top H = I$:

$$(X H)(W H)^\top = X (H H^\top) W^\top = X W^\top$$

Rotating activations and weights disperses concentrated outlier spikes uniformly across all feature dimensions prior to quantization while preserving exact linear layer outputs.

---

## 4. Implementation Reference

- **AWQ Grid Search & Hadamard Smoothing:** [`src/module_05_quantization/awq_quantization.py`](../src/module_05_quantization/awq_quantization.py)
- **BitsAndBytes Dynamic INT8/NF4 & Memory Calculator:** [`src/module_05_quantization/bitsandbytes_quant.py`](../src/module_05_quantization/bitsandbytes_quant.py)
- **Unit Verification:** [`tests/test_quantization.py`](../tests/test_quantization.py)
