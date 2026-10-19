# Module 02: Supervised Fine-Tuning (SFT) with LoRA & 4-Bit QLoRA

## 1. Low-Rank Adaptation (LoRA — Hu et al., ICLR 2022)

Full fine-tuning of a pre-trained language model requires storing FP32 optimizer states (momentum and variance in AdamW) and gradients for all parameters $W_0 \in \mathbb{R}^{d \times k}$, multiplying memory consumption by $4\times$ to $6\times$ the base weight footprint.

**Low-Rank Adaptation (LoRA)** freezes the pre-trained weight matrix $W_0$ and constrains the parameter update $\Delta W$ to a low-rank decomposition:

$$W = W_0 + \frac{\alpha}{r} B A, \quad A \in \mathbb{R}^{r \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad r \ll \min(d, k)$$

- **Initialization:** $A \sim \mathcal{N}(0, \sigma^2)$ and $B = 0$, ensuring $\Delta W = 0$ at the start of training.
- **All-Linear Target Modules:** Following Dettmers et al. (NeurIPS 2023, `arXiv:2305.14314`), injecting LoRA adapters across **all linear layers** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, i.e., `target_modules="all-linear"`) recovers full fine-tuning quality while updating $<1.5\%$ of total parameters.
- **Zero Inference Latency Overhead:** Post-training, adapter weights are folded back into the base weights analytically via $W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$ (`merge_lora_adapter`).

---

## 2. Quantized LoRA (QLoRA — Dettmers et al., NeurIPS 2023)

QLoRA enables fine-tuning 1.5B–7B models comfortably within an **8 GB VRAM** consumer GPU budget through three synergistic innovations:

1. **4-Bit NormalFloat (`NF4`):** An information-theoretically optimal data type for zero-mean normally distributed neural network weights $\mathcal{N}(0, \sigma^2)$, spacing $2^4 = 16$ quantization bins so each bin contains an equal expected number of weights under the standard normal cumulative distribution function $\Phi^{-1}$.
2. **Double Quantization (`bnb_4bit_use_double_quant=True`):** Quantizes the FP32 blockwise quantization scales ($c_2^{\text{FP32}}$) into 8-bit floats ($c_2^{\text{FP8}}$) with group size 256, slashing quantization metadata overhead from **0.50 bits/param** down to **0.127 bits/param**.
3. **BF16 Compute Dequantization:** Base weights remain stored in 4-bit NF4 and are dynamically dequantized to `bfloat16` only during forward and backward matrix multiplications, while gradients flow exclusively through the `bfloat16` LoRA adapters.

---

## 3. Instruction Formatting & OOM Safeguards

- **Template Standardization:** `src/utils/data_utils.py` supports both structured Alpaca instruction-input-response templates (`format_alpaca_prompt`) and native tokenizer chat templates (`format_chat_prompt`).
- **Dynamic VRAM Reclamation:** Training runs are wrapped in `GPUOOMHandler` (`src/utils/metrics.py`), which intercepts CUDA Out-Of-Memory runtime exceptions, triggers Python garbage collection, flushes the PyTorch CUDA allocator cache (`torch.cuda.empty_cache()`), and surfaces actionable batch-size/gradient-checkpointing diagnostics.

---

## 4. Implementation Reference

- **LoRA SFT Pipeline & Adapter Merger:** [`src/module_02_sft/sft_lora.py`](../src/module_02_sft/sft_lora.py)
- **4-Bit QLoRA SFT Pipeline:** [`src/module_02_sft/sft_qlora.py`](../src/module_02_sft/sft_qlora.py)
- **Dataset Preparation Utilities:** [`src/utils/data_utils.py`](../src/utils/data_utils.py)
- **Unit Verification:** [`tests/test_sft.py`](../tests/test_sft.py) & [`tests/test_data_utils.py`](../tests/test_data_utils.py)
