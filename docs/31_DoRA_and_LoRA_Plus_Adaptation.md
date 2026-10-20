# Module 31: Weight-Decomposed Low-Rank Adaptation (`DoRA`) & Asymmetric `LoRA+`

> **Module Path:** `src/module_31_dora_lora_plus/dora_adapter.py`  
> **Primary References:**  
> - Liu et al. (NVIDIA Research / ICML 2024 Oral), *"DoRA: Weight-Decomposed Low-Rank Adaptation"*, `arXiv:2402.09353`.  
> - Hayou et al. (ICML 2024), *"LoRA+: Efficient Low Rank Adaptation of Large Models"*, `arXiv:2402.12354`.

---

## 1. Weight-Decomposed Low-Rank Adaptation (`DoRALinear`)

Full fine-tuning updates weight magnitude and direction with negative correlation, whereas standard LoRA couples both updates proportionally. **DoRA (`DoRALinear`)** decomposes the pre-trained matrix $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ into a trainable row-wise magnitude vector $\mathbf{m} \in \mathbb{R}^{d_{\text{out}}}$ and a low-rank directional matrix:

$$\mathbf{W}_{\text{DoRA}} = \text{diag}(\mathbf{m}) \frac{\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}}{\left\|\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{B}\mathbf{A}\right\|_{\text{row}}}$$

After training, `merge_into_linear()` folds $\mathbf{W}_{\text{DoRA}}$ into a standard `nn.Linear` layer with **zero inference latency overhead**.

---

## 2. Asymmetric Learning Rates via `LoRA+` (`build_lora_plus_param_groups`)

By infinite-width feature learning theory ($\mu\text{P}$), setting equal learning rates $\eta_A = \eta_B$ causes slow feature learning on $\mathbf{B}$. `build_lora_plus_param_groups` assigns $\eta_B = \lambda_{\text{ratio}} \cdot \eta_A$ ($\lambda_{\text{ratio}} \in [8, 16]$), accelerating SFT convergence by $1.5\times$–$2\times$.
