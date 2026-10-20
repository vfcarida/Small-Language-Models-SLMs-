# Module 27: Multi-Head Latent Attention (MLA) & Low-Rank KV Joint Compression

> **Module Path:** `src/module_27_mla_latent_attention/mla_attention.py`  
> **Primary References:**  
> - Liu et al. (DeepSeek-AI, 2024), *"DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"*, `arXiv:2405.04434`.  
> - DeepSeek-AI (2024), *"DeepSeek-V3 Technical Report"*, `arXiv:2412.19437`.

---

## 1. Joint Low-Rank KV Compression vs. GQA

While Grouped-Query Attention (GQA) reduces KV-cache memory by forcing multiple query heads to share identical key/value projections, it lowers the rank of the per-layer attention kernel. **Multi-Head Latent Attention (`MultiHeadLatentAttention`)** projects hidden state $\mathbf{h}_t \in \mathbb{R}^d$ into a single shared low-dimensional latent vector $\mathbf{c}_t^{KV} \in \mathbb{R}^{d_c}$ ($d_c \ll n_h d_h$):

$$\mathbf{c}_t^{KV} = \mathbf{W}_{DKV} \mathbf{h}_t, \quad \mathbf{k}_{t, i}^C = \mathbf{W}_{UK, i} \mathbf{c}_t^{KV}, \quad \mathbf{v}_{t, i}^C = \mathbf{W}_{UV, i} \mathbf{c}_t^{KV}$$

---

## 2. Weight Absorption & Decoupled RoPE (`forward_absorbed`)

Because Rotary Position Embedding ($\text{RoPE}$) is position-dependent, placing $\text{RoPE}$ directly on $\mathbf{k}_{t, i}^C$ would prevent folding $\mathbf{W}_{UK, i}$ into $\mathbf{W}_{UQ, i}$. MLA resolves this via **Decoupled RoPE**:
- A small shared positional key $\mathbf{k}_t^R = \text{RoPE}(\mathbf{W}_{KR} \mathbf{h}_t) \in \mathbb{R}^{d_r}$ carries relative position.
- Content keys remain linear functions of $\mathbf{c}_s^{KV}$, allowing exact **matrix absorption**:

$$(\mathbf{q}_{t, i}^C)^\top \mathbf{k}_{s, i}^C = (\mathbf{q}_{t, i}^C)^\top (\mathbf{W}_{UK, i} \mathbf{c}_s^{KV}) = (\mathbf{W}_{UK, i}^\top \mathbf{q}_{t, i}^C)^\top \mathbf{c}_s^{KV}$$

Consequently, only $(\mathbf{c}_t^{KV}, \mathbf{k}_t^R)$ of total width $d_c + d_r$ is stored in the KV cache, achieving $6\times$ to $12\times$ memory compression over MHA with **zero numerical approximation error**.
