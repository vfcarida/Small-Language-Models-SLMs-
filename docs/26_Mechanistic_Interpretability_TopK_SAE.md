# Module 26: Mechanistic Interpretability via Top-$K$ Sparse Autoencoders (SAE)

> **Module Path:** `src/module_26_sparse_autoencoders/topk_sae.py`  
> **Primary References:**  
> - Gao et al. (OpenAI, 2024), *"Scaling and evaluating sparse autoencoders"*, `arXiv:2406.04093`.  
> - Templeton et al. (Anthropic, 2024), *"Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"*.

---

## 1. Superposition and Polysemanticity in SLMs

Because a Small Language Model must compress thousands of domain concepts into a narrow residual stream $\mathbf{x} \in \mathbb{R}^d$ (e.g., $d = 1024$ or $2048$), individual neurons exhibit **polysemanticity**: they fire on unrelated concepts packed in almost-orthogonal directions (*superposition*).

To audit an SLM under enterprise governance (**Gartner AI TRiSM / EU AI Act**), we decompose $\mathbf{x}$ into an overcomplete dictionary of $M \gg d$ **monosemantic features** using a **Top-$K$ Sparse Autoencoder (`TopKSparseAutoencoder`)**:

$$\bar{\mathbf{x}} = \mathbf{x} - \mathbf{b}_{\text{pre}}$$
$$\mathbf{a} = \text{ReLU}(\mathbf{W}_{\text{enc}} \bar{\mathbf{x}} + \mathbf{b}_{\text{enc}}) \in \mathbb{R}_{\ge 0}^M$$
$$\mathbf{z} = \text{TopK}(\mathbf{a}, k)$$
$$\hat{\mathbf{x}} = \mathbf{W}_{\text{dec}} \mathbf{z} + \mathbf{b}_{\text{pre}}, \quad \|\mathbf{w}_{\text{dec}, i}\|_2 = 1$$

Replacing the classic $L_1$ penalty with an exact $\text{TopK}(\cdot, k)$ activation gate directly enforces $\|\mathbf{z}\|_0 \le k$ per token while eliminating systematic *activation shrinkage*.

---

## 2. Preventing Dictionary Collapse via `AuxK` Dead-Latent Loss

In large overcomplete dictionaries, some latent atoms stop firing early in training (*dead latents*). `compute_sae_loss_with_auxk` tracks `steps_since_fired` for every feature and reconstructs the detached residual error $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$ using the top-$k_{\text{aux}}$ dead features:

$$\mathcal{L}_{\text{total}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \alpha_{\text{aux}} \|\mathbf{e} - \mathbf{W}_{\text{dec}} \text{TopK}_{\text{dead}}(\mathbf{a}, k_{\text{aux}})\|_2^2$$

---

## 3. Surgical Causal Feature Clamping (`clamp_sae_feature`)

Once a monosemantic feature $i$ is identified (e.g., a PII leakage feature or a domain-citation feature), `clamp_sae_feature` edits the residual stream while preserving the exact unmodeled SAE residual $\mathbf{e} = \mathbf{x} - \hat{\mathbf{x}}$:

$$\tilde{\mathbf{x}} = \mathbf{x} + (\alpha - z_i) \mathbf{w}_{\text{dec}, i}$$
