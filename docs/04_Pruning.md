# Module 04: One-Shot Structural & Weight-Activation Pruning (Magnitude & Wanda)

## 1. Taxonomy of Neural Network Sparsity

Model pruning removes redundant synaptic weights to reduce storage footprint and accelerate matrix multiplications:

1. **Unstructured Sparsity:** Zeroes individual weights across arbitrary coordinates. Achieves high compression ratios for storage/transmission without accuracy loss at 50% sparsity, though standard dense GPU kernels cannot skip arbitrary zeros.
2. **NVIDIA 2:4 Semi-Structured Sparsity:** Enforces a strict hardware constraint where **at least 2 out of every 4 contiguous weights** along the input dimension are zero. Supported natively by NVIDIA Ampere, Ada Lovelace, and Hopper Sparse Tensor Cores for up to **2x GEMM math acceleration**.
3. **Structured Pruning:** Removes entire attention heads, MLP channels, or transformer layers, accelerating any CPU/GPU without specialized sparse kernels at the cost of requiring post-pruning recovery fine-tuning.

---

## 2. Why Magnitude Pruning Fails on Transformers: Emergent Outliers

Classical Magnitude Pruning (Han et al., NeurIPS 2015) ranks weights solely by $S_{ij} = |W_{ij}|$. However, modern Transformers develop **emergent large-magnitude activation features** where specific hidden channels exhibit activation norms $10\times$ to $100\times$ larger than typical channels. Removing a numerically small weight $W_{ij}$ that multiplies a massive outlier activation $X_j$ destroys critical reasoning pathways.

---

## 3. State of the Art: Wanda (Sun et al., ICLR 2024, `arXiv:2306.11695`)

**Wanda (Pruning by Weights and Activations)** resolves the outlier failure mode in one shot—requiring **zero weight updates, zero gradient backpropagation, and zero Hessian inversions** (running up to $300\times$ faster than SparseGPT, Frantar & Alistarh, ICML 2023, `arXiv:2301.00774`).

### Mathematical Formulation & Per-Output Comparison Group

Given a linear layer weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and calibration input activations $X \in \mathbb{R}^{(N \cdot L) \times d_{\text{in}}}$ accumulated over $N$ calibration sequences of length $L$:

1. **Input Feature L2 Norm:**
   $$\|X_j\|_2 = \sqrt{\sum_{t=1}^{N \cdot L} X_{t, j}^2}, \quad j \in \{1, \dots, d_{\text{in}}\}$$
2. **Wanda Importance Score:**
   $$S_{ij} = |W_{ij}| \cdot \|X_j\|_2$$
3. **Per-Output Row Pruning (`dim=1`):**
   Crucially, Wanda compares and prunes weights **within each output row $i$ independently** (`dim=1`), rather than flattening the entire layer matrix. This guarantees that every output feature retains an identical fraction of active incoming connections.
4. **Zipf-Diversified Calibration (`rank_sequences_by_zipf_diversity`):**
   Natural language token frequencies follow a heavy-tailed Zipfian distribution. Selecting calibration windows with maximal token-frequency Shannon entropy prevents pruning rare syntactic and logical operators (`if`, `not`, `forall`).

---

## 4. Implementation Reference

- **Magnitude Pruning (`unstructured` & `2:4`):** [`src/module_04_pruning/magnitude_pruning.py`](../src/module_04_pruning/magnitude_pruning.py)
- **Wanda Pruning (`unstructured` per-row & `2:4`):** [`src/module_04_pruning/wanda_pruning.py`](../src/module_04_pruning/wanda_pruning.py)
- **Unit Verification:** [`tests/test_pruning.py`](../tests/test_pruning.py)
