# Module 28: Byte-Latent & Dynamic Patch Tokenization (BLT) for Tokenizer-Free SLMs

> **Module Path:** `src/module_28_byte_latent_patching/byte_latent_patcher.py`  
> **Primary Reference:**  
> - Pagnoni et al. (Meta FAIR, 2024), *"Byte Latent Transformer: Patches Scale Better Than Tokens"*, `arXiv:2412.09871`.

---

## 1. Information-Density Driven Dynamic Byte Patching

Heuristic subword tokenizers (BPE/WordPiece) freeze a static lookup vocabulary before training. **Byte-Latent Transformer (`BLT`)** operates directly on raw UTF-8 bytes $b_t \in \{0, \dots, 255\}$ and groups predictable continuation bytes into variable-length patches using next-byte Shannon entropy:

$$H(B_t \mid b_{<t}) = -\sum_{v=0}^{255} p_t(v) \log_2 p_t(v) \in [0, 8]\text{ bits}$$

`segment_bytes_into_dynamic_patches` starts a new latent patch at byte $t$ whenever:
1. **Global Threshold:** $H_t > \theta_{\text{global}}$ (high uncertainty at word/symbol boundaries), or
2. **Monotonic Spike:** $H_t - H_{t-1} > \theta_{\text{delta}}$ (sudden branching transition), or
3. **Max Span Cap:** Current patch length reaches `max_patch_bytes`.

---

## 2. Cross-Attention Byte-to-Patch Pooling (`ByteLatentPatchPooling`)

Each variable-length byte span $[s_k, e_k)$ is pooled via a learned query cross-attention head plus residual mean-pooling into a single $\mathbb{R}^{d_{\text{model}}}$ vector fed into the heavy Transformer backbone.
