# Module 01: Autoregressive Inference, Decoding Strategies & Latency Telemetry

## 1. Precision Formats & Memory Bandwidth

Autoregressive text generation is fundamentally **memory-bandwidth bound** during the token-by-token decoding phase. Selecting the appropriate numerical representation directly governs both VRAM footprint and token throughput:

- **`float32` (FP32 — 4 bytes/param):** Full IEEE-754 single precision; a 1.5B parameter model requires ~6.0 GB for weights alone.
- **`float16` (FP16 — 2 bytes/param):** 5-bit exponent, 10-bit mantissa; halves memory to ~3.0 GB but risks overflow/underflow in deep residual streams.
- **`bfloat16` (BF16 — 2 bytes/param):** Preserves the full 8-bit dynamic exponent range of FP32 while halving storage to 2 bytes per parameter, preventing `NaN` instabilities on modern Ampere/Hopper/Ada GPUs.

---

## 2. Decoding Algorithms & Mathematical Formulations

At generation step $t$, the language model maps context $x_{<t}$ to raw vocabulary logits $z \in \mathbb{R}^{|V|}$.

### 1. Greedy Decoding
Deterministically selects the maximum-likelihood token:
$$x_t = \arg\max_{v \in V} P(v \mid x_{<t})$$
Ideal for deterministic structured extraction, classification, and unit-tested routing, though susceptible to repetitive loops in open-ended generation.

### 2. Temperature-Scaled Multinomial Sampling
Applies temperature $T > 0$ prior to the softmax normalization:
$$P_T(v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{w \in V} \exp(z_w / T)}$$
As $T \to 0^+$, $P_T$ approaches greedy argmax; for $T \in (0.3, 0.8)$, low-probability tail noise is suppressed.

### 3. Top-K Sampling
Truncates the sampling support to the $K$ highest-logit vocabulary items $V^{(K)} \subset V$ before renormalizing probabilities.

### 4. Nucleus (Top-P) Sampling (Holtzman et al., ICLR 2020)
Dynamically selects the smallest candidate set $V^{(p)} \subset V$ whose cumulative probability mass exceeds threshold $p \in (0, 1)$:
$$\sum_{v \in V^{(p)}} P_T(v \mid x_{<t}) \ge p$$
Adapts candidate pool width automatically: narrowing to 1–2 tokens during high-certainty syntax transitions and expanding during open-ended semantic branching.

---

## 3. Production Operational Telemetry

1. **Time-To-First-Token (TTFT, ms):** Latency elapsed during the parallel **prefill phase** (processing the input prompt and populating the initial KV-Cache) until the first output token is emitted.
2. **Sustained Decode Throughput (tokens/sec):** Rate of sequential token emission during the memory-bandwidth-bound **decode phase**, measured with explicit `torch.cuda.synchronize()` barriers and median aggregation over post-warmup runs.

---

## 4. Implementation Reference

- **Inference & Decoding Engine:** [`src/module_01_inference/basic_inference.py`](../src/module_01_inference/basic_inference.py)
- **Latency & VRAM Instrumentation:** [`src/utils/metrics.py`](../src/utils/metrics.py)
- **Unit Verification:** [`tests/test_inference.py`](../tests/test_inference.py)
