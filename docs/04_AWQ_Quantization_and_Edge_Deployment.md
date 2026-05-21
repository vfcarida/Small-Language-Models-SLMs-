# 4. Activation-Aware Weight Quantization (AWQ)

Quantization compresses the precision of the model's weights. Training happens in rich 16-bit floating point (`FP16/BF16`), but quantization crushes these down to 8-bit or 4-bit integers (`INT8/INT4`). 

For an SLM (1.5B - 3B parameters), aggressively crushing everything to 4-bits usually induces "numerical collapse"—the model starts hallucinating or losing logical coherence because it lacks the parameter volume to absorb the precision loss.

## The Geometric Limits: AWQ vs. SmoothQuant

Earlier methods like **SmoothQuant** attempted to solve precision loss by mathematically smoothing out the massive activation outliers, shifting the quantization difficulty from the activations onto the weights. However, forcing this global smoothing across the entire matrix crushes the tiny, sensitive weights in an SLM.

**AWQ (Activation-aware Weight Quantization)** takes a completely different, highly targeted approach:
It analyzes the variance of activations and detects the tiny subset ($\approx 1\%$) of "salient channels" that are responsible for the massive outliers. 

Instead of blindly quantizing everything:
1. **Protection:** The 1% of salient channels are protected. They either remain in FP16 (conceptually) or are mathematically re-scaled to prevent precision loss.
2. **INT4 Compression:** The remaining 99% of "non-dominant" weights are aggressively quantized down to 4-bits ($w_{bit} = 4$).
3. **Grouped GEMM:** The quantization is applied in grouped blocks (`q_group_size = 128`) during General Matrix Multiply operations, localizing any scaling noise.

## Why AWQ is the Apex for Edge Deployment

While Pruning struggles to provide actual speedups without specialized sparse hardware, and Knowledge Distillation is just a training technique, **AWQ provides immediate, brutal efficiency gains on universal hardware.**

By locking the bit depth to INT4 with AWQ, we drastically reduce the memory bandwidth required to move tensors from VRAM to the compute cores. In Edge devices (mobile phones, local SoC hardware), memory bandwidth—not compute—is the primary bottleneck. 

AWQ results in an astonishing reduction in **Time-To-First-Token (TTFT)**, allowing SLMs to run locally on consumer hardware with zero API latency, zero cloud costs, and 100% data privacy compliance.

*(See `src/quantization/autoawq_cli.py` for the line-by-line code implementation).*
