# 4. Implementation 3: Model Quantization

Quantization attacks transfer bottlenecks and memory capacity limits by substantially decreasing the resolution of floating-point formats stored by the transformer (converting FP32 and BF16 types to reduced formats like INT8 or INT4). This directly reduces bandwidth requirements and the total VRAM capacity required to allocate the SLM in GPU memory.

## Dominant Quantization Paradigms

The ecosystem is critically divided into **weight-only** quantization methods versus joint **activation-and-weight** quantization.

### 1. GPTQ
A foundational paradigm based, similarly to SparseGPT, on reconstruction through Hessian matrices of second-order approximation. It applies column-by-column processes and notably preserves the original model's accuracy while reducing bit widths to 3-bit or 4-bit indices in a relatively short time (a few hours on A100 GPUs), quadrupling global inference accelerations without functional loss.

### 2. AWQ (Activation-aware Weight Quantization)
Currently considered a standard model of extreme efficacy, AWQ starts from an analytical premise of selective protection: **not all numerical components weigh equally**. 
Identifying salient weight channels requires observing offline pre-generated activation statistics rather than relying strictly on raw matrix weights. By rigorously protecting only about one percent of the most salient weights (mathematically scaling them via equivalent transformations so their quantized error decays considerably), the architecture enables executions on extreme infrastructures (like 4-bit on smartphones operating edge interfaces and consumer architecture boards, using native implementations with frameworks like TinyChat). This approach bypasses distortions and dataset over-fitting that methods reliant on constant backpropagation frequently generate.

### 3. QLoRA & SpinQuant
QLoRA combines low-rank adaptation with quantized **NF4 (NormalFloat)** theoretical compression schemes, which mathematically prove to be ideal for representations with weight behavior variables following standard normal distributions. This established simultaneous records in fine-tuning memory management (double quantization parameters), where limiting spikes are nullified by organized virtual paging schemes. Advanced techniques like **SpinQuant** push these limits further for models like Llama-3.2.

## Granularity: Per-Channel vs. SmoothQuant
Contemporary strategies explore detailed per-channel levels. Restrictions must be observed when methods attempt to encompass quantization in the "activations" themselves, such as the "SmoothQuant" frameworks. 

When extrapolated to massive models or even SLMs of extreme arithmetic complexity, activation quantizations cause unacceptable precision deltas (accuracy drops reported in the range of up to **9%** compared to the base format at native FP8 scale).

## Conclusion
The great technical virtue of quantization, notably in the AWQ approach, is allowing rapid offline allocations and unrestricted universal portability with zero detectable loss for end-users in banking SLM applications. However, the empirical need dictates absolute rigor in the quality of the data selected to calibrate metrics, analogous to Pruning.
