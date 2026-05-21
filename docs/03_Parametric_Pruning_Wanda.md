# 3. Implementation 2: Model Pruning

Pruning is the methodology responsible for identifying and obliterating non-critical parameters, with the dual objective of reducing storage size and mitigating computational redundancy without harming knowledge representation.

## Pruning Categories

Pruning topology is divided into three main formats:
1. **Unstructured:** Removes individual parameters that have the least impact based on heuristics. Although it creates extreme sparsity, the resulting sparse matrices do not offer direct latency acceleration on most standardized hardware without specialized kernels.
2. **Semi-structured:** Adopts fixed patterns (such as 2:4 sparsity in matrices), which are actively supported by hardware accelerators like NVIDIA Ampere and Ada Lovelace GPUs, directly accelerating tensor processing.
3. **Structured:** Eliminates complete channels, interconnected attention blocks, or entire layers. Offers immediate speed benefits and memory reduction on any hardware, but the disruption to architectural integrity requires extensive retraining.

## State of the Art: SparseGPT versus Wanda

In the paradigm of large models, the **SparseGPT** algorithm demonstrated for the first time the technical viability of pruning transformer models to 50% sparsity "one-shot", without requiring any compensatory fine-tuning (retraining). It relies on approximate second-order information calculations to infer the impact that deleting a weight will have on the global network.

The most recent and performant breakthrough for corporate SLMs goes by the name **Wanda** (Pruning by Weights and activations). Unlike SparseGPT, Wanda proposes that the degree of importance of a weight does not depend solely on its static intrinsic magnitude, but on the norm of the activations that interact with it. 

Notably, computing the pruning metric in Wanda is up to **300 times faster** than SparseGPT, and it does not require the heavy iterative updating of remaining weights. This suggests that exact and effective sparse sub-networks reside organically within the initial pre-trained structure.
*   **M-Wanda:** Extended metrics like M-Wanda have also demonstrated the ability to optimize pruning in a way that consistently preserves essential multilingual capacities (critical for Latin American contexts).
*   **Wanda++:** Explores regional gradients of decoder blocks to further leverage the baseline method's efficacy.

## The Calibration Dataset Paradox (Zipf Sampling)

The critical trap in implementing Pruning (and subsequent quantization) is the use of inadequate calibration data. The vast majority of algorithms conveniently use the generalist "C4" dataset. However, exhaustive testing demonstrates that using C4 causes invisible degradation, whereas datasets with an arithmetic focus often offer exceptionally superior calibrations.

Natural language activations follow a **Zipfian distribution**, where the bulk of the vocabulary is concentrated in a sparse long tail. Calibrating an SLM meant for the financial sector while ignoring this sparsity means omitting varied semantic contexts that trigger critical "outliers" in the model's activations.

Strategies like **Zipf Sampling** solve this by ensuring the calibration database is heuristically selected to maximize semantic diversity (lexical diversity) without extrapolating the number of samples. This aligns network pruning with the singular vocabularies of the corporate market.

The positive side of pruning is the effective decrease in storage size associated with latency improvements (in infrastructures supporting semi-structured sparsity). The negative side is the mathematical difficulty tied to calibration, where incorrect data causes drastic perplexity reductions in unexpected tasks.
