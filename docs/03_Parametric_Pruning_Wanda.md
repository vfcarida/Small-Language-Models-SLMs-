# 3. Parametric Pruning: The Wanda Algorithm

Pruning is the act of deleting weights (setting them to zero) from a neural network to make it smaller and faster. 

For Small Language Models (SLMs), traditional "Magnitude Pruning" (deleting weights that are closest to zero) is catastrophic. In heavily packed SLMs, a weight with a tiny magnitude might be mathematically critical if it frequently multiplies against massive activation values (outliers). Deleting it destroys the model's syntactic coherence.

## Wanda: Pruning by Weights AND Activations

To solve this, state-of-the-art unstructured pruning relies on the **Wanda** heuristic. Wanda calculates a pruning metric by looking at both the weight magnitude *and* the activation norm.

The analytical multiplier ($S_{ij}$) to evaluate the importance of a weight $W_{ij}$ is:
$$ S_{ij} = |W_{ij}| \cdot ||X_j||_2 $$

Where $X_j$ is the vector of activations hitting that specific weight during a calibration phase. 
*   If a weight is small ($|W_{ij}| \approx 0$), but the activation it processes is consistently massive ($||X_j||_2 \gg 0$), $S_{ij}$ remains high, and the weight is **saved**.

**The Massive Advantage:** Unlike complex methods like SparseGPT, Wanda does not require expensive, slow gradient backpropagation to recalibrate the remaining weights. It operates entirely in a forward pass.

## The C4 Dataset Failure and Zipf Sampling

Usually, engineers push a random subset of the **C4 dataset** through the model to calculate the activation norms ($X_j$). 
However, recent surveys have shown that for SLMs, random uniform web-text fails. It results in pruning fine-grained logical synapses (like Python operators).

**Our Implementation: Zipf Calibration**
Instead of random C4 data, we implement calibration dataloaders based on **Zipf's Law** sampling over arithmetic or code datasets. Zipf's law states that the frequency of any word is inversely proportional to its rank.
By pushing highly structured, repetitive code syntax (`def`, `class`, `=`, `if`) through the model, we ensure that the activations ($X_j$) associated with pure reasoning and logic spike massively. Wanda sees these spikes, protects those weights, and only prunes the "fluff" linguistic weights.

### Pros and Cons of Pruning
*   **Pros:** Requires no retraining (fine-tuning) if done correctly.
*   **Cons:** *Unstructured* pruning (setting random matrix values to zero) often masks its flaws. Without specialized hardware (like Nvidia's Sparse Tensor Cores), standard GPUs cannot easily accelerate sparse matrices, meaning you might not actually see the latency gains you expect in standard environments.

*(See `src/pruning/wanda_heuristics.py` for the line-by-line code implementation).*
