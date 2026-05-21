# 2. Generalized Knowledge Distillation (GKD): A Deep Dive

Knowledge Distillation (KD) is the process of transferring the "knowledge" of a massive, capable Teacher model (e.g., Llama-3.1-70B) into a smaller, faster Student model (e.g., Llama-3.2-1B). 

While standard Supervised Fine-Tuning (SFT) only teaches the model the "correct" answer (hard labels), Distillation teaches the model the *distribution* of possible answers (soft labels), allowing the student to learn nuance, grammar, and reasoning traces.

## The Math: KL Divergence vs. Generalized Jensen-Shannon (JSD)

### Standard KL Divergence
Historically, KD minimizes the **Forward Kullback-Leibler (KL) Divergence**:
$$ D_{KL}(P \parallel Q) = \sum_{x} P(x) \log\left(\frac{P(x)}{Q(x)}\right) $$
Where $P$ is the Teacher's distribution and $Q$ is the Student's. 
**The Problem:** This forces the student to perfectly mimic the teacher's exact probability distribution. If the teacher has a complex reasoning path the student lacks the capacity to follow, the student suffers from *mode collapse* or produces gibberish.

### Generalized KD (GKD)
GKD softens this by using a mixed objective, often inspired by the **Jensen-Shannon Divergence (JSD)**:
$$ JSD_{\pi}(P \parallel Q) = \pi D_{KL}(P \parallel M) + (1 - \pi) D_{KL}(Q \parallel M) $$
Where $M$ is a mixture distribution. GKD allows the student to explore its *own* generated trajectories (auto-regressive generation) and asks the teacher: *"Is this path acceptable?"* 

Instead of forcing the student to mimic the teacher blindly, GKD provides **reactive feedback**. This results in SLMs that are highly proficient in instruction-following without exceeding their parameter capacity.

## Engineering Architecture: Decoupling with vLLM

In a corporate cluster, running a 70B Teacher and a 1B Student on the same GPU for synchronous training destroys VRAM and training speed. 

**The Solution:**
1. **Asynchronous Generation:** The Student model generates candidate responses locally on a lightweight training node.
2. **vLLM Oracle:** The candidate text is sent (via API/base64) to a dedicated, high-throughput `vLLM` server hosting the 70B Teacher. 
3. **Logprob Feedback:** The vLLM server utilizes PagedAttention to instantly return the log-probabilities of that trajectory.
4. **Gradient Update:** The student calculates the divergence loss and updates its weights.

### Pros and Cons of KD
*   **Pros:** Preserves incredible systemic proficiency; the student learns "how" to think.
*   **Cons:** Extremely cluster-intensive during training. The infrastructure required to serve the teacher model via vLLM for millions of tokens is expensive.

*(See `src/distillation/gkd_pipeline.py` for the line-by-line code implementation).*
