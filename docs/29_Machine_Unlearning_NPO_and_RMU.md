# Module 29: Machine Unlearning & Targeted Knowledge Erasure (NPO & RMU)

> **Module Path:** `src/module_29_machine_unlearning/unlearning_objectives.py`  
> **Primary References:**  
> - Zhang et al. (COLM 2024), *"Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning"*, `arXiv:2404.05868`.  
> - Li et al. (ICML 2024), *"The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning (RMU)"*, `arXiv:2403.03218`.

---

## 1. Why Naive Gradient Ascent Collapses SLMs

When erasing revoked personal data (GDPR/LGPD *Right to be Forgotten*) or dual-use hazardous knowledge (AI TRiSM), naive **Gradient Ascent** maximizes $-\log \pi_\theta(y_f \mid x_f)$. Because $-\log p \to \infty$ as $p \to 0$, its gradient norm never vanishes, rapidly destroying general linguistic fluency.

---

## 2. Negative Preference Optimization (`compute_npo_unlearning_loss`)

**NPO (Zhang et al., COLM 2024)** bounds the forget-set objective using a reference-anchored Bradley-Terry negative log-sigmoid loss:

$$\mathcal{L}_{\text{NPO}}(\theta) = -\frac{2}{\beta} \mathbb{E}_{(x_f, y_f) \sim \mathcal{D}_{\text{forget}}}\!\left[\log \sigma\!\left(-\beta \log \frac{\pi_\theta(y_f \mid x_f)}{\pi_{\text{ref}}(y_f \mid x_f)}\right)\right]$$

Its adaptive gradient weight $w_{\text{NPO}} = \sigma\!\left(\beta \log \frac{\pi_\theta(y_f \mid x_f)}{\pi_{\text{ref}}(y_f \mid x_f)}\right)$ decays smoothly toward $0$ once $\pi_\theta(y_f \mid x_f) \ll \pi_{\text{ref}}(y_f \mid x_f)$, preventing post-unlearning collapse.

---

## 3. Representation Misdirection for Unlearning (`compute_rmu_representation_loss`)

**RMU (Li et al., ICML 2024)** intervenes directly on intermediate residual activations $\mathbf{h}_\theta(x)$:

$$\mathcal{L}_{\text{RMU}} = \mathbb{E}_{x_f \sim \mathcal{D}_{\text{forget}}}\!\left[\|\mathbf{h}_\theta(x_f) - c \cdot \mathbf{u}\|_2^2\right] + \alpha_{\text{retain}} \mathbb{E}_{x_r \sim \mathcal{D}_{\text{retain}}}\!\left[\|\mathbf{h}_\theta(x_r) - \mathbf{h}_{\text{ref}}(x_r)\|_2^2\right]$$

where $\mathbf{u} \in \mathbb{S}^{d-1}$ is a fixed random unit vector and $c > 0$ is a steering magnitude.
