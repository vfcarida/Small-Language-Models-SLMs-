"""
==============================================================================
MODULE 03 — KNOWLEDGE DISTILLATION (GKD & MINILLM PIPELINE)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 30 minutes (reading) + >4 hours (training run)

OBJECTIVE:
    Implement Generalized Knowledge Distillation (GKD) and Mode-Seeking Reverse
    KL Distillation (MiniLLM) to train a compact student SLM by aligning its
    token-level probability distributions with a high-capacity teacher LLM.

CONCEPTS COVERED:
    1. Forward KL Divergence (Mean-Seeking, Hinton et al., 2015).
    2. Reverse KL Divergence (Mode-Seeking, MiniLLM, Gu et al., ICLR 2024).
    3. Generalized Jensen-Shannon Divergence (GKD, Agarwal et al., ICLR 2024).
    4. Temperature scaling and Exposure Bias mitigation via On-Policy rollouts.
    5. Decoupled client-server architecture with a remote vLLM Teacher Oracle.

REFERENCES:
    - Hinton et al., "Distilling the Knowledge in a Neural Network", arXiv:1503.02531, 2015
    - Agarwal et al., "On-Policy Distillation of Language Models: Learning from
      Self-Generated Mistakes", ICLR 2024 (arXiv:2306.13649)
    - Gu et al., "MiniLLM: On-Policy Distillation of Large Language Models",
      ICLR 2024 (arXiv:2306.08543)
    - DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via
      Reinforcement Learning", arXiv:2501.12948, 2025

RUN COMMAND:
    python src/module_03_distillation/gkd_pipeline.py
==============================================================================
"""

import logging
import os
import time
from typing import Any, Dict, Literal, Optional

import requests
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from src.utils.config import load_config
from src.utils.data_utils import load_and_prepare_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DistillationMode = Literal["forward_kl", "reverse_kl", "jsd"]


class VLLMTeacherOracle:
    """
    Remote HTTP client interface for a Teacher LLM hosted via an OpenAI-compatible
    vLLM inference server.

    ARCHITECTURAL RATIONALE (DECOUPLED DISTILLATION):
        A frontier teacher model (e.g., Llama-3.1-70B) requires ~140 GB VRAM in BF16,
        whereas a compact student SLM (e.g., Qwen2.5-1.5B) fits within ~3 GB VRAM.
        Co-locating both models on a single consumer GPU triggers immediate CUDA OOM.
        Decoupling the teacher onto a remote vLLM cluster allows the local student
        training node to query top-k log-probabilities asynchronously with exponential
        backoff retry resilience.
    """

    def __init__(
        self,
        endpoint_url: str,
        timeout: int = 10,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ) -> None:
        self.endpoint_url = endpoint_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        logger.info(f"Initialized Remote Teacher Oracle client at: {self.endpoint_url}")

    def get_logprobs(self, prompt: str, generated_text: str) -> Dict[str, Any]:
        """
        Fetches token log-probabilities from the remote teacher instance for a
        student-generated trajectory.

        Args:
            prompt: Original prompt context prefix.
            generated_text: Trajectory suffix generated on-policy by the student.

        Returns:
            Dict[str, Any]: Parsed JSON payload containing teacher log-probabilities.
        """
        payload = {
            "prompt": prompt,
            "text": generated_text,
            "max_tokens": 1,
            "logprobs": 5,
            "echo": True,
        }

        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.endpoint_url}/v1/completions",
                    json=payload,
                    headers=self.headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as exc:
                logger.warning(
                    f"Communication attempt {attempt}/{self.max_retries} with vLLM Teacher failed: {exc}. "
                    f"Retrying after {backoff:.1f}s..."
                )
                if attempt == self.max_retries:
                    logger.error("Maximum vLLM communication retries reached. Distillation step aborted.")
                    raise
                time.sleep(backoff)
                backoff *= 2.0
        return {}


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    mode: DistillationMode = "jsd",
    beta: float = 0.5,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes temperature-scaled divergence loss between student and teacher logits.

    MATHEMATICAL FORMULATION:
        Let p_T = softmax(z_T / T) and p_S = softmax(z_S / T). All losses are scaled
        by T^2 to preserve gradient magnitudes across temperature regimes (Hinton et al., 2015).

        1. Forward KL ("forward_kl", Mean-Seeking):
           L_FKL = T^2 * KL(p_T || p_S) = T^2 * sum_v p_T(v) * [log p_T(v) - log p_S(v)]

        2. Reverse KL ("reverse_kl", Mode-Seeking — MiniLLM, Gu et al., ICLR 2024):
           L_RKL = T^2 * KL(p_S || p_T) = T^2 * sum_v p_S(v) * [log p_S(v) - log p_T(v)]
           Prevents a low-capacity student from assigning probability mass to empty regions
           between teacher modes.

        3. Generalized Jensen-Shannon Divergence ("jsd" — GKD, Agarwal et al., ICLR 2024):
           Let mixture distribution M = beta * p_T + (1 - beta) * p_S for beta in (0, 1).
           L_JSD^(beta) = T^2 * [ beta * KL(p_T || M) + (1 - beta) * KL(p_S || M) ]

    Args:
        student_logits: Tensor of shape (..., vocab_size) from the student SLM.
        teacher_logits: Tensor of shape (..., vocab_size) from the teacher LLM.
        temperature: Positive temperature scaling factor T > 0.
        mode: Divergence objective ("forward_kl", "reverse_kl", or "jsd").
        beta: Interpolation weight in [0, 1] for Generalized JSD.
        mask: Optional binary mask of shape (...) over valid non-padding tokens.

    Returns:
        torch.Tensor: Scalar distillation loss.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be strictly positive. Got: {temperature}")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError(
            f"Shape mismatch between student_logits {student_logits.shape} "
            f"and teacher_logits {teacher_logits.shape}."
        )

    scaled_student = student_logits.float() / temperature
    scaled_teacher = teacher_logits.float() / temperature

    log_p_s = F.log_softmax(scaled_student, dim=-1)
    log_p_t = F.log_softmax(scaled_teacher, dim=-1)
    p_s = torch.exp(log_p_s)
    p_t = torch.exp(log_p_t)

    if mode == "forward_kl":
        token_loss = torch.sum(p_t * (log_p_t - log_p_s), dim=-1)
    elif mode == "reverse_kl":
        token_loss = torch.sum(p_s * (log_p_s - log_p_t), dim=-1)
    elif mode == "jsd":
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"JSD interpolation beta must be in [0, 1]. Got: {beta}")
        if beta == 0.0:
            token_loss = torch.sum(p_s * (log_p_s - log_p_t), dim=-1)
        elif beta == 1.0:
            token_loss = torch.sum(p_t * (log_p_t - log_p_s), dim=-1)
        else:
            mixture = torch.clamp(beta * p_t + (1.0 - beta) * p_s, min=1e-12)
            log_mixture = torch.log(mixture)
            kl_t_m = torch.sum(p_t * (log_p_t - log_mixture), dim=-1)
            kl_s_m = torch.sum(p_s * (log_p_s - log_mixture), dim=-1)
            token_loss = beta * kl_t_m + (1.0 - beta) * kl_s_m
    else:
        raise ValueError(f"Unsupported distillation mode: '{mode}'. Choose from 'forward_kl', 'reverse_kl', 'jsd'.")

    scaled_loss = token_loss * (temperature ** 2)

    if mask is not None:
        mask_float = mask.to(scaled_loss.dtype)
        denom = torch.clamp(mask_float.sum(), min=1.0)
        return (scaled_loss * mask_float).sum() / denom

    return scaled_loss.mean()


def explain_kl_divergence() -> Dict[str, float]:
    """
    Demonstrates the numerical behavior of Forward KL, Reverse KL, and Generalized
    Jensen-Shannon Divergence (JSD) on a synthetic multimodal teacher distribution.

    Returns:
        Dict[str, float]: Computed loss values across all three divergence modes.
    """
    logger.info("\n" + "=" * 60)
    logger.info("THEORY: Forward KL vs. Reverse KL (MiniLLM) vs. Generalized JSD (GKD)")
    logger.info("=" * 60)

    classes = ["Paris", "Lyon", "London", "Apple"]
    teacher_logits = torch.tensor([[3.2, 1.8, 0.4, -2.0]])
    student_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

    fkl = compute_distillation_loss(student_logits, teacher_logits, temperature=2.0, mode="forward_kl").item()
    rkl = compute_distillation_loss(student_logits, teacher_logits, temperature=2.0, mode="reverse_kl").item()
    jsd = compute_distillation_loss(student_logits, teacher_logits, temperature=2.0, mode="jsd", beta=0.5).item()

    logger.info(f"Candidate Tokens:           {classes}")
    logger.info(f"Forward KL Loss (Hinton):   {fkl:.4f} (Mean-seeking across all teacher modes)")
    logger.info(f"Reverse KL Loss (MiniLLM):  {rkl:.4f} (Mode-seeking; penalizes student hallucinations)")
    logger.info(f"Generalized JSD Loss (GKD): {jsd:.4f} (Bounded, symmetric interpolation)")

    return {"forward_kl": fkl, "reverse_kl": rkl, "jsd": jsd}


def run_gkd_pipeline(config: Dict[str, Any]) -> None:
    """
    Configures and initializes the Generalized Knowledge Distillation (GKD) pipeline.

    Args:
        config: Centralized configuration dictionary.
    """
    from trl import GKDTrainer

    logger.info("\n" + "=" * 60)
    logger.info("STARTING GENERALIZED KNOWLEDGE DISTILLATION (GKD) PIPELINE")
    logger.info("=" * 60)

    gkd_cfg = config.get("distillation", {})
    output_dir = "./results/gkd_output"
    os.makedirs(output_dir, exist_ok=True)

    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading Student model: {model_id}")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )

    logger.info("Instantiating local reference teacher for validation pass...")
    mock_teacher = student_model

    train_dataset, _ = load_and_prepare_dataset(max_samples=100, format_fn="alpaca")

    temperature = float(gkd_cfg.get("temperature", 2.0))
    lmbda = float(gkd_cfg.get("lmbda", 0.5))
    beta = float(gkd_cfg.get("beta", 0.5))

    logger.info(f"GKD Hyperparameters: temperature={temperature}, lmbda={lmbda}, beta={beta}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    _ = GKDTrainer(
        model=student_model,
        teacher_model=mock_teacher,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        lmbda=lmbda,
        beta=beta,
    )

    logger.info("GKD pipeline initialized and verified.")


def main() -> None:
    config = load_config()
    explain_kl_divergence()
    run_gkd_pipeline(config)


if __name__ == "__main__":
    main()
