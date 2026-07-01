"""
==============================================================================
MODULE 06 — ALIGNMENT WITH DPO (Direct Preference Optimization)
==============================================================================
LEVEL: 🔴 Advanced
ESTIMATED TIME: 30 minutes (reading) + training run

OBJECTIVE:
    Align the SLM (which was instruction-tuned via SFT) with human preferences
    using Direct Preference Optimization (DPO), bypassing reinforcement learning.

CONCEPTS COVERED:
    1. Reinforcement Learning from Human Feedback (RLHF) vs. DPO.
    2. Implicit reward modeling (Bradley-Terry preference model).
    3. Formatting chosen/rejected dataset pairs.

REFERENCES:
    - Rafailov et al., "Direct Preference Optimization: Your Language Model 
      is Secretly a Reward Model", arXiv:2305.18290, 2023

RUN COMMAND:
    python src/module_06_alignment/dpo_trainer.py
==============================================================================
"""

import logging
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import DPOTrainer
from peft import LoraConfig, get_peft_model

from src.utils.config import load_config
from src.utils.data_utils import create_preference_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def explain_dpo_math():
    """
    Details the mathematical comparison of RLHF vs. DPO.
    
    TRADITIONAL RLHF:
    1. Train a Reward Model (RM) using binary choice datasets.
    2. Run PPO reinforcement learning loops to maximize reward scores.
    Problems: High parameter overhead, mathematical instability.
    
    DPO INTUITION:
        DPO shows that the policy weights themselves can implicitly define
        rewards, avoiding separate training loops for reward networks.
        
        Loss = -log sigmoid ( β * log(P_chosen/Ref_chosen) - β * log(P_reject/Ref_reject) )
    """
    pass


def run_dpo_alignment(config: dict, output_dir: str = "./results/dpo_output"):
    """
    Executes preference alignment using Direct Preference Optimization.
    """
    logger.info("=" * 60)
    logger.info("STARTING DIRECT PREFERENCE OPTIMIZATION (DPO) ALIGNMENT")
    logger.info("=" * 60)
    
    dpo_cfg = config.get("dpo", {})
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    logger.info("Loading tokenizer and base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Inject LoRA adapters to freeze main weights
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    logger.info("Fetching chosen and rejected preference data pairs...")
    dataset = create_preference_dataset(max_samples=100)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=dpo_cfg.get("learning_rate", 5e-7),  # Critical: Low LR
        bf16=True,
        logging_steps=10,
        report_to="none"
    )
    
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=dpo_cfg.get("beta", 0.1),
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
    )
    
    logger.info("\nStarting DPO optimization training runs (Simulated)...")
    logger.info("✅ DPO simulation completed successfully.")


def main():
    config = load_config()
    explain_dpo_math()
    run_dpo_alignment(config)


if __name__ == "__main__":
    main()
