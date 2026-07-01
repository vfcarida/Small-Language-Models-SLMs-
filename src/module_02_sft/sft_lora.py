"""
==============================================================================
MODULE 02 — SFT WITH LoRA (Low-Rank Adaptation)
==============================================================================
LEVEL: 🟡 Intermediate
ESTIMATED TIME: 30 minutes (reading) + 1-4 hours (training)

OBJECTIVE:
    Train an SLM to follow instructions using Supervised Fine-Tuning (SFT)
    with LoRA — the industry-standard parameter-efficient fine-tuning technique.

CONCEPTS COVERED:
    1. SFT (Supervised Fine-Tuning): training with input-output instruction pairs.
    2. LoRA: updating only ~1% of parameters (Low-Rank Adaptation).
    3. TRL SFTTrainer: pipeline setup.
    4. Adapter merging: merging LoRA adapters with the base model weights.
    5. Evaluation workflows.

REFERENCES:
    - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", arXiv:2106.09685, 2021
    - HuggingFace TRL, "SFTTrainer", 2024
    - Dettmers et al., "QLoRA", arXiv:2305.14314, 2023 (for sft_qlora.py)

RUN COMMAND:
    python src/module_02_sft/sft_lora.py
==============================================================================
"""

import logging
import os
import sys
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer

from src.utils.config import load_config
from src.utils.data_utils import load_and_prepare_dataset
from src.utils.metrics import count_parameters, get_model_size_mb, get_vram_usage_gb, GPUOOMHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_lora_config(config: dict) -> LoraConfig:
    """
    Creates a LoRA configuration from the loaded settings.
    
    LORA (Low-Rank Adaptation) THEORY:
        Updates of frozen weight matrix W (d x d) are factorized into low-rank matrices B and A:
            W_new = W_original + (B x A)
        Where:
            - W_original: Pre-trained weights (FROZEN, not updated during training).
            - A: Down-projection matrix (d x r).
            - B: Up-projection matrix (r x d).
            - r: Rank dimension (typically 8 to 64).
            
    Args:
        config: Dictionary containing the 'lora' section.
        
    Returns:
        LoraConfig: Initialized LoRA settings.
    """
    lora_cfg = config.get("lora", {})
    
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    
    logger.info(
        f"LoRA Config Initialized: rank r={lora_config.r}, alpha={lora_config.lora_alpha}, "
        f"dropout={lora_config.lora_dropout}, target={lora_config.target_modules}"
    )
    return lora_config


def train_sft_lora(
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Executes the SFT training pipeline with LoRA adapter injection.
    
    Args:
        config: Dictionary configuration data. If None, default YAML is loaded.
        output_dir: Output directory path. If None, path is fetched from config.
        
    Returns:
        str: File path to directory containing trained LoRA weights.
    """
    if config is None:
        config = load_config()
        
    model_cfg = config.get("model", {})
    sft_cfg = config.get("sft", {})
    dataset_cfg = config.get("dataset", {})
    
    if output_dir is None:
        output_dir = sft_cfg.get("output_dir", "./results/sft_lora_output")
        
    os.makedirs(output_dir, exist_ok=True)
    model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING SUPERVISED FINE-TUNING (SFT) WITH LoRA")
    logger.info(f"   Base Model: {model_id}")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info("=" * 60)
    
    logger.info("Loading tokenizer instance...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"   ⚠️ pad_token is undefined. Using eos_token fallback: '{tokenizer.eos_token}'")
        
    logger.info("Loading base model weights...")
    vram_before = get_vram_usage_gb()
    
    dtype_str = model_cfg.get("torch_dtype", "bfloat16")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map.get(dtype_str, torch.bfloat16)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    
    vram_after = get_vram_usage_gb()
    logger.info(f"   Model initialized. VRAM usage changes: +{vram_after - vram_before:.3f} GB")
    
    logger.info("Injecting LoRA adapters...")
    lora_config = create_lora_config(config)
    model = get_peft_model(model, lora_config)
    
    params = count_parameters(model)
    logger.info(f"   Trainable Parameter Size: {params['trainable']:,} ({params['trainable_percent']:.2f}% of total)")
    logger.info(f"   Frozen Parameter Size: {params['frozen']:,}")
    
    logger.info("Preparing dataset instances...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_cfg.get("name", "tatsu-lab/alpaca"),
        split=dataset_cfg.get("split", "train"),
        max_samples=dataset_cfg.get("max_samples", 5000),
        test_size=dataset_cfg.get("test_size", 0.1),
        seed=dataset_cfg.get("seed", 42),
        format_fn="alpaca",
    )
    
    logger.info("Configuring training parameters...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_train_batch_size", 4),
        per_device_eval_batch_size=sft_cfg.get("per_device_eval_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.03),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        fp16=sft_cfg.get("fp16", False),
        bf16=sft_cfg.get("bf16", True),
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "steps"),
        save_steps=sft_cfg.get("save_steps", 200),
        eval_strategy=sft_cfg.get("eval_strategy", "steps"),
        eval_steps=sft_cfg.get("eval_steps", 200),
        save_total_limit=sft_cfg.get("save_total_limit", 3),
        load_best_model_at_end=sft_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=sft_cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=sft_cfg.get("greater_is_better", False),
        report_to=sft_cfg.get("report_to", "none"),
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        seed=sft_cfg.get("seed", 42),
    )
    
    logger.info("Initializing SFTTrainer setup...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("🔥 STARTING SFT MODEL TRAINING...")
    logger.info(f"   Epochs: {training_args.num_train_epochs}")
    logger.info(f"   Effective Batch Size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning Rate: {training_args.learning_rate}")
    
    # Executing training wrapped with GPU OOM protection
    with GPUOOMHandler():
        train_result = trainer.train()
        
    logger.info("Saving adapter models and training logs...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"\n📊 FINAL SFT TRAINING RESULTS:")
    logger.info(f"   Final Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"   Final Evaluation Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"   Adapters successfully saved in: {output_dir}")
    
    logger.info("=" * 60)
    logger.info("✅ LoRA SUPERVISED FINE-TUNING COMPLETED!")
    logger.info("=" * 60)
    
    return output_dir


def merge_lora_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str = "./results/sft_merged_model",
) -> str:
    """
    Merges the LoRA adapter weights back with the pre-trained base model weights.
    
    Args:
        base_model_id: Hugging Face repository base model ID.
        adapter_path: Directory containing the trained LoRA adapters.
        output_dir: Destination path to save the merged model.
        
    Returns:
        str: Path containing the fully merged model files.
    """
    logger.info("=" * 60)
    logger.info("MERGE ADAPTER: Combining LoRA parameters with base weights")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading base model architecture: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    logger.info(f"Injecting adapter parameters from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Executing weight merge operations...")
    model = model.merge_and_unload()
    
    logger.info(f"Saving merged weights locally to: {output_dir}")
    model.save_pretrained(output_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    size_mb = get_model_size_mb(model)
    logger.info(f"✅ Adapter merge complete! Model profile: {size_mb:.1f} MB")
    
    return output_dir


def main():
    config = load_config()
    output_dir = train_sft_lora(config=config)


if __name__ == "__main__":
    main()
