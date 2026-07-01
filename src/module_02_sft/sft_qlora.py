"""
==============================================================================
MODULE 02 — SFT WITH QLoRA (Quantized LoRA)
==============================================================================
LEVEL: 🟡 Intermediate
ESTIMATED TIME: 30 minutes (reading) + 1-4 hours (training)

OBJECTIVE:
    Train an SLM using QLoRA — which combines LoRA with 4-bit quantization,
    enabling instruction fine-tuning on consumer-grade GPUs (8GB VRAM).

QLORA vs LoRA:
    - sft_lora.py: Loads base model in BF16 (2 bytes/param). Requires ~6GB+ VRAM.
    - sft_qlora.py: Loads base model in NF4 (0.5 bytes/param). Fits in ~4GB VRAM.
    
    QLoRA = 4-bit Base Model Quantization + BF16 LoRA Adapters.

REFERENCES:
    - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", arXiv:2305.14314, 2023
    - NormalFloat 4 (NF4) is information-theoretically optimal for zero-mean normal distributions.

RUN COMMAND:
    python src/module_02_sft/sft_qlora.py
==============================================================================
"""

import logging
import os
from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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


def create_bnb_config(config: dict) -> BitsAndBytesConfig:
    """
    Creates a bitsandbytes 4-bit quantization config.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        BitsAndBytesConfig: Configured 4-bit parameters.
    """
    qlora_cfg = config.get("qlora", {})
    compute_dtype_str = qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )
    
    logger.info(
        f"BitsAndBytes Config initialized: quant_type={bnb_config.bnb_4bit_quant_type}, "
        f"compute_dtype={compute_dtype}, double_quant={bnb_config.bnb_4bit_use_double_quant}"
    )
    return bnb_config


def train_sft_qlora(
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Executes instruction tuning via SFT with QLoRA.
    
    Args:
        config: Configuration dictionary parameters.
        output_dir: Directory to save the adapter weights.
        
    Returns:
        str: Output directory path.
    """
    if config is None:
        config = load_config()
        
    model_cfg = config.get("model", {})
    sft_cfg = config.get("sft", {})
    dataset_cfg = config.get("dataset", {})
    
    if output_dir is None:
        output_dir = "./results/sft_qlora_output"
        
    os.makedirs(output_dir, exist_ok=True)
    model_id = model_cfg.get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING SUPERVISED FINE-TUNING (SFT) WITH QLoRA (4-bit)")
    logger.info(f"   Base Model: {model_id}")
    logger.info(f"   Output Directory: {output_dir}")
    logger.info("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    logger.info("Loading base model in 4-bit precision via bitsandbytes...")
    bnb_config = create_bnb_config(config)
    vram_before = get_vram_usage_gb()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    
    vram_after = get_vram_usage_gb()
    logger.info(f"   4-bit Model Loaded. VRAM allocation: {vram_after:.3f} GB (+{vram_after - vram_before:.3f} GB)")
    
    logger.info("Preparing model layers for k-bit parameter updates...")
    model = prepare_model_for_kbit_training(model)
    
    logger.info("Injecting LoRA adapters...")
    lora_cfg = config.get("lora", {})
    
    lora_config = LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    
    model = get_peft_model(model, lora_config)
    params = count_parameters(model)
    logger.info(f"   Trainable parameter size: {params['trainable']:,} ({params['trainable_percent']:.2f}%)")
    
    logger.info("Formatting dataset splits...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_cfg.get("name", "tatsu-lab/alpaca"),
        max_samples=dataset_cfg.get("max_samples", 5000),
        test_size=dataset_cfg.get("test_size", 0.1),
        seed=dataset_cfg.get("seed", 42),
    )
    
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
        fp16=False,
        bf16=True,
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy="steps",
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("🔥 STARTING QLoRA MODEL TRAINING RUN...")
    with GPUOOMHandler():
        train_result = trainer.train()
        
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"\n📊 FINAL QLoRA TRAINING SUMMARY:")
    logger.info(f"   Final Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"   Final Evaluation Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"   Peak VRAM memory: {get_vram_usage_gb():.3f} GB")
    
    logger.info("=" * 60)
    logger.info("✅ QLoRA SUPERVISED FINE-TUNING COMPLETED!")
    logger.info("=" * 60)
    
    return output_dir


def main():
    config = load_config()
    train_sft_qlora(config=config)


if __name__ == "__main__":
    main()
