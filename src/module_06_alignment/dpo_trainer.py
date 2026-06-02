"""
==============================================================================
MÓDULO 06 — ALIGNMENT COM DPO (Direct Preference Optimization)
==============================================================================
NÍVEL: 🔴 Avançado
TEMPO ESTIMADO: 30 minutos (leitura) + treinamento

OBJETIVO:
    Alinhar o SLM (já instruído via SFT) aos valores e preferências desejadas,
    usando DPO, que substitui o antigo e complexo RLHF.

CONCEITOS ABORDADOS:
    1. RLHF vs DPO
    2. Reward Models Implícitos (Bradley-Terry)
    3. Construção do Dataset Chosen/Rejected
    
REFERÊNCIAS:
    - Rafailov et al., "Direct Preference Optimization: Your Language Model 
      is Secretly a Reward Model", arXiv:2305.18290, 2023

COMO EXECUTAR:
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
    Explica a intuição matemática do DPO versus RLHF.
    
    RLHF TRADICIONAL (Complexo):
    1. Treina um Reward Model (RM) separado baseado nas preferências humanas.
    2. Usa PPO (Reinforcement Learning) para maximizar o score do RM.
    Problemas: Instável, requer múltiplos modelos na RAM, tuning difícil.
    
    DPO (Elegante e Simples):
    Rafailov (2023) provou matematicamente que o RM implícito pode ser
    extraído DIRETAMENTE do modelo de linguagem.
    
    Equação DPO:
    Loss = -log sigmoid ( β * log(P_chosen/Ref_chosen) - β * log(P_reject/Ref_reject) )
    
    O que isso faz?
    Aumenta a probabilidade da resposta "Chosen" e diminui a da "Rejected",
    sempre mantendo o modelo amarrado ao modelo de Referência (para evitar
    que ele se degrade gerando jargões bizarros).
    """
    pass


def run_dpo_alignment(config: dict, output_dir: str = "./results/dpo_output"):
    """
    Executa o Direct Preference Optimization.
    """
    logger.info("=" * 60)
    logger.info("INICIANDO DPO ALIGNMENT")
    logger.info("=" * 60)
    
    dpo_cfg = config.get("dpo", {})
    
    # 1. Carregar Modelo (Idealmente, este seria o modelo após o SFT!)
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    logger.info("Carregando Modelos...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # Em DPO, usamos o mesmo modelo como "policy" (treinado) e "reference" (congelado)
    # Ao usar LoRA, o DPOTrainer inteligentemente desabilita o LoRA no forward do reference!
    
    lora_config = LoraConfig(
        r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 2. Dataset Chosen/Rejected
    dataset = create_preference_dataset(max_samples=100) # Pequeno para demo
    
    # 3. Training Arguments (Note o learning rate minúsculo)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=dpo_cfg.get("num_train_epochs", 1),
        per_device_train_batch_size=dpo_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=dpo_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=dpo_cfg.get("learning_rate", 5e-7),  # CRÍTICO: muito baixo
        bf16=True,
        logging_steps=10,
        report_to="none"
    )
    
    # 4. Iniciar DPOTrainer
    trainer = DPOTrainer(
        model,
        ref_model=None,  # Como estamos usando PEFT, None = base model frozen
        args=training_args,
        beta=dpo_cfg.get("beta", 0.1),
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=dpo_cfg.get("max_length", 1024),
        max_prompt_length=dpo_cfg.get("max_prompt_length", 512),
    )
    
    logger.info("\n🔥 Treinando (Alinhamento de Preferências)...")
    # trainer.train()  # Descomentado na pipeline real
    logger.info("✅ Simulação DPO concluída!")


def main():
    config = load_config()
    run_dpo_alignment(config)

if __name__ == "__main__":
    main()
