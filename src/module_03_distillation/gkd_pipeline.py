"""
==============================================================================
MÓDULO 03 — KNOWLEDGE DISTILLATION (GKD PIPELINE)
==============================================================================
NÍVEL: 🔴 Avançado
TEMPO ESTIMADO: 30 minutos (leitura) + >4 horas (treinamento)

OBJETIVO:
    Implementar Generalized Knowledge Distillation (GKD) para treinar um
    SLM (Student) copiando as probabilidades de um LLM gigante (Teacher).

CONCEITOS ABORDADOS:
    1. KL Divergence e Jensen-Shannon Divergence
    2. Temperature scaling (suavização de logits)
    3. Exposure Bias e On-Policy Distillation (GKD)
    4. Arquitetura cliente-servidor para destilação

REFERÊNCIAS:
    - Hinton et al., "Distilling the Knowledge in a Neural Network", 2015
    - Agarwal et al., "On-Policy Distillation of Language Models: 
      Learning from Self-Generated Mistakes", ICLR 2024
    - DeepSeek-AI, "DeepSeek-R1 Technical Report", 2025

COMO EXECUTAR:
    python src/module_03_distillation/gkd_pipeline.py
==============================================================================
"""

import logging
import os
import requests
from typing import Dict, Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import GKDTrainer

from src.utils.config import load_config
from src.utils.data_utils import load_and_prepare_dataset
from src.utils.metrics import get_vram_usage_gb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMTeacherOracle:
    """
    Interface remota para o modelo Teacher servido via vLLM.
    
    CONCEITO EDUCACIONAL — ARQUITETURA DESACOPLADA
    ================================================
    Problema:
    Um modelo Teacher bom (ex: Llama-3.1-70B) requer ~140GB de VRAM em BF16.
    O Student (ex: Qwen-1.5B) requer apenas ~3GB.
    Se tentarmos colocar ambos na mesma GPU para destilação, falharemos.
    
    Solução:
    1. Hospedar o Teacher em um cluster remoto (usando vLLM para max throughput)
    2. O Student roda localmente e envia suas gerações para o Teacher via API
    3. O Teacher retorna os "logprobs" (distribuição de probabilidade)
    4. O Student calcula o loss (KL Divergence) localmente
    """
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url
        logger.info(f"Conectado ao Teacher Oracle em: {endpoint_url}")

    def get_logprobs(self, prompt: str, generated_text: str):
        payload = {
            "prompt": prompt,
            "text": generated_text,
            "max_tokens": 1,
            "logprobs": 5, 
            "echo": True
        }
        try:
            response = requests.post(f"{self.endpoint_url}/v1/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Falha de comunicação com vLLM Teacher: {e}")
            # Em um cenário real, deveríamos fazer retry com backoff exponencial
            raise


def explain_kl_divergence():
    """
    Explica a matemática por trás da destilação (KL Divergence).
    
    TEORIA MATEMÁTICA:
    ──────────────────
    Em um treinamento padrão (SFT), usamos Cross-Entropy Loss com
    "Hard Labels" (One-hot encoding).
    Exemplo: A resposta certa é "Paris" (prob=1.0), todo o resto é 0.0.
    
    Na Destilação, usamos "Soft Labels" (KL Divergence).
    O Teacher não diz apenas que "Paris" é 100% certo. Ele diz:
    - "Paris": 80%
    - "Lyon": 15%
    - "Londres": 4%
    - "Maçã": 0.0001%
    
    Essa distribuição rica em informações é chamada de "Dark Knowledge" (Hinton).
    O Student aprende que "Lyon" é um erro aceitável (é uma cidade na França),
    mas "Maçã" é completamente absurdo.
    
    Fórmula KL Divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
    Onde: P = Distribuição do Teacher, Q = Distribuição do Student.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TEORIA: Kullback-Leibler (KL) Divergence e Soft Labels")
    logger.info("=" * 60)
    
    # Simulação de distribuições de probabilidade
    classes = ["Paris", "Lyon", "Londres", "Maçã"]
    
    # Hard label padrão (treinamento normal)
    hard_label = torch.tensor([1.0, 0.0, 0.0, 0.0])
    
    # Soft label do Teacher (conhecimento rico)
    teacher_probs = torch.tensor([0.80, 0.15, 0.04, 0.01])
    
    # Predição inicial (ruim) do Student
    student_probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    
    logger.info(f"Classes:         {classes}")
    logger.info(f"Teacher (Soft):  {teacher_probs.tolist()}")
    logger.info(f"Student (Atual): {student_probs.tolist()}")
    
    # Calcular KL Divergence (P || Q)
    # No PyTorch, kl_div espera log-probabilidades para o input (Student)
    # e probabilidades puras para o target (Teacher)
    log_student = torch.log(student_probs)
    kl_loss = F.kl_div(log_student, teacher_probs, reduction='sum')
    
    logger.info(f"\nKL Divergence Loss: {kl_loss.item():.4f}")
    logger.info("O otimizador ajustará os pesos do Student para minimizar este valor,")
    logger.info("fazendo a distribuição do Student imitar a do Teacher.")


def run_gkd_pipeline(config: Dict[str, Any]):
    """
    Executa o pipeline Generalized Knowledge Distillation.
    
    CONCEITO EDUCACIONAL — GKD vs KD TRADICIONAL
    ==============================================
    Problema do KD Tradicional (Exposure Bias):
    No KD normal (Forward KL), o Student recebe um texto do dataset
    (ex: escrito por humano) e tenta prever o próximo token imitando o Teacher.
    Mas na inferência real, o Student precisa prever o próximo token
    baseado no texto que ELE MESMO acabou de gerar (que pode conter erros).
    
    Solução GKD (On-Policy Distillation):
    O Student GERA ativamente as respostas (rollouts).
    Se ele cometer um erro, ele pede ao Teacher: "Dada essa frase torta
    que eu acabei de gerar, qual deveria ser o próximo token?"
    Isso ensina o Student a recuperar-se de seus próprios erros.
    """
    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO GENERALIZED KNOWLEDGE DISTILLATION (GKD)")
    logger.info("=" * 60)
    
    gkd_cfg = config.get("distillation", {})
    sft_cfg = config.get("sft", {})
    output_dir = "./results/gkd_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # --------------------------------------------------------------------------
    # 1. SETUP
    # --------------------------------------------------------------------------
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    teacher_id = gkd_cfg.get("teacher_model_id", "Qwen/Qwen2.5-7B-Instruct")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Carregando Student: {model_id}")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    # NOTA EDUCACIONAL:
    # Para o script rodar localmente sem estourar VRAM, usamos o Student como
    # "Mock Teacher". Em produção, use a classe VLLMTeacherOracle acima!
    logger.info(f"Carregando Mock Teacher (local) para demonstração...")
    mock_teacher = student_model 
    
    train_dataset, _ = load_and_prepare_dataset(
        max_samples=1000, format_fn="alpaca"
    )
    
    # --------------------------------------------------------------------------
    # 2. HYPERPARÂMETROS SENSÍVEIS (GKD)
    # --------------------------------------------------------------------------
    logger.info("\n⚙️ Configurando GKD Hyperparameters:")
    
    # Temperature: Valores altos (2.0 - 4.0) forçam a distribuição do Teacher
    # a ficar mais "plana", revelando as probabilidades secundárias.
    temperature = gkd_cfg.get("temperature", 2.0)
    logger.info(f"   Temperature: {temperature} (suaviza os logits)")
    
    # Lambda: Balanceia SFT e Distillation.
    # 0.0 = Distillation pura (ignora o texto real do dataset)
    # 1.0 = SFT puro (ignora o Teacher)
    lmbda = gkd_cfg.get("lmbda", 0.5)
    logger.info(f"   Lambda: {lmbda} (mix SFT/KD)")
    
    # Alpha: Define quem gera o texto durante o treino
    # 1.0 = Texto gerado 100% pelo Student (On-policy purista)
    # 0.0 = Texto 100% do dataset (Offline KD clássico)
    alpha = gkd_cfg.get("beta", 0.5)  # No TRL o param é alpha, no config chamamos de beta
    logger.info(f"   Alpha: {alpha} (fração de trajetórias do student)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        bf16=True,
        logging_steps=5,
        save_strategy="no",  # Demo
        report_to="none"
    )
    
    trainer = GKDTrainer(
        model=student_model,
        teacher_model=mock_teacher,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        temperature=temperature,
        lmbda=lmbda,
        alpha=alpha,
    )
    
    logger.info("\n🔥 Iniciando Distillation (Mock)...")
    logger.info("   [Aviso: O treinamento real pode levar semanas em 8x A100]")
    
    # trainer.train()  # Descomentar para treinar
    logger.info("   [Simulação concluída]")


def main():
    config = load_config()
    explain_kl_divergence()
    run_gkd_pipeline(config)


if __name__ == "__main__":
    main()
