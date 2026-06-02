"""
==============================================================================
MÓDULO 02 — SFT COM QLoRA (Quantized LoRA)
==============================================================================
NÍVEL: 🟡 Intermediário
TEMPO ESTIMADO: 30 minutos (leitura) + 1-4 horas (treinamento)

OBJETIVO:
    Treinar um SLM usando QLoRA — que combina LoRA com quantização 4-bit,
    permitindo fine-tuning em GPUs consumer (8GB VRAM).

DIFERENÇA PARA sft_lora.py:
    - sft_lora.py: Carrega modelo em BF16 (2 bytes/param). Precisa de ~6GB+ VRAM.
    - sft_qlora.py: Carrega modelo em NF4 (0.5 bytes/param). Funciona com ~4GB VRAM.
    
    QLoRA = Quantização 4-bit do modelo BASE + LoRA adapters em BF16

REFERÊNCIAS:
    - Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
      arXiv:2305.14314, 2023
    - Theorem 1 do paper: NF4 é informacionalmente ótimo para distribuições
      normais (que é a distribuição dos pesos de modelos pré-treinados)

COMO EXECUTAR:
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
from src.utils.metrics import count_parameters, get_model_size_mb, get_vram_usage_gb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_bnb_config(config: dict) -> BitsAndBytesConfig:
    """
    Cria a configuração de quantização 4-bit com bitsandbytes.
    
    TEORIA — QLoRA e NormalFloat 4 (NF4)
    ======================================
    A grande inovação do QLoRA é usar o formato NF4 (NormalFloat 4-bit)
    para quantizar o modelo base, em vez dos formatos INT4 tradicionais.
    
    POR QUE NF4?
    ─────────────
    Os pesos de redes neurais pré-treinadas seguem uma distribuição
    aproximadamente NORMAL (gaussiana) centrada em zero:
    
        P(w) ≈ N(0, σ²)
    
    NF4 divide os 16 níveis de quantização de forma ÓTIMA para esta
    distribuição, colocando mais níveis perto de zero (onde há mais pesos)
    e menos nos extremos:
    
        INT4 uniforme: [-8, -7, ..., 0, ..., 6, 7]  (espaçamento igual)
        NF4 ótimo:     [-1.0, -0.7, -0.5, -0.3, -0.1, 0.0, 0.1, 0.3, ...] 
                       (mais resolução perto de zero!)
    
    Resultado: NF4 mantém ~99% da informação vs FP16, enquanto
    INT4 uniforme perde ~3-5% (Theorem 1, Dettmers et al.).
    
    DOUBLE QUANTIZATION:
    ────────────────────
    Além de quantizar os pesos, QLoRA também quantiza os FATORES DE ESCALA
    usados na quantização (quantização da quantização!).
    
    Sem double quant: Cada grupo de 64 pesos precisa de 1 float32 (4 bytes)
    Com double quant: Esse float32 é quantizado para int8 (1 byte)
    Economia: ~0.37 bits/param adicionais
    
    Args:
        config: Dicionário de configuração com seção 'qlora'.
    
    Returns:
        BitsAndBytesConfig para quantização 4-bit.
    """
    qlora_cfg = config.get("qlora", {})
    
    # Mapear dtype string para torch type
    compute_dtype_str = qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    compute_dtype = dtype_map.get(compute_dtype_str, torch.bfloat16)
    
    bnb_config = BitsAndBytesConfig(
        # Ativar quantização 4-bit
        load_in_4bit=qlora_cfg.get("load_in_4bit", True),
        
        # NF4 vs FP4:
        # - "nf4": NormalFloat 4, ótimo para pesos de redes (RECOMENDADO)
        # - "fp4": Float ponto flutuante 4-bit (menos preciso para pesos)
        bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
        
        # Tipo de dado para computação durante o forward pass
        # Mesmo que os pesos estejam em 4-bit, a COMPUTAÇÃO é feita em bf16
        # para manter precisão numérica (dequantiza → computa → requantiza)
        bnb_4bit_compute_dtype=compute_dtype,
        
        # Double quantization: quantizar os fatores de escala da quantização
        # Economiza ~0.37 bits/param com impacto negligível na qualidade
        bnb_4bit_use_double_quant=qlora_cfg.get("bnb_4bit_use_double_quant", True),
    )
    
    logger.info(f"BitsAndBytes Config: quant_type={bnb_config.bnb_4bit_quant_type}, "
                f"compute_dtype={compute_dtype}, "
                f"double_quant={bnb_config.bnb_4bit_use_double_quant}")
    
    return bnb_config


def train_sft_qlora(
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Executa o pipeline completo de SFT com QLoRA.
    
    DIFERENÇAS PARA sft_lora.py:
    ──────────────────────────────
    1. Modelo carregado em 4-bit com BitsAndBytesConfig
    2. prepare_model_for_kbit_training() é chamado ANTES do LoRA
    3. Uso de VRAM ~50% menor que LoRA padrão
    4. Velocidade de treinamento similar (compute dtype = bf16)
    
    QUANDO USAR QLoRA vs LoRA?
    ────────────────────────────
    - GPU com ≥16GB VRAM: Use LoRA (sft_lora.py) — melhor precisão
    - GPU com 8-12GB VRAM: Use QLoRA (este script) — funcional
    - GPU com <8GB: Use QLoRA com gradient_checkpointing=True
    - Sem GPU: Não é viável para treinamento (use Google Colab)
    
    Args:
        config: Dicionário de configuração.
        output_dir: Diretório de saída.
    
    Returns:
        str: Caminho onde o modelo foi salvo.
    """
    # ==========================================================================
    # 1. CONFIGURAÇÃO
    # ==========================================================================
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
    logger.info("🚀 INICIANDO SFT COM QLoRA (4-bit)")
    logger.info(f"   Modelo: {model_id}")
    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 60)
    
    # ==========================================================================
    # 2. TOKENIZADOR
    # ==========================================================================
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # ==========================================================================
    # 3. CARREGAR MODELO EM 4-BIT (a grande diferença!)
    # ==========================================================================
    logger.info("\n🧠 Carregando modelo em 4-bit com bitsandbytes...")
    
    bnb_config = create_bnb_config(config)
    vram_before = get_vram_usage_gb()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,   # ← Quantização 4-bit aqui!
        device_map=model_cfg.get("device_map", "auto"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    
    vram_after = get_vram_usage_gb()
    logger.info(f"   VRAM após carregamento 4-bit: {vram_after:.3f} GB "
                f"(+{vram_after - vram_before:.3f} GB)")
    logger.info(f"   💡 Compare: em BF16 seriam ~{get_model_size_mb(model) * 4 / 1024:.1f} GB")
    
    # ==========================================================================
    # 4. PREPARAR MODELO PARA TREINAMENTO EM K-BIT
    # ==========================================================================
    # Esta função faz ajustes ESSENCIAIS para treinar um modelo quantizado:
    # 1. Converte layer norms para float32 (precisão necessária para estabilidade)
    # 2. Ativa gradient checkpointing (reduz VRAM ao custo de velocidade)
    # 3. Habilita input embeddings para receber gradientes
    logger.info("\n🔧 Preparando modelo para treinamento em 4-bit...")
    model = prepare_model_for_kbit_training(model)
    
    # ==========================================================================
    # 5. APLICAR LoRA (sobre o modelo quantizado)
    # ==========================================================================
    logger.info("\n🔧 Aplicando adaptadores LoRA...")
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
    logger.info(f"   Parâmetros treináveis: {params['trainable']:,} "
                f"({params['trainable_percent']:.2f}%)")
    
    # ==========================================================================
    # 6. DATASET
    # ==========================================================================
    logger.info("\n📦 Preparando dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_cfg.get("name", "tatsu-lab/alpaca"),
        max_samples=dataset_cfg.get("max_samples", 5000),
        test_size=dataset_cfg.get("test_size", 0.1),
        seed=dataset_cfg.get("seed", 42),
    )
    
    # ==========================================================================
    # 7. TRAINING ARGUMENTS
    # ==========================================================================
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
        save_steps=sft_cfg.get("save_steps", 200),
        eval_strategy="steps",
        eval_steps=sft_cfg.get("eval_steps", 200),
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        seed=42,
    )
    
    # ==========================================================================
    # 8. TREINAR
    # ==========================================================================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    logger.info("\n🔥 INICIANDO TREINAMENTO QLoRA...")
    train_result = trainer.train()
    
    # ==========================================================================
    # 9. SALVAR
    # ==========================================================================
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"\n📊 RESULTADOS QLoRA:")
    logger.info(f"   Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"   Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"   VRAM pico: {get_vram_usage_gb():.3f} GB")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ SFT COM QLoRA CONCLUÍDO!")
    logger.info(f"{'='*60}")
    
    return output_dir


def main():
    config = load_config()
    train_sft_qlora(config=config)


if __name__ == "__main__":
    main()
