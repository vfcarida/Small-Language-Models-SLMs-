"""
==============================================================================
MÓDULO 02 — SFT COM LoRA (Low-Rank Adaptation)
==============================================================================
NÍVEL: 🟡 Intermediário
TEMPO ESTIMADO: 30 minutos (leitura) + 1-4 horas (treinamento)

OBJETIVO:
    Treinar um SLM para seguir instruções usando Supervised Fine-Tuning (SFT)
    com LoRA — a técnica de fine-tuning mais eficiente da indústria.

POR QUE ESTE MÓDULO É ESSENCIAL:
    SFT + LoRA é o pipeline padrão (#1) da indústria para adaptar modelos.
    Sem SFT, o modelo base apenas "completa texto" — não segue instruções.
    Com SFT, ele aprende a responder perguntas, classificar, extrair dados, etc.

CONCEITOS ABORDADOS:
    1. SFT (Supervised Fine-Tuning): treinar com pares input→output
    2. LoRA: treinar apenas ~1% dos parâmetros (Parameter-Efficient)
    3. SFTTrainer do TRL: pipeline completo de treinamento
    4. Merge de adapters: combinar LoRA com o modelo base
    5. Avaliação pós-treinamento

REFERÊNCIAS:
    - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
      arXiv:2106.09685, 2021
    - TRL Documentation, "SFTTrainer", Hugging Face, 2024
    - Dettmers et al., "QLoRA", arXiv:2305.14314, 2023 (para sft_qlora.py)

COMO EXECUTAR:
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
from src.utils.metrics import count_parameters, get_model_size_mb, get_vram_usage_gb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_lora_config(config: dict) -> LoraConfig:
    """
    Cria a configuração LoRA a partir dos hiperparâmetros centralizados.
    
    TEORIA — LoRA (Low-Rank Adaptation)
    =====================================
    LoRA é baseada em uma observação matemática fundamental:
    
    "As matrizes de peso de modelos pré-treinados possuem um rank intrínseco
     baixo — ou seja, a maior parte da informação pode ser capturada por
     uma decomposição de baixo rank."
    
    Em vez de atualizar a matriz de pesos W completa (d × d), LoRA
    adiciona duas matrizes menores A e B:
    
        W_new = W_original + (B × A)
    
    Onde:
    - W_original: Pesos originais (CONGELADOS, não atualizam)
    - A: Matriz de projeção DOWN (d × r) onde r << d
    - B: Matriz de projeção UP (r × d)
    - r: "Rank" — dimensão do bottleneck (tipicamente 8-64)
    
    ECONOMIA DE PARÂMETROS:
    ────────────────────────
    Exemplo com d=2048 e r=16:
    - Full fine-tuning: 2048 × 2048 = 4,194,304 parâmetros
    - LoRA: (2048 × 16) + (16 × 2048) = 65,536 parâmetros
    - Redução: 98.4% menos parâmetros treináveis!
    
    HIPERPARÂMETROS CHAVE:
    ──────────────────────
    r (rank):
        - r=8: Mínimo viável. Bom para tarefas simples.
        - r=16: Sweet spot para a maioria dos casos. ← RECOMENDADO
        - r=64: Mais expressivo, mas perde eficiência.
        - r=256: Quase tão caro quanto full fine-tuning.
    
    lora_alpha:
        - Fator de escala. O scaling real é alpha/r.
        - alpha=32 com r=16 → scaling=2.0 (padrão recomendado).
        - Scaling muito alto causa instabilidade; muito baixo limita aprendizado.
    
    target_modules:
        - "all-linear": Aplica LoRA em TODAS as camadas lineares.
          Hu et al. (2021) mostrou que restringir a q,v é subótimo.
          Dettmers et al. (2023) confirmou: all-linear ≈ full fine-tuning.
    
    Args:
        config: Dicionário de configuração com seção 'lora'.
    
    Returns:
        LoraConfig configurado.
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
    
    logger.info(f"LoRA Config: r={lora_config.r}, alpha={lora_config.lora_alpha}, "
                f"dropout={lora_config.lora_dropout}, "
                f"target={lora_config.target_modules}")
    
    return lora_config


def train_sft_lora(
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
) -> str:
    """
    Executa o pipeline completo de SFT com LoRA.
    
    PIPELINE:
        1. Carregar configuração
        2. Carregar modelo base em BF16
        3. Aplicar adaptadores LoRA
        4. Preparar dataset
        5. Treinar com SFTTrainer
        6. Salvar modelo (adapter weights)
        7. (Opcional) Merge adapter + base model
    
    CONCEITO EDUCACIONAL — SFTTrainer
    ====================================
    O SFTTrainer (do TRL) é um wrapper sobre o Trainer do Transformers
    que simplifica o treinamento de modelos de linguagem para seguir instruções.
    
    Ele automaticamente:
    - Formata o campo "text" como input + labels para language modeling
    - Aplica o chat template do tokenizador se configurado
    - Faz packing de sequências para eficiência (múltiplas amostras em 1 seq)
    - Calcula loss apenas sobre os tokens de resposta (masking)
    
    Args:
        config: Dicionário de configuração. Se None, carrega do arquivo.
        output_dir: Diretório de saída. Se None, usa o da configuração.
    
    Returns:
        str: Caminho do diretório onde o modelo foi salvo.
    """
    # ==========================================================================
    # 1. CARREGAR CONFIGURAÇÃO
    # ==========================================================================
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
    logger.info("🚀 INICIANDO SFT COM LoRA")
    logger.info(f"   Modelo: {model_id}")
    logger.info(f"   Output: {output_dir}")
    logger.info("=" * 60)
    
    # ==========================================================================
    # 2. CARREGAR TOKENIZADOR
    # ==========================================================================
    logger.info("\n📝 Carregando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # CORREÇÃO CRÍTICA: pad_token
    # Muitos modelos modernos (Qwen, Llama, Mistral) NÃO definem pad_token.
    # Sem ele, o DataCollator não consegue criar batches com padding.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"   ⚠️ pad_token não definido. Usando eos_token: '{tokenizer.eos_token}'")
    
    # ==========================================================================
    # 3. CARREGAR MODELO BASE
    # ==========================================================================
    logger.info("\n🧠 Carregando modelo base...")
    vram_before = get_vram_usage_gb()
    
    # Mapear dtype string para torch type
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
    logger.info(f"   Modelo carregado. VRAM: {vram_after:.3f} GB (+{vram_after - vram_before:.3f} GB)")
    
    # ==========================================================================
    # 4. APLICAR LoRA
    # ==========================================================================
    logger.info("\n🔧 Aplicando adaptadores LoRA...")
    lora_config = create_lora_config(config)
    
    # get_peft_model() congela os pesos originais e adiciona os adaptadores
    model = get_peft_model(model, lora_config)
    
    # Reportar parâmetros
    params = count_parameters(model)
    logger.info(f"   Parâmetros treináveis: {params['trainable']:,} "
                f"({params['trainable_percent']:.2f}% do total)")
    logger.info(f"   Parâmetros congelados: {params['frozen']:,}")
    
    # ==========================================================================
    # 5. PREPARAR DATASET
    # ==========================================================================
    logger.info("\n📦 Preparando dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_cfg.get("name", "tatsu-lab/alpaca"),
        split=dataset_cfg.get("split", "train"),
        max_samples=dataset_cfg.get("max_samples", 5000),
        test_size=dataset_cfg.get("test_size", 0.1),
        seed=dataset_cfg.get("seed", 42),
        format_fn="alpaca",
    )
    
    # ==========================================================================
    # 6. CONFIGURAR TREINAMENTO
    # ==========================================================================
    logger.info("\n⚙️ Configurando TrainingArguments...")
    
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
    
    # ==========================================================================
    # 7. INICIALIZAR SFTTrainer
    # ==========================================================================
    logger.info("\n🏋️ Inicializando SFTTrainer...")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # O SFTTrainer automaticamente usa o campo "text" do dataset
        # para criar input_ids e labels para language modeling causal
    )
    
    # ==========================================================================
    # 8. TREINAR!
    # ==========================================================================
    logger.info("\n🔥 INICIANDO TREINAMENTO...")
    logger.info(f"   Épocas: {training_args.num_train_epochs}")
    logger.info(f"   Batch efetivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"   Learning rate: {training_args.learning_rate}")
    logger.info(f"   Steps por época: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    
    train_result = trainer.train()
    
    # ==========================================================================
    # 9. SALVAR MODELO E MÉTRICAS
    # ==========================================================================
    logger.info("\n💾 Salvando modelo e métricas...")
    
    # Salvar adapter weights (apenas os pesos LoRA, muito pequenos ~50MB)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Salvar métricas de treinamento
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Avaliação final
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    logger.info(f"\n📊 RESULTADOS DO TREINAMENTO:")
    logger.info(f"   Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    logger.info(f"   Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    logger.info(f"   Modelo salvo em: {output_dir}")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ SFT COM LoRA CONCLUÍDO!")
    logger.info("📚 Próximo passo: Fazer merge do adapter ou executar sft_qlora.py")
    logger.info(f"{'='*60}")
    
    return output_dir


def merge_lora_adapter(
    base_model_id: str,
    adapter_path: str,
    output_dir: str = "./results/sft_merged_model",
) -> str:
    """
    Faz o merge dos pesos LoRA com o modelo base.
    
    CONCEITO EDUCACIONAL — POR QUE FAZER MERGE?
    ==============================================
    Após o treinamento com LoRA, você tem DOIS conjuntos de pesos:
    1. Modelo base original (congelado, ~3 GB)
    2. Adapter LoRA (apenas os pesos novos, ~50 MB)
    
    Opções de deploy:
    
    A) MANTER SEPARADOS (Multi-Adapter):
       ✅ Memória: Carrega 1 base + múltiplos adapters para diferentes tarefas
       ✅ Flexibilidade: Troca de adapter em tempo real (sem recarregar)
       ❌ Latência: Ligeiramente mais lento (soma durante forward pass)
    
    B) MERGE (Combinar em 1 modelo):
       ✅ Velocidade: Inferência nativa, sem overhead de soma
       ✅ Simplicidade: 1 modelo = 1 diretório
       ❌ Memória: Cada task precisa de um modelo completo separado
    
    Para ENTREGA INICIAL, merge é recomendado (simplicidade).
    
    Args:
        base_model_id: ID do modelo base no HuggingFace.
        adapter_path: Caminho do adapter LoRA salvo.
        output_dir: Onde salvar o modelo merged.
    
    Returns:
        str: Caminho do modelo merged.
    """
    logger.info(f"{'='*60}")
    logger.info("MERGE: Combinando LoRA Adapter com Modelo Base")
    logger.info(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Carregar modelo base
    logger.info(f"Carregando modelo base: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Carregar adapter LoRA
    logger.info(f"Carregando adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Merge: W_new = W_base + B @ A (operação matemática permanente)
    logger.info("Executando merge (W_new = W_base + B @ A)...")
    model = model.merge_and_unload()
    
    # Salvar modelo merged
    logger.info(f"Salvando modelo merged em: {output_dir}")
    model.save_pretrained(output_dir)
    
    # Salvar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    
    size_mb = get_model_size_mb(model)
    logger.info(f"✅ Merge concluído! Tamanho: {size_mb:.1f} MB")
    
    return output_dir


def main():
    """
    Executa o pipeline completo de SFT com LoRA.
    """
    # Carregar configuração centralizada
    config = load_config()
    
    # Executar treinamento
    output_dir = train_sft_lora(config=config)
    
    # Merge (opcional — descomentar para produção)
    # model_id = config["model"]["base_model_id"]
    # merge_lora_adapter(model_id, output_dir, "./results/sft_merged_model")


if __name__ == "__main__":
    main()
