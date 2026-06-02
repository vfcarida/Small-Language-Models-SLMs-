"""
==============================================================================
PIPELINE END-TO-END — ENTREGA DO PROJETO SLM
==============================================================================
NÍVEL: 🟣 Produção
TEMPO ESTIMADO DE EXECUÇÃO: ~4 horas (em GPU de 8GB)

OBJETIVO:
    Este script orquestra toda a entrega do projeto. Ele une os módulos
    individuais em um fluxo de trabalho único e contínuo:
    
    1. Setup e Validação de Ambiente
    2. Avaliação Zero-Shot do Modelo Base (Baseline)
    3. Treinamento SFT + LoRA no dataset Alpaca
    4. Merge do Adapter
    5. Avaliação do Modelo Finetuned
    6. Geração dos Gráficos de Resultado
    
COMO EXECUTAR:
    python pipeline/run_full_pipeline.py
==============================================================================
"""

import os
import sys
import logging
import time

# Adicionar a raiz do projeto ao path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.utils.config import load_config, ensure_output_dirs
from src.utils.metrics import get_vram_usage_gb
from src.module_02_sft.sft_lora import train_sft_lora, merge_lora_adapter
from src.module_07_evaluation.benchmark_runner import run_lm_eval
from src.module_07_evaluation.results_visualizer import plot_training_loss, plot_benchmark_radar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 80)
    logger.info("🚀 INICIANDO PIPELINE END-TO-END: ENTREGA DO SLM".center(80))
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # --------------------------------------------------------------------------
    # FASE 0: SETUP E CONFIGURAÇÃO
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 0/5] SETUP DE AMBIENTE")
    config = load_config()
    ensure_output_dirs(config)
    
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    logger.info(f"Modelo Alvo: {model_id}")
    logger.info(f"VRAM Inicial: {get_vram_usage_gb():.2f} GB")
    
    # --------------------------------------------------------------------------
    # FASE 1: AVALIAÇÃO BASELINE (O que o modelo já sabe?)
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 1/5] AVALIAÇÃO DO MODELO BASE")
    baseline_dir = "./results/eval_baseline"
    run_lm_eval(
        model_id=model_id,
        tasks="mmlu,gsm8k",
        num_fewshot=0,
        batch_size=4,
        output_dir=baseline_dir
    )
    
    # --------------------------------------------------------------------------
    # FASE 2: TREINAMENTO (O coração do projeto)
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 2/5] TREINAMENTO (SFT + LoRA)")
    logger.info("Iniciando treinamento no dataset Alpaca-GPT4. Isso vai demorar...")
    
    adapter_dir = train_sft_lora(config=config)
    
    logger.info(f"Treinamento concluído. Adapters salvos em: {adapter_dir}")
    
    # --------------------------------------------------------------------------
    # FASE 3: MERGE (Preparando para deploy)
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 3/5] MERGE DO ADAPTER COM MODELO BASE")
    merged_dir = "./results/sft_merged_model"
    
    merge_lora_adapter(
        base_model_id=model_id,
        adapter_path=adapter_dir,
        output_dir=merged_dir
    )
    
    # --------------------------------------------------------------------------
    # FASE 4: AVALIAÇÃO PÓS-TREINO (Melhoramos?)
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 4/5] AVALIAÇÃO DO MODELO FINETUNED")
    eval_finetuned_dir = "./results/eval_finetuned"
    
    run_lm_eval(
        model_id=merged_dir, # Apontando para o nosso modelo local!
        tasks="mmlu,gsm8k",
        num_fewshot=0,
        batch_size=4,
        output_dir=eval_finetuned_dir
    )
    
    # --------------------------------------------------------------------------
    # FASE 5: VISUALIZAÇÃO E RELATÓRIO
    # --------------------------------------------------------------------------
    logger.info("\n[FASE 5/5] GERAÇÃO DE GRÁFICOS")
    
    os.makedirs("./results/figures", exist_ok=True)
    
    plot_training_loss(
        metrics_path=os.path.join(adapter_dir, "trainer_state.json"),
        output_path="./results/figures/training_loss.png"
    )
    
    plot_benchmark_radar(
        results_path=os.path.join(eval_finetuned_dir, "results.json"),
        output_path="./results/figures/benchmark_radar.png"
    )
    
    elapsed = time.time() - start_time
    hours, rem = divmod(elapsed, 3600)
    minutes, seconds = divmod(rem, 60)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"✅ PIPELINE CONCLUÍDA COM SUCESSO!".center(80))
    logger.info(f"Tempo total: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("=" * 80)
    logger.info("\nPróximo passo: Execute 'python pipeline/generate_report.py' para gerar")
    logger.info("o relatório executivo em Markdown e PDF para sua apresentação.")


if __name__ == "__main__":
    main()
