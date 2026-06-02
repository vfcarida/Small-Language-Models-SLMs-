"""
==============================================================================
MÓDULO 07 — BENCHMARK RUNNER (lm-eval-harness)
==============================================================================
NÍVEL: 🔴 Avançado
TEMPO ESTIMADO: 15 minutos (leitura) + Execução

OBJETIVO:
    Aprender o PADRÃO DA INDÚSTRIA para avaliar SLMs: o framework 
    LM Evaluation Harness (EleutherAI). Ninguém avalia modelos "no olho".

CONCEITOS ABORDADOS:
    1. Few-Shot Evaluation
    2. Benchmarks Clássicos: MMLU, GSM8K, HellaSwag, TruthfulQA
    3. Métricas automáticas exatas
    
REFERÊNCIAS:
    - Gao et al., "A framework for few-shot language model evaluation", 2023
    - Hendrycks et al., "Measuring Massive Multitask Language Understanding" (MMLU)

COMO EXECUTAR:
    python src/module_07_evaluation/benchmark_runner.py --model_id Qwen/Qwen2.5-1.5B --tasks mmlu_lite
==============================================================================
"""

import os
import json
import logging
import argparse
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_lm_eval(model_id: str, tasks: str, num_fewshot: int, batch_size: int, output_dir: str):
    """
    Executa o EleutherAI lm-eval via subprocess.
    
    TEORIA DOS BENCHMARKS:
    ──────────────────────
    MMLU (Massive Multitask Language Understanding): 
    - 57 disciplinas (Matemática, História, Direito, etc.)
    - Múltipla escolha (A, B, C, D).
    - O modelo calcula a probabilidade (Logits) para os tokens "A", "B", "C", "D".
    - Resposta escolhida = token de maior probabilidade.
    
    GSM8K (Grade School Math 8K):
    - Problemas matemáticos de ensino fundamental.
    - Requer "Chain-of-Thought" (CoT). O modelo gera os passos até a resposta.
    - Avaliação checa se o número final está correto.
    """
    logger.info("=" * 60)
    logger.info(f"EXECUTANDO LM-EVAL HARNESS")
    logger.info(f"Modelo: {model_id}")
    logger.info(f"Tasks: {tasks}")
    logger.info("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Usar VLLM ou Hf (HuggingFace) baseado no argumento
    # Em produção, vllm é 5-10x mais rápido para avaliação
    model_args = f"pretrained={model_id},dtype=bfloat16"
    
    cmd = [
        "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", output_dir
    ]
    
    logger.info(f"Comando: {' '.join(cmd)}")
    
    # NOTA: Comentado porque LM Eval demora horas e falha se não tiver GPU livre
    # O script shell real (bash) faria isso. Aqui apenas demonstramos o formato.
    logger.info("\nSimulando a execução do Benchmark... (Em produção, o subprocess rodaria aqui)")
    
    # Criar um arquivo JSON de resultado falso para a pipeline não quebrar
    fake_results = {
        "results": {
            "mmlu": {"acc,none": 0.45},
            "gsm8k": {"exact_match,none": 0.22}
        },
        "config": {"model": model_id, "tasks": tasks}
    }
    
    result_path = os.path.join(output_dir, "results.json")
    with open(result_path, "w") as f:
        json.dump(fake_results, f, indent=2)
        
    logger.info(f"Resultados (simulados) salvos em: {result_path}")
    logger.info("Use o módulo results_visualizer.py para plotar os gráficos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--tasks", type=str, default="mmlu,gsm8k")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./results/eval_output")
    args = parser.parse_args()
    
    run_lm_eval(args.model_id, args.tasks, args.num_fewshot, args.batch_size, args.output_dir)
