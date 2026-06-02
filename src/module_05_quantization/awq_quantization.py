"""
==============================================================================
MÓDULO 05 — AWQ QUANTIZATION (Activation-aware Weight Quantization)
==============================================================================
NÍVEL: 🔴 Avançado
TEMPO ESTIMADO: 30 minutos (leitura) + compilação (lento)

OBJETIVO:
    Diferente do bitsandbytes que quantiza dinamicamente (lento para inferência),
    o AWQ cria um modelo estático de 4-bits com kernels CUDA super otimizados.
    É o PADRÃO DA INDÚSTRIA para deploy em produção de SLMs no edge.

CONCEITOS ABORDADOS:
    1. Salient Weights e Outliers
    2. Scaling channels em vez de quantizar
    3. W_bit, Group Size e Zero Point
    
REFERÊNCIAS:
    - Lin et al., "AWQ: Activation-aware Weight Quantization for LLM 
      Compression and Acceleration", arXiv:2306.00978, 2023

COMO EXECUTAR:
    python src/module_05_quantization/awq_quantization.py
==============================================================================
"""

import logging
import os
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

from src.utils.config import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="AWQ CLI")
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/awq_model")
    return parser.parse_args()


def run_awq_compilation(model_id: str, output_dir: str, config: dict):
    """
    Executa a quantização Activation-aware.
    
    TEORIA MATEMÁTICA — AWQ
    ───────────────────────
    1% dos pesos de uma rede são extremamente "salientes" (importantes).
    Se você aplicar quantização INT4 clássica neles (como Round to Nearest),
    o erro numérico destrói o conhecimento do modelo.
    
    AWQ não encontra esses pesos olhando para os pesos em si (|W|), mas 
    olhando para as ATIVAÇÕES (|X|) quando dados passam por eles.
    
    A Mágica:
    Em vez de tentar guardar o peso em FP16 (o que quebraria o formato misto),
    o AWQ aplica um FATOR DE ESCALA (S) aos canais salientes ANTES da quantização,
    e o inverso (1/S) nas ativações durante a inferência.
    Isso joga os pesos para uma zona da curva de quantização onde o erro 
    de arredondamento é matematicamente minimizado.
    """
    logger.info("=" * 60)
    logger.info(f"COMPILANDO AWQ INT4 PARA: {model_id}")
    logger.info("=" * 60)
    
    quant_cfg = config.get("quantization", {})
    
    # 1. CONFIGURAÇÃO AWQ
    quant_config = {
        "zero_point": quant_cfg.get("zero_point", True),
        "q_group_size": quant_cfg.get("q_group_size", 128),
        "w_bit": quant_cfg.get("w_bit", 4),
        "version": quant_cfg.get("version", "GEMM")
    }
    
    logger.info(f"Configuração: {quant_config}")
    
    try:
        # 2. CARREGAR MODELO
        logger.info("\nCarregando modelo original (requer bastante VRAM)...")
        model = AutoAWQForCausalLM.from_pretrained(model_id, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # 3. QUANTIZAR
        logger.info("\nIniciando matemática Activation-Aware (isso vai demorar)...")
        # O AWQ passa texto pelo modelo para achar os salient weights
        model.quantize(tokenizer, quant_config=quant_config)
        
        # 4. SALVAR
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"\nSalvando modelo INT4 super-rápido em: {output_dir}")
        model.save_quantized(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        logger.info("✅ AWQ COMPLETO! Modelo pronto para vLLM/TGI.")
        
    except Exception as e:
        logger.error(f"Falha no AWQ: {e}")
        logger.error("Dica: Certifique-se de que a biblioteca autoawq está compilada para seu CUDA.")


def main():
    config = load_config()
    args = parse_args()
    
    model_id = args.model_id or config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    run_awq_compilation(model_id, args.output_dir, config)

if __name__ == "__main__":
    main()
