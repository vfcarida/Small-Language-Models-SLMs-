"""
==============================================================================
MÓDULO 05 — QUANTIZAÇÃO ON-THE-FLY (bitsandbytes)
==============================================================================
NÍVEL: 🟢 Iniciante
TEMPO ESTIMADO: 15 minutos

OBJETIVO:
    Demonstrar a quantização dinâmica em 8-bit e 4-bit usando bitsandbytes.
    É a forma mais rápida e fácil de comprimir um modelo, pois ocorre no
    momento do carregamento (on-the-fly) e não requer GPUs de alto nível.

CONCEITOS ABORDADOS:
    1. Float16 vs Int8 vs NF4
    2. Trade-off: Menos VRAM mas inferência levemente mais lenta.
    
REFERÊNCIAS:
    - Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for 
      Transformers at Scale", 2022

COMO EXECUTAR:
    python src/module_05_quantization/bitsandbytes_quant.py
==============================================================================
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.metrics import get_vram_usage_gb, get_model_size_mb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_quantization(model_id: str, bit_target: int):
    """
    Carrega o modelo na precisão solicitada e mede VRAM.
    """
    logger.info(f"\n{'-'*40}")
    logger.info(f"CARREGANDO EM {bit_target}-BIT")
    logger.info(f"{'-'*40}")
    
    vram_before = get_vram_usage_gb()
    
    if bit_target == 16:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
    elif bit_target == 8:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", load_in_8bit=True
        )
    elif bit_target == 4:
        # Usando NF4 que é superior ao INT4 padrão
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=quant_config
        )
        
    vram_after = get_vram_usage_gb()
    size_mb = get_model_size_mb(model)
    
    logger.info(f"VRAM Consumida: {vram_after - vram_before:.2f} GB")
    logger.info(f"Tamanho na RAM: {size_mb/1024:.2f} GB")
    
    del model
    torch.cuda.empty_cache()


def main():
    logger.info("=" * 60)
    logger.info("MÓDULO 05: BITSANDBYTES QUANTIZATION (ON-THE-FLY)")
    logger.info("=" * 60)
    
    model_id = "Qwen/Qwen2.5-1.5B"
    
    if not torch.cuda.is_available():
        logger.error("ERRO: Quantização bitsandbytes requer GPU.")
        return
        
    # 1. Baseline FP16
    evaluate_quantization(model_id, 16)
    
    # 2. INT8 Quantization
    evaluate_quantization(model_id, 8)
    
    # 3. NF4 Quantization (4-bit)
    evaluate_quantization(model_id, 4)
    
    logger.info("\n💡 NOTA: bitsandbytes é perfeito para uso imediato.")
    logger.info("Para produção com máxima velocidade (kernels otimizados),")
    logger.info("usamos AWQ. Veja awq_quantization.py!")

if __name__ == "__main__":
    main()
