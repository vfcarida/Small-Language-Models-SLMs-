"""
==============================================================================
MÓDULO 07 — AVALIAÇÃO DE PERPLEXIDADE E LATÊNCIA
==============================================================================
NÍVEL: 🟡 Intermediário
TEMPO ESTIMADO: 15 minutos

OBJETIVO:
    Aprender como quantificar a "confusão" (Perplexidade) de um modelo
    em textos específicos e como medir sua velocidade (Latência).

CONCEITOS ABORDADOS:
    1. Cross-Entropy Loss em sequências
    2. Sliding Window Perplexity (para textos > Context Window)
    3. Time-To-First-Token (TTFT)
    
REFERÊNCIAS:
    - Jurafsky & Martin, "Speech and Language Processing", 2024
    - HuggingFace Evaluate Library

COMO EXECUTAR:
    python src/module_07_evaluation/perplexity_eval.py
==============================================================================
"""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.metrics import calculate_perplexity, measure_inference_latency

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("MÓDULO 07: PERPLEXIDADE E LATÊNCIA")
    logger.info("=" * 60)
    
    model_id = "Qwen/Qwen2.5-1.5B"
    
    logger.info(f"Carregando {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    
    # TEXTOS DE TESTE
    texto_facil = "O céu é azul e a grama é verde. O sol brilha durante o dia."
    texto_dificil = "A mecânica quântica descreve sobreposições de autoestados de Hamiltonianos hermitianos em espaços de Hilbert."
    texto_nonsense = "Sapato voa porque abacaxi não sabe matemática quântica do pneu."
    
    # 1. PERPLEXIDADE
    logger.info("\n📊 TESTANDO PERPLEXIDADE (Quão confuso o modelo está?)")
    logger.info("   Valores Menores = Melhor (1 = perfeição impossível)")
    
    for nome, texto in [("Fácil", texto_facil), ("Difícil", texto_dificil), ("Absurdo", texto_nonsense)]:
        ppl = calculate_perplexity(model, tokenizer, texto)
        logger.info(f"   [{nome}]: PPL = {ppl:.2f} | Texto: '{texto[:40]}...'")
        
    logger.info("\n   Conclusão: O modelo acha o texto normal muito provável (ppl baixa),")
    logger.info("   mas textos confusos/aleatórios são improváveis (ppl alta).")
    
    # 2. LATÊNCIA
    logger.info("\n⏱️ TESTANDO LATÊNCIA E THROUGHPUT")
    prompt = "Explique a teoria da relatividade em 3 parágrafos:"
    
    metrics = measure_inference_latency(
        model, tokenizer, prompt, max_new_tokens=128, num_warmup=1, num_runs=3
    )
    
    logger.info(f"   Prompt: {prompt}")
    logger.info(f"   Time To First Token (TTFT): {metrics['ttft_ms']:.1f} ms")
    logger.info(f"   Throughput: {metrics['throughput_tokens_per_sec']:.1f} tokens/s")
    
    if metrics['throughput_tokens_per_sec'] < 20:
        logger.info("   💡 Dica: < 20 t/s é lento para interação humana.")
        logger.info("      Solução: AWQ, vLLM, ou comprar GPUs melhores!")

if __name__ == "__main__":
    main()
