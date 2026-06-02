"""
==============================================================================
MÓDULO 01 — INFERÊNCIA BÁSICA COM SMALL LANGUAGE MODELS
==============================================================================
NÍVEL: 🟢 Iniciante
TEMPO ESTIMADO: 15 minutos

OBJETIVO:
    Aprender a carregar um SLM do Hugging Face Hub e gerar texto usando
    diferentes estratégias de decodificação (greedy, sampling, top-k, top-p).

CONCEITOS ABORDADOS:
    1. Como carregar um modelo pré-treinado com AutoModelForCausalLM
    2. Estratégias de decodificação e seus trade-offs
    3. Medição de uso de memória (VRAM/RAM)
    4. Geração com diferentes temperaturas

PRÉ-REQUISITOS:
    pip install transformers torch accelerate

REFERÊNCIAS:
    - Holtzman et al., "The Curious Case of Neural Text Degeneration", 2020
      (Paper sobre top-p/nucleus sampling)
    - HuggingFace, "How to generate text", 2024

COMO EXECUTAR:
    python src/module_01_inference/basic_inference.py
==============================================================================
"""

import logging
import time
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.metrics import (
    count_parameters,
    get_model_size_mb,
    get_vram_usage_gb,
    measure_inference_latency,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(
    model_id: str = "Qwen/Qwen2.5-1.5B",
    torch_dtype: str = "bfloat16",
    device_map: str = "auto",
) -> tuple:
    """
    Carrega um Small Language Model do Hugging Face Hub.
    
    CONCEITO EDUCACIONAL — CARREGAMENTO DE MODELOS
    =================================================
    AutoModelForCausalLM.from_pretrained() faz automaticamente:
    1. Download dos pesos do modelo (se não estiver em cache)
    2. Mapeamento dos pesos para as camadas da arquitetura
    3. Distribuição entre dispositivos (GPU/CPU) via device_map
    
    OPÇÕES IMPORTANTES:
    
    torch_dtype — Precisão numérica dos pesos:
    ─────────────────────────────────────────────
    - float32 (FP32): Máxima precisão, 4 bytes/param. Modelo 1.5B ≈ 6 GB
    - float16 (FP16): Boa precisão, 2 bytes/param. Modelo 1.5B ≈ 3 GB
      ⚠️ Pode causar NaN em some architectures (overflow no exp())
    - bfloat16 (BF16): Mesma memória que FP16, mas range maior.
      ✅ RECOMENDADO para Qwen, Llama, Phi. Previne NaN.
    
    device_map — Onde colocar os pesos:
    ─────────────────────────────────────
    - "auto": Distribui automaticamente entre GPUs disponíveis
    - "cpu": Força tudo para CPU (lento, mas funciona sem GPU)
    - "cuda:0": Força para uma GPU específica
    
    Args:
        model_id: ID do modelo no Hugging Face Hub.
        torch_dtype: Tipo numérico ("float32", "float16", "bfloat16").
        device_map: Estratégia de distribuição.
    
    Returns:
        Tupla (model, tokenizer) prontos para uso.
    """
    logger.info(f"{'='*60}")
    logger.info(f"CARREGANDO MODELO: {model_id}")
    logger.info(f"{'='*60}")
    
    # Mapear string para tipo PyTorch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)
    
    # Medir VRAM antes do carregamento
    vram_before = get_vram_usage_gb()
    
    # --------------------------------------------------------------------------
    # CARREGAR TOKENIZADOR
    # --------------------------------------------------------------------------
    logger.info("Carregando tokenizador...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    # FIX CRÍTICO: Definir pad_token se não existir
    # Sem isso, batching de sequências de tamanhos diferentes FALHA
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"pad_token não definido. Usando eos_token: '{tokenizer.eos_token}'")
    
    # --------------------------------------------------------------------------
    # CARREGAR MODELO
    # --------------------------------------------------------------------------
    logger.info(f"Carregando modelo em {torch_dtype}...")
    start_time = time.perf_counter()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    
    load_time = time.perf_counter() - start_time
    vram_after = get_vram_usage_gb()
    
    # --------------------------------------------------------------------------
    # REPORTAR ESTATÍSTICAS
    # --------------------------------------------------------------------------
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    logger.info(f"\n📊 ESTATÍSTICAS DO MODELO:")
    logger.info(f"   Parâmetros totais: {params['total']:,}")
    logger.info(f"   Tamanho em memória: {size_mb:.1f} MB ({size_mb/1024:.2f} GB)")
    logger.info(f"   VRAM utilizada: {vram_after - vram_before:.3f} GB")
    logger.info(f"   Tempo de carregamento: {load_time:.2f}s")
    logger.info(f"   Device: {next(model.parameters()).device}")
    
    return model, tokenizer


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    strategy: str = "greedy",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
) -> str:
    """
    Gera texto usando diferentes estratégias de decodificação.
    
    TEORIA — ESTRATÉGIAS DE DECODIFICAÇÃO
    ========================================
    Dado um prompt, o modelo gera uma distribuição de probabilidade sobre
    todos os tokens do vocabulário para a PRÓXIMA posição. A estratégia
    de decodificação define COMO escolher o próximo token.
    
    1. GREEDY (determinístico):
       Sempre escolhe o token com MAIOR probabilidade.
       ✅ Consistente e reprodutível
       ❌ Texto repetitivo e previsível
       Uso: Quando precisa de saída determinística (classificação, extração)
    
    2. SAMPLING (estocástico):
       Amostra aleatoriamente da distribuição de probabilidade.
       ✅ Texto diverso e criativo
       ❌ Pode gerar nonsense se temperatura muito alta
       Uso: Geração criativa, chatbots
    
    3. TOP-K SAMPLING (Fan et al., 2018):
       Restringe a amostragem aos K tokens mais prováveis.
       ✅ Evita tokens muito improváveis
       ❌ K fixo: pode incluir tokens ruins (se dist. é flat) ou excluir
          bons (se dist. é peaked)
       Uso: K=50 é um bom default
    
    4. TOP-P / NUCLEUS SAMPLING (Holtzman et al., 2020):
       Restringe aos tokens cuja probabilidade acumulada ≤ p.
       ✅ Adapta-se automaticamente à forma da distribuição
       ❌ Menos intuitivo que top-k
       Uso: p=0.9 é o padrão mais usado na indústria
    
    TEMPERATURA:
    ─────────────
    Escala os logits ANTES do softmax: softmax(logits / T)
    - T < 1.0: Distribuição mais "peaked" → mais determinístico
    - T = 1.0: Sem efeito (distribuição original)
    - T > 1.0: Distribuição mais "flat" → mais diverso/caótico
    - T → 0: Equivale a greedy
    
    Args:
        model: Modelo de linguagem carregado.
        tokenizer: Tokenizador correspondente.
        prompt: Texto de entrada.
        strategy: "greedy", "sample", "top_k", "top_p", "beam".
        max_new_tokens: Quantidade máxima de tokens a gerar.
        temperature: Controle de aleatoriedade (0.0-2.0).
        top_k: Número de tokens para top-k sampling.
        top_p: Limite de probabilidade acumulada para nucleus sampling.
    
    Returns:
        Texto gerado como string.
    """
    # Tokenizar o prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Configurar parâmetros de geração baseado na estratégia
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    if strategy == "greedy":
        # Greedy: do_sample=False força seleção determinística
        gen_kwargs["do_sample"] = False
        
    elif strategy == "sample":
        # Sampling puro: amostrar da distribuição completa
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        
    elif strategy == "top_k":
        # Top-K: restringir aos K tokens mais prováveis
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_k"] = top_k
        
    elif strategy == "top_p":
        # Nucleus Sampling: restringir por probabilidade acumulada
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
        gen_kwargs["top_k"] = 0  # Desabilitar top-k quando usando top-p
        
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    
    # Gerar texto
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    # Decodificar apenas os tokens NOVOS (remover o prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text


def demonstrate_decoding_strategies(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
):
    """
    Demonstra visualmente como diferentes estratégias produzem textos distintos.
    """
    prompt = "The future of artificial intelligence in healthcare will"
    
    logger.info(f"\n{'='*60}")
    logger.info("COMPARAÇÃO DE ESTRATÉGIAS DE DECODIFICAÇÃO")
    logger.info(f"Prompt: \"{prompt}\"")
    logger.info(f"{'='*60}")
    
    strategies = [
        ("greedy", {"temperature": 1.0}),
        ("sample", {"temperature": 0.3}),
        ("sample", {"temperature": 0.7}),
        ("sample", {"temperature": 1.5}),
        ("top_k", {"temperature": 0.7, "top_k": 50}),
        ("top_p", {"temperature": 0.7, "top_p": 0.9}),
    ]
    
    for strategy, kwargs in strategies:
        label = f"{strategy} (T={kwargs.get('temperature', 1.0)}"
        if "top_k" in kwargs:
            label += f", K={kwargs['top_k']}"
        if "top_p" in kwargs:
            label += f", P={kwargs['top_p']}"
        label += ")"
        
        text = generate_text(
            model, tokenizer, prompt,
            strategy=strategy,
            max_new_tokens=50,
            **kwargs,
        )
        
        logger.info(f"\n   📝 {label}:")
        logger.info(f"      {text[:200]}...")


def main():
    """
    Pipeline principal de demonstração de inferência.
    """
    logger.info("🚀 MÓDULO 01: INFERÊNCIA BÁSICA COM SLMs")
    logger.info("=" * 60)
    
    # --------------------------------------------------------------------------
    # 1. CARREGAR MODELO
    # --------------------------------------------------------------------------
    model, tokenizer = load_model(
        model_id="Qwen/Qwen2.5-1.5B",
        torch_dtype="bfloat16",
    )
    
    # --------------------------------------------------------------------------
    # 2. GERAÇÃO SIMPLES
    # --------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("GERAÇÃO SIMPLES DE TEXTO")
    logger.info(f"{'='*60}")
    
    prompt = "Explain what a Small Language Model is in simple terms:"
    text = generate_text(model, tokenizer, prompt, strategy="top_p", max_new_tokens=100)
    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Resposta: {text}")
    
    # --------------------------------------------------------------------------
    # 3. COMPARAÇÃO DE ESTRATÉGIAS
    # --------------------------------------------------------------------------
    demonstrate_decoding_strategies(model, tokenizer)
    
    # --------------------------------------------------------------------------
    # 4. MEDIÇÃO DE LATÊNCIA
    # --------------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("MEDIÇÃO DE LATÊNCIA")
    logger.info(f"{'='*60}")
    
    latency = measure_inference_latency(
        model, tokenizer,
        prompt="Solve this: 2 + 2 = ",
        max_new_tokens=50,
        num_warmup=1,
        num_runs=3,
    )
    
    logger.info(f"\n📊 RESULTADOS DE LATÊNCIA:")
    logger.info(f"   TTFT: {latency['ttft_ms']:.2f} ms")
    logger.info(f"   Throughput: {latency['throughput_tokens_per_sec']:.2f} tokens/s")
    logger.info(f"   Tempo total (50 tokens): {latency['total_time_ms']:.2f} ms")
    
    # --------------------------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------------------------
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ MÓDULO 01 CONCLUÍDO!")
    logger.info("📚 Próximo passo: Módulo 02 (SFT com LoRA)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
