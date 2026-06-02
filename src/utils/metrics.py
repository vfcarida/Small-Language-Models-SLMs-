"""
==============================================================================
MÓDULO DE MÉTRICAS — Funções Reutilizáveis de Avaliação
==============================================================================
Este módulo centraliza todas as métricas de avaliação utilizadas no projeto.
Cada métrica é implementada com explicação matemática completa.

CONCEITO EDUCACIONAL:
    Métricas são a linguagem com que comunicamos a qualidade de um modelo.
    Em SLMs, as métricas mais importantes são:
    
    1. PERPLEXIDADE: Mede quão "surpreso" o modelo fica com texto novo.
       Menor é melhor. Um modelo perfeito teria perplexidade = 1.
       Fórmula: PPL = exp(-1/N * Σ log P(token_i | contexto))
    
    2. ACCURACY: Fração de respostas corretas em benchmarks.
    
    3. LATÊNCIA (TTFT): Tempo até o primeiro token ser gerado.
    
    4. THROUGHPUT: Tokens gerados por segundo.
    
    5. VRAM: Memória GPU utilizada pelo modelo.

REFERÊNCIAS:
    - Jurafsky & Martin, "Speech and Language Processing", Cap. 3 (Perplexidade)
    - EleutherAI, "Language Model Evaluation Harness", 2024
==============================================================================
"""

import time
import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def calculate_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """
    Calcula a perplexidade de um modelo em um texto dado.
    
    TEORIA MATEMÁTICA:
        A perplexidade é definida como a exponencial da entropia cruzada
        negativa média do modelo sobre uma sequência de tokens:
        
            PPL = exp( -1/N * Σᵢ log P(xᵢ | x₁, ..., xᵢ₋₁) )
        
        Onde:
        - N é o número total de tokens
        - P(xᵢ | contexto) é a probabilidade que o modelo atribui ao token xᵢ
        - O somatório percorre todos os tokens da sequência
        
        INTERPRETAÇÃO:
        - PPL = 1: O modelo prevê perfeitamente cada token (impossível na prática)
        - PPL = 10: O modelo está "tão confuso" quanto se tivesse 10 opções
                     igualmente prováveis para cada token
        - PPL = 1000: O modelo está muito confuso (modelo ruim ou texto fora do domínio)
    
    IMPLEMENTAÇÃO:
        Para textos longos, usamos uma "janela deslizante" (sliding window)
        com stride, para evitar estouro de memória. Cada janela tem sobreposição
        com a anterior para manter contexto.
    
    Args:
        model: Modelo de linguagem causal (e.g., Qwen2.5-1.5B).
        tokenizer: Tokenizador correspondente ao modelo.
        text: Texto sobre o qual calcular a perplexidade.
        max_length: Comprimento máximo da janela de contexto.
        stride: Passo da janela deslizante (sobreposição = max_length - stride).
    
    Returns:
        float: Valor da perplexidade. Menor é melhor.
    
    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        >>> ppl = calculate_perplexity(model, tokenizer, "Hello world")
    """
    # Tokeniza o texto completo
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids
    seq_len = input_ids.size(1)
    
    device = next(model.parameters()).device
    
    # Lista para acumular as log-probabilidades negativas
    nlls = []  # NLL = Negative Log-Likelihood
    
    # Janela deslizante para textos maiores que max_length
    # Isso evita processar o texto inteiro de uma vez (estouro de VRAM)
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        
        # Definir quais tokens devem ser avaliados nesta janela
        # (target_len exclui a parte sobreposta com a janela anterior)
        target_len = end_loc - prev_end_loc
        
        # Extrair a janela de input
        input_chunk = input_ids[:, begin_loc:end_loc].to(device)
        
        # Labels são os mesmos tokens (tarefa de próximo token)
        # Tokens na zona de sobreposição são marcados com -100 (ignorados no loss)
        target_ids = input_chunk.clone()
        target_ids[:, :-target_len] = -100
        
        with torch.no_grad():
            outputs = model(input_chunk, labels=target_ids)
            # outputs.loss é a entropia cruzada média sobre os tokens válidos
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    # Perplexidade = exp(média das NLLs)
    ppl = torch.exp(torch.stack(nlls).mean())
    
    return ppl.item()


def measure_inference_latency(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> Dict[str, float]:
    """
    Mede métricas de latência de inferência do modelo.
    
    MÉTRICAS MEDIDAS:
        1. TTFT (Time-To-First-Token): Tempo para gerar o primeiro token.
           Crítico para UX — o usuário quer ver resposta imediata.
        
        2. Throughput (tokens/segundo): Velocidade de geração sustentada.
           Importante para processamento em lote (batch).
        
        3. Total Time: Tempo total para gerar max_new_tokens.
    
    BOAS PRÁTICAS:
        - Sempre fazer "warmup" antes de medir (GPU precisa compilar kernels)
        - Medir múltiplas vezes e reportar a mediana (evita outliers)
        - Usar torch.cuda.synchronize() para timing preciso em GPU
    
    Args:
        model: Modelo carregado na GPU.
        tokenizer: Tokenizador correspondente.
        prompt: Texto de entrada para geração.
        max_new_tokens: Quantidade de tokens a gerar.
        num_warmup: Rodadas de aquecimento (ignoradas na medição).
        num_runs: Rodadas de medição.
    
    Returns:
        Dict com 'ttft_ms', 'throughput_tokens_per_sec', 'total_time_ms'.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # --------------------------------------------------------------------------
    # WARMUP: Execuções iniciais para "aquecer" a GPU
    # --------------------------------------------------------------------------
    # A primeira execução em GPU é sempre mais lenta porque:
    # 1. CUDA precisa compilar kernels JIT (Just-In-Time)
    # 2. Caches de memória da GPU estão frios
    # 3. torch.compile() pode estar fazendo otimizações
    logger.info(f"Executando {num_warmup} rodadas de warmup...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=10)
    
    # Sincronizar GPU para garantir que warmup terminou
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # --------------------------------------------------------------------------
    # MEDIÇÃO DE TTFT (Time-To-First-Token)
    # --------------------------------------------------------------------------
    ttft_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        ttft_times.append((time.perf_counter() - start) * 1000)  # em ms
    
    # --------------------------------------------------------------------------
    # MEDIÇÃO DE THROUGHPUT
    # --------------------------------------------------------------------------
    total_times = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=max_new_tokens,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_times.append((time.perf_counter() - start) * 1000)  # em ms
    
    # Usar mediana para robustez contra outliers
    median_ttft = float(np.median(ttft_times))
    median_total = float(np.median(total_times))
    throughput = max_new_tokens / (median_total / 1000)  # tokens por segundo
    
    results = {
        "ttft_ms": round(median_ttft, 2),
        "throughput_tokens_per_sec": round(throughput, 2),
        "total_time_ms": round(median_total, 2),
    }
    
    logger.info(f"Resultados de latência: {results}")
    return results


def get_model_size_mb(model: PreTrainedModel) -> float:
    """
    Calcula o tamanho do modelo em megabytes.
    
    TEORIA:
        O tamanho de um modelo é determinado por:
        - Número de parâmetros × bytes por parâmetro
        - FP32 = 4 bytes/param, FP16/BF16 = 2 bytes/param, INT8 = 1 byte, INT4 = 0.5 bytes
        
        Exemplo: Qwen2.5-1.5B em BF16 = 1.5B × 2 bytes = ~3 GB
    
    Args:
        model: Modelo PyTorch.
    
    Returns:
        float: Tamanho do modelo em MB.
    """
    total_bytes = 0
    for param in model.parameters():
        # param.nelement() = número de elementos no tensor
        # param.element_size() = bytes por elemento (4 para float32, 2 para float16)
        total_bytes += param.nelement() * param.element_size()
    
    size_mb = total_bytes / (1024 ** 2)
    return round(size_mb, 2)


def get_vram_usage_gb() -> float:
    """
    Retorna o uso atual de VRAM da GPU em gigabytes.
    
    NOTA:
        torch.cuda.memory_allocated() retorna a memória alocada pelo PyTorch,
        não a memória total usada pela GPU (que inclui overhead do CUDA runtime,
        buffers do driver, etc.).
    
    Returns:
        float: VRAM utilizada em GB. Retorna 0.0 se não houver GPU.
    """
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / (1024 ** 3), 3)
    return 0.0


def get_peak_vram_usage_gb() -> float:
    """
    Retorna o pico de uso de VRAM desde o último reset.
    
    Returns:
        float: Pico de VRAM em GB. Retorna 0.0 se não houver GPU.
    """
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    return 0.0


def count_parameters(model: PreTrainedModel) -> Dict[str, int]:
    """
    Conta os parâmetros do modelo, separando treináveis de congelados.
    
    CONCEITO EDUCACIONAL:
        Em técnicas como LoRA, a maioria dos parâmetros é "congelada"
        (frozen) e apenas os adaptadores de baixo rank são treinados.
        Por exemplo, em um modelo de 1.5B parâmetros com LoRA rank=16:
        - Parâmetros totais: ~1.5 bilhão
        - Parâmetros treináveis: ~4-8 milhões (< 1%)
        
        Essa eficiência é o que permite fine-tuning em GPUs consumer.
    
    Args:
        model: Modelo PyTorch.
    
    Returns:
        Dict com 'total', 'trainable', 'frozen', 'trainable_percent'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    
    trainable_pct = (trainable / total * 100) if total > 0 else 0.0
    
    result = {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": round(trainable_pct, 4),
    }
    
    logger.info(
        f"Parâmetros — Total: {total:,} | Treináveis: {trainable:,} "
        f"({trainable_pct:.2f}%) | Congelados: {frozen:,}"
    )
    
    return result


def calculate_sparsity(model: PreTrainedModel) -> Dict[str, float]:
    """
    Calcula a esparsidade do modelo (fração de pesos que são exatamente zero).
    
    CONCEITO EDUCACIONAL:
        Após pruning, muitos pesos são zerados. A esparsidade mede
        a proporção de zeros:
        
            Esparsidade = (Nº de zeros) / (Total de parâmetros)
        
        - Esparsidade 0%: Modelo denso (nenhum peso zerado)
        - Esparsidade 50%: Metade dos pesos são zero (típico Wanda)
        - Esparsidade 90%: Modelo muito esparso (pode perder qualidade)
        
        NOTA: Esparsidade não garante aceleração sem hardware especializado.
        GPUs Ampere/Hopper suportam esparsidade estruturada 2:4 nativamente.
    
    Args:
        model: Modelo PyTorch (possivelmente com pruning aplicado).
    
    Returns:
        Dict com 'total_params', 'zero_params', 'sparsity_percent'.
    """
    total_params = 0
    zero_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    sparsity = (zero_params / total_params * 100) if total_params > 0 else 0.0
    
    result = {
        "total_params": total_params,
        "zero_params": zero_params,
        "sparsity_percent": round(sparsity, 2),
    }
    
    logger.info(f"Esparsidade do modelo: {sparsity:.2f}%")
    return result
