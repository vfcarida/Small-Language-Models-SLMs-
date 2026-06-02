"""
==============================================================================
MÓDULO 04 — MAGNITUDE PRUNING (Baseline)
==============================================================================
NÍVEL: 🟡 Intermediário
TEMPO ESTIMADO: 15 minutos

OBJETIVO:
    Implementar o método mais simples de compressão: Pruning por Magnitude.
    Este módulo serve como baseline para entender por que métodos mais
    avançados (como Wanda) são necessários.

CONCEITOS ABORDADOS:
    1. O que é Unstructured Pruning
    2. Hipótese do Bilhete de Loteria (Lottery Ticket Hypothesis)
    3. Pruning baseado puramente no valor absoluto dos pesos
    4. Cálculo de Esparsidade

REFERÊNCIAS:
    - Han et al., "Learning both Weights and Connections for Efficient
      Neural Networks", 2015
    - Frankle & Carbin, "The Lottery Ticket Hypothesis", 2019

COMO EXECUTAR:
    python src/module_04_pruning/magnitude_pruning.py
==============================================================================
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from src.utils.metrics import calculate_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_magnitude_pruning(model: nn.Module, sparsity_ratio: float = 0.5):
    """
    Aplica Pruning por Magnitude em todas as camadas lineares do modelo.
    
    TEORIA MATEMÁTICA:
    ──────────────────
    A premissa da magnitude é simples:
    "Se um peso w tem valor próximo a 0 (ex: w = 0.000001), ele não
    tem impacto significativo na saída da rede e pode ser removido."
    
    MÉTODO:
    1. Para cada camada, coletamos todos os pesos.
    2. Calculamos o valor absoluto: |W|
    3. Encontramos o limiar (threshold) que separa os X% menores valores.
    4. Criamos uma máscara: 1 para |W| ≥ threshold, 0 para |W| < threshold.
    5. Aplicamos a máscara: W_novo = W * máscara.
    
    O PROBLEMA DESSE MÉTODO:
    Um peso pode ser numericamente pequeno, mas se conectar a uma ativação
    gigantesca (outlier). O produto (peso * ativação) será grande!
    Magnitude pruning ignora ativações, frequentemente destruindo conexões
    lógicas críticas do SLM.
    
    Args:
        model: Modelo PyTorch.
        sparsity_ratio: Fração de pesos a remover (0.5 = 50%).
    """
    logger.info(f"\nAplicando Magnitude Pruning (Target: {sparsity_ratio*100}%)")
    
    # 1. Identificar camadas alvo (geralmente nn.Linear)
    linear_layers = [module for name, module in model.named_modules() 
                     if isinstance(module, nn.Linear)]
    
    logger.info(f"Encontradas {len(linear_layers)} camadas lineares.")
    
    # 2. Aplicar pruning camada por camada
    for module in linear_layers:
        W = module.weight.data
        
        # Valor absoluto dos pesos
        abs_W = torch.abs(W)
        
        # Flatten para ordenação
        flat_W = torch.flatten(abs_W)
        
        # Encontrar o threshold
        # kthvalue encontra o k-ésimo menor elemento
        # Se queremos 50% de zeros, k = 50% do total de elementos
        k = int(sparsity_ratio * flat_W.numel())
        
        # Prevenção: não podar 0 pesos
        if k == 0: continue
            
        threshold, _ = torch.kthvalue(flat_W, k)
        
        # Criar máscara booleana e converter para float
        mask = (abs_W >= threshold).float()
        
        # Aplicar máscara in-place (zerar os pesos menores que threshold)
        module.weight.data.mul_(mask)
        
        # O bias (se existir) geralmente NÃO é podado
    
    logger.info("Pruning concluído.")
    return model


def main():
    logger.info("=" * 60)
    logger.info("MÓDULO 04: MAGNITUDE PRUNING BASELINE")
    logger.info("=" * 60)
    
    model_id = "Qwen/Qwen2.5-1.5B"
    logger.info(f"Carregando modelo {model_id}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16
    )
    
    # Avaliar esparsidade antes (deve ser 0%)
    logger.info("\nEsparsidade ANTES:")
    calculate_sparsity(model)
    
    # Aplicar pruning de 50%
    apply_magnitude_pruning(model, sparsity_ratio=0.5)
    
    # Avaliar esparsidade depois (deve ser 50%)
    logger.info("\nEsparsidade APÓS:")
    calculate_sparsity(model)
    
    # NOTA EDUCACIONAL
    logger.info("\n💡 LIMITAÇÕES DO PRUNING UNSTRUCTURED:")
    logger.info("Apesar de 50% dos pesos serem zero, o modelo não ficará 2x")
    logger.info("mais rápido. GPUs padrão não conseguem pular multiplicações")
    logger.info("por zero aleatórias. Para ter ganho de velocidade real,")
    logger.info("precisamos de hardware específico ou Pruning Semi-Estruturado (2:4).")
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ Veja wanda_pruning.py para a técnica State of the Art.")

if __name__ == "__main__":
    main()
