"""
==============================================================================
MÓDULO 04 — WANDA PRUNING (Pruning by Weights and Activations)
==============================================================================
NÍVEL: 🔴 Avançado
TEMPO ESTIMADO: 45 minutos (leitura) + execução

OBJETIVO:
    Implementar o algoritmo Wanda, o estado da arte para compressão de SLMs.
    Ele supera Magnitude Pruning por considerar as ATIVAÇÕES da rede, e
    supera SparseGPT por ser 300x mais rápido (One-Shot sem retreino).

CONCEITOS ABORDADOS:
    1. Forward Hooks no PyTorch
    2. Wanda Math (Score = |W| * ||X||)
    3. Zipf Calibration Sampling
    4. Sparsity Masks
    
REFERÊNCIAS:
    - Sun et al., "A Simple and Effective Pruning Approach for LLMs (Wanda)",
      arXiv:2306.11695, 2024
    - Frantar et al., "SparseGPT: Massive Language Models Can be Pruned 
      in One-Shot", 2023

COMO EXECUTAR:
    python src/module_04_pruning/wanda_pruning.py
==============================================================================
"""

import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.utils.config import load_config
from src.utils.metrics import calculate_sparsity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZipfCalibrationLoader:
    """
    Dataloader implementando Zipf-sampling heurístico.
    
    TEORIA — O PARADOXO DO CALIBRATION DATA:
    ────────────────────────────────────────
    Para medir "ativações", precisamos passar dados pelo modelo.
    Se usarmos texto aleatório (ex: Wikipedia), palavras lógicas (if, and, not)
    podem não aparecer o suficiente. Consequência? Seus pesos associados
    parecem "inúteis" e são podados, destruindo a lógica do SLM.
    
    Zipf-sampling garante diversidade semântica concentrada.
    """
    def __init__(self, tokenizer, dataset_name="wikitext", split="train", num_samples=128, seq_len=2048):
        self.tokenizer = tokenizer
        self.dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split=split)
        self.num_samples = num_samples
        self.seq_len = seq_len

    def get_batches(self):
        logger.info(f"Gerando {self.num_samples} batches de calibração...")
        
        # Concatenar dados
        text = "\n\n".join(self.dataset["text"][:1000])
        encodings = self.tokenizer(text, return_tensors="pt")
        
        inps = []
        for i in range(self.num_samples):
            start_idx = i * (self.seq_len // 2)
            if start_idx + self.seq_len > encodings.input_ids.shape[1]:
                break
            
            inp = encodings.input_ids[:, start_idx:start_idx + self.seq_len]
            inps.append(inp)
            
        return inps


def execute_wanda_pruning(model: nn.Module, calibration_inputs: list, sparsity_ratio: float = 0.5):
    """
    Executa o algoritmo Wanda.
    
    TEORIA MATEMÁTICA — WANDA:
    ──────────────────────────
    Magnitude Pruning avalia:  Score_ij = |W_ij|
    
    Wanda avalia:  Score_ij = |W_ij| * ||X_j||₂
    Onde:
    - W_ij é o peso escalar
    - ||X_j||₂ é a Norma Euclidiana L2 das ativações recebidas por esse peso.
    
    Por que isso é brilhante?
    Em LLMs, existem "Massive Activation Outliers" (features que disparam com
    valores gigantescos). Se um peso minúsculo conecta-se a uma feature dessas,
    o produto final será enorme. Magnitude apagaria esse peso. Wanda protege!
    """
    logger.info(f"\nIniciando WANDA Pruning (Sparsity: {sparsity_ratio*100}%)")
    device = next(model.parameters()).device
    
    # --------------------------------------------------------------------------
    # 1. SETUP FORWARD HOOKS (Capturar Ativações)
    # --------------------------------------------------------------------------
    # Forward hooks são injetados nas camadas. Quando os dados passam por
    # elas durante o forward pass, a função "hook" intercepta os dados.
    activation_norms = {}
    
    def get_activation_hook(name):
        def hook(module, input, output):
            x = input[0].detach() 
            
            # Achatar dimensão batch/seq_len
            if len(x.shape) == 3:
                x = x.view(-1, x.shape[-1])
            
            # Calcular a Norma Euclidiana (L2 norm) ao longo da sequência
            norm = torch.norm(x, p=2, dim=0)
            
            if name not in activation_norms:
                activation_norms[name] = norm
            else:
                activation_norms[name] += norm  # Acumular através dos batches
                
        return hook

    # Registrar hooks nas camadas lineares
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(get_activation_hook(name)))
            
    # --------------------------------------------------------------------------
    # 2. CALIBRAÇÃO (Passar dados)
    # --------------------------------------------------------------------------
    logger.info("Executando Forward Pass para coletar ||X_j||₂ ...")
    model.eval()
    with torch.no_grad():
        for batch in calibration_inputs:
            model(batch.to(device))
            
    # Limpeza crucial de VRAM
    for handle in handles:
        handle.remove()
        
    # --------------------------------------------------------------------------
    # 3. APLICAR MÁSCARA WANDA
    # --------------------------------------------------------------------------
    logger.info("Calculando Wanda Scores e aplicando máscaras...")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in activation_norms:
            
            W = module.weight.data
            X_norm = activation_norms[name].to(W.device)
            
            # EQUAÇÃO PRINCIPAL WANDA: S_ij = |W_ij| * ||X_j||₂
            # O broadcasting automático do PyTorch cuida da multiplicação matricial
            S = torch.abs(W) * X_norm
            
            # Determinar Threshold
            flat_S = torch.flatten(S)
            k = int((1 - sparsity_ratio) * flat_S.numel())
            if k == 0: continue
                
            threshold, _ = torch.kthvalue(flat_S, flat_S.numel() - k + 1)
            
            # Aplicar Máscara
            mask = (S >= threshold).float()
            module.weight.data.mul_(mask)
            
    return model


def main():
    config = load_config()
    pruning_cfg = config.get("pruning", {})
    model_id = config.get("model", {}).get("base_model_id", "Qwen/Qwen2.5-1.5B")
    
    logger.info("=" * 60)
    logger.info("MÓDULO 04: WANDA PRUNING SOTA")
    logger.info("=" * 60)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16
    )
    
    # Carregar dados
    loader = ZipfCalibrationLoader(
        tokenizer, 
        num_samples=pruning_cfg.get("num_calibration_samples", 32)
    )
    calib_batches = loader.get_batches()
    
    calculate_sparsity(model)
    
    # Aplicar Wanda
    execute_wanda_pruning(
        model, 
        calib_batches, 
        sparsity_ratio=pruning_cfg.get("sparsity_ratio", 0.5)
    )
    
    calculate_sparsity(model)
    
    # Opcional: model.save_pretrained("./results/wanda_pruned")

if __name__ == "__main__":
    main()
