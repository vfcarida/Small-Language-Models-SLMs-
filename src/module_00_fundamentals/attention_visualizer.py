"""
==============================================================================
MÓDULO 00 — VISUALIZADOR DE MECANISMO DE ATENÇÃO
==============================================================================
NÍVEL: 🟢 Iniciante → 🟡 Intermediário
TEMPO ESTIMADO: 20 minutos de leitura + execução

OBJETIVO:
    Entender visualmente o mecanismo de SELF-ATTENTION, que é o coração
    dos modelos Transformer (e portanto de todos os SLMs modernos).

CONCEITOS ABORDADOS:
    1. Self-Attention: como o modelo "decide" quais palavras são importantes
    2. Multi-Head Attention vs Grouped Query Attention (GQA)
    3. Visualização de mapas de atenção (heatmaps)
    4. Por que a atenção tem complexidade O(n²)

PRÉ-REQUISITOS:
    pip install transformers torch matplotlib numpy

REFERÊNCIAS:
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints", 2023
    - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019

COMO EXECUTAR:
    python src/module_00_fundamentals/attention_visualizer.py
==============================================================================
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def demonstrate_self_attention_math():
    """
    Demonstra o cálculo matemático do Self-Attention passo a passo.
    
    TEORIA COMPLETA — SELF-ATTENTION
    ==================================
    O Self-Attention permite que cada token em uma sequência "olhe" para
    todos os outros tokens e decida quais são mais relevantes.
    
    ANALOGIA:
        Imagine que você está lendo a frase:
        "O gato sentou no tapete porque ele estava cansado."
        
        Quando o modelo processa "ele", precisa decidir: "ele" se refere
        a "gato" ou a "tapete"? O Self-Attention calcula scores de
        similaridade entre "ele" e todas as outras palavras, resultando
        em um score alto para "gato" e baixo para "tapete".
    
    MATEMÁTICA (Vaswani et al., 2017):
    ====================================
    Dado um input X (sequência de embeddings), calculamos:
    
    1. Q = X · W_Q  (Queries — "o que estou procurando?")
    2. K = X · W_K  (Keys — "o que eu ofereço?")
    3. V = X · W_V  (Values — "qual informação eu carrego?")
    
    4. Scores = Q · K^T / √d_k  (Similaridade escalonada)
    5. Weights = softmax(Scores)  (Normalizar para somar 1)
    6. Output = Weights · V  (Combinação ponderada dos valores)
    
    Onde:
    - d_k é a dimensão das keys (√d_k previne scores muito grandes)
    - softmax normaliza os scores para uma distribuição de probabilidade
    
    COMPLEXIDADE: O(n² · d)
    - n² porque cada token compara com todos os outros
    - d é a dimensão do modelo
    - Para n=2048 tokens: 2048² = ~4 milhões de comparações por camada!
    """
    logger.info("=" * 60)
    logger.info("SELF-ATTENTION: CÁLCULO PASSO A PASSO")
    logger.info("=" * 60)
    
    # --------------------------------------------------------------------------
    # SETUP: Criar uma sequência de exemplo
    # --------------------------------------------------------------------------
    # Simular 4 tokens com embeddings de dimensão 8
    # Na prática, d_model = 768 (BERT), 2048 (Llama-1B), 1536 (Qwen-1.5B)
    seq_len = 4      # 4 tokens na sequência
    d_model = 8      # Dimensão do embedding
    d_k = 4          # Dimensão das queries/keys (tipicamente d_model / num_heads)
    
    # Token embeddings simulados (em um modelo real, vêm da camada de embedding)
    torch.manual_seed(42)  # Para reprodutibilidade
    X = torch.randn(seq_len, d_model)
    
    tokens = ["O", "gato", "dormiu", "tranquilamente"]
    
    logger.info(f"\nTokens: {tokens}")
    logger.info(f"Shape do input X: {X.shape} (seq_len={seq_len}, d_model={d_model})")
    
    # --------------------------------------------------------------------------
    # PASSO 1: Projeções Q, K, V
    # --------------------------------------------------------------------------
    # As matrizes de peso W_Q, W_K, W_V são APRENDIDAS durante o treinamento
    # Elas transformam o embedding genérico em representações especializadas
    W_Q = torch.randn(d_model, d_k)  # Projeta para queries
    W_K = torch.randn(d_model, d_k)  # Projeta para keys
    W_V = torch.randn(d_model, d_k)  # Projeta para values
    
    Q = X @ W_Q  # Queries: "O que cada token está procurando?"
    K = X @ W_K  # Keys: "O que cada token oferece como match?"
    V = X @ W_V  # Values: "Qual informação cada token carrega?"
    
    logger.info(f"\nPASSO 1 — Projeções:")
    logger.info(f"  Q (Queries) shape: {Q.shape}")
    logger.info(f"  K (Keys) shape: {K.shape}")
    logger.info(f"  V (Values) shape: {V.shape}")
    
    # --------------------------------------------------------------------------
    # PASSO 2: Calcular Scores de Atenção
    # --------------------------------------------------------------------------
    # Score(i,j) = Q_i · K_j^T / √d_k
    # Mede a similaridade entre o query do token i e a key do token j
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Q @ K^T produz uma matriz (seq_len × seq_len) de scores
    scores = (Q @ K.T) / scale
    
    logger.info(f"\nPASSO 2 — Scores de Atenção (antes do softmax):")
    logger.info(f"  Shape: {scores.shape} ({seq_len}x{seq_len} = cada token vs todos)")
    for i, token in enumerate(tokens):
        formatted_scores = [f"{s:.3f}" for s in scores[i].tolist()]
        logger.info(f"  '{token}' → {formatted_scores}")
    
    # --------------------------------------------------------------------------
    # PASSO 3: Softmax (Normalização)
    # --------------------------------------------------------------------------
    # Transforma os scores brutos em pesos de probabilidade (somam 1.0)
    # Tokens com scores altos recebem peso alto; scores baixos → peso ~0
    weights = F.softmax(scores, dim=-1)
    
    logger.info(f"\nPASSO 3 — Pesos de Atenção (após softmax):")
    for i, token in enumerate(tokens):
        formatted_weights = [f"{w:.3f}" for w in weights[i].tolist()]
        logger.info(f"  '{token}' presta atenção em: {formatted_weights}")
        # O peso mais alto indica qual token é mais relevante
        max_idx = weights[i].argmax().item()
        logger.info(f"    → Mais atenção em: '{tokens[max_idx]}' ({weights[i][max_idx]:.1%})")
    
    # --------------------------------------------------------------------------
    # PASSO 4: Combinação Ponderada dos Values
    # --------------------------------------------------------------------------
    # Output = Weights @ V
    # Cada token recebe uma mistura ponderada das informações de todos os tokens
    output = weights @ V
    
    logger.info(f"\nPASSO 4 — Output da Atenção:")
    logger.info(f"  Shape: {output.shape} (mesma que o input)")
    logger.info("  Cada token agora contém informação contextual de toda a sequência!")
    
    return weights, tokens


def visualize_attention_heatmap(weights, tokens, save_path=None):
    """
    Gera um heatmap visual dos pesos de atenção.
    
    CONCEITO EDUCACIONAL:
        Heatmaps de atenção são a principal ferramenta de interpretabilidade
        para Transformers. Eles mostram VISUALMENTE quais tokens o modelo
        "olha" quando processa cada posição.
        
        Cores mais quentes (vermelho/amarelo) = mais atenção
        Cores mais frias (azul/roxo) = menos atenção
    
    Args:
        weights: Tensor de pesos de atenção (seq_len x seq_len).
        tokens: Lista de strings representando os tokens.
        save_path: Caminho para salvar o plot (opcional).
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    
    # Configurar rótulos
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(tokens, fontsize=12)
    
    # Adicionar valores numéricos em cada célula
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text_color = "white" if weights[i, j] > 0.5 else "black"
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color=text_color)
    
    ax.set_xlabel("Keys (tokens de origem)", fontsize=12)
    ax.set_ylabel("Queries (tokens de destino)", fontsize=12)
    ax.set_title("Mapa de Atenção — Self-Attention\n(cada linha mostra para onde o token olha)",
                 fontsize=14, fontweight="bold")
    
    plt.colorbar(im, label="Peso de Atenção")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Heatmap salvo em: {save_path}")
    
    plt.close()


def explain_mha_vs_gqa():
    """
    Explica a diferença entre Multi-Head Attention (MHA) e
    Grouped Query Attention (GQA) — usada em SLMs modernos.
    
    CONCEITO EDUCACIONAL:
    
    MULTI-HEAD ATTENTION (MHA) — Original, Vaswani 2017
    =====================================================
    Em vez de uma única operação de atenção, dividimos Q, K, V em
    múltiplas "cabeças" independentes. Cada cabeça pode aprender
    diferentes padrões de atenção:
    - Cabeça 1: Relações sintáticas (sujeito ↔ verbo)
    - Cabeça 2: Relações semânticas (sinônimos)
    - Cabeça 3: Posição relativa (proximidade)
    
    PROBLEMA: Cada cabeça precisa de seu próprio K e V, o que consome
    muita memória durante a geração autorregressiva (KV-Cache).
    
    GROUPED QUERY ATTENTION (GQA) — Ainslie et al., 2023
    =====================================================
    Solução: Compartilhar K e V entre GRUPOS de cabeças de query.
    
    Exemplo com 32 cabeças de query e 8 grupos:
    - MHA: 32 Q + 32 K + 32 V = 96 projeções
    - GQA: 32 Q + 8 K + 8 V = 48 projeções (50% menos memória!)
    
    MULTI-QUERY ATTENTION (MQA) — Caso extremo:
    - 32 Q + 1 K + 1 V = 34 projeções (muito eficiente, mas menos expressivo)
    
    IMPACTO EM SLMs:
    - Qwen2.5-1.5B: Usa GQA com 12 Q-heads e 2 KV-heads
    - Llama-3.2-1B: Usa GQA com 32 Q-heads e 8 KV-heads
    - Phi-3.5-mini: Usa MHA completo (mais expressivo, mais VRAM)
    """
    logger.info("\n" + "=" * 60)
    logger.info("MHA vs GQA: POR QUE SLMs USAM GQA?")
    logger.info("=" * 60)
    
    # Simular tamanhos para comparação
    d_model = 2048
    num_q_heads = 32
    d_head = d_model // num_q_heads  # 64
    seq_len = 2048
    
    configs = {
        "MHA (Multi-Head Attention)": {
            "kv_heads": 32,
            "description": "Cada head de query tem seu próprio K e V",
        },
        "GQA-8 (Grouped Query Attention)": {
            "kv_heads": 8,
            "description": "4 heads de query compartilham 1 K e 1 V",
        },
        "GQA-2 (como Qwen2.5-1.5B)": {
            "kv_heads": 2,
            "description": "16 heads de query compartilham 1 K e 1 V",
        },
        "MQA (Multi-Query Attention)": {
            "kv_heads": 1,
            "description": "Todas as heads compartilham 1 K e 1 V",
        },
    }
    
    logger.info(f"\nParâmetros: d_model={d_model}, Q-heads={num_q_heads}, "
                f"d_head={d_head}, seq_len={seq_len}")
    logger.info(f"{'─' * 60}")
    
    for name, cfg in configs.items():
        kv_heads = cfg["kv_heads"]
        
        # KV-Cache size = 2 (K+V) × kv_heads × seq_len × d_head × 2 bytes (fp16)
        kv_cache_bytes = 2 * kv_heads * seq_len * d_head * 2
        kv_cache_mb = kv_cache_bytes / (1024 ** 2)
        
        # Para 24 camadas (típico de 1.5B)
        total_cache_mb = kv_cache_mb * 24
        
        logger.info(f"\n  📊 {name}")
        logger.info(f"     {cfg['description']}")
        logger.info(f"     KV-heads: {kv_heads}")
        logger.info(f"     KV-Cache por camada: {kv_cache_mb:.2f} MB")
        logger.info(f"     KV-Cache total (24 camadas): {total_cache_mb:.2f} MB")
    
    logger.info(f"\n  💡 CONCLUSÃO:")
    logger.info(f"     GQA reduz drasticamente o KV-Cache, permitindo")
    logger.info(f"     contextos mais longos e batching maior em SLMs.")
    logger.info(f"     Isso é CRÍTICO para deploy em edge/mobile.")


def main():
    """
    Executa todas as demonstrações do visualizador de atenção.
    """
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 1: Self-Attention Matemática
    # --------------------------------------------------------------------------
    weights, tokens = demonstrate_self_attention_math()
    
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 2: Visualização (Heatmap)
    # --------------------------------------------------------------------------
    logger.info("\n" + "📊 " * 20)
    logger.info("GERANDO HEATMAP DE ATENÇÃO")
    logger.info("📊 " * 20)
    
    save_path = os.path.join("results", "attention_heatmap.png")
    visualize_attention_heatmap(weights, tokens, save_path=save_path)
    
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 3: MHA vs GQA
    # --------------------------------------------------------------------------
    explain_mha_vs_gqa()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ VISUALIZAÇÃO DE ATENÇÃO CONCLUÍDA!")
    logger.info("📚 Próximo passo: Veja o módulo 01 (basic_inference.py)")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
