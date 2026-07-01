"""
==============================================================================
MODULE 00 — ATTENTION MECHANISM VISUALIZER
==============================================================================
LEVEL: 🟢 Beginner → 🟡 Intermediate
ESTIMATED TIME: 20 minutes (reading) + execution

OBJECTIVE:
    Understand the self-attention mechanism visually, which represents the core
    of Transformer architectures (and thus all modern SLMs).

CONCEPTS COVERED:
    1. Self-Attention: How the model determines relationships and importance.
    2. Multi-Head Attention vs. Grouped Query Attention (GQA).
    3. Visualizing attention weights via heatmaps.
    4. Complexity of self-attention calculations (O(n^2)).

PREREQUISITES:
    pip install transformers torch matplotlib numpy

REFERENCES:
    - Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    - Ainslie et al., "GQA: Training Generalized Multi-Query Transformer
      Models from Multi-Head Checkpoints", 2023
    - Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019

RUN COMMAND:
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
    Demonstrates the step-by-step mathematical calculations of Self-Attention.
    
    SELF-ATTENTION THEORY
    ======================
    Self-attention enables each token in a sequence to attend to all other
    tokens to assign context relevance weights.
    
    ANALOGY:
        Consider the sentence:
        "The animal didn't cross the street because it was too tired."
        
        When computing the word "it", the model must resolve if "it" refers to
        "animal" or "street". Self-attention calculates the similarity scores
        between "it" and all other tokens, leading to high scores for "animal".
    
    MATHEMATICS:
    =============
    Given an input matrix X of embeddings:
    
    1. Q = X · W_Q  (Queries — "What am I looking for?")
    2. K = X · W_K  (Keys — "What do I offer?")
    3. V = X · W_V  (Values — "What information do I contain?")
    
    4. Scores = Q · K^T / √d_k  (Scaled dot-product similarity)
    5. Weights = softmax(Scores)  (Normalize to sum to 1.0)
    6. Output = Weights · V  (Weighted representation of values)
    
    Where:
    - d_k is the dimension of the keys (scale factor √d_k prevents gradient saturation).
    
    COMPLEXITY: O(n^2 · d)
    - n^2 because every token is compared against every other token.
    """
    logger.info("=" * 60)
    logger.info("SELF-ATTENTION: STEP-BY-STEP CALCULATION")
    logger.info("=" * 60)
    
    seq_len = 4
    d_model = 8
    d_k = 4
    
    torch.manual_seed(42)
    X = torch.randn(seq_len, d_model)
    
    tokens = ["The", "cat", "slept", "peacefully"]
    
    logger.info(f"\nTokens: {tokens}")
    logger.info(f"Input matrix X shape: {X.shape} (seq_len={seq_len}, d_model={d_model})")
    
    W_Q = torch.randn(d_model, d_k)
    W_K = torch.randn(d_model, d_k)
    W_V = torch.randn(d_model, d_k)
    
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V
    
    logger.info(f"\nSTEP 1 — Projections:")
    logger.info(f"  Q (Queries) shape: {Q.shape}")
    logger.info(f"  K (Keys) shape: {K.shape}")
    logger.info(f"  V (Values) shape: {V.shape}")
    
    scale = torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    scores = (Q @ K.T) / scale
    
    logger.info(f"\nSTEP 2 — Attention Scores (Pre-Softmax):")
    logger.info(f"  Shape: {scores.shape} ({seq_len}x{seq_len} grid)")
    for i, token in enumerate(tokens):
        formatted_scores = [f"{s:.3f}" for s in scores[i].tolist()]
        logger.info(f"  '{token}' → {formatted_scores}")
        
    weights = F.softmax(scores, dim=-1)
    
    logger.info(f"\nSTEP 3 — Attention Weights (Post-Softmax):")
    for i, token in enumerate(tokens):
        formatted_weights = [f"{w:.3f}" for w in weights[i].tolist()]
        logger.info(f"  '{token}' attends to: {formatted_weights}")
        max_idx = weights[i].argmax().item()
        logger.info(f"    → Primary attention target: '{tokens[max_idx]}' ({weights[i][max_idx]:.1%})")
        
    output = weights @ V
    
    logger.info(f"\nSTEP 4 — Attention Output Representation:")
    logger.info(f"  Shape: {output.shape} (Matches input dimensions)")
    logger.info("  Each token representation now incorporates context from the entire sequence.")
    
    return weights, tokens


def visualize_attention_heatmap(weights, tokens, save_path=None):
    """
    Generates a visual heatmap of self-attention weights.
    
    Args:
        weights: Tensor of attention weights (seq_len x seq_len).
        tokens: List of string tokens.
        save_path: Optional output path to save the generated image.
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(tokens, fontsize=12)
    
    # Render weight values inside the heatmap grid cells
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            text_color = "white" if weights[i, j] > 0.5 else "black"
            ax.text(j, i, f"{weights[i, j]:.2f}",
                    ha="center", va="center", fontsize=10, color=text_color)
                    
    ax.set_xlabel("Keys (Source Tokens)", fontsize=12)
    ax.set_ylabel("Queries (Target Tokens)", fontsize=12)
    ax.set_title("Self-Attention Map\n(Each row details token focus distributions)",
                 fontsize=14, fontweight="bold")
                 
    plt.colorbar(im, label="Attention Weight")
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Attention heatmap saved successfully to: {save_path}")
        
    plt.close()


def explain_mha_vs_gqa():
    """
    Compares Multi-Head Attention (MHA) vs Grouped Query Attention (GQA).
    """
    logger.info("\n" + "=" * 60)
    logger.info("MHA vs GQA: WHY SLMs DEPLOY GROUPED QUERY ATTENTION")
    logger.info("=" * 60)
    
    d_model = 2048
    num_q_heads = 32
    d_head = d_model // num_q_heads
    seq_len = 2048
    
    configs = {
        "MHA (Multi-Head Attention)": {
            "kv_heads": 32,
            "description": "Each Query head maps to unique Key and Value heads.",
        },
        "GQA-8 (Grouped Query Attention)": {
            "kv_heads": 8,
            "description": "4 Query heads share 1 Key and 1 Value head.",
        },
        "GQA-2 (e.g. Qwen2.5-1.5B structure)": {
            "kv_heads": 2,
            "description": "16 Query heads share 1 Key and 1 Value head.",
        },
        "MQA (Multi-Query Attention)": {
            "kv_heads": 1,
            "description": "All Query heads share a single Key and Value head.",
        },
    }
    
    logger.info(f"Parameters: d_model={d_model}, Q-heads={num_q_heads}, d_head={d_head}, context={seq_len}")
    logger.info("─" * 60)
    
    for name, cfg in configs.items():
        kv_heads = cfg["kv_heads"]
        # KV-Cache size calculation: 2 (K+V) * kv_heads * seq_len * d_head * 2 bytes (FP16)
        kv_cache_bytes = 2 * kv_heads * seq_len * d_head * 2
        kv_cache_mb = kv_cache_bytes / (1024 ** 2)
        total_cache_mb = kv_cache_mb * 24  # Assuming a standard 24-layer network
        
        logger.info(f"\n  📊 {name}")
        logger.info(f"     {cfg['description']}")
        logger.info(f"     KV-heads: {kv_heads}")
        logger.info(f"     KV-Cache size per layer: {kv_cache_mb:.2f} MB")
        logger.info(f"     Total KV-Cache size (24 layers): {total_cache_mb:.2f} MB")
        
    logger.info("\n  💡 SUMMARY:")
    logger.info("     GQA dramatically scales down KV-Cache footprints, enabling")
    logger.info("     longer context execution and larger batch scales for edge deployment.")


def main():
    weights, tokens = demonstrate_self_attention_math()
    
    logger.info("\n" + "📊 " * 20)
    logger.info("GENERATING ATTENTION HEATMAP")
    logger.info("📊 " * 20)
    
    save_path = os.path.join("results", "attention_heatmap.png")
    visualize_attention_heatmap(weights, tokens, save_path=save_path)
    
    explain_mha_vs_gqa()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ SELF-ATTENTION VISUALIZATION COMPLETE!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
