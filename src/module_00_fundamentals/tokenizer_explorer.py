"""
==============================================================================
MODULE 00 — TOKENIZER EXPLORER
==============================================================================
LEVEL: 🟢 Beginner
ESTIMATED TIME: 15 minutes (reading) + execution

OBJECTIVE:
    Understand how language models perceive text. Before processing any words,
    the model must convert them to integers. This conversion step is called
    TOKENIZATION.

CONCEPTS COVERED:
    1. Tokenization foundations.
    2. Byte-Pair Encoding (BPE) algorithm — used by GPT, Llama, Qwen.
    3. Special control tokens (EOS, PAD, BOS).
    4. Comparing vocabulary spaces across model families.
    5. Impact of tokenization on speed and computing costs.

PREREQUISITES:
    pip install transformers

REFERENCES:
    - Sennrich et al., "Neural Machine Translation of Rare Words with
      Subword Units", ACL 2016 (BPE algorithm)
    - Kudo & Richardson, "SentencePiece: A simple and language independent
      subword tokenizer", EMNLP 2018
    - HuggingFace, "Tokenizers Documentation", 2024

RUN COMMAND:
    python src/module_00_fundamentals/tokenizer_explorer.py
==============================================================================
"""

import logging
from typing import Dict, List

from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def explore_tokenizer(model_id: str, sample_texts: List[str]) -> Dict:
    """
    Explores how a specific model tokenizer parses text strings into tokens.
    
    TOKENIZATION FUNDAMENTALS:
    ===========================
    Language models do not read characters or text directly. They process
    numerical sequences. Tokenization divides strings into subword units, and
    maps each unique subword token to an integer index.
    
    BPE ALGORITHM SUMMARY:
    =======================
    1. Initialize vocabulary with raw characters.
    2. Identify and count the most frequent adjacent token pairs.
    3. Merge the highest frequency token pairs to create new subword tokens.
    4. Iterate until the target vocabulary size limit is reached.
    
    Args:
        model_id: Hugging Face model registry identifier.
        sample_texts: List of strings to tokenize.
        
    Returns:
        Dict: Metadata and tokenization results.
    """
    logger.info(f"{'='*60}")
    logger.info(f"EXPLORING TOKENIZER: {model_id}")
    logger.info(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    
    vocab_size = tokenizer.vocab_size
    logger.info(f"\n📊 VOCABULARY INFORMATION:")
    logger.info(f"   Vocabulary Size: {vocab_size:,} tokens")
    logger.info(f"   Tokenizer Class: {type(tokenizer).__name__}")
    logger.info(f"   Execution Backend: {tokenizer.is_fast and 'Fast (Rust)' or 'Slow (Python)'}")
    
    logger.info(f"\n🏷️  SPECIAL CONTROL TOKENS:")
    logger.info(f"   BOS Token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(f"   EOS Token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"   PAD Token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"   UNK Token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    if hasattr(tokenizer, "additional_special_tokens"):
        special = tokenizer.additional_special_tokens[:10]
        if special:
            logger.info(f"   Additional Control Tokens: {special}")
            
    logger.info(f"\n📝 SAMPLE TOKENIZATION RUNS:")
    
    results = []
    for text in sample_texts:
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        compression_ratio = len(text) / len(tokens) if tokens else 0
        
        logger.info(f"\n   Text: \"{text}\"")
        logger.info(f"   Tokens ({len(tokens)}): {tokens}")
        logger.info(f"   IDs: {token_ids}")
        logger.info(f"   Characters: {len(text)} → Tokens: {len(tokens)}")
        logger.info(f"   Compression Ratio: {compression_ratio:.2f} chars/token")
        
        reconstructed = tokenizer.decode(token_ids)
        logger.info(f"   Reconstruction: \"{reconstructed}\"")
        
        results.append({
            "text": text,
            "tokens": tokens,
            "token_ids": token_ids,
            "num_tokens": len(tokens),
            "compression_ratio": round(compression_ratio, 2),
        })
        
    return {
        "model_id": model_id,
        "vocab_size": vocab_size,
        "tokenizer_type": type(tokenizer).__name__,
        "examples": results,
    }


def compare_tokenizers(
    model_ids: List[str],
    text: str,
) -> None:
    """
    Compares tokenization outputs and efficiency statistics across models.
    
    Args:
        model_ids: List of Hugging Face model IDs.
        text: Test string to tokenize.
    """
    logger.info(f"\n{'='*60}")
    logger.info("TOKENIZER COMPARISON STATE")
    logger.info(f"Test Text: \"{text}\"")
    logger.info(f"Total Characters: {len(text)}")
    logger.info(f"{'='*60}")
    
    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            tokens = tokenizer.tokenize(text)
            ratio = len(text) / len(tokens) if tokens else 0
            
            logger.info(f"\n   📦 {model_id}")
            logger.info(f"      Vocab Size: {tokenizer.vocab_size:,}")
            logger.info(f"      Tokens ({len(tokens)}): {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
            logger.info(f"      Compression: {ratio:.2f} chars/token")
            
        except Exception as e:
            logger.warning(f"   ❌ {model_id}: Failed to load tokenizer — {e}")


def demonstrate_special_tokens_impact():
    """
    Demonstrates the practical impact of configuring padding and end-of-sequence tokens.
    """
    logger.info(f"\n{'='*60}")
    logger.info("DEMO: CONTROL TOKENS OPERATION")
    logger.info(f"{'='*60}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        logger.info(f"\n   Model: {model_id}")
        logger.info(f"   Is PAD token defined? {tokenizer.pad_token is not None}")
        
        if tokenizer.pad_token is None:
            logger.info("   ⚠️  PAD TOKEN UNDEFINED — Sequence batch operations would fail!")
            logger.info("   🔧 Resolving PAD = EOS token fallback.")
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"   ✅ Active PAD token is now: '{tokenizer.pad_token}'")
            
        texts = ["Short string.", "This represents a significantly longer text input to observe padding behavior."]
        
        batch = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=32,
            return_tensors="pt",
        )
        
        logger.info(f"\n   📦 BATCH OPERATION RESULTS:")
        logger.info(f"   Input Tensor Shape: {batch['input_ids'].shape}")
        logger.info(f"   Sequence 1 Token IDs: {batch['input_ids'][0].tolist()}")
        logger.info(f"   Sequence 2 Token IDs: {batch['input_ids'][1].tolist()}")
        logger.info(f"   Sequence 1 Attention Mask: {batch['attention_mask'][0].tolist()}")
        logger.info("   ℹ️ Zeros in the attention mask represent padded regions.")
        
    except Exception as e:
        logger.error(f"Special tokens test cycle crashed: {e}")


def main():
    sample_texts = [
        "Hello, how are you?",
        "Natural language processing is transitioning digital products globally.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The Transformer architecture uses multi-head self-attention with O(n²) complexity.",
        "3.14159 * (2^10) = 3216.99",
        "The neural robot processed 1.5B parameters successfully.",
    ]
    
    explore_tokenizer("Qwen/Qwen2.5-1.5B", sample_texts)
    
    models_to_compare = [
        "Qwen/Qwen2.5-1.5B",
        "meta-llama/Llama-3.2-1B",
        "microsoft/Phi-3.5-mini-instruct",
    ]
    
    comparison_text = (
        "Natural language processing focuses on enabling interactions "
        "between computer algorithms and human communication interfaces."
    )
    
    compare_tokenizers(models_to_compare, comparison_text)
    demonstrate_special_tokens_impact()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ TOKENIZER EXPLORATION COMPLETED!")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
