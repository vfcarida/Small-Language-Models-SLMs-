"""
==============================================================================
DATA UTILS MODULE — Dataset Formatting, Templates and Preparation
==============================================================================
This module contains helper functions for loading, formatting and pre-processing
instruction-following and preference datasets for SLMs.

EDUCATIONAL CONCERN:
    Data quality is the most crucial factor in language model training.
    A small, clean dataset (e.g. 50-100 high-quality samples) often beats
    a massive, noisy one.
    
    Reference: OpenAI, "Fine-tuning Best Practices", 2024
    
    COMMON SCHEMAS:
    1. Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    2. ShareGPT: {"conversations": [{"role": "user", "content": "..."}]}
    3. Chat Template: Native markup boundaries (<|im_start|>, etc.)
==============================================================================
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ==============================================================================
# PROMPT TEMPLATES
# ==============================================================================
# Templates define how instructions are structured prior to entering the model.
# Alpaca remains a highly robust format.

ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_TEMPLATE_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_alpaca_prompt(example: Dict[str, str]) -> Dict[str, str]:
    """
    Formats a single dataset example into the Alpaca prompt style.
    
    Args:
        example: Dictionary containing 'instruction', 'input', and 'output' keys.
        
    Returns:
        Dict: Dictionary with a single 'text' key containing the formatted string prompt.
    """
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()
    
    if not instruction:
        logger.warning("Empty instruction field detected in dataset example.")
        
    if inp:
        text = ALPACA_TEMPLATE.format(
            instruction=instruction,
            input=inp,
            output=output,
        )
    else:
        text = ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=instruction,
            output=output,
        )
        
    return {"text": text}


def format_chat_prompt(
    example: Dict[str, str],
    tokenizer: Any,
) -> Dict[str, str]:
    """
    Formats an instruction sample utilizing the native chat template of the tokenizer.
    
    Args:
        example: Input dictionary with keys 'instruction', 'input', and 'output'.
        tokenizer: Tokenizer object with a defined chat_template schema.
        
    Returns:
        Dict: Dictionary with a single 'text' key containing the parsed template string.
    """
    user_message = example.get("instruction", "")
    inp = example.get("input", "").strip()
    if inp:
        user_message += f"\n\nInput: {inp}"
        
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example.get("output", "")},
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    
    return {"text": text}


def load_and_prepare_dataset(
    dataset_name: str = "tatsu-lab/alpaca",
    split: str = "train",
    max_samples: Optional[int] = None,
    test_size: float = 0.1,
    seed: int = 42,
    format_fn: str = "alpaca",
    tokenizer: Any = None,
) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset from the Hugging Face Hub, performs shuffling, splits it,
    formats input data using prompt templates, and filters out short records.
    
    Args:
        dataset_name: Registry identifier for HF datasets.
        split: Split sequence identifier.
        max_samples: Limit count for training subsets.
        test_size: Evaluation split ratio (e.g. 0.1 for 10%).
        seed: Random state seed.
        format_fn: Formatting routine option ('alpaca' or 'chat').
        tokenizer: Tokenizer instance (required for 'chat' templates).
        
    Returns:
        Tuple[Dataset, Dataset]: Formatted training and validation dataset splits.
    """
    logger.info(f"Loading dataset: {dataset_name} (split={split})")
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Loaded dataset size: {len(dataset)} examples")
    
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Dataset limited to {max_samples} examples.")
        
    split_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(f"Split breakdown: {len(train_dataset)} training | {len(eval_dataset)} evaluation")
    
    if format_fn == "chat":
        if tokenizer is None:
            raise ValueError("Tokenizer instance must be supplied when format_fn is set to 'chat'.")
        fmt_function = lambda ex: format_chat_prompt(ex, tokenizer)
    else:
        fmt_function = format_alpaca_prompt
        
    train_dataset = train_dataset.map(
        fmt_function,
        desc="Formatting training dataset splits",
    )
    eval_dataset = eval_dataset.map(
        fmt_function,
        desc="Formatting evaluation dataset splits",
    )
    
    min_length = 20
    train_dataset = train_dataset.filter(
        lambda x: len(x.get("text", "")) >= min_length,
        desc="Filtering short records (train)",
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x.get("text", "")) >= min_length,
        desc="Filtering short records (eval)",
    )
    
    logger.info(f"Post-filtering count: {len(train_dataset)} training | {len(eval_dataset)} evaluation")
    return train_dataset, eval_dataset


def create_preference_dataset(
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences",
    max_samples: Optional[int] = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Loads preference dataset pairs containing prompt, chosen, and rejected options.
    
    Args:
        dataset_name: Registry identifier for HF preference datasets.
        max_samples: Sample limit.
        seed: Random state seed.
        
    Returns:
        Dataset: Loaded preference dataset instance.
    """
    logger.info(f"Loading preference feedback dataset: {dataset_name}")
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logger.warning(
            f"Failed loading {dataset_name} ({e}). "
            "Falling back to synthetic demonstration preference dataset."
        )
        dataset = _create_synthetic_preference_data()
        
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        
    logger.info(f"Preference dataset size: {len(dataset)} examples")
    return dataset


def _create_synthetic_preference_data() -> Dataset:
    """
    Builds a synthetic validation preference dataset for testing or local run fallback scenarios.
    """
    data = {
        "prompt": [
            "Explain what machine learning is in simple terms.",
            "What is the difference between Python and JavaScript?",
            "How does the attention mechanism work in Transformers?",
            "What is overfitting and how can it be avoided?",
            "Explain gradient descent intuitively.",
        ],
        "chosen": [
            "Machine learning is a subset of AI where computers learn patterns directly from data, rather than being explicitly programmed with rules. Think of it as teaching a computer through examples.",
            "Python is widely used for backend, data science, and AI development, with a simple syntax. JavaScript is the standard language of web development (frontend and backend via Node.js).",
            "The attention mechanism allows a model to look at all words in a sequence simultaneously and weigh which words are most relevant. For example, it associates 'it' with 'cat' instead of 'mat'.",
            "Overfitting occurs when a model memorizes training noise instead of general patterns. It can be prevented using more data, regularizations (dropout, weight decay), and early stopping.",
            "Imagine you are lost in a foggy mountain and need to find the valley. Gradient descent is like feeling the steepness under your feet and taking a step downhill iteratively. The step size is the learning rate.",
        ],
        "rejected": [
            "Machine learning is when computers learn on their own.",
            "Python and JavaScript are different coding languages.",
            "Attention is a neural network weighting technique.",
            "Overfitting is bad. Use more training data.",
            "Gradient descent is an optimization algorithm that minimizes the loss function.",
        ],
    }
    return Dataset.from_dict(data)
