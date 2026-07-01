import pytest
from unittest.mock import MagicMock
from src.utils.data_utils import format_alpaca_prompt, format_chat_prompt, _create_synthetic_preference_data


def test_format_alpaca_prompt_with_input():
    example = {
        "instruction": "Summarize this text",
        "input": "This is a long text.",
        "output": "Short summary."
    }
    result = format_alpaca_prompt(example)
    assert "text" in result
    assert "### Instruction:\nSummarize this text" in result["text"]
    assert "### Input:\nThis is a long text." in result["text"]
    assert "### Response:\nShort summary." in result["text"]


def test_format_alpaca_prompt_without_input():
    example = {
        "instruction": "Say hello",
        "input": "",
        "output": "Hello!"
    }
    result = format_alpaca_prompt(example)
    assert "text" in result
    assert "### Instruction:\nSay hello" in result["text"]
    assert "### Input:" not in result["text"]
    assert "### Response:\nHello!" in result["text"]


def test_format_chat_prompt():
    example = {
        "instruction": "Explain quantum computing",
        "input": "Briefly",
        "output": "Superposition and entanglement."
    }
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "Formatted Chat Template String"
    
    result = format_chat_prompt(example, tokenizer)
    assert result["text"] == "Formatted Chat Template String"
    tokenizer.apply_chat_template.assert_called_once()


def test_create_synthetic_preference_data():
    dataset = _create_synthetic_preference_data()
    assert len(dataset) > 0
    assert "prompt" in dataset.column_names
    assert "chosen" in dataset.column_names
    assert "rejected" in dataset.column_names
