import pytest
from unittest.mock import MagicMock
from src.module_01_inference.basic_inference import generate_text


class MockBatchEncoding(dict):
    """
    Mock class inheriting from dict to simulate Hugging Face BatchEncoding,
    which has a .to() method to move tensors to devices.
    """
    def to(self, device):
        return self


def test_generate_text_greedy():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    inputs = MockBatchEncoding({"input_ids": MagicMock()})
    inputs["input_ids"].shape = [1, 5]
    tokenizer.return_value = inputs
    tokenizer.decode.return_value = "Greedy Output"
    
    # Verify that model.generate is called with do_sample=False
    generate_text(model, tokenizer, "Test Prompt", strategy="greedy")
    
    model.generate.assert_called_once()
    kwargs = model.generate.call_args[1]
    assert kwargs["do_sample"] is False


def test_generate_text_sample():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    inputs = MockBatchEncoding({"input_ids": MagicMock()})
    inputs["input_ids"].shape = [1, 5]
    tokenizer.return_value = inputs
    tokenizer.decode.return_value = "Sample Output"
    
    generate_text(model, tokenizer, "Test Prompt", strategy="sample", temperature=0.8)
    
    model.generate.assert_called_once()
    kwargs = model.generate.call_args[1]
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.8


def test_generate_text_top_k():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    inputs = MockBatchEncoding({"input_ids": MagicMock()})
    inputs["input_ids"].shape = [1, 5]
    tokenizer.return_value = inputs
    tokenizer.decode.return_value = "Top K Output"
    
    generate_text(model, tokenizer, "Test Prompt", strategy="top_k", top_k=40)
    
    model.generate.assert_called_once()
    kwargs = model.generate.call_args[1]
    assert kwargs["do_sample"] is True
    assert kwargs["top_k"] == 40


def test_generate_text_top_p():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    
    inputs = MockBatchEncoding({"input_ids": MagicMock()})
    inputs["input_ids"].shape = [1, 5]
    tokenizer.return_value = inputs
    tokenizer.decode.return_value = "Top P Output"
    
    generate_text(model, tokenizer, "Test Prompt", strategy="top_p", top_p=0.95)
    
    model.generate.assert_called_once()
    kwargs = model.generate.call_args[1]
    assert kwargs["do_sample"] is True
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 0
