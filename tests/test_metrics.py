import pytest
from unittest.mock import MagicMock
import torch
from src.utils.metrics import count_parameters, calculate_sparsity, oom_protected, get_model_size_mb


def test_count_parameters():
    model = MagicMock()
    param1 = MagicMock()
    param1.numel.return_value = 100
    param1.requires_grad = True
    
    param2 = MagicMock()
    param2.numel.return_value = 200
    param2.requires_grad = False
    
    model.parameters.return_value = [param1, param2]
    
    counts = count_parameters(model)
    assert counts["total"] == 300
    assert counts["trainable"] == 100
    assert counts["frozen"] == 200
    assert counts["trainable_percent"] == 33.3333


def test_calculate_sparsity():
    model = MagicMock()
    
    param1 = torch.tensor([0.0, 1.0, 0.0, 2.0])
    param2 = torch.tensor([1.0, 1.0, 0.0, 0.0])
    
    model.named_parameters.return_value = [("layer1", param1), ("layer2", param2)]
    
    sparsity = calculate_sparsity(model)
    assert sparsity["total_params"] == 8
    assert sparsity["zero_params"] == 4
    assert sparsity["sparsity_percent"] == 50.0


def test_oom_protected_success():
    @oom_protected
    def dummy_func(x):
        return x + 1
        
    assert dummy_func(5) == 6


def test_oom_protected_failure():
    @oom_protected
    def dummy_func_oom():
        raise RuntimeError("CUDA out of memory. Tried to allocate...")
        
    with pytest.raises(RuntimeError, match="OOM error in dummy_func_oom"):
        dummy_func_oom()
