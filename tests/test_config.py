import pytest
import os
import yaml
from src.utils.config import load_config, validate_config, ConfigurationError, get_project_root


def test_get_project_root():
    root = get_project_root()
    assert root.exists()
    assert (root / "setup.py").exists() or (root / "configs").exists()


def test_validate_config_missing_section():
    invalid_config = {
        "model": {"base_model_id": "test"},
        "lora": {"r": 8}
        # missing sft
    }
    with pytest.raises(ConfigurationError, match="Missing mandatory configuration section"):
        validate_config(invalid_config)


def test_validate_config_invalid_model_id():
    invalid_config = {
        "model": {"base_model_id": ""},
        "sft": {"num_train_epochs": 1},
        "lora": {"r": 8}
    }
    with pytest.raises(ConfigurationError, match="Invalid or missing 'model.base_model_id'"):
        validate_config(invalid_config)


def test_validate_config_invalid_epochs():
    invalid_config = {
        "model": {"base_model_id": "test_model"},
        "sft": {"num_train_epochs": -1},
        "lora": {"r": 8}
    }
    with pytest.raises(ConfigurationError, match="SFT 'num_train_epochs' must be positive"):
        validate_config(invalid_config)


def test_validate_config_invalid_batch_size():
    invalid_config = {
        "model": {"base_model_id": "test_model"},
        "sft": {"per_device_train_batch_size": 0},
        "lora": {"r": 8}
    }
    with pytest.raises(ConfigurationError, match="SFT 'per_device_train_batch_size' must be a positive integer"):
        validate_config(invalid_config)


def test_validate_config_invalid_learning_rate():
    invalid_config = {
        "model": {"base_model_id": "test_model"},
        "sft": {"learning_rate": -0.01},
        "lora": {"r": 8}
    }
    with pytest.raises(ConfigurationError, match="SFT 'learning_rate' must be a positive float"):
        validate_config(invalid_config)


def test_validate_config_invalid_lora_r():
    invalid_config = {
        "model": {"base_model_id": "test_model"},
        "sft": {"learning_rate": 2e-4},
        "lora": {"r": 0}
    }
    with pytest.raises(ConfigurationError, match="LoRA rank 'r' must be a positive integer"):
        validate_config(invalid_config)


def test_load_default_config():
    config = load_config()
    assert isinstance(config, dict)
    assert "model" in config
    assert "sft" in config
    assert "lora" in config
