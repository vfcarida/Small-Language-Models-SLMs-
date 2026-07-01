"""
==============================================================================
CONFIGURATION MODULE — Centralized Hyperparameter Loading
==============================================================================
This module provides functions to load and strictly validate the centralized
YAML configurations of the project. It ensures all scripts run with uniform
hyperparameters to support reproducibility.

EDUCATIONAL CONCERN:
    In machine learning projects, hyperparameters are often scattered across
    multiple scripts. Centralizing configurations enables:
    - Modifying parameter values in a single location.
    - Tracking exactly which parameters generated which output.
    - Simplifying hyperparameter sweep operations.

USAGE:
    from src.utils.config import load_config
    config = load_config()  # Loads configs/default_config.yaml
    model_id = config["model"]["base_model_id"]
==============================================================================
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Exception raised when configuration parameters are invalid or missing."""
    pass


def get_project_root() -> Path:
    """
    Finds the project root directory by searching upwards from the current file.
    
    The root directory is identified by the presence of 'setup.py' or a 'configs' folder.
    
    Returns:
        Path: Absolute path to the project root.
    
    Raises:
        FileNotFoundError: If the project root cannot be determined.
    """
    current = Path(__file__).resolve().parent
    
    # Traverse up to 5 parent directories looking for setup.py or configs folder
    for _ in range(5):
        if (current / "setup.py").exists() or (current / "configs").exists():
            return current
        current = current.parent
    
    raise FileNotFoundError(
        "Could not determine the project root directory. "
        "Ensure 'setup.py' or a 'configs/' folder exists in the project path."
    )


def validate_config(config: Dict[str, Any]) -> None:
    """
    Strictly validates the integrity and structure of loaded configuration parameters.
    
    Args:
        config: The configuration dictionary to validate.
        
    Raises:
        ConfigurationError: If validation checks fail.
    """
    # 1. Section Presence Validation
    required_sections = ["model", "sft", "lora"]
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing mandatory configuration section: '{section}'")

    # 2. Model Settings Validation
    model_cfg = config["model"]
    if "base_model_id" not in model_cfg or not isinstance(model_cfg["base_model_id"], str) or not model_cfg["base_model_id"].strip():
        raise ConfigurationError("Invalid or missing 'model.base_model_id' config parameter.")

    # 3. SFT Settings Validation
    sft_cfg = config["sft"]
    if "num_train_epochs" in sft_cfg:
        epochs = sft_cfg["num_train_epochs"]
        if not isinstance(epochs, (int, float)) or epochs <= 0:
            raise ConfigurationError(f"SFT 'num_train_epochs' must be positive. Got: {epochs}")
            
    if "per_device_train_batch_size" in sft_cfg:
        batch_size = sft_cfg["per_device_train_batch_size"]
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ConfigurationError(f"SFT 'per_device_train_batch_size' must be a positive integer. Got: {batch_size}")

    if "learning_rate" in sft_cfg:
        lr = sft_cfg["learning_rate"]
        if not isinstance(lr, float) or lr <= 0:
            raise ConfigurationError(f"SFT 'learning_rate' must be a positive float. Got: {lr}")

    # 4. LoRA Settings Validation
    lora_cfg = config["lora"]
    if "r" in lora_cfg:
        r = lora_cfg["r"]
        if not isinstance(r, int) or r <= 0:
            raise ConfigurationError(f"LoRA rank 'r' must be a positive integer. Got: {r}")

    logger.debug("Configuration validation passed successfully.")


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads and validates the YAML configuration file.
    
    Args:
        config_path: The file path to the YAML configuration. If None, the default path is resolved.
    
    Returns:
        Dict: Parsed configuration dictionary.
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found.
        yaml.YAMLError: If the YAML formatting is invalid.
        ConfigurationError: If validation checks fail.
    """
    if config_path is None:
        project_root = get_project_root()
        config_path = str(project_root / "configs" / "default_config.yaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please run the project from the repository root."
        )
    
    logger.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(f"Error parsing YAML configuration file: {exc}")
            raise
    
    # Run structured configuration validation
    validate_config(config)
    
    logger.info("Configuration loaded and validated successfully.")
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the model configuration section.
    
    Args:
        config: Central configuration dictionary.
    
    Returns:
        Dict: Model configuration parameters.
    """
    return config.get("model", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the SFT (Supervised Fine-Tuning) configuration section.
    
    Args:
        config: Central configuration dictionary.
    
    Returns:
        Dict: SFT training parameters.
    """
    return config.get("sft", {})


def get_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts the LoRA (Low-Rank Adaptation) configuration section.
    
    Args:
        config: Central configuration dictionary.
    
    Returns:
        Dict: LoRA configuration parameters.
    """
    return config.get("lora", {})


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Creates default output directories specified in the configuration if they do not exist.
    
    Args:
        config: Central configuration dictionary.
    """
    dirs_to_create = [
        config.get("sft", {}).get("output_dir", "./results/sft_output"),
        config.get("evaluation", {}).get("output_dir", "./results/eval_output"),
        config.get("logging", {}).get("results_dir", "./results"),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Directory verified/created: {dir_path}")
