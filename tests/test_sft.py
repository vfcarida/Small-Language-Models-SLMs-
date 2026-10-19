import torch
from src.module_02_sft.sft_lora import create_lora_config
from src.module_02_sft.sft_qlora import create_bnb_config


def test_create_lora_config_custom_values():
    cfg = {
        "lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1,
            "bias": "none",
            "target_modules": "all-linear",
            "task_type": "CAUSAL_LM",
        }
    }
    lora_cfg = create_lora_config(cfg)
    assert lora_cfg.r == 32
    assert lora_cfg.lora_alpha == 64
    assert lora_cfg.lora_dropout == 0.1
    assert lora_cfg.target_modules == "all-linear"


def test_create_bnb_config_defaults():
    cfg = {
        "qlora": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    }
    bnb_cfg = create_bnb_config(cfg)
    assert bnb_cfg.load_in_4bit is True
    assert bnb_cfg.bnb_4bit_quant_type == "nf4"
    assert bnb_cfg.bnb_4bit_use_double_quant is True
    assert bnb_cfg.bnb_4bit_compute_dtype == torch.bfloat16
