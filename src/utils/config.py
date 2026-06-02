"""
==============================================================================
MÓDULO DE CONFIGURAÇÃO — Carregamento Centralizado de Hiperparâmetros
==============================================================================
Este módulo fornece funções para carregar e validar a configuração YAML
centralizada do projeto. Garante que todos os scripts usem os mesmos
hiperparâmetros, promovendo reprodutibilidade.

CONCEITO EDUCACIONAL:
    Em projetos de ML, é comum ver hiperparâmetros espalhados em dezenas
    de scripts. Isso dificulta reprodutibilidade e experimentação.
    A abordagem centralizada permite:
    - Mudar um valor em um único lugar
    - Rastrear exatamente quais parâmetros geraram quais resultados
    - Facilitar hyperparameter sweeps (varrer combinações)

USO:
    from src.utils.config import load_config
    config = load_config()  # Carrega configs/default_config.yaml
    model_id = config["model"]["base_model_id"]
==============================================================================
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """
    Encontra o diretório raiz do projeto navegando para cima a partir deste arquivo.
    
    O diretório raiz é identificado pela presença do arquivo 'setup.py' ou
    'configs/default_config.yaml'.
    
    Returns:
        Path: Caminho absoluto para a raiz do projeto.
    
    Raises:
        FileNotFoundError: Se não encontrar o diretório raiz.
    """
    # Navega para cima a partir do diretório atual do arquivo
    current = Path(__file__).resolve().parent
    
    # Sobe no máximo 5 níveis procurando o setup.py
    for _ in range(5):
        if (current / "setup.py").exists() or (current / "configs").exists():
            return current
        current = current.parent
    
    raise FileNotFoundError(
        "Não foi possível encontrar o diretório raiz do projeto. "
        "Certifique-se de que o arquivo setup.py ou configs/ existe."
    )


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega o arquivo de configuração YAML.
    
    O YAML é um formato de serialização legível por humanos, amplamente
    usado em projetos de ML para configuração. Vantagens sobre JSON:
    - Suporta comentários (essencial para documentar hiperparâmetros)
    - Sintaxe mais limpa (sem chaves e aspas obrigatórias)
    - Suporta âncoras e referências (evita duplicação)
    
    Args:
        config_path: Caminho para o arquivo YAML. Se None, usa o padrão.
    
    Returns:
        Dict contendo toda a configuração do projeto.
    
    Raises:
        FileNotFoundError: Se o arquivo de configuração não existir.
        yaml.YAMLError: Se o YAML estiver malformado.
    """
    if config_path is None:
        # Usa o arquivo de configuração padrão na raiz do projeto
        project_root = get_project_root()
        config_path = str(project_root / "configs" / "default_config.yaml")
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}\n"
            f"Execute o projeto a partir da raiz do repositório."
        )
    
    logger.info(f"Carregando configuração de: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # Validação básica: verifica se as seções obrigatórias existem
    required_sections = ["model", "sft", "lora"]
    for section in required_sections:
        if section not in config:
            logger.warning(
                f"Seção '{section}' não encontrada na configuração. "
                f"Usando valores padrão."
            )
    
    logger.info("Configuração carregada com sucesso.")
    return config


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai e retorna apenas a configuração do modelo.
    
    Args:
        config: Dicionário completo de configuração.
    
    Returns:
        Dict com configuração do modelo.
    """
    return config.get("model", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai e retorna a configuração de treinamento (SFT).
    
    Args:
        config: Dicionário completo de configuração.
    
    Returns:
        Dict com configuração de treinamento.
    """
    return config.get("sft", {})


def get_lora_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extrai e retorna a configuração de LoRA.
    
    Args:
        config: Dicionário completo de configuração.
    
    Returns:
        Dict com configuração de LoRA.
    """
    return config.get("lora", {})


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Cria os diretórios de saída especificados na configuração.
    
    Args:
        config: Dicionário completo de configuração.
    """
    dirs_to_create = [
        config.get("sft", {}).get("output_dir", "./results/sft_output"),
        config.get("evaluation", {}).get("output_dir", "./results/eval_output"),
        config.get("logging", {}).get("results_dir", "./results"),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Diretório garantido: {dir_path}")
