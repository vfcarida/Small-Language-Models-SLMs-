"""
==============================================================================
MÓDULO DE DADOS — Funções de Preparação e Formatação de Datasets
==============================================================================
Este módulo centraliza toda a lógica de carregamento, formatação e
pré-processamento de datasets para treinamento de SLMs.

CONCEITO EDUCACIONAL:
    A qualidade dos dados é o fator MAIS IMPORTANTE no treinamento de LMs.
    Um dataset pequeno mas bem curado (50-100 exemplos de alta qualidade)
    frequentemente supera um dataset enorme e ruidoso.
    
    Referência: OpenAI, "Fine-tuning Best Practices", 2024
    
    FORMATOS COMUNS:
    1. Alpaca: {"instruction": "...", "input": "...", "output": "..."}
    2. ShareGPT: {"conversations": [{"role": "user", "content": "..."}]}
    3. Chat Template: Usa tokens especiais do modelo (<|im_start|>, etc.)

REFERÊNCIAS:
    - Peng et al., "Instruction Tuning with GPT-4", 2023
    - Zhou et al., "LIMA: Less Is More for Alignment", 2023
==============================================================================
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

logger = logging.getLogger(__name__)


# ==============================================================================
# TEMPLATES DE PROMPT
# ==============================================================================
# Templates definem como estruturar as instruções para o modelo.
# Cada modelo pode ter seu formato preferido, mas o formato Alpaca é universal.

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
    Formata um exemplo no formato Alpaca para treinamento.
    
    CONCEITO EDUCACIONAL:
        O formato Alpaca foi introduzido por Stanford (2023) e se tornou
        o padrão de facto para fine-tuning de instruções. A estrutura é:
        
        1. Instruction: O que o modelo deve fazer
        2. Input (opcional): Contexto adicional
        3. Response: A resposta desejada
        
        O modelo aprende a padrão "dada esta instrução → gere esta resposta".
        
    DETALHE TÉCNICO:
        Durante o treinamento, o loss é calculado APENAS sobre os tokens
        da Response. Os tokens da Instruction/Input servem como contexto
        mas não geram gradiente (isso é feito automaticamente pelo SFTTrainer
        quando se usa o campo "text" formatado).
    
    Args:
        example: Dicionário com chaves 'instruction', 'input', 'output'.
    
    Returns:
        Dicionário com chave 'text' contendo o prompt formatado.
    """
    # Se o campo 'input' existe e não está vazio, usar template com input
    if example.get("input", "").strip():
        text = ALPACA_TEMPLATE.format(
            instruction=example["instruction"],
            input=example["input"],
            output=example["output"],
        )
    else:
        # Sem input adicional — usar template simplificado
        text = ALPACA_TEMPLATE_NO_INPUT.format(
            instruction=example["instruction"],
            output=example["output"],
        )
    
    return {"text": text}


def format_chat_prompt(
    example: Dict[str, str],
    tokenizer: Any,
) -> Dict[str, str]:
    """
    Formata um exemplo usando o chat template nativo do tokenizador.
    
    CONCEITO EDUCACIONAL:
        Cada modelo tem um "chat template" específico que define como
        delimitar mensagens de diferentes roles (system, user, assistant).
        
        Exemplos de chat templates:
        - Qwen2.5: <|im_start|>user\n{msg}<|im_end|>
        - Llama 3: <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        - Phi-3: <|user|>\n{msg}<|end|>
        
        Usar o template CORRETO é CRÍTICO. Um template errado pode
        degradar a performance do modelo em até 30% (mesmo com bons dados).
        
        O tokenizer.apply_chat_template() faz isso automaticamente.
    
    Args:
        example: Dicionário com 'instruction', 'input', 'output'.
        tokenizer: Tokenizador do modelo com chat_template configurado.
    
    Returns:
        Dicionário com chave 'text' contendo o prompt formatado.
    """
    # Construir a mensagem do usuário
    user_message = example["instruction"]
    if example.get("input", "").strip():
        user_message += f"\n\nInput: {example['input']}"
    
    # Estrutura de mensagens no formato OpenAI/HuggingFace
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": example["output"]},
    ]
    
    # apply_chat_template converte para o formato nativo do modelo
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,      # Retornar string, não tokens
        add_generation_prompt=False,  # Não adicionar prompt de geração
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
    Carrega e prepara um dataset para fine-tuning de SLMs.
    
    PIPELINE DE PREPARAÇÃO:
        1. Carregar dataset do Hugging Face Hub
        2. Limitar número de amostras (para testes rápidos)
        3. Dividir em treino/validação
        4. Formatar prompts no template escolhido
        5. Filtrar exemplos vazios ou muito curtos
    
    CONCEITO EDUCACIONAL:
        A divisão treino/validação é ESSENCIAL para detectar overfitting.
        Se o loss de treino diminui mas o loss de validação sobe,
        o modelo está memorizando em vez de aprender padrões gerais.
        
        Para SFT com poucos dados (< 10k exemplos), use 10% para validação.
        Para datasets maiores, 5% é suficiente.
    
    Args:
        dataset_name: Nome do dataset no Hugging Face Hub.
        split: Divisão do dataset a carregar.
        max_samples: Número máximo de amostras (None = usar tudo).
        test_size: Fração para validação (0.1 = 10%).
        seed: Semente aleatória para reprodutibilidade.
        format_fn: Função de formatação ("alpaca" ou "chat").
        tokenizer: Tokenizador (obrigatório se format_fn="chat").
    
    Returns:
        Tupla (train_dataset, eval_dataset) já formatados.
    
    Raises:
        ValueError: Se format_fn="chat" e tokenizer não for fornecido.
    """
    logger.info(f"Carregando dataset: {dataset_name} (split={split})")
    
    # --------------------------------------------------------------------------
    # 1. CARREGAR DATASET
    # --------------------------------------------------------------------------
    dataset = load_dataset(dataset_name, split=split)
    logger.info(f"Dataset carregado: {len(dataset)} exemplos")
    
    # --------------------------------------------------------------------------
    # 2. LIMITAR AMOSTRAS (para testes rápidos ou hardware limitado)
    # --------------------------------------------------------------------------
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        logger.info(f"Dataset limitado a {max_samples} exemplos")
    
    # --------------------------------------------------------------------------
    # 3. DIVIDIR EM TREINO E VALIDAÇÃO
    # --------------------------------------------------------------------------
    # A divisão é estratificada e reprodutível graças à seed
    split_dataset = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    logger.info(
        f"Divisão: {len(train_dataset)} treino | {len(eval_dataset)} validação"
    )
    
    # --------------------------------------------------------------------------
    # 4. FORMATAR PROMPTS
    # --------------------------------------------------------------------------
    if format_fn == "chat":
        if tokenizer is None:
            raise ValueError(
                "Tokenizer é obrigatório quando format_fn='chat'. "
                "Passe o tokenizer como argumento."
            )
        fmt_function = lambda ex: format_chat_prompt(ex, tokenizer)
    else:
        fmt_function = format_alpaca_prompt
    
    train_dataset = train_dataset.map(
        fmt_function,
        desc="Formatando dataset de treino",
    )
    eval_dataset = eval_dataset.map(
        fmt_function,
        desc="Formatando dataset de validação",
    )
    
    # --------------------------------------------------------------------------
    # 5. FILTRAR EXEMPLOS INVÁLIDOS
    # --------------------------------------------------------------------------
    # Remover exemplos onde o texto formatado ficou muito curto
    min_length = 20  # Mínimo de caracteres
    train_dataset = train_dataset.filter(
        lambda x: len(x["text"]) >= min_length,
        desc="Filtrando exemplos curtos (treino)",
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["text"]) >= min_length,
        desc="Filtrando exemplos curtos (validação)",
    )
    
    logger.info(
        f"Após filtragem: {len(train_dataset)} treino | {len(eval_dataset)} validação"
    )
    
    return train_dataset, eval_dataset


def create_preference_dataset(
    dataset_name: str = "argilla/ultrafeedback-binarized-preferences",
    max_samples: Optional[int] = 1000,
    seed: int = 42,
) -> Dataset:
    """
    Carrega e formata um dataset de preferências para DPO.
    
    CONCEITO EDUCACIONAL:
        O DPO (Direct Preference Optimization) requer dados no formato:
        - prompt: A instrução/pergunta
        - chosen: A resposta PREFERIDA (melhor qualidade)
        - rejected: A resposta REJEITADA (pior qualidade)
        
        O modelo aprende a aumentar a probabilidade de gerar respostas
        similares às "chosen" e diminuir para as "rejected".
        
        Fontes comuns de dados de preferência:
        1. UltraFeedback: Pares gerados por múltiplos LLMs e ranqueados
        2. Anthropic HH-RLHF: Pares de humanos (helpful vs harmful)
        3. Dados próprios: Anotação manual ou com LLM como juiz
    
    Args:
        dataset_name: Nome do dataset de preferências.
        max_samples: Número máximo de exemplos.
        seed: Semente aleatória.
    
    Returns:
        Dataset com colunas 'prompt', 'chosen', 'rejected'.
    """
    logger.info(f"Carregando dataset de preferências: {dataset_name}")
    
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        logger.warning(
            f"Erro ao carregar {dataset_name}: {e}. "
            f"Usando dataset sintético de demonstração."
        )
        # Fallback: criar dataset sintético para demonstração
        dataset = _create_synthetic_preference_data()
    
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
    
    logger.info(f"Dataset de preferências: {len(dataset)} exemplos")
    return dataset


def _create_synthetic_preference_data() -> Dataset:
    """
    Cria um dataset sintético de preferências para demonstração.
    
    NOTA: Este dataset é apenas para fins educacionais e de teste.
    Em produção, use dados reais anotados ou gerados por LLMs.
    """
    data = {
        "prompt": [
            "Explique o que é machine learning em termos simples.",
            "Qual a diferença entre Python e JavaScript?",
            "Como funciona o mecanismo de atenção em Transformers?",
            "O que é overfitting e como evitá-lo?",
            "Explique gradient descent de forma intuitiva.",
        ],
        "chosen": [
            "Machine learning é uma área da IA onde computadores aprendem padrões a partir de dados, sem serem explicitamente programados para cada tarefa. Pense nisso como ensinar um computador pelo exemplo, não pela regra.",
            "Python é usado principalmente para backend, ciência de dados e IA, com sintaxe simples e legível. JavaScript domina o desenvolvimento web (frontend e backend via Node.js), com execução assíncrona nativa.",
            "O mecanismo de atenção permite que o modelo 'olhe' para todas as palavras de uma frase simultaneamente e decida quais são mais relevantes para cada palavra. É como ler uma frase e saber que 'ele' se refere a 'João' porque ambos estão relacionados.",
            "Overfitting ocorre quando o modelo memoriza os dados de treinamento em vez de aprender padrões gerais. Para evitá-lo: use mais dados, aplique regularização (dropout, weight decay), faça validação cruzada e pare o treinamento cedo (early stopping).",
            "Imagine que você está no topo de uma montanha com nevoeiro e precisa descer ao vale. Gradient descent funciona assim: a cada passo, você sente a inclinação do terreno e caminha na direção mais íngreme para baixo. O tamanho do passo é o learning rate.",
        ],
        "rejected": [
            "Machine learning é quando o computador aprende sozinho.",
            "Python e JavaScript são linguagens de programação diferentes.",
            "Atenção é uma técnica usada em redes neurais.",
            "Overfitting é ruim. Use mais dados.",
            "Gradient descent é um algoritmo de otimização que minimiza a função de perda.",
        ],
    }
    
    return Dataset.from_dict(data)
