"""
==============================================================================
MÓDULO 00 — EXPLORADOR DE TOKENIZADORES
==============================================================================
NÍVEL: 🟢 Iniciante
TEMPO ESTIMADO: 15 minutos de leitura + execução

OBJETIVO:
    Entender como modelos de linguagem "enxergam" texto. Antes de processar
    qualquer palavra, o modelo precisa convertê-la em números. Esse processo
    se chama TOKENIZAÇÃO.

CONCEITOS ABORDADOS:
    1. O que é tokenização e por que é necessária
    2. Algoritmo BPE (Byte-Pair Encoding) — usado por GPT, Llama, Qwen
    3. Tokens especiais (EOS, PAD, BOS) e seus papéis
    4. Comparação de vocabulários entre modelos diferentes
    5. Impacto da tokenização na eficiência e custo

PRÉ-REQUISITOS:
    pip install transformers

REFERÊNCIAS:
    - Sennrich et al., "Neural Machine Translation of Rare Words with
      Subword Units", ACL 2016 (Paper original do BPE)
    - Kudo & Richardson, "SentencePiece: A simple and language independent
      subword tokenizer", EMNLP 2018
    - HuggingFace, "Tokenizers Documentation", 2024

COMO EXECUTAR:
    python src/module_00_fundamentals/tokenizer_explorer.py
==============================================================================
"""

import logging
from typing import Dict, List

from transformers import AutoTokenizer

# Configuração de logging para output educacional
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def explore_tokenizer(model_id: str, sample_texts: List[str]) -> Dict:
    """
    Explora em detalhe como um tokenizador específico processa texto.
    
    CONCEITO EDUCACIONAL — O QUE É TOKENIZAÇÃO?
    =============================================
    Modelos de linguagem não processam texto diretamente. Eles trabalham
    com NÚMEROS (tensores). A tokenização é o processo de converter
    texto em uma sequência de inteiros, onde cada inteiro representa
    um "token" no vocabulário do modelo.
    
    EXEMPLO:
        Texto: "Olá mundo"
        Tokens: ["Ol", "á", " mundo"]
        IDs:    [1234, 567, 8901]
    
    POR QUE NÃO USAR LETRAS OU PALAVRAS?
    
    1. LETRAS: Sequências muito longas (ineficiente para atenção O(n²))
    2. PALAVRAS: Vocabulário infinito (cada palavra nova seria desconhecida)
    3. SUBPALAVRAS (BPE): Equilíbrio perfeito — palavras comuns são um token,
       palavras raras são divididas em pedaços conhecidos
    
    ALGORITMO BPE (Byte-Pair Encoding):
    =====================================
    1. Começar com todos os caracteres individuais como vocabulário
    2. Contar pares de caracteres adjacentes mais frequentes
    3. Unir o par mais frequente em um novo token
    4. Repetir até atingir o tamanho desejado de vocabulário
    
    Exemplo do BPE em ação:
        Corpus: ["low", "lower", "lowest", "new", "newer"]
        Passo 1: Tokens = ['l','o','w','e','r','s','t','n']
        Passo 2: Par mais frequente = ('l','o') → merge em 'lo'
        Passo 3: Par mais frequente = ('lo','w') → merge em 'low'
        ... e assim por diante
    
    Args:
        model_id: ID do modelo no Hugging Face Hub (ex: "Qwen/Qwen2.5-1.5B").
        sample_texts: Lista de textos para tokenizar.
    
    Returns:
        Dicionário com informações detalhadas do tokenizador.
    """
    logger.info(f"{'='*60}")
    logger.info(f"EXPLORANDO TOKENIZADOR: {model_id}")
    logger.info(f"{'='*60}")
    
    # --------------------------------------------------------------------------
    # 1. CARREGAR O TOKENIZADOR
    # --------------------------------------------------------------------------
    # AutoTokenizer detecta automaticamente o tipo correto de tokenizador
    # para o modelo especificado (BPE, WordPiece, SentencePiece, etc.)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,  # Necessário para modelos com código custom
    )
    
    # --------------------------------------------------------------------------
    # 2. INFORMAÇÕES GERAIS DO VOCABULÁRIO
    # --------------------------------------------------------------------------
    vocab_size = tokenizer.vocab_size
    logger.info(f"\n📊 INFORMAÇÕES DO VOCABULÁRIO:")
    logger.info(f"   Tamanho do vocabulário: {vocab_size:,} tokens")
    logger.info(f"   Tipo do tokenizador: {type(tokenizer).__name__}")
    logger.info(f"   Modelo de tokenização: {tokenizer.is_fast and 'Fast (Rust)' or 'Slow (Python)'}")
    
    # --------------------------------------------------------------------------
    # 3. TOKENS ESPECIAIS
    # --------------------------------------------------------------------------
    # Tokens especiais são sinais de controle que o modelo usa para
    # entender a estrutura da entrada:
    #
    # - BOS (Beginning of Sequence): Marca o início de uma sequência
    # - EOS (End of Sequence): Marca o fim (o modelo para de gerar aqui)
    # - PAD (Padding): Preenche sequências curtas para igualar tamanhos no batch
    # - UNK (Unknown): Substitui tokens desconhecidos (raro em BPE moderno)
    logger.info(f"\n🏷️  TOKENS ESPECIAIS:")
    logger.info(f"   BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    logger.info(f"   EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    logger.info(f"   PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    logger.info(f"   UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    
    # Listar todos os tokens especiais adicionais
    if hasattr(tokenizer, "additional_special_tokens"):
        special = tokenizer.additional_special_tokens[:10]  # Limitar a 10
        if special:
            logger.info(f"   Tokens adicionais: {special}")
    
    # --------------------------------------------------------------------------
    # 4. TOKENIZAÇÃO DE EXEMPLOS
    # --------------------------------------------------------------------------
    logger.info(f"\n📝 TOKENIZAÇÃO DE EXEMPLOS:")
    
    results = []
    for text in sample_texts:
        # Tokenizar o texto
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        
        # Calcular a taxa de compressão (caracteres por token)
        # Uma taxa maior significa tokenização mais eficiente
        compression_ratio = len(text) / len(tokens) if tokens else 0
        
        logger.info(f"\n   Texto: \"{text}\"")
        logger.info(f"   Tokens ({len(tokens)}): {tokens}")
        logger.info(f"   IDs: {token_ids}")
        logger.info(f"   Caracteres: {len(text)} → Tokens: {len(tokens)}")
        logger.info(f"   Taxa de compressão: {compression_ratio:.2f} chars/token")
        
        # Mostrar a reconstrução inversa (decode)
        # Isso demonstra que a tokenização é REVERSÍVEL
        reconstructed = tokenizer.decode(token_ids)
        logger.info(f"   Reconstrução: \"{reconstructed}\"")
        
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
    Compara como diferentes tokenizadores processam o mesmo texto.
    
    CONCEITO EDUCACIONAL — POR QUE COMPARAR?
    ==========================================
    Diferentes modelos usam vocabulários diferentes. Isso afeta:
    
    1. CUSTO: APIs cobram por token. Um tokenizador que usa menos tokens
       para o mesmo texto é mais barato.
    
    2. EFICIÊNCIA: Menos tokens = sequências menores = atenção mais rápida
       (atenção é O(n²), onde n = número de tokens).
    
    3. CAPACIDADE: Modelos têm limite de contexto em TOKENS, não em palavras.
       Se o tokenizador é eficiente, cabe mais texto no contexto.
    
    4. IDIOMA: Tokenizadores treinados em pouco texto em português podem
       dividir palavras em muitos tokens, reduzindo a eficiência.
    
    EXEMPLO PRÁTICO:
        Texto em português: "A inteligência artificial transformará o mundo"
        - GPT-2 (inglês): 8 tokens (português é "caro")
        - Llama-3: 6 tokens (melhor suporte multilíngue)
        - Qwen2.5: 5 tokens (excelente em múltiplos idiomas)
    
    Args:
        model_ids: Lista de IDs de modelos para comparar.
        text: Texto a ser tokenizado por todos os modelos.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPARAÇÃO DE TOKENIZADORES")
    logger.info(f"Texto: \"{text}\"")
    logger.info(f"Caracteres: {len(text)}")
    logger.info(f"{'='*60}")
    
    for model_id in model_ids:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True
            )
            tokens = tokenizer.tokenize(text)
            ratio = len(text) / len(tokens) if tokens else 0
            
            logger.info(f"\n   📦 {model_id}")
            logger.info(f"      Vocabulário: {tokenizer.vocab_size:,}")
            logger.info(f"      Tokens: {len(tokens)} → {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
            logger.info(f"      Compressão: {ratio:.2f} chars/token")
            
        except Exception as e:
            logger.warning(f"   ❌ {model_id}: Erro ao carregar — {e}")


def demonstrate_special_tokens_impact():
    """
    Demonstra o impacto dos tokens especiais no treinamento.
    
    CONCEITO EDUCACIONAL:
        Tokens especiais parecem triviais, mas causam bugs sérios se
        configurados incorretamente:
        
        1. PAD TOKEN AUSENTE:
           - Modelos como Qwen2.5 e Llama não definem pad_token por padrão
           - Sem pad_token, batching de sequências de tamanhos diferentes FALHA
           - Solução: pad_token = eos_token (mais comum) ou um token custom
        
        2. EOS TOKEN ERRADO:
           - Se o modelo não reconhece o EOS, ele nunca para de gerar
           - Resulta em textos infinitos e incoerentes
        
        3. CHAT TEMPLATE:
           - Modelos instruction-tuned esperam delimitadores específicos
           - Usar o template errado degrada severamente a qualidade
    """
    logger.info(f"\n{'='*60}")
    logger.info("DEMONSTRAÇÃO: IMPACTO DOS TOKENS ESPECIAIS")
    logger.info(f"{'='*60}")
    
    model_id = "Qwen/Qwen2.5-1.5B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # Verificar se pad_token está definido
        logger.info(f"\n   Modelo: {model_id}")
        logger.info(f"   pad_token definido? {tokenizer.pad_token is not None}")
        
        if tokenizer.pad_token is None:
            logger.info("   ⚠️  PAD TOKEN NÃO DEFINIDO — Isso causaria erro no batching!")
            logger.info("   🔧 FIX: Definindo pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"   ✅ pad_token agora é: '{tokenizer.pad_token}'")
        
        # Demonstrar batching com padding
        texts = ["Frase curta.", "Esta é uma frase consideravelmente mais longa para demonstrar padding."]
        
        batch = tokenizer(
            texts,
            padding=True,       # Preencher sequências curtas com pad_token
            truncation=True,    # Cortar sequências longas
            max_length=32,
            return_tensors="pt",
        )
        
        logger.info(f"\n   📦 BATCHING COM PADDING:")
        logger.info(f"   Input shape: {batch['input_ids'].shape}")
        logger.info(f"   Frase 1 (tokens): {batch['input_ids'][0].tolist()}")
        logger.info(f"   Frase 2 (tokens): {batch['input_ids'][1].tolist()}")
        logger.info(f"   Attention mask 1: {batch['attention_mask'][0].tolist()}")
        logger.info(
            f"   ℹ️  Zeros no attention_mask = tokens de padding (ignorados pelo modelo)"
        )
        
    except Exception as e:
        logger.error(f"Erro na demonstração: {e}")


def main():
    """
    Executa todas as demonstrações do explorador de tokenizadores.
    
    FLUXO EDUCACIONAL:
    1. Explorar o tokenizador do Qwen2.5-1.5B em detalhe
    2. Comparar tokenizadores de diferentes famílias
    3. Demonstrar o impacto dos tokens especiais
    """
    # --------------------------------------------------------------------------
    # TEXTOS DE EXEMPLO
    # --------------------------------------------------------------------------
    # Incluímos textos variados para mostrar como diferentes tipos de conteúdo
    # são tokenizados de formas diferentes:
    sample_texts = [
        # Texto simples em inglês (muito eficiente — poucas divisões)
        "Hello, how are you?",
        
        # Texto em português (pode ser menos eficiente dependendo do tokenizador)
        "A inteligência artificial está transformando o mundo corporativo.",
        
        # Código Python (tokenizadores modernos são otimizados para código)
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        
        # Texto técnico com termos especializados
        "The Transformer architecture uses multi-head self-attention with O(n²) complexity.",
        
        # Números e símbolos matemáticos (desafiadores para tokenizadores)
        "3.14159 * (2^10) = 3216.99",
        
        # Emoji e caracteres Unicode (teste de robustez)
        "O modelo 🤖 processou 1.5B parâmetros ✅",
    ]
    
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 1: Explorar Tokenizador Principal
    # --------------------------------------------------------------------------
    logger.info("\n" + "🔍 " * 20)
    logger.info("PARTE 1: EXPLORAÇÃO DETALHADA DO TOKENIZADOR")
    logger.info("🔍 " * 20)
    
    explore_tokenizer("Qwen/Qwen2.5-1.5B", sample_texts)
    
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 2: Comparar Tokenizadores
    # --------------------------------------------------------------------------
    logger.info("\n" + "⚖️  " * 20)
    logger.info("PARTE 2: COMPARAÇÃO ENTRE TOKENIZADORES")
    logger.info("⚖️  " * 20)
    
    models_to_compare = [
        "Qwen/Qwen2.5-1.5B",        # Excelente multilíngue
        "meta-llama/Llama-3.2-1B",   # Forte em inglês
        "microsoft/Phi-3.5-mini-instruct",  # Otimizado para raciocínio
    ]
    
    comparison_text = (
        "O processamento de linguagem natural (PLN) é uma subárea da "
        "inteligência artificial que foca na interação entre computadores "
        "e linguagem humana."
    )
    
    compare_tokenizers(models_to_compare, comparison_text)
    
    # --------------------------------------------------------------------------
    # DEMONSTRAÇÃO 3: Tokens Especiais
    # --------------------------------------------------------------------------
    logger.info("\n" + "🏷️  " * 20)
    logger.info("PARTE 3: TOKENS ESPECIAIS E SEUS IMPACTOS")
    logger.info("🏷️  " * 20)
    
    demonstrate_special_tokens_impact()
    
    logger.info(f"\n{'='*60}")
    logger.info("✅ EXPLORAÇÃO DE TOKENIZADORES CONCLUÍDA!")
    logger.info("📚 Próximo passo: Veja o módulo attention_visualizer.py")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
