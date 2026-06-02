# Módulo 00: Fundamentos de SLMs e NLP

Neste módulo, exploramos os componentes essenciais que formam a base dos Small Language Models (SLMs) e dos Transformers em geral.

## O que é um SLM?
Small Language Models (SLMs) são modelos baseados na arquitetura Transformer com uma quantidade reduzida de parâmetros (geralmente entre 1 a 7 bilhões) em comparação com Large Language Models (LLMs) como GPT-4 (trilhões de parâmetros). A vantagem dos SLMs é que eles podem ser treinados, ajustados e executados em hardwares acessíveis, como GPUs de consumidor ou até mesmo em dispositivos móveis (Edge AI).

## Tokenização
Antes de processar qualquer texto, o modelo precisa convertê-lo em números. A tokenização é o processo de dividir o texto em pedaços (tokens) e mapeá-los para um dicionário de IDs inteiros (Vocabulário).
- **Subword Tokenization (BPE):** Algoritmos como Byte-Pair Encoding (BPE) dividem palavras raras em sub-palavras conhecidas, lidando eficientemente com vocabulários abertos.
- **Tokens Especiais:** Modelos utilizam tokens especiais como `<|endoftext|>` ou `<|im_start|>` para demarcar o fim de uma sequência ou a estrutura de uma conversa.

**Consulte o código:** `src/module_00_fundamentals/tokenizer_explorer.py`

## Mecanismo de Atenção (Self-Attention)
O coração da arquitetura Transformer. O Self-Attention permite que o modelo calcule a relevância (score) de cada token em relação a todos os outros tokens na sequência de entrada, permitindo que o modelo capture dependências de longo alcance.
- **Queries (Q), Keys (K) e Values (V):** A atenção é calculada mapeando a entrada em três matrizes. O score é o produto escalar de Q e K, normalizado por softmax, e multiplicado por V.
- **GQA (Grouped Query Attention):** Para otimizar a memória durante a geração, modelos modernos como o Qwen e o Llama utilizam GQA, compartilhando Keys e Values entre múltiplas cabeças de atenção.

**Consulte o código:** `src/module_00_fundamentals/attention_visualizer.py`
