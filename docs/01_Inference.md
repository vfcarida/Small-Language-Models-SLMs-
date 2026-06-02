# Módulo 01: Inferência Básica

A inferência é o processo de utilizar um modelo pré-treinado para gerar previsões (neste caso, texto) a partir de um prompt fornecido.

## Carregando Modelos com Transformers
A biblioteca `transformers` da HuggingFace facilita o carregamento de SLMs.
Ao carregar, devemos atentar a:
- **`torch_dtype`:** Usar `bfloat16` ou `float16` economiza metade da memória da GPU (VRAM) em comparação com `float32`, mantendo a precisão necessária para inferência.
- **`device_map`:** Utilizar `"auto"` permite que a biblioteca distribua os pesos do modelo de forma ótima através das GPUs e memória RAM disponíveis.

## Estratégias de Decodificação
A forma como o modelo escolhe o "próximo token" afeta grandemente a qualidade e criatividade do texto gerado.
1. **Greedy Decoding:** Seleciona sempre o token com a maior probabilidade. É rápido e determinístico, mas pode gerar textos repetitivos.
2. **Sampling com Temperatura:** Amostra o próximo token a partir da distribuição de probabilidade.
   - `Temperature < 1`: Torna as escolhas mais focadas/previsíveis.
   - `Temperature > 1`: Aumenta a criatividade, podendo levar a textos incoerentes.
3. **Top-K Sampling:** Restringe a amostragem aos K tokens mais prováveis.
4. **Top-P (Nucleus) Sampling:** Restringe a amostragem aos tokens cuja soma das probabilidades cumulativas atinja o valor P. É o método preferido em produção, ajustando o número de tokens candidatos de acordo com a confiança do modelo.

## Métricas Críticas em Produção
- **TTFT (Time To First Token):** O tempo desde o envio do prompt até a geração do primeiro token. Crucial para a experiência do usuário.
- **Throughput:** A velocidade de geração, medida em tokens por segundo (t/s).

**Consulte o código:** `src/module_01_inference/basic_inference.py`
