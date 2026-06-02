# Módulo 07: Avaliação e Benchmarks

A avaliação baseada puramente na sensação visual de quem usa o modelo (vibe check) não tem validade técnica ou acadêmica. Para provar o valor do projeto, precisamos de métricas sólidas.

## Avaliações Intrínsecas
- **Perplexidade (PPL):** Mede o quão "surpreso" o modelo fica com uma amostra de texto. Uma perplexidade baixa (próxima a 1) indica que o modelo prevê bem o texto. É calculada exponenciando a Entropia Cruzada negativa da sequência de tokens.
- **Latência / Throughput:** Crucial para produção. Medimos o tempo para a primeira resposta (TTFT - Time To First Token) e a quantidade de tokens gerados por segundo.

## EleutherAI LM Evaluation Harness
O padrão da indústria para avaliação de Modelos de Linguagem baseia-se em datasets estáticos extensos de múltipla escolha ou raciocínio.
- **MMLU (Massive Multitask Language Understanding):** Mede o conhecimento acadêmico bruto através de 57 áreas do conhecimento humano.
- **GSM8K:** Avalia a capacidade de pensamento estruturado ("Chain-of-Thought") através de matemática do ensino fundamental. O foco não está nas contas em si, mas sim na habilidade lógica sequencial.

A avaliação nesses benchmarks permite plotar gráficos radar comparativos demonstrando empiricamente o ganho gerado pelo treinamento.

**Consulte os códigos:** 
- `src/module_07_evaluation/perplexity_eval.py`
- `src/module_07_evaluation/benchmark_runner.py`
- `src/module_07_evaluation/results_visualizer.py`
