# Módulo 03: Knowledge Distillation

A Destilação de Conhecimento (KD) é a arte de transferir a "inteligência" de um modelo grande e lento (Teacher) para um modelo pequeno e rápido (Student).

## A Teoria dos "Soft Labels"
Num treinamento tradicional, dizemos ao modelo que a resposta correta é A e todas as outras (B, C, D) estão 100% erradas (Hard Labels). 
Na destilação, o Teacher fornece as probabilidades reais (Soft Labels). Por exemplo, o Teacher sabe que embora "Maçã" seja o melhor exemplo de fruta, "Pera" também é aceitável, mas "Carro" não é. O Student aprende essa distribuição sutil de conhecimento.

## KL Divergence
A função de perda principal na destilação é a Divergência de Kullback-Leibler (KL Divergence). Ela mede quão diferente é a distribuição de probabilidade do Student em relação à do Teacher. O objetivo do otimizador é minimizar essa divergência.

## GKD (Generalized Knowledge Distillation)
O KD clássico sofre de "Exposure Bias". O Student só vê os exemplos perfeitos do dataset de treino e não aprende o que fazer quando ele próprio comete um erro de geração.
O GKD (On-policy Distillation) resolve isso permitindo que o Student *gere* sequências. Durante a geração, se o Student cometer um pequeno desvio, consultamos o Teacher para saber qual é o melhor caminho a partir daquele desvio.

**Consulte o código:** `src/module_03_distillation/gkd_pipeline.py`
