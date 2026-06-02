# Módulo 06: Alignment com DPO

Após o modelo aprender a seguir instruções (SFT), ele pode gerar respostas verbosas, perigosas ou que não soam como um assistente profissional. Para isso, aplicamos técnicas de alinhamento (Alignment).

## O Problema do RLHF
Historicamente, usava-se o RLHF (Reinforcement Learning from Human Feedback), onde se treinava um modelo separado (Reward Model) para avaliar as respostas e, em seguida, usava-se algoritmos de PPO para atualizar a rede. Essa infraestrutura é notoriamente complexa, instável e propensa a travar.

## A Solução: DPO (Direct Preference Optimization)
O DPO provou matematicamente que podemos extrair as preferências diretamente da probabilidade cruzada (logits) do modelo de linguagem, eliminando totalmente a necessidade de um Reward Model separado.

**Como Funciona:**
Fornecemos ao modelo um dataset contendo:
- A pergunta (Prompt)
- Uma resposta escolhida/boa (Chosen)
- Uma resposta rejeitada/ruim (Rejected)

A função de perda do DPO aumenta a probabilidade dos tokens que formam a resposta "Chosen" enquanto diminui penaliza a probabilidade da resposta "Rejected". O resultado é um modelo comportado, de maneira extremamente simples de treinar.

**Consulte o código:** `src/module_06_alignment/dpo_trainer.py`
