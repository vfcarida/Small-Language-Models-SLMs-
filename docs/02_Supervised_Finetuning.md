# Módulo 02: Supervised Fine-Tuning (SFT) e LoRA

Para transformar um modelo base (que apenas completa texto) em um assistente útil, aplicamos o Supervised Fine-Tuning (SFT). No entanto, treinar todos os parâmetros (Full Fine-Tuning) é muito custoso. É aqui que entra o LoRA.

## LoRA (Low-Rank Adaptation)
O LoRA é um método Parameter-Efficient Fine-Tuning (PEFT).
- **Como funciona:** O LoRA congela os pesos originais do modelo pré-treinado ($W_0$) e treina apenas matrizes de decomposição de posto (rank) baixo ($A$ e $B$) que são adicionadas aos pesos originais durante o forward pass: $W = W_0 + \Delta W = W_0 + BA$.
- **Vantagem:** Reduz o número de parâmetros treináveis em até 99%, permitindo o treinamento em hardware convencional e produzindo checkpoints minúsculos (os adapters).

## QLoRA
O QLoRA leva a eficiência ao extremo ao quantizar os pesos congelados do modelo base para 4-bits usando o formato NormalFloat4 (NF4).
- O NF4 é matematicamente otimizado para a distribuição normal dos pesos de redes neurais.
- Permite o fine-tuning de um modelo de 7B parâmetros em uma única GPU de 8GB-12GB de VRAM.

## Datasets e Templates
Para o SFT, os dados devem ter o formato Instrução -> Resposta. O formato Alpaca é amplamente adotado. Modelos mais modernos usam formatos baseados em "conversas" (Chat Templates), onde o tokenizador é responsável por injetar os tokens de turno de fala e marcadores especiais.

**Consulte os códigos:** 
- `src/module_02_sft/sft_lora.py`
- `src/module_02_sft/sft_qlora.py`
