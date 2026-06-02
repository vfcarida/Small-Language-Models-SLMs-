# Módulo 04: Pruning de Modelos (Poda)

O Pruning (Poda) visa reduzir o tamanho e o custo computacional do modelo removendo parâmetros (pesos) que são "inúteis" para o desempenho final.

## Magnitude Pruning (A Abordagem Simples)
O Magnitude Pruning assume que se o valor absoluto de um peso for próximo de zero ($|W| \approx 0$), ele não afeta o modelo e pode ser transformado em exatamente 0. 
**Problema:** Se esse peso pequeno for multiplicado por uma ativação (input) gigantesca (um outlier comum em LLMs), a sua remoção causará um erro imenso na rede.

## Wanda (Weight and Activation Pruning)
O algoritmo Wanda resolve o problema do Magnitude Pruning. Ele avalia a importância de um peso não apenas pelo seu próprio valor, mas também pela norma das ativações que passam por ele.
**A fórmula do Wanda:** $Score = |W| \times ||X||_2$
Dessa forma, pesos pequenos que lidam com ativações enormes são protegidos (não são podados).

## Calibração com Zipf-sampling
Para que o Wanda meça corretamente as ativações, precisamos passar dados reais pelo modelo. Se usarmos amostras enviesadas, as ativações estarão incorretas. O "Zipf Calibration" garante uma distribuição balanceada de palavras para calibrar a poda com máxima fidelidade.

**Consulte os códigos:** 
- `src/module_04_pruning/magnitude_pruning.py`
- `src/module_04_pruning/wanda_pruning.py`
