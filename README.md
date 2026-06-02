# 🚀 Small Language Models (SLMs) Masterclass

Bem-vindo ao repositório definitivo para aprendizado e implementação de **Small Language Models (SLMs)**. 

Este projeto foi desenhado com dois propósitos principais:
1. **Educacional:** Servir como uma base de conhecimento para a equipe (desde fundamentos teóricos até implementações avançadas State-of-the-Art).
2. **Entrega de Valor:** Fornecer um pipeline de produção completo e funcional (End-to-End) capaz de treinar, quantizar, avaliar e gerar relatórios executivos de um SLM.

O modelo padrão adotado pela infraestrutura é o **Qwen2.5-1.5B**, escolhido por sua excelente relação entre tamanho, velocidade e capacidade lógica.

---

## 📚 Trilha de Conhecimento (Módulos Educacionais)

O código fonte está dividido em módulos progressivos. Todo o código é extensamente comentado em português com a teoria matemática e conceitual por trás das técnicas.

- 🟢 **[Módulo 00: Fundamentos](docs/00_Fundamentals.md)** (`src/module_00_fundamentals/`) - Tokenização, BPE e Self-Attention (Visualizador).
- 🟢 **[Módulo 01: Inferência](docs/01_Inference.md)** (`src/module_01_inference/`) - Geração de texto, Latência (TTFT) e Estratégias de Decodificação.
- 🟡 **[Módulo 02: SFT & LoRA](docs/02_Supervised_Finetuning.md)** (`src/module_02_sft/`) - Supervised Fine-Tuning usando PEFT (LoRA e QLoRA/NF4).
- 🔴 **[Módulo 03: Destilação](docs/03_Distillation.md)** (`src/module_03_distillation/`) - Generalized Knowledge Distillation (GKD) e KL Divergence.
- 🔴 **[Módulo 04: Pruning](docs/04_Pruning.md)** (`src/module_04_pruning/`) - Compressão de pesos (Magnitude vs Wanda SOTA).
- 🟡 **[Módulo 05: Quantização](docs/05_Quantization.md)** (`src/module_05_quantization/`) - Redução numérica 4-bits e 8-bits (bitsandbytes vs AWQ).
- 🔴 **[Módulo 06: Alignment](docs/06_Alignment.md)** (`src/module_06_alignment/`) - Direct Preference Optimization (DPO).
- 🟡 **[Módulo 07: Avaliação](docs/07_Evaluation.md)** (`src/module_07_evaluation/`) - LM-Eval Harness (MMLU, GSM8K), Perplexidade e Gráficos Radar.

---

## 🛠️ Pipeline de Produção (End-to-End)

Para cumprir com o objetivo de **Entrega**, desenvolvemos uma pipeline unificada. Ao rodá-la, o sistema fará automaticamente o download do modelo base, avaliará suas capacidades iniciais (baseline), realizará o treinamento SFT+LoRA, salvará os adaptadores, os combinará ao modelo, reavaliará a nova versão, e gerará os gráficos comparativos.

### Como Rodar a Entrega Inicial:

1. Instale as dependências (recomendado usar um ambiente virtual):
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. Execute o orquestrador principal:
   ```bash
   python pipeline/run_full_pipeline.py
   ```
   *(Nota: O treinamento pode levar algumas horas dependendo da sua GPU. Uma GPU com no mínimo 8GB de VRAM é recomendada, embora 12GB+ seja o ideal para batches maiores).*

3. Gere o Relatório Executivo Markdown:
   ```bash
   python pipeline/generate_report.py
   ```

O relatório final e os gráficos (Radar e Loss) estarão disponíveis na pasta `results/`.

---

## ⚙️ Configuração Centralizada

Todos os hiperparâmetros (modelos, learning rates, epochs, estratégias de quantização) estão centralizados no arquivo **`configs/default_config.yaml`**. 
Para mudar o modelo base (por exemplo, de Qwen para Llama-3.2-1B), basta alterar este arquivo sem tocar no código-fonte.

---

## 👥 Para a Equipe
Se você é novo no mundo de SLMs/LLMs, recomendo fortemente ler os arquivos na pasta `docs/` e executar os scripts do módulo `00` e `01` antes de tentar treinar um modelo na pipeline completa. Os comentários nos códigos-fonte em `src/` servem como verdadeiras "aulas" sobre o assunto. Bons estudos!
