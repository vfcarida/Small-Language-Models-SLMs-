# Imagem base oficial do PyTorch com suporte a CUDA 12.1
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# Variáveis de ambiente para build do flash-attention e runtime do vLLM
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=4
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROCESS_METHOD=spawn

# Instalação de dependências do sistema
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copiando apenas o requirements primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instalação das bibliotecas Python, assegurando o flash_attn
# Compilar o flash_attn pode ser demorado, por isso a variável MAX_JOBS
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia do código fonte do projeto
COPY . .

# Comando padrão
CMD ["/bin/bash"]
