# Módulo 05: Quantização Numérica

A quantização é a técnica de converter os pesos (e possivelmente as ativações) do modelo, de alta precisão (FP32 ou FP16), para formatos de menor precisão, como INT8 ou INT4.

## Bitsandbytes (Quantização Dinâmica)
O formato `bitsandbytes` (ex: QLoRA) quantiza os pesos "on-the-fly". Quando a matriz precisa ser multiplicada, ela é dequantizada para FP16/BF16, a multiplicação ocorre e o resultado é mantido.
- **Vantagem:** Muito fácil de usar; reduz incrivelmente o uso de memória.
- **Desvantagem:** O overhead de dequantização torna a inferência mais lenta.

## AWQ (Activation-aware Weight Quantization)
Para implantações em produção e inferência ultra-rápida (edge computing), usamos a quantização AWQ. 
O AWQ descobre que 1% dos pesos ("salient weights") são cruciais para a inteligência da rede. Em vez de simplesmente aplicar uma quantização Round-to-Nearest e destruir esses 1%, o AWQ escala esses pesos antes da quantização. O resultado é um modelo que pode rodar inteiramente em INT4 em núcleos otimizados de GPU.
- **Vantagem:** Modelos estáticos muito rápidos e leves.
- **Desvantagem:** Exige uma etapa de compilação/calibração pesada.

**Consulte os códigos:** 
- `src/module_05_quantization/bitsandbytes_quant.py`
- `src/module_05_quantization/awq_quantization.py`
