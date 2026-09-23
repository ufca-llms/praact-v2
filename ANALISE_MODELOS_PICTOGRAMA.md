# Comparativo dos modelos de previsão de pictogramas

Este documento resume os modelos testados com o fluxo de `src/praact_pictogram_prediction/README.md`: preparação do tokenizer de IDs ARASAAC, treino causal sobre o campo `pictos`, `score` com o prompt `6632 5441 6456` e `evaluate` no `valid.json`.

## Configuração comum

- Tarefa: previsão do próximo pictograma ARASAAC.
- Tokenização: vocabulário de IDs de pictogramas com `--output-vocabulary pictogram-id` e `--token-format raw-id`.
- Dados: `data/data/arasaac_en.json` e `data/data/starting kit text2picto/{train,valid}.json`.
- Validação durante o treino: `--valid-json` mantido em todos os experimentos.
- Comprimento máximo: `--max-length 128`.
- Precisão usada nos treinos novos: `bf16`.
- Otimizador/configuração base: `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--epochs 1`, salvo quando indicado nas variantes.
- Métricas principais: menor `loss`/`perplexity` é melhor; maior `accuracy@k` e `mrr` é melhor; menor `mean_rank`/`median_rank` é melhor.

## Resultado geral

| Modelo | Loss | Perplexity | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR | Mean rank | Median rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma-2-2B-it | 3.8480 | 46.90 | 0.3333 | 0.4767 | 0.5361 | 0.6155 | 0.4360 | 180.91 | 4 |
| Gemma-4-E2B base | 3.8631 | 47.61 | 0.3334 | 0.4757 | 0.5362 | 0.6140 | 0.4306 | 195.33 | 4 |
| Llama-3.2-3B-Instruct | 3.9312 | 50.97 | 0.3310 | 0.4672 | 0.5263 | 0.6046 | 0.4261 | 207.97 | 4 |
| Qwen2.5-3B-Instruct | 3.9763 | 53.32 | 0.3257 | 0.4641 | 0.5250 | 0.6023 | 0.4226 | 213.50 | 4 |
| SmolLM3-3B | 4.0199 | 55.70 | 0.3245 | 0.4652 | 0.5213 | 0.5965 | 0.4241 | 227.97 | 4 |
| Qwen2.5-1.5B-Instruct | 4.0446 | 57.09 | 0.3229 | 0.4587 | 0.5171 | 0.5913 | 0.4175 | 224.92 | 5 |
| DeepSeek-R1-Distill-Qwen-1.5B | 4.2242 | 68.32 | 0.3081 | 0.4415 | 0.4996 | 0.5742 | 0.4014 | 274.49 | 5 |
| Gemma-4-E2B 3 épocas | 4.4148 | 82.67 | 0.3265 | 0.4644 | 0.5235 | 0.5981 | 0.4208 | 253.72 | 5 |
| Gemma-4-E2B 6 épocas | 8.6322 | 5609.50 | 0.3174 | 0.4370 | 0.4884 | 0.5500 | 0.4013 | 454.81 | 6 |
| Qwen2.5-3B-Instruct v1 | 4.0042 | 54.83 | 0.3226 | 0.4641 | 0.5224 | 0.5981 | 0.4208 | 209.29 | 4 |
| Qwen2.5-3B-Instruct v2 | 4.5045 | 90.43 | 0.2800 | 0.4124 | 0.4696 | 0.5480 | 0.3734 | 354.12 | 7 |
| Qwen2.5-3B-Instruct v3 | 4.7216 | 112.34 | 0.2686 | 0.4006 | 0.4567 | 0.5304 | 0.3608 | 488.90 | 8 |

Todos os `evaluate` usaram 4.364 exemplos válidos, 33 exemplos ignorados e 27.693 alvos de predição.

## Modelos e características

### Gemma-2-2B-it

- Modelo base: `google/gemma-2-2b-it`.
- Característica: modelo instruction-tuned de 2B, forte para o tamanho e bem eficiente.
- Treino: tokenizer de pictogramas preparado via `prepare-model`; fine-tuning completo em `bf16`; batch de treino recomendado/usado para 48 GB: `8-16`, batch de avaliação `16-32`, com acumulação para manter batch efetivo maior.
- Resultado: melhor modelo do conjunto. Teve menor loss, menor perplexity, maior Acc@1, Acc@10 e MRR, além do menor `mean_rank`.
- Ponto positivo: melhor equilíbrio entre qualidade e tamanho.
- Ponto negativo: exige aceitar licença/acesso no Hugging Face.

### Gemma-4-E2B

- Modelo base: `google/gemma-4-E2B`.
- Característica: modelo Gemma 4 de geração condicional, com vocabulário/text config adaptado ao tokenizer de pictogramas. O experimento exigiu correção manual dos tokens especiais (`pad=0`, `bos=2`, `eos=3`) dentro do `config.json`.
- Treino base: `bf16`, `--epochs 1`, `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--max-length 128`; para 80 GB de VRAM foram usados batches maiores, como `--per-device-train-batch-size 32`, `--per-device-eval-batch-size 64`, `--gradient-accumulation-steps 1`.
- Resultado base: ficou muito forte, praticamente empatado com o Gemma 2. Teve `loss` 3.8631, `perplexity` 47.61, `accuracy@1` 0.3334 e `accuracy@10` 0.6140.
- Resultado com 3 épocas: piorou em loss/perplexity e ranking médio, apesar de manter Acc@1 razoável. Indica que mais treino começou a prejudicar a generalização.
- Resultado com 6 épocas: degradou bastante, com perplexity 5609.50 e queda forte em Acc@10. Esse regime não é recomendado.
- Ponto positivo: o treino base de 1 época é um dos melhores resultados gerais.
- Ponto negativo: mais épocas pioraram o modelo; o melhor uso foi o treinamento curto.

### Llama-3.2-3B-Instruct

- Modelo base: `meta-llama/Llama-3.2-3B-Instruct`.
- Característica: modelo instruction-tuned de 3B da família Llama, gated no Hugging Face e com boa capacidade geral para tarefas de sequência.
- Treino: tokenizer de pictogramas preparado via `prepare-model`; fine-tuning completo em `bf16`; configuração usada para 48 GB de VRAM: `--epochs 1`, `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--per-device-train-batch-size 8`, `--per-device-eval-batch-size 16`, `--gradient-accumulation-steps 4`.
- Resultado: segundo melhor modelo geral. Ficou abaixo do Gemma, mas acima do Qwen 3B em loss, perplexity, Acc@1, Acc@10, MRR e mean rank.
- Ponto positivo: desempenho forte e consistente, com ranking médio melhor que Qwen 3B e SmolLM3.
- Ponto negativo: exige acesso/licença no Hugging Face e ocupa mais espaço que modelos de 1.5B/2B.

### Qwen2.5-3B-Instruct

- Modelo base: `Qwen/Qwen2.5-3B-Instruct`.
- Característica: instruction model de 3B, estável e muito competitivo.
- Treino base: `bf16`, `--epochs 1`, `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--max-length 128`. Nos treinos com mais VRAM foram usados batches maiores, como `--per-device-train-batch-size 32/64`, avaliação `64/128` e baixa acumulação.
- Resultado: segundo melhor resultado geral entre os modelos principais, ficando pouco atrás do Gemma.
- Ponto positivo: consistente, robusto e com boa Acc@10.
- Ponto negativo: ocupa mais disco/VRAM que modelos de 1.5B/2B.

### SmolLM3-3B

- Modelo base: `HuggingFaceTB/SmolLM3-3B`.
- Característica: modelo aberto Apache-2.0 de 3B, leve para uso e sem gate de licença como Gemma/Llama.
- Treino: `bf16`, `--epochs 1`, `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--per-device-train-batch-size 8`, `--per-device-eval-batch-size 16`, `--gradient-accumulation-steps 4`.
- Observação técnica: precisou corrigir IDs especiais no `config.json` após `prepare-model` (`pad=0`, `bos=2`, `eos=3`). O salvamento final falhou por falta de disco, então os resultados foram gerados a partir de `checkpoint-1000`.
- Resultado: muito próximo do Qwen 3B em Acc@3/MRR, mas com loss e perplexity um pouco piores.
- Ponto positivo: licença permissiva e desempenho competitivo.
- Ponto negativo: exigiu ajuste manual de config e cuidado com espaço em disco.

### Qwen2.5-1.5B-Instruct

- Modelo base: `Qwen/Qwen2.5-1.5B-Instruct`.
- Característica: modelo menor, rápido e barato de treinar.
- Treino: mesmo fluxo de pictogramas, com `valid-json`; usado como baseline menor da família Qwen.
- Resultado: ficou abaixo do Qwen 3B e SmolLM3, mas ainda competitivo para o tamanho.
- Ponto positivo: boa relação custo/resultado.
- Ponto negativo: menor capacidade que 2B/3B melhores.

### DeepSeek-R1-Distill-Qwen-1.5B

- Modelo base: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.
- Característica: modelo destilado a partir de DeepSeek-R1 sobre base Qwen 1.5B; mais voltado a raciocínio do que a previsão simples de sequência.
- Treino: `bf16`, `--epochs 1`, `--learning-rate 5e-5`, `--weight-decay 0.01`, `--warmup-ratio 0.03`, `--per-device-train-batch-size 32`, `--per-device-eval-batch-size 64`, `--gradient-accumulation-steps 1`.
- Resultado: pior que Qwen 1.5B puro neste conjunto, com maior perplexity e menor accuracy.
- Ponto positivo: treino rápido e licença permissiva.
- Ponto negativo: o viés de modelo de raciocínio/distill não pareceu ajudar nesta tarefa de próximo pictograma.

## Variantes Qwen 3B

Foram testadas três variantes adicionais do Qwen2.5-3B-Instruct para tentar melhorar o resultado com hiperparâmetros diferentes.

| Variante | Característica do experimento | Resultado |
|---|---|---|
| v1 | Variante próxima do baseline, com ajuste alternativo de treino. | Ficou levemente abaixo do Qwen 3B principal, mas ainda competitiva. |
| v2 | Treino mais agressivo/alterado. | Piorou bastante: perplexity subiu para 90.43 e Acc@10 caiu para 0.5480. |
| v3 | Variante ainda mais distante/longa. | Pior resultado entre as variantes: perplexity 112.34 e Acc@10 0.5304. |

A conclusão dessas variantes é que aumentar ou alterar demais o treino não melhorou o resultado. O Qwen 3B principal foi o melhor da família Qwen testada.

## Ranking prático

1. Gemma-2-2B-it: melhor qualidade geral.
2. Gemma-4-E2B base: praticamente empatado com o Gemma 2, mas levemente pior em MRR, Acc@10 e mean rank.
3. Llama-3.2-3B-Instruct: forte e consistente.
4. Qwen2.5-3B-Instruct: melhor alternativa robusta entre os Qwen.
5. SmolLM3-3B: competitivo, aberto e interessante pela licença Apache-2.0.
6. Qwen2.5-1.5B-Instruct: melhor opção leve da família Qwen.
7. DeepSeek-R1-Distill-Qwen-1.5B: não compensou para esta tarefa.

## Gemma 4 por épocas

| Variante | Épocas | Loss | Perplexity | Acc@1 | Acc@10 | MRR | Mean rank | Avaliação |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Gemma-4-E2B base | 1 | 3.8631 | 47.61 | 0.3334 | 0.6140 | 0.4306 | 195.33 | Melhor configuração do Gemma 4. |
| Gemma-4-E2B 3 épocas | 3 | 4.4148 | 82.67 | 0.3265 | 0.5981 | 0.4208 | 253.72 | Piorou a generalização. |
| Gemma-4-E2B 6 épocas | 6 | 8.6322 | 5609.50 | 0.3174 | 0.5500 | 0.4013 | 454.81 | Forte degradação; não recomendado. |

O melhor resultado do Gemma 4 foi o treino base de 1 época. Os treinos de 3 e 6 épocas indicam overfitting ou degradação do ajuste para esta tarefa.

## Arquivos de resultado

- Qwen: `outputs/qwen-pictogram-results/`
- Gemma: `outputs/imported-pictogram-results/gemma-pictogram-results/`
- Gemma 4: `outputs/imported-pictogram-results/gemma4-pictogram-results/`
- Llama: `outputs/imported-pictogram-results/llama-pictogram-results/`
- DeepSeek: `outputs/imported-pictogram-results/deepseek-pictogram-results/`
- SmolLM3: `outputs/imported-pictogram-results/smollm-pictogram-results/`

## Recomendação

Para seguir com apenas um modelo, o melhor candidato continua sendo o `Gemma-2-2B-it`. O `Gemma-4-E2B` base ficou muito próximo e é a melhor alternativa dentro dos novos testes, mas os treinos de 3 e 6 épocas não devem ser usados. O `Llama-3.2-3B-Instruct` ficou como a alternativa de maior qualidade depois dos Gemma. Se a licença/acesso do Gemma ou Llama for um problema, o `SmolLM3-3B` vira uma alternativa muito boa por ser Apache-2.0. Se a prioridade for continuar dentro da família Qwen, use o `Qwen2.5-3B-Instruct` principal, não as variantes v2/v3.


| Modelo | Característica | Parâmetros principais | Loss | Perplexity | Accuracy@1 | Accuracy@10 | MRR | Mean rank | Median rank | Avaliação |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-3B-Instruct original | Melhor Qwen testado; treino base mais equilibrado | bf16; epochs=1; learning-rate=5e-5; weight-decay=0.01; warmup-ratio=0.03; max-length=128 | 3.9763 | 53.32 | 0.3257 | 0.6023 | 0.4226 | 213.50 | 4 | Melhor resultado da família Qwen |
| Qwen2.5-3B-Instruct v1 | Variante próxima do baseline, com ajuste alternativo de treino | bf16; variação leve dos hiperparâmetros do treino base | 4.0042 | 54.83 | 0.3226 | 0.5981 | 0.4208 | 209.29 | 4 | Muito próxima do original, mas não superou nas métricas principais |
| Qwen2.5-3B-Instruct v2 | Variante mais agressiva/alterada | bf16; alteração mais forte nos parâmetros de treino | 4.5045 | 90.43 | 0.2800 | 0.5480 | 0.3734 | 354.12 | 7 | Piorou claramente o desempenho |
