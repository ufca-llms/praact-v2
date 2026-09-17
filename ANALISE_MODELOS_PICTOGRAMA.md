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
| Llama-3.2-3B-Instruct | 3.9312 | 50.97 | 0.3310 | 0.4672 | 0.5263 | 0.6046 | 0.4261 | 207.97 | 4 |
| Qwen2.5-3B-Instruct | 3.9763 | 53.32 | 0.3257 | 0.4641 | 0.5250 | 0.6023 | 0.4226 | 213.50 | 4 |
| SmolLM3-3B | 4.0199 | 55.70 | 0.3245 | 0.4652 | 0.5213 | 0.5965 | 0.4241 | 227.97 | 4 |
| Qwen2.5-1.5B-Instruct | 4.0446 | 57.09 | 0.3229 | 0.4587 | 0.5171 | 0.5913 | 0.4175 | 224.92 | 5 |
| DeepSeek-R1-Distill-Qwen-1.5B | 4.2242 | 68.32 | 0.3081 | 0.4415 | 0.4996 | 0.5742 | 0.4014 | 274.49 | 5 |
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
2. Llama-3.2-3B-Instruct: segundo melhor resultado geral, forte e consistente.
3. Qwen2.5-3B-Instruct: melhor alternativa robusta entre os Qwen.
4. SmolLM3-3B: competitivo, aberto e interessante pela licença Apache-2.0.
5. Qwen2.5-1.5B-Instruct: melhor opção leve da família Qwen.
6. DeepSeek-R1-Distill-Qwen-1.5B: não compensou para esta tarefa.

## Arquivos de resultado

- Qwen: `outputs/qwen-pictogram-results/`
- Gemma: `outputs/imported-pictogram-results/gemma-pictogram-results/`
- Llama: `outputs/imported-pictogram-results/llama-pictogram-results/`
- DeepSeek: `outputs/imported-pictogram-results/deepseek-pictogram-results/`
- SmolLM3: `outputs/imported-pictogram-results/smollm-pictogram-results/`

## Recomendação

Para seguir com apenas um modelo, o melhor candidato é o `Gemma-2-2B-it`. Ele foi melhor em praticamente todas as métricas e ainda é menor que os modelos de 3B. O `Llama-3.2-3B-Instruct` ficou como a alternativa de maior qualidade depois do Gemma. Se a licença/acesso do Gemma ou Llama for um problema, o `SmolLM3-3B` vira uma alternativa muito boa por ser Apache-2.0. Se a prioridade for continuar dentro da família Qwen, use o `Qwen2.5-3B-Instruct` principal, não as variantes v2/v3.
