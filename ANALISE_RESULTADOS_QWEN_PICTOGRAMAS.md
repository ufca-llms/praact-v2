# Análise dos resultados dos modelos Qwen

## Escopo da avaliação

Todos os arquivos `evaluate.json` principais usam o mesmo conjunto de validação:

- exemplos avaliados: `4.364`;
- exemplos ignorados: `33`;
- posições de previsão: `27.693`.

Isso permite comparar os resultados diretamente. Quanto menor `loss`, `perplexity`, `mean_rank` e `median_rank`, melhor. Quanto maior `accuracy@k` e `mrr`, melhor.

## Comparação geral

| Modelo/versão | Loss | Perplexity | Accuracy@1 | Accuracy@3 | Accuracy@5 | Accuracy@10 | MRR | Mean rank | Mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 4,0446 | 57,09 | 32,29% | 45,87% | 51,71% | 59,13% | 41,75% | 224,92 | 5 |
| Qwen2.5-3B original | **3,9763** | **53,32** | 32,57% | 46,41% | **52,50%** | **60,23%** | 42,26% | 213,50 | **4** |
| Qwen2.5-3B v1 | 4,0042 | 54,83 | 32,26% | 46,41% | 52,24% | 59,81% | 42,08% | **209,29** | **4** |
| Qwen2.5-3B v2 | 4,5045 | 90,43 | 28,00% | 41,24% | 46,96% | 54,80% | 37,34% | 354,12 | 7 |
| Qwen2.5-3B v3 | 4,7216 | 112,34 | 26,86% | 40,06% | 45,67% | 53,04% | 36,08% | 488,90 | 8 |
| Resultado legado `results/` | 3,9845 | 53,76 | **32,79%** | **46,53%** | 52,35% | 60,06% | **42,35%** | 219,08 | **4** |

Os valores de `accuracy` e `mrr` estão apresentados como percentuais. `Mean rank` e `median rank` são posições; posições menores são melhores.

## Interpretação por modelo

### Qwen2.5-1.5B

Pontos positivos:

- desempenho razoável considerando o menor tamanho;
- `accuracy@1` de 32,29% e `accuracy@10` de 59,13%;
- modelo mais leve e barato para executar.

Pontos negativos:

- pior `loss` e `perplexity` entre os modelos nomeados;
- `median_rank` igual a 5, enquanto o Qwen2.5-3B original chega a 4;
- fica atrás do 3B original em todas as métricas principais, exceto por pequenas diferenças de ranking.

Conclusão: é uma boa opção quando memória, latência ou custo são prioridade, mas não é o melhor modelo em qualidade.

### Qwen2.5-3B original

Pontos positivos:

- melhor `loss` e `perplexity` entre os modelos nomeados;
- melhor `accuracy@5` e `accuracy@10` entre os modelos nomeados;
- `median_rank` igual a 4;
- desempenho consistente em todas as métricas.

Pontos negativos:

- a melhora em relação ao 1.5B é positiva, mas relativamente pequena;
- o `mean_rank` é um pouco pior que o da v1;
- ainda existe forte tendência a prever tokens funcionais, como “of”, “the” e “a”, no exemplo individual de `score`.

Conclusão: é o melhor modelo geral entre os modelos identificados corretamente. Deve ser a escolha principal.

### Qwen2.5-3B v1

Pontos positivos:

- melhor `mean_rank` entre os modelos nomeados: 209,29;
- `accuracy@3` e `median_rank` iguais aos do 3B original;
- no teste individual de `score`, apresentou a maior confiança no primeiro candidato: 22,36%.

Pontos negativos:

- `loss`, `perplexity`, `accuracy@1`, `accuracy@5` e `accuracy@10` ficaram ligeiramente piores que no 3B original;
- a maior confiança em um único candidato não se traduziu em melhor desempenho global.

Conclusão: é uma alternativa competitiva para aplicações que valorizem a posição média dos candidatos, mas não supera o 3B original no conjunto de métricas.

### Qwen2.5-3B v2

Pontos positivos:

- continua produzindo previsões válidas dentro do vocabulário de pictogramas;
- é uma variação útil para mostrar o efeito de uma configuração de treino diferente.

Pontos negativos:

- piora significativa em todas as métricas;
- `perplexity` sobe de 53,32 no 3B original para 90,43;
- `accuracy@1` cai para 28,00%;
- `median_rank` aumenta de 4 para 7.

Conclusão: a configuração da v2 não é recomendada.

### Qwen2.5-3B v3

Pontos positivos:

- mantém o funcionamento do pipeline e gera previsões válidas;
- a configuração pode ser útil como experimento de regularização, mas não como modelo final.

Pontos negativos:

- pior resultado em todas as métricas avaliadas;
- `perplexity` de 112,34;
- `accuracy@1` de apenas 26,86%;
- `mean_rank` de 488,90 e mediana 8.

Conclusão: é a pior versão. O aumento de épocas combinado com learning rate baixo, maior regularização e contexto de 192 tokens não trouxe benefício. O resultado é compatível com uma configuração que não se adaptou bem ao objetivo ou com sobreajuste/treinamento excessivo.

### Resultado legado `outputs/results/`

Os arquivos:

- `outputs/results/evaluate.json`;
- `outputs/evaluate_qwen5B.json`;
- `outputs/results/score.json`;
- `outputs/score_qwen5B.json`;

representam o mesmo resultado: os arquivos de `score` são idênticos e os arquivos de avaliação diferem apenas por espaço em branco. Portanto, não devem ser tratados como dois modelos diferentes.

Esse resultado legado é ligeiramente melhor em `accuracy@1`, `accuracy@3` e `mrr`, mas não há metadado suficiente nesses JSONs para confirmar qual modelo o produziu. O nome `qwen5B` sozinho não é evidência suficiente para atribuí-lo a um modelo específico.

## Análise dos resultados de `score`

O prompt usado foi:

```text
6632 5441 6456
```

Os principais candidatos foram majoritariamente palavras funcionais:

- `of`;
- `the`;
- `a`;
- `some`;
- `one`;
- `my`.

Isso revela uma tendência do modelo, mas não permite afirmar se o primeiro candidato está correto sem conhecer o pictograma esperado para esse prompt específico.

Comparação do primeiro candidato:

| Modelo/versão | Primeiro pictograma | Palavra associada | Probabilidade |
|---|---:|---|---:|
| Qwen2.5-1.5B | 8476 | the | 8,89% |
| Qwen2.5-3B original | 7074 | of | 19,53% |
| Qwen2.5-3B v1 | 2627 | one | **22,36%** |
| Qwen2.5-3B v2 | 7074 | of | 9,38% |
| Qwen2.5-3B v3 | 7074 | of | 7,13% |
| Resultado legado | 7074 | of | 9,13% |

A v1 foi a mais confiante nesse exemplo isolado, mas as métricas de validação mostram que confiança maior não significa necessariamente melhor desempenho geral.

## Recomendação final

1. **Modelo recomendado:** `Qwen2.5-3B original`.
2. **Alternativa:** `Qwen2.5-3B v1`, caso o `mean_rank` seja a métrica prioritária.
3. **Modelo leve:** `Qwen2.5-1.5B`, quando custo ou latência forem mais importantes que qualidade.
4. **Evitar:** Qwen2.5-3B v2 e v3.
5. **Não considerar como modelo separado:** os arquivos `results/` e `qwen5B`, pois são duplicados/legados sem identificação confiável.

O 3B original é a melhor escolha equilibrada porque combina a menor perplexidade, a maior `accuracy@5` e `accuracy@10` entre os modelos identificados, além de mediana de ranking igual a 4.
