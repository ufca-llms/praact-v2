# Resumo dos candidatos — Gemma 2 e Gemma 4

Análise realizada com 50 exemplos de `valid.json`, usando os 25 candidatos mais prováveis para o próximo pictograma.

| Modelo | Exemplos | Top-k | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@25 | Rank médio* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B Instruct — 4 épocas | 50 | 25 | 14% | 24% | 26% | 32% | 44% | 6,64 |
| Gemma 4 E2B Instruct — 4 épocas | 50 | 25 | 14% | 20% | 32% | 38% | 48% | 5,88 |

\* Rank médio calculado somente nos exemplos em que a resposta correta apareceu no top-25.

## Configuração da análise

| Item | Valor |
|---|---|
| Conjunto de dados | `valid.json` |
| Número de exemplos | 50 por modelo |
| Número de candidatos | Top-25 |
| Precisão numérica | BF16 |
| Dispositivo | CUDA via `--device auto` |
| Resposta avaliada | Último pictograma da sequência |
| Resultado salvo em | `top25_candidates_50_examples.json` |

## Interpretação resumida

| Critério | Melhor modelo | Observação |
|---|---|---|
| Top-1 | Empate | Ambos acertaram 7 de 50 exemplos (14%). |
| Top-3 | Gemma 2 | Acertou 12 exemplos, contra 10 do Gemma 4. |
| Top-5 | Gemma 4 | Acertou 16 exemplos, contra 13 do Gemma 2. |
| Top-10 | Gemma 4 | Acertou 19 exemplos, contra 16 do Gemma 2. |
| Top-25 | Gemma 4 | Acertou 24 exemplos, contra 22 do Gemma 2. |
| Posição média | Gemma 4 | Quando acertou, colocou a resposta em posição média 5,88. |

## Arquivos detalhados

| Modelo | Relatório detalhado |
|---|---|
| Gemma 2 | `Gemma2_top25_candidates_50_examples.md` |
| Gemma 4 | `Gemma4_top25_candidates_50_examples.md` |

## Análise final — base completa

Esta é a análise final usando todos os exemplos rotulados disponíveis em `valid.json`. O script processou 4.397 exemplos por modelo, carregando cada modelo uma única vez e usando inferência em batches de 512.

| Item | Valor |
|---|---|
| Base | `data/data/starting kit text2picto/valid.json` |
| Exemplos analisados | 4.397 por modelo |
| Candidatos por exemplo | Top-25 |
| Batch de inferência | 512 |
| Precisão | BF16 |
| Modelos | Gemma 2 2B e Gemma 4 E2B Instruct, ambos com 4 épocas |
| Arquivo JSON completo | `outputs/gemma-candidates-results/full-base/` |

### Métricas na base completa

| Modelo | Exemplos | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@25 | Rank médio* | MRR** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemma 2 2B Instruct — 4 épocas | 4.397 | 24,11% (1.060) | 33,18% (1.459) | 37,53% (1.650) | 42,76% (1.880) | 48,94% (2.152) | 4,30 | 0,3027 |
| Gemma 4 E2B Instruct — 4 épocas | 4.397 | 24,29% (1.068) | 33,09% (1.455) | 37,50% (1.649) | 42,94% (1.888) | 50,92% (2.239) | 4,70 | 0,3038 |

\* Rank médio calculado apenas nos exemplos em que a resposta correta apareceu no top-25.  
\*\* MRR calculado sobre todos os 4.397 exemplos; exemplos sem a resposta no top-25 contribuem com zero.

### Comparação final

| Critério | Resultado | Interpretação |
|---|---|---|
| Hit@1 | Gemma 4 | Pequena vantagem: 1.068 contra 1.060 acertos. |
| Hit@3 | Gemma 2 | Pequena vantagem: 1.459 contra 1.455 acertos. |
| Hit@5 | Gemma 2 | Praticamente empate: 1.650 contra 1.649 acertos. |
| Hit@10 | Gemma 4 | Vantagem pequena: 1.888 contra 1.880 acertos. |
| Hit@25 | Gemma 4 | Melhor cobertura: 2.239 contra 2.152 acertos. |
| Rank médio quando encontrado | Gemma 2 | A resposta aparece mais acima quando é encontrada: 4,30 contra 4,70. |
| MRR global | Gemma 4 | Ligeira vantagem global: 0,3038 contra 0,3027. |

### Conclusão da base completa

Na base completa, os modelos ficaram muito próximos. O Gemma 4 foi ligeiramente melhor no primeiro candidato, no top-10, no top-25 e no MRR global. Sua principal vantagem foi encontrar a resposta correta em mais 87 exemplos no top-25.

O Gemma 2 foi ligeiramente melhor no top-3, top-5 e no rank médio condicionado aos casos encontrados. Isso indica que, quando ambos encontram a resposta, o Gemma 2 tende a posicioná-la um pouco mais alto; porém, o Gemma 4 cobre mais exemplos quando são considerados 25 candidatos.

Para uma aplicação que utiliza uma lista ampla de alternativas, o Gemma 4 é a melhor escolha marginal entre os dois. Para uma aplicação que prioriza a posição média da resposta quando ela aparece, o Gemma 2 permanece competitivo.

### Arquivos da análise completa

| Modelo | Markdown detalhado | JSON detalhado |
|---|---|---|
| Gemma 2 | `outputs/gemma-candidates-results/full-base/Gemma2_top25_full_base.md` | `outputs/gemma-candidates-results/full-base/Gemma2_top25_full_base.json` |
| Gemma 4 | `outputs/gemma-candidates-results/full-base/Gemma4_top25_full_base.md` | `outputs/gemma-candidates-results/full-base/Gemma4_top25_full_base.json` |

Os arquivos detalhados contêm, para cada exemplo, os 25 pictogramas candidatos, seus IDs, textos, probabilidades e a posição da resposta correta quando ela aparece no top-25.
