# Resumo dos candidatos — todos os modelos

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

## Comparação dos modelos adicionais — base completa

Além dos resultados do Gemma 2 e do Gemma 4, foram analisados os cinco modelos treinados em 4 épocas. A análise usou `valid.json`, top-25 candidatos, BF16 e batch de inferência 512.

Nos modelos desta execução, 4.375 exemplos foram processados. Os 22 exemplos restantes foram ignorados porque não possuíam contexto tokenizável no vocabulário preparado; portanto, as porcentagens abaixo usam 4.375 como denominador.

| Modelo | Parâmetros | Exemplos | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@25 | Rank médio* | MRR** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct — 4 épocas | 3B | 4.375 | 26,10% (1.142) | 35,82% (1.567) | 40,00% (1.750) | 45,05% (1.971) | **53,44% (2.338)** | 4,63 | **0,3252** |
| Llama 3.2 3B Instruct — 4 épocas | 3B | 4.375 | **25,44% (1.113)** | 34,58% (1.513) | 38,72% (1.694) | 44,48% (1.946) | 52,82% (2.311) | 4,73 | 0,3179 |
| SmolLM3-3B — 4 épocas | 3B | 4.375 | 25,42% (1.112) | 34,17% (1.495) | 38,33% (1.677) | 43,93% (1.922) | 50,26% (2.199) | **4,30** | 0,3136 |
| Qwen2.5-1.5B-Instruct — 4 épocas | 1,5B | 4.375 | 25,33% (1.108) | **35,02% (1.532)** | **38,97% (1.705)** | 44,32% (1.939) | 52,25% (2.286) | 4,65 | 0,3172 |
| DeepSeek-R1-Distill-Qwen-1.5B — 4 épocas | 1,5B | 4.375 | 22,31% (976) | 30,90% (1.352) | 35,09% (1.535) | 41,23% (1.804) | 49,39% (2.161) | 4,91 | 0,2847 |

\* Rank médio calculado apenas nos exemplos em que a resposta correta apareceu no top-25. Quanto menor, melhor.  
\*\* MRR calculado sobre os 4.375 exemplos processados; exemplos sem a resposta no top-25 contribuem com zero.

### Interpretação dos modelos adicionais

| Modelo | Pontos positivos | Pontos negativos |
|---|---|---|
| Qwen2.5-3B-Instruct | Melhor resultado geral: maior Hit@25, Hit@5, Hit@10 e MRR. | Não teve o melhor rank médio condicionado aos acertos. |
| Llama 3.2 3B Instruct | Maior Hit@1 entre os cinco modelos novos e segunda maior cobertura no Hit@25. | Ficou abaixo do Qwen2.5-3B nas métricas de cobertura a partir do top-3. |
| SmolLM3-3B | Melhor rank médio quando a resposta correta aparece. | Cobertura inferior aos dois Qwen e ao Llama. |
| Qwen2.5-1.5B-Instruct | Desempenho próximo dos modelos 3B; melhor Hit@3 e Hit@5 entre os cinco modelos novos. | Fica atrás do Qwen2.5-3B em Hit@1, Hit@10, Hit@25 e MRR. |
| DeepSeek-R1-Distill-Qwen-1.5B | Modelo menor e ainda alcança quase 50% de Hit@25. | Pior resultado nas métricas de cobertura e menor MRR. |

### Comparação geral entre todos os modelos

Considerando também Gemma 2 e Gemma 4, o **Qwen2.5-3B-Instruct** é o melhor modelo para uma aplicação que utiliza até 25 candidatos: alcançou 53,44% de Hit@25 e MRR 0,3252. Ele supera o Gemma 4 no Hit@25 (50,92%) e no MRR (0,3038).

| Modelo | Exemplos | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@25 | Rank médio | MRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | 4.375 | **26,10%** | 35,82% | **40,00%** | **45,05%** | **53,44%** | 4,63 | **0,3252** |
| Llama 3.2 3B Instruct | 4.375 | 25,44% | 34,58% | 38,72% | 44,48% | 52,82% | 4,73 | 0,3179 |
| Qwen2.5-1.5B-Instruct | 4.375 | 25,33% | **35,02%** | 38,97% | 44,32% | 52,25% | 4,65 | 0,3172 |
| Gemma 4 E2B Instruct | 4.397 | 24,29% | 33,09% | 37,50% | 42,94% | 50,92% | 4,70 | 0,3038 |
| SmolLM3-3B | 4.375 | 25,42% | 34,17% | 38,33% | 43,93% | 50,26% | **4,30** | 0,3136 |
| Gemma 2 2B Instruct | 4.397 | 24,11% | 33,18% | 37,53% | 42,76% | 48,94% | 4,30 | 0,3027 |
| DeepSeek-R1-Distill-Qwen-1.5B | 4.375 | 22,31% | 30,90% | 35,09% | 41,23% | 49,39% | 4,91 | 0,2847 |

Observação: Gemma 2 e Gemma 4 foram calculados anteriormente com 4.397 exemplos, enquanto os cinco modelos novos tiveram 22 exemplos sem contexto tokenizável e foram calculados com 4.375. As comparações são úteis, mas não são perfeitamente controladas pelo mesmo subconjunto.

O **Gemma 4 E2B Instruct** continua competitivo, mas fica abaixo do Qwen2.5-3B nesta análise completa. O **Qwen2.5-1.5B-Instruct** é a alternativa mais interessante quando se deseja reduzir o tamanho do modelo sem perder muito desempenho.

Para priorizar uma lista ampla de alternativas, recomenda-se o Qwen2.5-3B. Para priorizar o rank médio entre respostas encontradas, o SmolLM3-3B teve o melhor resultado entre os cinco modelos novos.

### Arquivos dos modelos adicionais

| Modelo | Markdown detalhado | JSON detalhado |
|---|---|---|
| Llama 3.2 3B Instruct | `outputs/all-models-4ep-candidates/Llama-3.2-3B-Instruct_top25_full_base.md` | `outputs/all-models-4ep-candidates/Llama-3.2-3B-Instruct_top25_full_base.json` |
| Qwen2.5 3B Instruct | `outputs/all-models-4ep-candidates/Qwen2.5-3B-Instruct_top25_full_base.md` | `outputs/all-models-4ep-candidates/Qwen2.5-3B-Instruct_top25_full_base.json` |
| SmolLM3-3B | `outputs/all-models-4ep-candidates/SmolLM3-3B_top25_full_base.md` | `outputs/all-models-4ep-candidates/SmolLM3-3B_top25_full_base.json` |
| Qwen2.5 1.5B Instruct | `outputs/all-models-4ep-candidates/Qwen2.5-1.5B-Instruct_top25_full_base.md` | `outputs/all-models-4ep-candidates/Qwen2.5-1.5B-Instruct_top25_full_base.json` |
| DeepSeek-R1-Distill-Qwen 1.5B | `outputs/all-models-4ep-candidates/DeepSeek-R1-Distill-Qwen-1.5B_top25_full_base.md` | `outputs/all-models-4ep-candidates/DeepSeek-R1-Distill-Qwen-1.5B_top25_full_base.json` |
