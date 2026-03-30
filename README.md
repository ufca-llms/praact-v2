# praact

Pacote principal do repositorio Praact.

## Instalacao

Crie um ambiente virtual e instale o pacote em modo editavel:

```bash
python3 -m venv .venv312
.venv312/bin/python -m pip install -U pip
.venv312/bin/python -m pip install -e .
```

## Como funciona

O PRAACT parte da ideia de adaptar um modelo de linguagem causal para operar com um vocabulario de Comunicacao Aumentativa e Alternativa (CAA). Em vez de gerar livremente no vocabulario completo do modelo original, o sistema passa a trabalhar com um conjunto de keywords e termos pictograficos extraidos do acervo do Praact.

O objetivo dessa adaptacao e aproximar o espaco de saida do modelo do tipo de representacao usado em CAA, permitindo que a geracao seja feita diretamente sobre termos mais proximos do dominio de pictogramas.

O fluxo do projeto tem tres etapas principais:

### 1. Expansao do modelo

O comando `expand` le um arquivo como `data/arasaac_en.json`, extrai as keywords do Praact e expande um modelo causal para que esse vocabulario possa ser usado durante a geracao.

Para cada keyword, o expansor tenta primeiro reaproveitar um token que ja exista naturalmente no tokenizer original. Isso e importante porque, em muitos casos, o proprio modelo ja possui uma representacao interna adequada para palavras comuns do vocabulario.

Quando nao existe um token adequado, o expansor adiciona um token novo ao tokenizer. Nesse caso, o token nao e inicializado aleatoriamente: sua embedding e alinhada ao espaco vetorial original do modelo por meio da media das embeddings dos subtokens que representam aquela keyword na tokenizacao original. O mesmo principio e usado para manter a compatibilidade com a camada de saida quando aplicavel.

Na pratica, isso faz com que os novos tokens de CAA sejam inseridos em uma regiao do espaco semantico coerente com o modelo ja treinado, em vez de surgirem como ids isolados sem relacao com o vocabulario existente.

Ao final, o diretorio salvo contem:

- o tokenizer atualizado
- o modelo atualizado
- um arquivo `praact_vocab.json` com o mapeamento entre keywords e ids de token

### 2. Geracao restrita ao vocabulario do Praact

O comando `decode` carrega o modelo expandido e usa o `praact_vocab.json` para restringir a geracao aos tokens permitidos. Na pratica, os logits do modelo sao mascarados para que a saida seja produzida dentro do vocabulario do Praact, em vez de usar livremente todo o vocabulario original do modelo.

Isso transforma o processo de geracao em uma forma de decodificacao controlada por vocabulario: o modelo continua usando seu conhecimento linguistico e contextual, mas a escolha do proximo token passa a ser limitada ao conjunto de termos de CAA definidos pelo Praact.

O `decode` pode ser usado de duas formas:

- para uma unica frase com `--prompt`
- para um dataset inteiro com `--input-json`

Quando `--prompt-file` e usado, o arquivo de prompt funciona como template e `{sentence}` e substituido pela frase de entrada.

### 3. Avaliacao

O comando `evaluate` compara um arquivo de predicoes no formato:

```json
[
  { "id": "...", "hyp": "..." }
]
```

com um arquivo de referencia contendo `id` e `tgt`.

As metricas calculadas sao:

- `sacrebleu`
- `meteor`
- `pictoer`

Essas metricas seguem o estilo da task ToPicto, comparando as hipoteses geradas com as sequencias de termos pictograficos de referencia. Assim, a avaliacao mede nao apenas fluencia textual, mas principalmente o quao proxima a sequencia produzida esta da representacao esperada no dominio de CAA.

## Como executar

A CLI exposta pelo pacote possui tres subcomandos principais:

- `expand`: expande o tokenizer/modelo com o vocabulario do Praact.
- `decode`: gera uma hipotese restrita ao vocabulario salvo em `praact_vocab.json`.
- `evaluate`: avalia as hipoteses com metricas no estilo da task ToPicto.

### 1. Expandir um modelo

Exemplo com `Qwen/Qwen2.5-0.5B`:

```bash
.venv312/bin/praact expand data/arasaac_en.json Qwen/Qwen2.5-0.5B --dtype fp32 --device cpu
```

Isso salva o modelo expandido em um diretorio como:

```text
outputs/Qwen--Qwen2.5-0.5B
```

No final, o comando imprime um resumo com quantas keywords ja existiam e quantas foram adicionadas.

### 2. Gerar uma hipotese para uma frase

Exemplo usando um prompt direto:

```bash
.venv312/bin/praact decode outputs/Qwen--Qwen2.5-0.5B \
  --prompt "Transform this sentence into a telegraphic sentence used in Augmentative and Alternative Communication.
Sentence: They are attacked by a bird
Telegraphic:" \
  --max-new-tokens 16 \
  --repetition-penalty 1.2 \
  --dtype fp32 \
  --device cpu
```

### 3. Gerar usando um prompt few-shot salvo em arquivo

O repositorio inclui um prompt reutilizavel em:

```text
prompts/telegraphic_few_shot.txt
```

Esse arquivo usa `{sentence}` como placeholder. Exemplo:

```bash
.venv312/bin/praact decode outputs/Qwen--Qwen2.5-0.5B \
  --prompt-file prompts/telegraphic_few_shot.txt \
  --prompt "They are attacked by a bird" \
  --max-new-tokens 16 \
  --repetition-penalty 1.2 \
  --dtype fp32 \
  --device cpu
```

### 4. Gerar em lote a partir do dataset de validacao

O modo em lote espera um JSON com itens contendo `id` e `src`, e grava um JSON contendo `id` e `hyp`.

Exemplo com o `valid.json`:

```bash
.venv312/bin/praact decode outputs/Qwen--Qwen2.5-0.5B \
  --prompt-file prompts/telegraphic_few_shot.txt \
  --input-json "data/starting kit text2picto/valid.json" \
  --output-json outputs/qwen25_05b_valid_predictions.json \
  --batch-size 8 \
  --max-new-tokens 16 \
  --repetition-penalty 1.2 \
  --dtype fp32 \
  --device cpu
```

### 5. Avaliar as predicoes

Depois de gerar o arquivo de predicoes, voce pode avaliar contra o arquivo de referencia:

```bash
.venv312/bin/praact evaluate \
  outputs/qwen25_05b_valid_predictions.json \
  "data/starting kit text2picto/valid.json"
```

A saida e um JSON com:

- `num_samples`
- `sacrebleu`
- `meteor`
- `pictoer`

### 6. Modelos instruct

Para modelos instruction-tuned, use `--chat-template` no `decode`:

```bash
.venv312/bin/praact decode outputs/Qwen--Qwen2.5-0.5B-Instruct \
  --prompt-file prompts/telegraphic_few_shot.txt \
  --prompt "Its label bears the logo." \
  --chat-template \
  --max-new-tokens 16 \
  --repetition-penalty 1.2 \
  --dtype fp32 \
  --device cpu
```

## Observacoes

- `--dtype` aceita `auto`, `fp16`, `bf16` e `fp32`.
- `--device` aceita `auto`, `cpu`, `mps` e `cuda`.
- Em Mac com Apple Silicon, `mps` pode funcionar bem, mas alguns modelos podem ser mais estaveis em `cpu`.
- Se adicionar uma dependencia nova ao projeto, reinstale com:

```bash
.venv312/bin/python -m pip install -e .
```
