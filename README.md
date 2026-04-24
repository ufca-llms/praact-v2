# praact

Pacote principal do repositorio Praact.

## Instalacao

Crie um ambiente virtual e instale o pacote em modo editavel:

```bash
python3 -m venv .venv312
.venv312/bin/python -m pip install -U pip
.venv312/bin/python -m pip install -e .
```

Se preferir instalar via arquivos de requirements:

```bash
.venv312/bin/python -m pip install -r requirements.txt
```

Para treinamento, instale as dependencias extras:

```bash
.venv312/bin/python -m pip install -r requirements-training.txt
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

A CLI exposta pelo pacote possui quatro subcomandos principais:

- `expand`: expande o tokenizer/modelo com o vocabulario do Praact.
- `train`: faz o fine-tuning supervisionado em pares `src -> tgt`.
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

### 2. Treinar o modelo expandido

O comando `train` faz supervised fine-tuning (SFT) em exemplos `src -> tgt`. O prompt de entrada e formatado antes da tokenizacao, a perda e mascarada na parte do prompt, e o modelo aprende apenas a prever a saida `tgt`. Por padrao, o treinamento usa LoRA, e em modelos instruct o ideal e ativar `--chat-template` para manter o formato esperado pelo tokenizer.

#### Instalar dependencias de treino

LoRA exige o pacote `peft`. Para instalar tudo o que o fluxo de treino precisa:

```bash
.venv312/bin/python -m pip install -r requirements-training.txt
```

#### Smoke test local no Mac

Esse comando roda um teste curto usando um modelo pequeno e apenas uma amostra reduzida do dataset:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv312/bin/praact train \
  outputs/Qwen--Qwen2.5-0.5B-Instruct \
  --train-json "data/starting kit text2picto/train.json" \
  --valid-json "data/starting kit text2picto/valid.json" \
  --output-dir outputs/train-smoke-qwen25-05b-instruct \
  --prompt-file prompts/telegraphic_instruction.txt \
  --chat-template \
  --dtype fp32 \
  --max-length 256 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-train-samples 256 \
  --max-eval-samples 64 \
  --logging-steps 10 \
  --eval-steps 50 \
  --save-steps 50 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing
```

#### Treino maior no Mac

Para um teste mais representativo, mas ainda viavel localmente, aumente a amostragem:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv312/bin/praact train \
  outputs/Qwen--Qwen2.5-0.5B-Instruct \
  --train-json "data/starting kit text2picto/train.json" \
  --valid-json "data/starting kit text2picto/valid.json" \
  --output-dir outputs/train-qwen25-05b-instruct-medium \
  --prompt-file prompts/telegraphic_instruction.txt \
  --chat-template \
  --dtype fp32 \
  --max-length 256 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --per-device-train-batch-size 2 \
  --per-device-eval-batch-size 2 \
  --gradient-accumulation-steps 4 \
  --max-train-samples 2000 \
  --max-eval-samples 256 \
  --logging-steps 10 \
  --eval-steps 100 \
  --save-steps 100 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --gradient-checkpointing
```

#### Treino completo

O arquivo `data/starting kit text2picto/train.json` tem `59.278` exemplos de treino, e `data/starting kit text2picto/valid.json` tem `4.397` exemplos de validacao. Para usar o conjunto inteiro, basta remover `--max-train-samples` e `--max-eval-samples` do comando.

Se o treino terminou rapido demais, o motivo mais provavel e justamente o uso desses limites.

#### Argumentos que mais afetam custo e duracao

- `--max-train-samples`: limita quantos exemplos do treino serao usados.
- `--max-eval-samples`: limita quantos exemplos da validacao serao usados.
- `--per-device-train-batch-size`: controla o batch por dispositivo.
- `--gradient-accumulation-steps`: aumenta o batch efetivo sem exigir mais memoria de uma vez.
- `--epochs`: controla quantas passadas pelo dataset serao feitas.
- `--gradient-checkpointing`: reduz uso de memoria, normalmente ao custo de mais tempo.

O checkpoint treinado e salvo no `--output-dir`. Depois disso, esse diretorio pode ser usado normalmente com `praact decode`, da mesma forma que um modelo expandido.

#### Testar um checkpoint treinado

Se o treino foi feito com LoRA, o `output-dir` salva um adapter. Ainda assim, ele pode ser usado diretamente com `praact decode`: o comando detecta o adapter, carrega automaticamente o modelo base e aplica o checkpoint treinado.

Exemplo com uma frase:

```bash
.venv312/bin/praact decode outputs/train-smoke-qwen25-05b-instruct \
  --prompt-file prompts/telegraphic_few_shot.txt \
  --prompt "They are attacked by a bird" \
  --chat-template \
  --max-new-tokens 4 \
  --dtype fp32 \
  --device cpu
```

Exemplo em lote no conjunto de validacao:

```bash
.venv312/bin/praact decode outputs/train-smoke-qwen25-05b-instruct \
  --prompt-file prompts/telegraphic_few_shot.txt \
  --input-json "data/starting kit text2picto/valid.json" \
  --output-json outputs/train-smoke-qwen25-05b-instruct_valid_predictions.json \
  --chat-template \
  --batch-size 8 \
  --max-new-tokens 16 \
  --dtype fp32 \
  --device cpu
```

Depois disso, voce pode avaliar normalmente:

```bash
.venv312/bin/praact evaluate \
  outputs/train-smoke-qwen25-05b-instruct_valid_predictions.json \
  "data/starting kit text2picto/valid.json"
```

Atalhos prontos no repositorio:

- [scripts/train_qwen25_05b_instruct_lora.sh](/Users/jayra/dev/praact-v2/scripts/train_qwen25_05b_instruct_lora.sh)
- [scripts/train_qwen3_4b_instruct_lora.sh](/Users/jayra/dev/praact-v2/scripts/train_qwen3_4b_instruct_lora.sh)

### 3. Gerar uma hipotese para uma frase

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

### 4. Gerar usando um prompt few-shot salvo em arquivo

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

### 5. Gerar em lote a partir do dataset de validacao

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

### 6. Avaliar as predicoes

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

### 7. Modelos instruct

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
