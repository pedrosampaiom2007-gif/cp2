# Halter + DocMind RAG · Treino de academia

**Prompt Engineering & AI · FIAP · 2º Semestre 2026 · Módulos 1 e 2 (CKP01 + CKP02)**

**Integrantes:**
Luan de Araujo Carneiro (RM573691) · Pedro Sampaio Mochnacs Arruda (RM573522) ·
Raul Sampaio Mochnacs Arruda (RM573523) · Pedro Ribeiro Lopes (RM570083) ·
Kevin Rodrigues de Melo (RM571777) · Pedro Vianna (RM570747) · Lana Ozeki (RM569795)

---

# CKP02 — DocMind RAG

Pipeline RAG completo sobre documentos reais de treino e atividade física, adicionado ao
projeto do CKP01 como o pacote `app/rag/` e como a aba **📚 DocMind RAG** da interface.

```
documentos/*.pdf ──► LOAD ──► SPLIT ──► EMBED ──────────► STORE
                    pypdf    Recursive  nomic-embed-text   ChromaDB
                             Character  (Ollama)           treino_academia_<chunk>
                             TextSplitter                      │
                                                               ▼
 pergunta ──► embed_query ──► RETRIEVE (top 12, where=filtros) ──► RERANK (cross-encoder, top 4)
                                                                        │
                                                                        ▼
                                    resposta citando [n] ◄── GENERATE (gemma4:cloud, temperatura 0)
```

| Componente | Implementação |
|---|---|
| Base de conhecimento | 5 PDFs reais (tabela abaixo), até 5 páginas de cada — `app/rag/fontes.py`, `app/rag/baixar_documentos.py` |
| Load | `app/rag/carregador.py` — um `Document` por página, com título, tipo, categoria, público, ano, URL e página do original |
| Split | `app/rag/divisor.py` — `RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ". ", " ", ""])`, chunk 256 / 512 / 1024 com overlap de 12,5% (32 / 64 / 128) |
| Embed | `app/rag/vetores.py` — `OllamaEmbeddings(model="nomic-embed-text")` com `embed_documents()` e `embed_query()` |
| Store | `app/rag/vetores.py` — `chromadb.PersistentClient`, coleção `treino_academia_<chunk_size>` (distância cosseno) |
| Retrieve + metadata filtering | `BaseVetorial.buscar(..., where=)` e `montar_filtro(tipo, categoria, publico, ano_minimo, fonte)` |
| Reranking | `app/rag/reranker.py` — `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` reordena 12 candidatos e entrega 4 ao LLM |
| Generate | `app/rag/pipeline.py` — `ChatPromptTemplate \| ChatOllama(gemma4:cloud, temperature=0) \| StrOutputParser`, resposta cita `[n]` e lista documento, página e chunk |
| Tool para o CKP03 | `app.rag.pipeline.buscar(consulta)` |
| RAGAS | `app/rag/avaliacao.py` — faithfulness + answer_relevancy, 8 perguntas × 3 configurações |
| Interface | `app/main.py`, aba **📚 DocMind RAG** — chat, filtros de metadados, reranking liga/desliga e fontes citadas |
| Notebook | `CKP02_DocMind_RAG.ipynb` — o pipeline inteiro, etapa por etapa, com a tabela RAGAS |

## Base de conhecimento

| # | Documento | Autoria | Tipo | Público | Fonte |
|---:|---|---|---|---|---|
| 1 | Guia de Atividade Física para a População Brasileira (2021) | Ministério da Saúde | guia_oficial | adultos | [PDF](https://bvsms.saude.gov.br/bvs/publicacoes/guia_atividade_fisica_populacao_brasileira.pdf) |
| 2 | Diretrizes da OMS para atividade física e comportamento sedentário: num piscar de olhos (2020) | Organização Mundial da Saúde | guia_oficial | geral | [PDF](https://iris.who.int/bitstream/handle/10665/337001/9789240014886-por.pdf) |
| 3 | Recuperação entre séries no treino de força: revisão sistemática e meta-análise | Revista Brasileira de Medicina do Esporte | artigo_cientifico | adultos | [SciELO](https://www.scielo.br/j/rbme/a/Y9vYkwhHhbzKcKNSG9Ft85s/?format=pdf&lang=pt) |
| 4 | Influência de variáveis do treinamento contra-resistência sobre a força muscular de idosos: uma revisão sistemática com ênfase nas relações dose-resposta | Revista Brasileira de Medicina do Esporte | artigo_cientifico | idosos | [SciELO](https://www.scielo.br/j/rbme/a/8z4PZxrP4fPvJgfccndzx8M/?format=pdf&lang=pt) |
| 5 | Sessão de treinamento de força supervisionada aumenta a carga total levantada e as respostas subjetivas em sujeitos treinados | Journal of Physical Education | artigo_cientifico | adultos | [SciELO](https://www.scielo.br/j/jpe/a/5fnPtNjMh8Swt3g8kcHTgkt/?format=pdf&lang=pt) |

`python -m app.rag.baixar_documentos` baixa os cinco PDFs e grava em `documentos/` um recorte
de **no máximo 5 páginas** de cada um: as páginas recebem uma nota pela frequência das
palavras-chave do tema e as cinco melhores entram, na ordem original. O arquivo
`documentos/fontes.json` registra quais páginas do original foram usadas, e a citação de cada
resposta aponta a página do documento original.

Se algum site bloquear o download, baixe o PDF pelo navegador e importe:

```bash
python -m app.rag.baixar_documentos --arquivo ms_guia_atividade_fisica ~/Downloads/guia.pdf
```

## Como executar o CKP02

```bash
pip install -r requirements.txt
cp .env.example .env                 # preencha OLLAMA_API_KEY
python -m app.rag.baixar_documentos  # baixa e recorta os 5 PDFs em documentos/
python -m app.rag.avaliacao          # indexa 256/512/1024 e gera a tabela RAGAS em resultados/
python -m app.main                   # interface, aba "📚 DocMind RAG"
```

Uma pergunta direto pelo terminal:

```bash
python -m app.rag.pipeline "Quantas vezes por semana devo fazer fortalecimento muscular?"
```

No **Colab**: abra `CKP02_DocMind_RAG.ipynb`, cadastre `OLLAMA_API_KEY` em *Secrets* e rode
*Executar tudo*. A primeira célula clona este repositório e instala as dependências.

## Como adicionar novos documentos à base

1. Coloque o arquivo (`.pdf`, `.txt` ou `.md`) na pasta `documentos/`.
2. Crie ao lado um `.json` com o mesmo nome, com os metadados usados nos filtros e na citação:

   ```json
   {
     "titulo": "Nome do documento",
     "autoria": "Órgão ou revista",
     "tipo": "artigo_cientifico",
     "categoria": "prescricao_treino_forca",
     "publico": "adultos",
     "ano": 2024,
     "url": "https://link-da-fonte"
   }
   ```

   Sem o `.json` o documento entra com tipo `outro` e público `geral`.
3. Reindexe: botão **Indexar documentos** na aba DocMind, ou `python -m app.rag.avaliacao`.
   A coleção é recriada sempre que o conjunto de chunks muda.

Para uma fonte que deva ser baixada automaticamente, acrescente um `FonteDocumento` em
`app/rag/fontes.py` (URL, metadados e palavras-chave para a seleção de páginas) e rode
`python -m app.rag.baixar_documentos`.

## Comparação de chunking com RAGAS

As três configurações respondem às mesmas 8 perguntas (`PERGUNTAS_TESTE` em
`app/rag/avaliacao.py`), com o mesmo reranker e o mesmo modelo. O RAGAS usa o `gemma4:cloud`
como juiz e o `nomic-embed-text` para o answer_relevancy.

| chunk_size | overlap | faithfulness médio | answer_relevancy médio |
|---:|---:|---:|---:|
| 256 | 32 | _rodar_ | _rodar_ |
| 512 | 64 | _rodar_ | _rodar_ |
| 1024 | 128 | _rodar_ | _rodar_ |

`python -m app.rag.avaliacao` grava em `resultados/`:

- `ragas_por_pergunta.csv` — pergunta, resposta, fontes e as duas métricas por configuração;
- `ragas_resumo.csv` — médias por configuração;
- `ragas_relatorio.md` — as duas tabelas e a escolha final justificada pelos números.

O vencedor é o de maior faithfulness médio, com answer_relevancy como desempate, e vira o
padrão da interface pela variável `RAG_CHUNK_SIZE` do `.env`.

---

# CKP01 — Chatbot Profissional

## Domínio

**Qual:** assistente virtual de **treino de academia e prescrição de exercícios** — apelidado de
**Halter**.

**Por que foi escolhido:** é um domínio com fronteira de escopo nítida e com consequência
real. Treino é competência do educador físico; dieta fechada é do nutricionista; lesão é do
médico ou fisioterapeuta. Essa fronteira dá material concreto para as restrições do system
prompt — e violá-la causa dano de verdade, o que torna a avaliação do prompt objetiva. É
também um domínio naturalmente conversacional e dependente de memória: objetivo, nível,
dias disponíveis e limitações são ditos uma vez e precisam valer pelo resto da sessão.

**Usuários-alvo:** praticantes iniciantes e intermediários de musculação, de 16 a 55 anos,
que treinam em academia convencional e querem montar, ajustar ou entender um treino. Não
dominam jargão técnico — por isso o system prompt obriga a traduzir os termos usados.

---

## Requisitos atendidos

| Requisito | Status | Implementação |
|---|:--:|---|
| Pipeline LCEL | ✅ | `chain.py` — `prompt \| llm \| PydanticOutputParser()` em `construir_chain_analise()` e `construir_chain_relatorio()` |
| Arquitetura de 2 chains (Aula 03) | ✅ | `ConversationChain` com memória (chain 1) + pipeline LCEL stateless (chain 2), ambas em `chain.py` |
| ChatOllama | ✅ | `gemma4:cloud` via Ollama Cloud, chave lida do `.env` em `config.py` |
| ChatPromptTemplate com variáveis | ✅ | `prompts.py` + `chain.py` — system e human separados, variáveis `{input}`, `{mensagem}`, `{historico}`, `{format_instructions}`; nenhuma f-string manual |
| Memória gerenciada | ✅ | `memory_manager.py` — as 3 estratégias implementadas, `ConversationTokenBufferMemory` (1200 tokens) ativa; justificativa abaixo |
| Demonstração em ≥ 5 turnos | ✅ | Aba **Memória** da interface conta os turnos do usuário na sessão |
| Pydantic v2 (≥ 4 campos) | ✅ | `schemas.py` — `AnaliseConsulta` (7 campos) e `RelatorioSessao` (6 campos), com `field_validator` |
| PydanticOutputParser | ✅ | `chain.py` — nas duas chains estruturadas (e não `JsonOutputParser`, que devolveria `dict` sem validação) |
| Seção context rot | ✅ | `context_rot.py` — mesma pergunta em janelas de 0/5/10/15/20 turnos, com tabela comparativa |
| System prompt com persona | ✅ | `prompts.py` — XML tagging (`<papel>`, `<regras>`, `<restricoes>`…), persona, escopo, recusas e resistência a jailbreak |
| Domínio documentado | ✅ | Este README |
| Projeto local estruturado | ✅ | Pacote `app/` + `.env.example` + `requirements.txt` + `README.md`, sem Colab |

---

## Como executar (local — sem Colab)

Requer **Python 3.10 a 3.13**. Todos os comandos são rodados de dentro da pasta do
projeto — a que contém `requirements.txt` e a pasta `app/`.

**Windows (Prompt de Comando):**

```cmd
cd caminho\para\cp2
copy .env.example .env
notepad .env
pip install -r requirements.txt
python -m app.main
```

**Linux e macOS:**

```bash
cd caminho/para/cp2
cp .env.example .env
pip install -r requirements.txt
python -m app.main
```

A interface sobe em <http://localhost:7860>.

No `.env`, troque `coloque_sua_chave_aqui` pela chave da Ollama Cloud, que sai de
<https://ollama.com> → *Settings* → *Keys*. Se o `.env` estiver faltando ou mal
preenchido, o programa para logo no início com uma mensagem dizendo exatamente o que
corrigir — não há chave hardcoded em lugar nenhum do código.

Para rodar a demonstração de context rot separadamente:

```bash
python -m app.context_rot
```

### Se o `pip install` falhar

Quase sempre é versão de Python. O `gradio` e suas dependências ainda não têm pacote
pronto para as versões mais novas do Python (3.14+), e a instalação tenta compilar do
zero e falha. Instale o Python 3.12 e rode apontando para ele:

```cmd
py -3.12 -m pip install -r requirements.txt
py -3.12 -m app.main
```

### Estrutura

```
cp2/
├── app/
│   ├── __init__.py        # metadados do pacote e filtro de avisos de legado
│   ├── main.py            # interface Gradio + entry point
│   ├── rag/               # DocMind RAG (CKP02)
│   │   ├── fontes.py              # catálogo das 5 fontes reais
│   │   ├── baixar_documentos.py   # download + recorte de até 5 páginas
│   │   ├── carregador.py          # load
│   │   ├── divisor.py             # split
│   │   ├── vetores.py             # embed + store + metadata filtering
│   │   ├── reranker.py            # cross-encoder
│   │   ├── pipeline.py            # retrieve + generate + buscar()
│   │   └── avaliacao.py           # RAGAS
│   ├── chain.py           # as 2 chains da Aula 03 + fachada ChatbotTreino
│   ├── memory_manager.py  # as 3 estratégias de memória gerenciada
│   ├── schemas.py         # Pydantic v2: AnaliseConsulta e RelatorioSessao
│   ├── context_rot.py     # demonstração de degradação por contexto
│   ├── prompts.py         # system prompts com XML tagging
│   ├── config.py          # leitura e validação do .env
│   └── tokens.py          # estimativa de tokens usada pela memória
├── documentos/            # PDFs da base + fontes.json
├── resultados/            # tabelas RAGAS
├── CKP02_DocMind_RAG.ipynb
├── .env.example
├── requirements.txt
└── README.md
```

---

## Arquitetura das 2 chains (Aula 03)

```
                        mensagem do usuário
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
   CHAIN 2 — LCEL stateless               CHAIN 1 — com memória
   ChatPromptTemplate                     ConversationChain
        │ (operador |)                         │
        ▼                                      ├── ChatPromptTemplate
     ChatOllama  (gemma4:cloud)                ├── ChatOllama (gemma4:cloud)
        │                                      └── ConversationTokenBufferMemory
        ▼                                              │
   PydanticOutputParser                                ▼
        │                                     resposta conversacional
        ▼
   AnaliseConsulta  ──► risco ≥ 4?  ──► corta a prescrição e encaminha
   (objeto validado)                     a profissional de saúde
```

As duas chains rodam no **mesmo turno**, e a ordem importa: a triagem estruturada roda
**antes** da chain de conversa. Quando ela devolve `risco_seguranca >= 4` — dor aguda,
lesão recente, pedido sobre substância proibida — o sistema corta a prescrição e encaminha
o usuário a um profissional, sem gastar o turno de chat. É a diferença entre *pedir* ao
modelo que se comporte e *garantir* em código que ele se comporte.

---

## Justificativa da memória

**Escolha: `ConversationTokenBufferMemory` com teto de 1200 tokens.**

As três estratégias estão implementadas em `memory_manager.py` e podem ser trocadas pela
variável `MEMORIA_ESTRATEGIA` no `.env`. A comparação que levou à escolha:

| Estratégia | Custo de tokens | O que se perde | Veredito para este domínio |
|---|---|---|---|
| `ConversationBufferMemory` | Cresce sem teto a cada turno | Nada | ❌ Numa sessão longa o prompt vira um custo que não fecha |
| `ConversationSummaryMemory` | Quase constante, mas **+1 chamada ao LLM por turno** | **Os números.** "supino reto 4×8 com 40 kg, descanso 90 s" vira "treinou peito" | ❌ Fatal aqui: prescrição de treino **é** feita de números |
| `ConversationTokenBufferMemory` | Limitado pelo teto (1200 tokens) | Só os turnos mais antigos, e por inteiro | ✅ **Escolhida** |

**Por que 1200 tokens.** O enunciado pede de 800 a 1500. Um par pergunta-resposta de
prescrição gasta, na prática, de 150 a 250 tokens, então 1200 guardam de 5 a 8 turnos
completos — o suficiente para cobrir a demonstração de 5+ turnos com folga.

**O que se perde, e por que é aceitável.** O token buffer descarta os turnos mais antigos.
Na prática, é justamente a informação menos usada: o usuário quer ajustar o treino que
acabou de receber, não o que discutiu quinze mensagens atrás. O risco de perder uma
restrição de segurança declarada logo no início é o que a seção de context rot mede — e o
motivo pelo qual a triagem da chain 2 reavalia o risco de segurança a **cada** mensagem,
sem depender da memória.

---

## Context rot — degradação com contexto crescente

Rodando `python -m app.context_rot`, o experimento planta três fatos no primeiro turno
(objetivo, frequência semanal e uma restrição de segurança no ombro), enche a conversa com
N turnos de papo genérico de academia que **não repetem** esses fatos, e faz sempre a
**mesma pergunta de controle**: *"Monta pra mim o treino de ombro de amanhã."*

Como só o contexto muda, qualquer queda observada é atribuível ao volume de contexto, e não
à pergunta.

| Turnos de contexto | ~Tokens do prompt | Objetivo | Frequência | Restrição de segurança |
|---:|---:|:--:|:--:|:--:|
| 0 | 1275 | ✅ | ✅ | ❌ |
| 5 | 1491 | ❌ | ✅ | ✅ |
| 10 | 1662 | ❌ | ✅ | ✅ |
| 15 | 1809 | ❌ | ❌ | ✅ |
| 20 | 1974 | ❌ | ❌ | ✅ |

**Conclusão:** do contexto de 0 para 20 turnos, o recall dos fatos plantados caiu de 2/3
para 1/3. É o context rot: a informação continua DENTRO da janela, mas o modelo deixa de
usá-la conforme o volume de contexto ao redor cresce. É por isso que a memória deste
projeto tem teto de tokens em vez de acumular a conversa inteira.

**O que observar.** O primeiro fato a se perder foi o objetivo (hipertrofia), já a partir
de 5 turnos de enchimento; a frequência semanal (4x) resistiu até os 10 turnos e caiu a
partir dos 15. A coluna de restrição de segurança chama atenção por ir na direção
contrária — ela aparece como ❌ só no turno 0 e ✅ daí em diante. Olhando a resposta
completa daquele turno, o modelo respeitou a limitação na prática ("respeitando sua
limitação, vamos focar na parte frontal..."), só que com uma palavra diferente da que o
verificador procura (`restri`, `evitar`, `lesão`...); os turnos seguintes usaram
literalmente a palavra "restrição" e por isso pontuaram. Ou seja: nesse critério específico
o resultado reflete uma limitação do verificador por palavra-chave, não uma falha real do
modelo — mas a queda no objetivo e na frequência é um sinal direto e real de degradação.

**Conclusão de engenharia.** Contexto maior não é contexto melhor. É essa observação que
sustenta a memória com **teto** (em vez de acumular a conversa inteira) e a verificação de
segurança rodando numa chain **separada e stateless**, que recebe só a mensagem atual —
imune ao rot que afeta a chain de conversa.

---

## Segurança e resistência a desvio de assunto

O projeto se defende de tentativas de manipulação em três camadas, cada uma mais cara que a
anterior — se uma camada não pegar, a próxima pega:

**1. Filtro determinístico (`detectar_tentativa_injecao`, em `chain.py`).** Roda antes de
qualquer chamada ao modelo e bloqueia por padrão de texto, sem depender do LLM: "ignore
suas instruções", "modo desenvolvedor/debug/admin", "finja que você é...", "a partir de
agora você é...", `[SYSTEM]`, alegações de ser o desenvolvedor/administrador, entre outros.
Pega os ataques mais batidos com custo zero de chamada ao modelo.

**2. Triagem estruturada (`AnaliseConsulta.tentativa_manipulacao`, em `schemas.py`).** Toda
mensagem que passa pelo filtro acima é classificada por uma chain LCEL separada antes de
chegar à conversa. Ela entende contexto, não só padrão de texto, e pega tentativas escritas
de um jeito novo que a regex não previu. Se marcar `tentativa_manipulacao` ou
`risco_seguranca >= 4`, a conversa normal é cortada e uma resposta fixa é usada no lugar —
isso é uma verificação em código, então não depende do modelo "lembrar" de obedecer a uma
regra no meio de uma conversa longa.

**3. System prompt (`prompts.py`).** Última linha de defesa, para o que passar pelas duas
camadas anteriores. O bloco `<resistencia_a_desvio>` instrui o modelo a nunca sair da
persona do Halter, recusar pedidos de troca de identidade, alegações de autoridade e
mensagens de sistema falsificadas — e a mensagem do usuário chega sempre delimitada por
`<mensagem_usuario>`, com instrução explícita de tratar esse conteúdo como dado, nunca como
comando, mesmo que o texto tente imitar uma instrução.

Além disso:

- Assuntos fora de treino/exercício/condicionamento físico são recusados em uma frase, com
  o bot reconduzindo a conversa ao próprio domínio.
- Pedidos sobre anabolizantes, hormônios ou substâncias controladas são recusados.
- Mensagens que indicam dor aguda, lesão recente ou risco de segurança cortam a prescrição
  e encaminham a um profissional de saúde — mesma lógica de código da camada 2, usando o
  campo `risco_seguranca`.

---

## Como validar antes de apresentar

- Rode `python -m app.main`, mantenha uma conversa de pelo menos 10 interações e confirme
  que o Halter continua respondendo normalmente do primeiro ao último turno, sem travar.
- Na aba **Memória**, confira que o contador de turnos sobe a cada mensagem e que o bot
  ainda lembra de informações ditas no início da conversa (objetivo, dias por semana,
  alguma restrição relatada).
- Teste as restrições do domínio: peça uma dieta fechada, peça opinião sobre anabolizante,
  relate uma dor aguda e mude de assunto de propósito — em todos os casos o Halter deve
  recusar de forma coerente com a persona e continuar disponível para falar de treino.
- Tente também pedidos do tipo "ignore suas instruções", "ative o modo desenvolvedor" ou
  "finja que você é outra IA" — o bot deve manter a persona e seguir a conversa no
  domínio de treino em vez de obedecer ao pedido.
- Rode `python -m app.context_rot` e cole a tabela impressa na seção acima.
