"""Gestão de memória conversacional — as 3 estratégias da Aula 02.

O checkpoint exige **1 tipo** de memória gerenciada, escolhido e justificado.
Este módulo implementa as três para permitir a comparação, e a escolhida para
este domínio (`token_buffer`) é a configurada por padrão no .env.

    buffer        -> ConversationBufferMemory
                     Guarda TODOS os turnos. Fidelidade total, custo crescente
                     sem teto: o prompt cresce a cada mensagem até estourar a
                     janela do modelo. Útil em conversas curtas.

    summary       -> ConversationSummaryMemory
                     Resume o histórico com o próprio LLM. Custo quase
                     constante, mas cada turno gasta uma chamada extra e o
                     resumo PERDE número: "3x12 com 20kg" vira "treinou peito".
                     Ruim para prescrição de treino, que é feita de números.

    token_buffer  -> ConversationTokenBufferMemory  (ESCOLHA DO GRUPO)
                     Mantém os turnos recentes na íntegra e descarta os mais
                     antigos quando o histórico ultrapassa o teto de tokens.
                     Preserva os números do treino recente, dá teto duro de
                     custo e não gasta chamada extra ao modelo.

Justificativa completa no README.md, seção "Justificativa da memória".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory

from app import silenciar_avisos_de_legado

# Tem de vir depois dos imports do langchain — ver a docstring da função.
silenciar_avisos_de_legado()

# Chave usada no prompt da ConversationChain (MessagesPlaceholder("history")).
CHAVE_MEMORIA = "history"
CHAVE_ENTRADA = "input"

ESTRATEGIAS_DISPONIVEIS = ("buffer", "summary", "token_buffer")


@dataclass(frozen=True)
class DescricaoEstrategia:
    """Metadados exibidos na interface para justificar a escolha."""

    nome: str
    classe: str
    custo: str
    perda: str
    indicada_para: str


DESCRICOES: Dict[str, DescricaoEstrategia] = {
    "buffer": DescricaoEstrategia(
        nome="Buffer completo",
        classe="ConversationBufferMemory",
        custo="Cresce linearmente e sem teto a cada turno.",
        perda="Nenhuma — mantém a conversa inteira.",
        indicada_para="Conversas curtas (até ~10 turnos) ou depuração.",
    ),
    "summary": DescricaoEstrategia(
        nome="Resumo incremental",
        classe="ConversationSummaryMemory",
        custo="Quase constante, mas +1 chamada ao LLM por turno.",
        perda="Alta em detalhe numérico (séries, cargas, repetições).",
        indicada_para="Conversas longas em que só o tema importa.",
    ),
    "token_buffer": DescricaoEstrategia(
        nome="Buffer com teto de tokens",
        classe="ConversationTokenBufferMemory",
        custo="Limitado pelo teto configurado (800–1500 tokens).",
        perda="Só os turnos mais antigos, na íntegra.",
        indicada_para="Prescrição de treino: preserva os números recentes.",
    ),
}


def criar_memoria(
    estrategia: str,
    llm: BaseChatModel,
    max_tokens: int = 1200,
) -> BaseMemory:
    """Fábrica da memória gerenciada usada pela ConversationChain.

    Args:
        estrategia: "buffer", "summary" ou "token_buffer".
        llm: modelo usado pelas estratégias que precisam contar ou resumir.
        max_tokens: teto do token_buffer (o enunciado exige de 800 a 1500).

    Returns:
        Instância de memória pronta para a ConversationChain.

    Raises:
        ValueError: se a estratégia for desconhecida.
    """
    estrategia = estrategia.strip().lower()

    # return_messages=True porque o prompt do chat usa MessagesPlaceholder,
    # ou seja, a memória precisa devolver objetos de mensagem e não uma string.
    comum = {
        "memory_key": CHAVE_MEMORIA,
        "input_key": CHAVE_ENTRADA,
        "return_messages": True,
    }

    if estrategia == "buffer":
        return ConversationBufferMemory(**comum)

    if estrategia == "summary":
        return ConversationSummaryMemory(llm=llm, **comum)

    if estrategia == "token_buffer":
        return ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=max_tokens,
            **comum,
        )

    raise ValueError(
        f"Estratégia de memória desconhecida: '{estrategia}'. "
        f"Use uma de {ESTRATEGIAS_DISPONIVEIS}."
    )


def historico_como_texto(memoria: BaseMemory, limite_chars: int = 4000) -> str:
    """Serializa a memória em texto para alimentar as chains estruturadas.

    A chain de análise e a de relatório não têm memória própria (são stateless,
    como manda o LCEL): elas recebem o histórico como uma variável de template.
    """
    try:
        variaveis = memoria.load_memory_variables({})
    except Exception:  # memória ainda vazia ou backend indisponível
        return "(sem histórico)"

    mensagens = variaveis.get(CHAVE_MEMORIA, [])

    if isinstance(mensagens, str):
        texto = mensagens
    else:
        linhas: List[str] = []
        for msg in mensagens:
            papel = "Usuário" if msg.type in ("human", "user") else "Halter"
            linhas.append(f"{papel}: {msg.content}")
        texto = "\n".join(linhas)

    texto = texto.strip() or "(sem histórico)"

    # Mantém o FIM do histórico, que é a parte mais relevante para o turno atual.
    if len(texto) > limite_chars:
        texto = "[...trecho antigo descartado...]\n" + texto[-limite_chars:]
    return texto


def contar_turnos_usuario(memoria: BaseMemory) -> int:
    """Conta quantas mensagens o usuário enviou — vai no RelatorioSessao."""
    try:
        mensagens = memoria.load_memory_variables({}).get(CHAVE_MEMORIA, [])
    except Exception:
        return 0
    if isinstance(mensagens, str):
        return mensagens.count("Human:")
    return sum(1 for m in mensagens if m.type in ("human", "user"))


def resumo_estrategia(estrategia: str, max_tokens: int) -> str:
    """Texto em Markdown com a justificativa, exibido na aba da interface."""
    d = DESCRICOES.get(estrategia)
    if d is None:
        return f"Estratégia desconhecida: {estrategia}"
    teto = f"{max_tokens} tokens" if estrategia == "token_buffer" else "sem teto"
    return (
        f"**Estratégia ativa:** {d.nome} (`{d.classe}`)\n\n"
        f"- **Teto:** {teto}\n"
        f"- **Custo de tokens:** {d.custo}\n"
        f"- **Perda de informação:** {d.perda}\n"
        f"- **Indicada para:** {d.indicada_para}\n"
    )
