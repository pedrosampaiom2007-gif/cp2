"""Arquitetura de 2 chains da Aula 03.

    CHAIN 1 — conversa com memória
        ConversationChain(llm=ChatOllama, memory=<memória gerenciada>,
                          prompt=ChatPromptTemplate)
        É ela que mantém o fio da conversa entre os turnos.

    CHAIN 2 — saída estruturada em LCEL puro
        ChatPromptTemplate | ChatOllama | PydanticOutputParser
        Stateless, composta com o operador `|`, devolve um objeto Pydantic v2
        já validado (e não um dict solto como faria o JsonOutputParser).

Há ainda uma terceira chain LCEL, de mesmo formato, que fecha a sessão com o
RelatorioSessao — mesma técnica, outra saída validada.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

from langchain.chains import ConversationChain
from langchain_core.language_models import BaseChatModel
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.config import Config, carregar_config
from app.memory_manager import (
    CHAVE_ENTRADA,
    CHAVE_MEMORIA,
    criar_memoria,
    historico_como_texto,
)
from app.prompts import (
    HUMAN_PROMPT_ANALISE,
    HUMAN_PROMPT_RELATORIO,
    SYSTEM_PROMPT_ANALISE,
    SYSTEM_PROMPT_CHAT,
    SYSTEM_PROMPT_RELATORIO,
)
from app.schemas import AnaliseConsulta, RelatorioSessao
from app.tokens import token_ids


# ---------------------------------------------------------------------------
# Camada 1 de segurança — filtro por padrão de texto contra prompt injection
# ---------------------------------------------------------------------------
# Roda antes de qualquer chamada ao modelo: pega os pedidos de jailbreak mais
# comuns ("ignore suas instruções", "modo desenvolvedor", "finja que você
# é...") na hora. A camada 2 (campo `tentativa_manipulacao` da triagem, em
# prompts.py) cobre o que for escrito de um jeito diferente destes padrões.
_PADROES_INJECAO = [
    r"ignor[ea]\w*\s+(as\s+|todas\s+as\s+|suas\s+)?instru[cç][oõ]es",
    r"ignore\s+(all\s+|previous\s+|prior\s+|the\s+above\s+)*instructions",
    r"esque[cç]a\s+(as\s+|suas\s+|todas\s+as\s+)?(regras|instru[cç][oõ]es)",
    r"forget\s+(all\s+|your\s+|previous\s+)?(rules|instructions)",
    r"modo\s+(desenvolvedor|debug|admin|deus|god)\b",
    r"developer\s+mode",
    r"\bdan\s+mode\b",
    r"jailbreak",
    r"(revele|mostre|repita|reescreva|traduza|imprima)\s+(o\s+|seu\s+)?system\s*prompt",
    r"(revele|mostre|repita|imprima)\s+suas?\s+instru[cç][oõ]es",
    r"finja\s+que\s+(voc[eê])\s+[eé]",
    r"a\s+partir\s+de\s+agora\s+(voc[eê])\s+[eé]",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you|a[n]?\s)",
    r"you\s+are\s+now\b",
    r"\[\s*system\s*\]",
    r"###\s*(nova\s+)?instru",
    r"sou\s+(o\s+)?(desenvolvedor|administrador|admin|programador)\s+(deste|do)\s+"
    r"(projeto|sistema|chatbot|bot)",
]
_REGEX_INJECAO = re.compile("|".join(_PADROES_INJECAO), flags=re.IGNORECASE | re.UNICODE)

RESPOSTA_BLOQUEIO = (
    "Não posso atender esse pedido. Sou o Halter, assistente de treino, e "
    "sigo assim. Como posso te ajudar com o seu treino hoje?"
)


def detectar_tentativa_injecao(texto: str) -> bool:
    """True quando o texto bate com um padrão conhecido de jailbreak."""
    return _REGEX_INJECAO.search(texto or "") is not None


# ---------------------------------------------------------------------------
# Modelo — ChatOllama apontando para a Ollama Cloud
# ---------------------------------------------------------------------------
def construir_llm(config: Optional[Config] = None, temperatura: Optional[float] = None) -> ChatOllama:
    """Instancia o ChatOllama com gemma4:cloud via Ollama Cloud.

    A autenticação vai no header Authorization, com a chave lida do .env —
    nunca hardcoded no código.
    """
    config = config or carregar_config()
    return ChatOllama(
        model=config.modelo,
        base_url=config.base_url,
        temperature=config.temperatura if temperatura is None else temperatura,
        # Autenticação via header Authorization com a chave do .env.
        client_kwargs={"headers": {"Authorization": f"Bearer {config.api_key}"}},
        # Contador de tokens usado pela memória com teto (app/tokens.py).
        custom_get_token_ids=token_ids,
    )


# ---------------------------------------------------------------------------
# CHAIN 1 — conversa com memória gerenciada (ConversationChain)
# ---------------------------------------------------------------------------
def construir_prompt_chat() -> ChatPromptTemplate:
    """ChatPromptTemplate com system e human SEPARADOS e com variáveis.

    Nada de f-string manual: `{input}` é variável do template e `history` é o
    slot preenchido pela memória a cada turno. A mensagem do usuário é
    delimitada por `<mensagem_usuario>` — o `<resistencia_a_desvio>` do
    system prompt instrui o modelo a tratar tudo dentro dessa tag como fala
    do usuário, nunca como uma instrução nova, mesmo que o texto tente
    imitar uma.
    """
    return ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_CHAT),
            MessagesPlaceholder(variable_name=CHAVE_MEMORIA),
            HumanMessagePromptTemplate.from_template(
                "<mensagem_usuario>\n{" + CHAVE_ENTRADA + "}\n</mensagem_usuario>"
            ),
        ]
    )


def construir_chain_conversa(
    llm: BaseChatModel,
    memoria: BaseMemory,
    verbose: bool = False,
) -> ConversationChain:
    """Monta a ConversationChain — a chain COM memória da arquitetura da Aula 03."""
    return ConversationChain(
        llm=llm,
        memory=memoria,
        prompt=construir_prompt_chat(),
        input_key=CHAVE_ENTRADA,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# CHAIN 2 — saída estruturada em LCEL (prompt | llm | parser)
# ---------------------------------------------------------------------------
def construir_chain_analise(llm: BaseChatModel) -> Runnable:
    """Pipeline LCEL que devolve um `AnaliseConsulta` já validado.

    O `format_instructions` do PydanticOutputParser é injetado como variável
    parcial do template — o modelo recebe o JSON Schema do BaseModel e é
    obrigado a respeitá-lo; o parser rejeita o que não couber no contrato.
    """
    parser = PydanticOutputParser(pydantic_object=AnaliseConsulta)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_ANALISE),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_ANALISE),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    # with_retry cobre o caso de o modelo devolver JSON quebrado na 1ª tentativa.
    return (prompt | llm | parser).with_retry(stop_after_attempt=2)


def construir_chain_relatorio(llm: BaseChatModel) -> Runnable:
    """Pipeline LCEL que fecha a sessão devolvendo um `RelatorioSessao`."""
    parser = PydanticOutputParser(pydantic_object=RelatorioSessao)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_RELATORIO),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_RELATORIO),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return (prompt | llm | parser).with_retry(stop_after_attempt=2)


# ---------------------------------------------------------------------------
# Fachada que a interface usa
# ---------------------------------------------------------------------------
class ChatbotTreino:
    """Reúne as duas chains + a memória em um objeto só, usado pelo Gradio."""

    def __init__(self, config: Optional[Config] = None, verbose: bool = False) -> None:
        self.config = config or carregar_config()
        self.llm = construir_llm(self.config)
        self.memoria = criar_memoria(
            estrategia=self.config.memoria_estrategia,
            llm=self.llm,
            max_tokens=self.config.memoria_max_tokens,
        )
        self.chain_conversa = construir_chain_conversa(self.llm, self.memoria, verbose)
        self.chain_analise = construir_chain_analise(self.llm)
        self.chain_relatorio = construir_chain_relatorio(self.llm)

    # -- conversa ----------------------------------------------------------
    def responder(self, mensagem: str) -> str:
        """Um turno de conversa. A memória é atualizada pela própria chain."""
        return self.chain_conversa.predict(**{CHAVE_ENTRADA: mensagem})

    # -- saídas estruturadas ----------------------------------------------
    def analisar(self, mensagem: str) -> AnaliseConsulta:
        """Roda a chain LCEL de triagem sobre a mensagem do usuário."""
        return self.chain_analise.invoke(
            {
                "mensagem": mensagem,
                "historico": historico_como_texto(self.memoria, limite_chars=1500),
            }
        )

    def responder_com_analise(self, mensagem: str) -> Tuple[str, AnaliseConsulta]:
        """Executa as 2 chains no mesmo turno: triagem + resposta conversacional.

        Três camadas rodam nesta ordem, cada uma mais cara que a anterior:

        1. Filtro determinístico (regex) — pega jailbreaks batidos sem
           chamar o modelo.
        2. Triagem estruturada (`AnaliseConsulta`) — se marcar
           `tentativa_manipulacao` ou risco de segurança alto, corta a
           conversa normal sem gastar o turno de chat.
        3. Chain de conversa — só roda se as duas primeiras liberarem.
        """
        if detectar_tentativa_injecao(mensagem):
            analise = AnaliseConsulta(
                objetivo_treino="indefinido",
                nivel_experiencia="indefinido",
                grupos_musculares=[],
                risco_seguranca=1,
                fora_do_escopo=True,
                tentativa_manipulacao=True,
                resumo_intencao="Tentativa de manipular as instruções do assistente.",
            )
            self.memoria.save_context(
                {CHAVE_ENTRADA: mensagem}, {"response": RESPOSTA_BLOQUEIO}
            )
            return RESPOSTA_BLOQUEIO, analise

        analise = self.analisar(mensagem)

        if analise.exige_bloqueio():
            self.memoria.save_context(
                {CHAVE_ENTRADA: mensagem}, {"response": RESPOSTA_BLOQUEIO}
            )
            return RESPOSTA_BLOQUEIO, analise

        if analise.exige_encaminhamento():
            resposta = (
                "Pelo que você descreveu, isso precisa de avaliação presencial "
                "antes de qualquer treino. Procure um médico ou fisioterapeuta "
                "e, enquanto isso, não force o movimento que dói. Quando tiver "
                "a liberação, eu monto o treino adaptado com você."
            )
            # Grava o turno na memória mesmo sem passar pela chain de conversa,
            # para o histórico não ficar com buraco.
            self.memoria.save_context({CHAVE_ENTRADA: mensagem}, {"response": resposta})
            return resposta, analise

        return self.responder(mensagem), analise

    def gerar_relatorio(self) -> RelatorioSessao:
        """Fecha a sessão com a 2ª saída validada por Pydantic."""
        return self.chain_relatorio.invoke(
            {"historico": historico_como_texto(self.memoria, limite_chars=6000)}
        )

    # -- utilidades --------------------------------------------------------
    def limpar_memoria(self) -> None:
        """Zera o histórico, começando uma nova sessão de atendimento."""
        self.memoria.clear()
