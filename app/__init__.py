"""CKP01 + CKP02 — Halter e DocMind RAG (FIAP · 2º Semestre).

Pacote do chatbot de domínio "Treino de academia e prescrição de exercícios".
Módulos:
    config          -> leitura do .env e configuração central
    prompts         -> system prompts com XML tagging (Aula 04)
    schemas         -> modelos Pydantic v2 que validam as saídas
    memory_manager  -> as 3 estratégias de memória gerenciada (Aula 02)
    chain           -> arquitetura de 2 chains da Aula 03 (conversa + LCEL)
    context_rot     -> demonstração da degradação por contexto crescente
    rag             -> DocMind: pipeline RAG sobre documentos reais (CKP02)
    main            -> interface Gradio + entry point (python -m app.main)
"""

import warnings


def silenciar_avisos_de_legado() -> None:
    """Silencia o aviso de depreciação do LangChain sobre ConversationChain
    e as memórias de langchain.memory — usadas de propósito, pois são a
    arquitetura pedida pela Aula 03. Elas funcionam normalmente; o aviso só
    poluiria o terminal e a interface.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*(ConversationChain|ConversationBufferMemory|"
        r"ConversationSummaryMemory|ConversationTokenBufferMemory|"
        r"migration guide|migrating_memory).*",
    )


__version__ = "2.0.0"
__all__ = [
    "config",
    "prompts",
    "schemas",
    "memory_manager",
    "chain",
    "context_rot",
    "rag",
]
