"""Pipeline RAG completo: load → split → embed → store → retrieve → generate.

    from app.rag.pipeline import buscar, responder
    buscar("quantos minutos por semana de atividade moderada?")
    responder("quantas vezes por semana devo fazer fortalecimento muscular?")

`buscar(consulta)` devolve os trechos mais relevantes já formatados com a
fonte — é a função que vira tool do agente no CKP03.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable

from app.chain import construir_llm
from app.config import Config, carregar_config
from app.prompts import HUMAN_PROMPT_RAG, RESPOSTA_SEM_CONTEXTO, SYSTEM_PROMPT_RAG
from app.rag.carregador import carregar_documentos
from app.rag.divisor import dividir_documentos
from app.rag.reranker import Reranker
from app.rag.vetores import BaseVetorial, Trecho, criar_embeddings


@dataclass
class RespostaRAG:
    """Resposta gerada e os trechos que a fundamentaram."""

    pergunta: str
    resposta: str
    trechos: List[Trecho]

    @property
    def contextos(self) -> List[str]:
        return [t.texto for t in self.trechos]

    def fontes_markdown(self) -> str:
        if not self.trechos:
            return "_Nenhum trecho recuperado._"
        linhas = []
        for indice, trecho in enumerate(self.trechos, start=1):
            url = trecho.metadados.get("url") or ""
            titulo = f"[{trecho.titulo}]({url})" if url else trecho.titulo
            nota = (
                f"rerank {trecho.score_rerank:.2f}"
                if trecho.score_rerank is not None
                else f"similaridade {trecho.similaridade:.2f}"
            )
            linhas.append(
                f"**[{indice}]** {titulo}, p. {trecho.pagina} · `{trecho.chunk_id}` · {nota}"
            )
        return "\n\n".join(linhas)

    def para_markdown(self) -> str:
        return f"{self.resposta}\n\n---\n**Fontes**\n\n{self.fontes_markdown()}"


def formatar_contexto(trechos: Sequence[Trecho]) -> str:
    """Numera os trechos para o modelo citar [1], [2]..."""
    return "\n\n".join(
        f"[{indice}] {t.titulo} — p. {t.pagina}\n{t.texto}"
        for indice, t in enumerate(trechos, start=1)
    )


def construir_chain_rag(config: Config) -> Runnable:
    """prompt | ChatOllama (temperatura 0) | StrOutputParser."""
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_RAG),
            HumanMessagePromptTemplate.from_template(HUMAN_PROMPT_RAG),
        ]
    )
    llm = construir_llm(config, temperatura=config.rag_temperatura)
    return prompt | llm | StrOutputParser()


class DocMindRAG:
    """Base de conhecimento indexada para um chunk_size específico."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        config: Optional[Config] = None,
        reranker: Optional[Reranker] = None,
        documentos: Optional[Sequence[Document]] = None,
    ) -> None:
        self.config = config or carregar_config()
        self.chunk_size = chunk_size or self.config.rag_chunk_size
        self.base = BaseVetorial(self.chunk_size, self.config, embeddings=criar_embeddings(self.config))
        self.reranker = reranker if reranker is not None else Reranker()
        self.chain = construir_chain_rag(self.config)
        self.documentos: List[Document] = list(documentos) if documentos is not None else []
        self.chunks: List[Document] = []

    def preparar(self, recriar: bool = False) -> int:
        """Load → split → embed → store. Devolve quantos chunks foram indexados."""
        if not self.documentos:
            self.documentos = carregar_documentos()
        self.chunks = dividir_documentos(self.documentos, self.chunk_size)
        return self.base.indexar(self.chunks, recriar=recriar)

    def recuperar(
        self,
        consulta: str,
        filtros: Optional[Dict[str, object]] = None,
        k: Optional[int] = None,
        reranking: Optional[bool] = None,
    ) -> List[Trecho]:
        """Busca vetorial com metadata filtering e, opcionalmente, reranking."""
        k = k or self.config.rag_top_k
        usar_rerank = self.config.rag_reranking if reranking is None else reranking
        candidatos = self.base.buscar(
            consulta,
            k=self.config.rag_candidatos if usar_rerank else k,
            where=filtros,
        )
        if usar_rerank:
            return self.reranker.reordenar(consulta, candidatos, top_k=k)
        return candidatos[:k]

    def responder(
        self,
        pergunta: str,
        filtros: Optional[Dict[str, object]] = None,
        reranking: Optional[bool] = None,
    ) -> RespostaRAG:
        """Retrieve → generate, com a resposta citando os trechos de origem."""
        trechos = self.recuperar(pergunta, filtros=filtros, reranking=reranking)
        if not trechos:
            return RespostaRAG(pergunta, RESPOSTA_SEM_CONTEXTO, [])
        resposta = self.chain.invoke({"contexto": formatar_contexto(trechos), "pergunta": pergunta})
        return RespostaRAG(pergunta, resposta.strip(), trechos)


_INSTANCIA: Optional[DocMindRAG] = None


def obter_rag() -> DocMindRAG:
    """Instância única, preparada na primeira chamada."""
    global _INSTANCIA
    if _INSTANCIA is None:
        _INSTANCIA = DocMindRAG()
        _INSTANCIA.preparar()
    return _INSTANCIA


def buscar(consulta: str, filtros: Optional[Dict[str, object]] = None, k: int = 4) -> str:
    """Trechos relevantes da base de treino, cada um com a sua fonte."""
    trechos = obter_rag().recuperar(consulta, filtros=filtros, k=k)
    if not trechos:
        return RESPOSTA_SEM_CONTEXTO
    return "\n\n".join(f"[Fonte: {t.citacao()}]\n{t.texto}" for t in trechos)


def responder(pergunta: str, filtros: Optional[Dict[str, object]] = None) -> RespostaRAG:
    """Resposta gerada pelo LLM com base nos documentos."""
    return obter_rag().responder(pergunta, filtros=filtros)


def main() -> None:
    """python -m app.rag.pipeline "sua pergunta" """
    import sys

    pergunta = " ".join(sys.argv[1:]) or "Quantos minutos de atividade física moderada um adulto deve fazer por semana?"
    print(responder(pergunta).para_markdown())


if __name__ == "__main__":
    main()
