"""Etapas EMBED e STORE: nomic-embed-text via Ollama + ChromaDB local.

O nomic-embed-text foi treinado com prefixos de tarefa: textos indexados
recebem `search_document:` e consultas recebem `search_query:`. Usar os dois
prefixos coloca documento e pergunta no mesmo espaço de busca e melhora a
similaridade medida.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import chromadb
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

from app.config import PASTA_CHROMA, Config, carregar_config

PREFIXO_DOCUMENTO = "search_document: "
PREFIXO_CONSULTA = "search_query: "
TAMANHO_LOTE = 32


def criar_embeddings(config: Optional[Config] = None) -> OllamaEmbeddings:
    """OllamaEmbeddings autenticado na Ollama Cloud com a chave do .env."""
    config = config or carregar_config()
    return OllamaEmbeddings(
        model=config.modelo_embedding,
        base_url=config.base_url,
        client_kwargs={"headers": {"Authorization": f"Bearer {config.api_key}"}},
    )


@dataclass
class Trecho:
    """Um chunk recuperado da base, com metadados e pontuações."""

    texto: str
    metadados: Dict[str, object]
    similaridade: float
    score_rerank: Optional[float] = None
    extras: Dict[str, object] = field(default_factory=dict)

    @property
    def titulo(self) -> str:
        """Título do documento de origem."""
        return str(self.metadados.get("titulo", "documento"))

    @property
    def pagina(self) -> int:
        """Página do documento original."""
        return int(self.metadados.get("pagina", 0) or 0)

    @property
    def chunk_id(self) -> str:
        """Identificador do chunk na coleção."""
        return str(self.metadados.get("chunk_id", ""))

    def citacao(self) -> str:
        """Ex.: "Guia de Atividade Física..., p. 23 (chunk ms_guia-p23-c2)"."""
        return f"{self.titulo}, p. {self.pagina} (chunk {self.chunk_id})"


def _metadados_chroma(metadados: Dict[str, object]) -> Dict[str, object]:
    """O Chroma só aceita str, int, float e bool como valor de metadado."""
    limpos: Dict[str, object] = {}
    for chave, valor in metadados.items():
        if chave.startswith("_") or valor is None:
            continue
        limpos[chave] = valor if isinstance(valor, (str, int, float, bool)) else str(valor)
    return limpos


class BaseVetorial:
    """Coleção do ChromaDB para uma configuração de chunking."""

    def __init__(
        self,
        chunk_size: int,
        config: Optional[Config] = None,
        embeddings: Optional[OllamaEmbeddings] = None,
        cliente: Optional[chromadb.ClientAPI] = None,
    ) -> None:
        """Conecta ao ChromaDB e abre (ou cria) a coleção do chunk_size."""
        self.config = config or carregar_config()
        self.chunk_size = chunk_size
        self.embeddings = embeddings or criar_embeddings(self.config)
        self.cliente = cliente or chromadb.PersistentClient(path=str(PASTA_CHROMA))
        self.nome_colecao = f"{self.config.rag_colecao}_{chunk_size}"
        self.colecao = self.cliente.get_or_create_collection(
            name=self.nome_colecao,
            metadata={"hnsw:space": "cosine", "dominio": "treino de academia"},
        )

    def total(self) -> int:
        """Quantidade de chunks indexados na coleção."""
        return self.colecao.count()

    def indexar(self, chunks: Sequence[Document], recriar: bool = False) -> int:
        """Gera os embeddings dos chunks e grava na coleção.

        Se a coleção já tiver exatamente os mesmos chunks, nada é refeito.
        """
        ids = [c.metadata["chunk_id"] for c in chunks]
        if not recriar and self.total() == len(ids):
            existentes = set(self.colecao.get(ids=ids, include=[])["ids"])
            if existentes == set(ids):
                return 0

        self.cliente.delete_collection(self.nome_colecao)
        self.colecao = self.cliente.create_collection(
            name=self.nome_colecao,
            metadata={"hnsw:space": "cosine", "dominio": "treino de academia"},
        )

        for inicio in range(0, len(chunks), TAMANHO_LOTE):
            lote = chunks[inicio : inicio + TAMANHO_LOTE]
            textos = [c.page_content for c in lote]
            vetores = self.embeddings.embed_documents([PREFIXO_DOCUMENTO + t for t in textos])
            self.colecao.add(
                ids=[c.metadata["chunk_id"] for c in lote],
                documents=textos,
                embeddings=vetores,
                metadatas=[_metadados_chroma(c.metadata) for c in lote],
            )
        return len(chunks)

    def buscar(
        self,
        consulta: str,
        k: int = 4,
        where: Optional[Dict[str, object]] = None,
    ) -> List[Trecho]:
        """Busca semântica: os k chunks mais próximos da consulta."""
        if self.total() == 0:
            return []
        vetor = self.embeddings.embed_query(PREFIXO_CONSULTA + consulta)
        resultado = self.colecao.query(
            query_embeddings=[vetor],
            n_results=min(k, self.total()),
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            Trecho(texto=texto, metadados=dict(meta), similaridade=round(1 - distancia, 4))
            for texto, meta, distancia in zip(
                resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
            )
        ]

    def valores_distintos(self, campo: str) -> List[str]:
        """Valores existentes de um metadado, para montar os filtros da interface."""
        metadados = self.colecao.get(include=["metadatas"])["metadatas"] or []
        return sorted({str(m[campo]) for m in metadados if m.get(campo) not in (None, "")})


def montar_filtro(
    tipo: Optional[str] = None,
    categoria: Optional[str] = None,
    publico: Optional[str] = None,
    ano_minimo: Optional[int] = None,
    fonte: Optional[str] = None,
) -> Optional[Dict[str, object]]:
    """Monta o `where=` do Chroma a partir dos filtros preenchidos."""
    condicoes: List[Dict[str, object]] = []
    for campo, valor in (("tipo", tipo), ("categoria", categoria), ("publico", publico), ("id", fonte)):
        if valor:
            condicoes.append({campo: {"$eq": valor}})
    if ano_minimo:
        condicoes.append({"ano": {"$gte": int(ano_minimo)}})
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}
