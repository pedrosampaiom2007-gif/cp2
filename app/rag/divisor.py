"""Etapa SPLIT: divide os documentos em chunks com RecursiveCharacterTextSplitter.

Cada configuração de chunking usa overlap de 12,5% do chunk_size. O splitter
tenta quebrar primeiro em parágrafos, depois em linhas, frases e palavras,
e só corta no meio de uma palavra se não houver outra saída.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Sequence

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARADORES = ["\n\n", "\n", ". ", " ", ""]
PROPORCAO_OVERLAP = 0.125


@dataclass(frozen=True)
class ConfigChunking:
    """Uma configuração de chunking comparada no experimento."""

    chunk_size: int

    @property
    def chunk_overlap(self) -> int:
        """Overlap em caracteres: 12,5% do chunk_size."""
        return int(self.chunk_size * PROPORCAO_OVERLAP)

    @property
    def nome(self) -> str:
        """Nome curto da configuração, ex.: chunk_512."""
        return f"chunk_{self.chunk_size}"


CONFIGURACOES: Dict[int, ConfigChunking] = {
    tamanho: ConfigChunking(tamanho) for tamanho in (256, 512, 1024)
}


def criar_splitter(config: ConfigChunking) -> RecursiveCharacterTextSplitter:
    """Splitter recursivo configurado com o tamanho e o overlap da configuração."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=SEPARADORES,
        length_function=len,
        keep_separator=True,
    )


def dividir_documentos(documentos: Sequence[Document], chunk_size: int) -> List[Document]:
    """Divide os documentos e numera cada chunk com um id estável."""
    config = CONFIGURACOES.get(chunk_size, ConfigChunking(chunk_size))
    chunks = criar_splitter(config).split_documents(list(documentos))

    contadores: Dict[str, int] = {}
    for chunk in chunks:
        chave = f"{chunk.metadata['id']}-p{chunk.metadata['pagina']}"
        contadores[chave] = contadores.get(chave, 0) + 1
        chunk.metadata["chunk_id"] = f"{chave}-c{contadores[chave]}"
        chunk.metadata["chunk_size"] = config.chunk_size
    return chunks


def estatisticas_chunks(chunks: Sequence[Document]) -> Dict[str, float]:
    """Quantidade e tamanho médio, mínimo e máximo dos chunks."""
    tamanhos = [len(c.page_content) for c in chunks] or [0]
    return {
        "chunks": len(chunks),
        "media_caracteres": round(mean(tamanhos), 1),
        "min_caracteres": min(tamanhos),
        "max_caracteres": max(tamanhos),
    }
