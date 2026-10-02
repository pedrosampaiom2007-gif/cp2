"""Reranking com cross-encoder.

A busca vetorial compara embeddings calculados separadamente para a pergunta
e para o chunk. O cross-encoder lê os dois juntos e dá uma nota de
relevância mais precisa, então ele reordena os candidatos que o Chroma
trouxe antes de eles irem para o LLM.
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Sequence

from app.rag.vetores import Trecho

MODELO_RERANKER = os.getenv("RAG_RERANKER_MODELO", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class Reranker:
    """Carrega o cross-encoder sob demanda e reordena trechos."""

    def __init__(self, modelo: str = MODELO_RERANKER) -> None:
        self.nome_modelo = modelo
        self._modelo = None
        self.erro: Optional[str] = None

    @property
    def disponivel(self) -> bool:
        if self._modelo is None and self.erro is None:
            try:
                from sentence_transformers import CrossEncoder

                self._modelo = CrossEncoder(self.nome_modelo, max_length=512)
            except Exception as erro:
                self.erro = str(erro)
                warnings.warn(f"Reranker indisponível ({erro}); usando a ordem da busca vetorial.")
        return self._modelo is not None

    def reordenar(self, consulta: str, trechos: Sequence[Trecho], top_k: int) -> List[Trecho]:
        """Devolve os `top_k` trechos mais relevantes segundo o cross-encoder."""
        trechos = list(trechos)
        if not trechos or not self.disponivel:
            return trechos[:top_k]
        notas = self._modelo.predict([(consulta, t.texto) for t in trechos])
        for trecho, nota in zip(trechos, notas):
            trecho.score_rerank = round(float(nota), 4)
        return sorted(trechos, key=lambda t: t.score_rerank, reverse=True)[:top_k]
