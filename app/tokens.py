"""Estimativa de tokens usada pela memória com teto (ConversationTokenBufferMemory).

O ChatOllama não conta tokens sozinho, então a memória precisa dessa
estimativa (~4 caracteres por token) para saber quando descartar o
histórico mais antigo. Não é uma contagem exata, mas é suficiente pro que
a memória precisa decidir.
"""

from __future__ import annotations

from typing import List

CARACTERES_POR_TOKEN = 4


def token_ids(texto: str) -> List[int]:
    """IDs de token — assinatura exigida por `custom_get_token_ids` do LangChain.

    Os IDs são fictícios (zeros); só a CONTAGEM deles importa para a memória.
    """
    return [0] * max(1, len(texto) // CARACTERES_POR_TOKEN)


def contar_tokens(texto: str) -> int:
    """Número estimado de tokens do texto."""
    return len(token_ids(texto))
