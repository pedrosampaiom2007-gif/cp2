"""Configuração central do projeto, lida exclusivamente do arquivo .env.

Regra do checkpoint: a OLLAMA_API_KEY nunca pode estar hardcoded no código.
Ela é carregada com python-dotenv a partir do .env, que não é versionado nem
enviado no .zip da entrega (apenas o .env.example).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto (a pasta que contém app/, .env, requirements.txt...).
RAIZ_PROJETO = Path(__file__).resolve().parent.parent

# Carrega o .env da raiz. Não sobrescreve variáveis já exportadas no shell,
# o que permite rodar em CI/containers sem arquivo .env.
load_dotenv(dotenv_path=RAIZ_PROJETO / ".env", override=False)

# Modelo obrigatório do checkpoint — não usar modelos deprecated.
MODELO_OBRIGATORIO = "gemma4:cloud"

# Faixa de tokens exigida pelo enunciado para a memória com teto.
MEMORIA_TOKENS_MIN = 800
MEMORIA_TOKENS_MAX = 1500


class ConfiguracaoInvalida(RuntimeError):
    """Erro levantado quando o .env está ausente ou mal preenchido."""


def _ler_int(nome: str, padrao: int) -> int:
    """Lê uma variável de ambiente inteira, caindo no padrão se inválida."""
    bruto = os.getenv(nome)
    if bruto is None or not bruto.strip():
        return padrao
    try:
        return int(bruto)
    except ValueError:
        return padrao


def _ler_float(nome: str, padrao: float) -> float:
    """Lê uma variável de ambiente float, caindo no padrão se inválida."""
    bruto = os.getenv(nome)
    if bruto is None or not bruto.strip():
        return padrao
    try:
        return float(bruto)
    except ValueError:
        return padrao


@dataclass(frozen=True)
class Config:
    """Snapshot imutável da configuração da aplicação."""

    api_key: str
    modelo: str
    base_url: str
    temperatura: float
    memoria_estrategia: str
    memoria_max_tokens: int

    def validar(self) -> None:
        """Falha rápido e com mensagem em PT-BR quando o .env está errado."""
        if not self.api_key or self.api_key.startswith("coloque_sua_chave"):
            raise ConfiguracaoInvalida(
                "OLLAMA_API_KEY não configurada.\n"
                "Rode:  cp .env.example .env  e edite o .env com a sua chave "
                "da Ollama Cloud (https://ollama.com -> Settings -> Keys)."
            )
        if self.modelo != MODELO_OBRIGATORIO:
            raise ConfiguracaoInvalida(
                f"O checkpoint exige o modelo '{MODELO_OBRIGATORIO}' via Ollama "
                f"Cloud, mas o .env pede '{self.modelo}'. Ajuste OLLAMA_MODEL."
            )
        if self.memoria_estrategia not in {"buffer", "summary", "token_buffer"}:
            raise ConfiguracaoInvalida(
                "MEMORIA_ESTRATEGIA deve ser 'buffer', 'summary' ou "
                f"'token_buffer' (recebido: '{self.memoria_estrategia}')."
            )
        if not MEMORIA_TOKENS_MIN <= self.memoria_max_tokens <= MEMORIA_TOKENS_MAX:
            raise ConfiguracaoInvalida(
                "MEMORIA_MAX_TOKENS precisa ficar entre "
                f"{MEMORIA_TOKENS_MIN} e {MEMORIA_TOKENS_MAX} tokens "
                f"(recebido: {self.memoria_max_tokens})."
            )


def carregar_config() -> Config:
    """Monta a Config a partir do ambiente e a valida antes de devolver."""
    config = Config(
        api_key=os.getenv("OLLAMA_API_KEY", "").strip(),
        modelo=os.getenv("OLLAMA_MODEL", MODELO_OBRIGATORIO).strip(),
        base_url=os.getenv("OLLAMA_BASE_URL", "https://ollama.com").strip(),
        temperatura=_ler_float("OLLAMA_TEMPERATURE", 0.3),
        memoria_estrategia=os.getenv("MEMORIA_ESTRATEGIA", "token_buffer").strip().lower(),
        memoria_max_tokens=_ler_int("MEMORIA_MAX_TOKENS", 1200),
    )
    config.validar()
    return config
