"""Etapa LOAD: lê os documentos da pasta `documentos/` como `Document`s.

Aceita PDF, TXT e Markdown. Os metadados de cada arquivo vêm de:

1. `documentos/fontes.json`, para as fontes do catálogo (gerado pelo
   `baixar_documentos`);
2. um arquivo `<nome>.json` ao lado do documento, para documentos novos;
3. valores padrão, quando nenhum dos dois existir.

Cada página de PDF vira um `Document` com o número da página no original,
para a resposta poder citar "documento, p. X".
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import PASTA_DOCUMENTOS

EXTENSOES_SUPORTADAS = (".pdf", ".txt", ".md")
CAMPOS_METADADOS = ("id", "titulo", "autoria", "tipo", "categoria", "publico", "ano", "url")


def limpar_texto(texto: str) -> str:
    """Desfaz hifenização de fim de linha e normaliza espaços."""
    texto = texto.replace("\x00", " ")
    texto = re.sub(r"(\w)-\n(\w)", r"\1\2", texto)
    texto = re.sub(r"[ \t]+", " ", texto)
    texto = re.sub(r" *\n *", "\n", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def _registro_fontes(pasta: Path) -> Dict[str, Dict[str, object]]:
    """Metadados do `fontes.json`, indexados pelo nome do arquivo."""
    arquivo = pasta / "fontes.json"
    if not arquivo.exists():
        return {}
    return {item["arquivo"]: item for item in json.loads(arquivo.read_text("utf-8"))}


def metadados_do_arquivo(caminho: Path, registro: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Monta os metadados de um arquivo, com padrão para campos ausentes."""
    dados: Dict[str, object] = dict(registro.get(caminho.name, {}))
    lateral = caminho.with_suffix(".json")
    if not dados and lateral.exists():
        dados = json.loads(lateral.read_text("utf-8"))

    metadados = {
        "id": dados.get("id", caminho.stem),
        "titulo": dados.get("titulo", caminho.stem.replace("_", " ")),
        "autoria": dados.get("autoria", "não informada"),
        "tipo": dados.get("tipo", "outro"),
        "categoria": dados.get("categoria", "geral"),
        "publico": dados.get("publico", "geral"),
        "ano": int(dados.get("ano") or 0),
        "url": dados.get("url", ""),
        "arquivo": caminho.name,
    }
    paginas = dados.get("paginas_originais")
    metadados["_paginas_originais"] = list(paginas) if paginas else None
    return metadados


def carregar_pdf(caminho: Path, metadados: Dict[str, object]) -> List[Document]:
    """Um `Document` por página com texto."""
    paginas_originais: Optional[List[int]] = metadados.pop("_paginas_originais", None)
    documentos: List[Document] = []
    for indice, pagina in enumerate(PdfReader(str(caminho)).pages):
        texto = limpar_texto(pagina.extract_text() or "")
        if not texto:
            continue
        numero = paginas_originais[indice] if paginas_originais and indice < len(paginas_originais) else indice + 1
        documentos.append(Document(page_content=texto, metadata={**metadados, "pagina": numero}))
    return documentos


def carregar_texto(caminho: Path, metadados: Dict[str, object]) -> List[Document]:
    """TXT e Markdown entram como um único `Document`."""
    metadados.pop("_paginas_originais", None)
    texto = limpar_texto(caminho.read_text(encoding="utf-8", errors="ignore"))
    return [Document(page_content=texto, metadata={**metadados, "pagina": 1})] if texto else []


def carregar_documentos(pasta: Path = PASTA_DOCUMENTOS) -> List[Document]:
    """Carrega todos os documentos suportados da pasta."""
    if not pasta.exists():
        raise FileNotFoundError(
            f"A pasta {pasta} não existe. Rode: python -m app.rag.baixar_documentos"
        )
    registro = _registro_fontes(pasta)
    documentos: List[Document] = []
    for caminho in sorted(pasta.iterdir()):
        if caminho.suffix.lower() not in EXTENSOES_SUPORTADAS:
            continue
        metadados = metadados_do_arquivo(caminho, registro)
        if caminho.suffix.lower() == ".pdf":
            documentos.extend(carregar_pdf(caminho, metadados))
        else:
            documentos.extend(carregar_texto(caminho, metadados))
    if not documentos:
        raise FileNotFoundError(
            f"Nenhum documento encontrado em {pasta}. "
            "Rode: python -m app.rag.baixar_documentos"
        )
    return documentos


def resumo_documentos(documentos: List[Document]) -> List[Dict[str, object]]:
    """Uma linha por arquivo: título, páginas carregadas e caracteres."""
    resumo: Dict[str, Dict[str, object]] = {}
    for doc in documentos:
        chave = doc.metadata["arquivo"]
        item = resumo.setdefault(
            chave,
            {"arquivo": chave, "titulo": doc.metadata["titulo"], "tipo": doc.metadata["tipo"],
             "paginas": 0, "caracteres": 0},
        )
        item["paginas"] += 1
        item["caracteres"] += len(doc.page_content)
    return list(resumo.values())
