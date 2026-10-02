"""Baixa as fontes do catálogo e grava o recorte de até 5 páginas de cada uma.

    python -m app.rag.baixar_documentos            # baixa o que ainda não existe
    python -m app.rag.baixar_documentos --forcar   # baixa tudo de novo

Para cada fonte, as páginas do PDF original recebem uma pontuação pela
frequência das palavras-chave do catálogo; as `MAX_PAGINAS` mais bem
pontuadas são gravadas em `documentos/<id>.pdf`, na ordem original. O
arquivo `documentos/fontes.json` guarda os metadados e quais páginas do
original entraram no recorte.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import ssl
import sys
import unicodedata
import urllib.request
from typing import Dict, List, Optional, Sequence

from pypdf import PdfReader, PdfWriter

from app.config import PASTA_DOCUMENTOS
from app.rag.fontes import FONTES, MAX_PAGINAS, FonteDocumento

ARQUIVO_FONTES = PASTA_DOCUMENTOS / "fontes.json"
MIN_CARACTERES_PAGINA = 400
CABECALHOS_HTTP = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


def _sem_acento(texto: str) -> str:
    normalizado = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in normalizado if not unicodedata.combining(c))


def baixar_pdf(urls: Sequence[str], timeout: int = 60) -> bytes:
    """Tenta cada URL até receber um PDF válido."""
    erros: List[str] = []
    contexto = ssl.create_default_context()
    for url in urls:
        try:
            requisicao = urllib.request.Request(url, headers=CABECALHOS_HTTP)
            with urllib.request.urlopen(requisicao, timeout=timeout, context=contexto) as resposta:
                conteudo = resposta.read()
            if conteudo[:5] == b"%PDF-":
                return conteudo
            erros.append(f"{url}: resposta não é PDF")
        except Exception as erro:
            erros.append(f"{url}: {erro}")
    raise RuntimeError("Não foi possível baixar o PDF.\n  " + "\n  ".join(erros))


def pontuar_pagina(texto: str, palavras_chave: Sequence[str]) -> int:
    """Quantas vezes as palavras-chave aparecem no texto da página."""
    alvo = _sem_acento(texto)
    if len(alvo.strip()) < MIN_CARACTERES_PAGINA:
        return 0
    return sum(len(re.findall(re.escape(_sem_acento(p)), alvo)) for p in palavras_chave)


def selecionar_paginas(
    leitor: PdfReader,
    palavras_chave: Sequence[str],
    maximo: int = MAX_PAGINAS,
) -> List[int]:
    """Índices (base 0) das páginas mais relevantes, na ordem do original."""
    if len(leitor.pages) <= maximo:
        return list(range(len(leitor.pages)))
    pontuacoes = [
        (pontuar_pagina(pagina.extract_text() or "", palavras_chave), indice)
        for indice, pagina in enumerate(leitor.pages)
    ]
    melhores = sorted(pontuacoes, key=lambda par: (-par[0], par[1]))[:maximo]
    return sorted(indice for _, indice in melhores)


def detectar_ano(leitor: PdfReader) -> Optional[int]:
    """Ano de publicação: o ano mais citado na primeira página do original
    (cabeçalho do periódico), ou a data de criação do PDF."""
    texto = (leitor.pages[0].extract_text() or "") if len(leitor.pages) else ""
    anos = [int(a) for a in re.findall(r"\b(19[89]\d|20[0-2]\d)\b", texto)]
    if anos:
        return max(set(anos), key=anos.count)
    try:
        data = leitor.metadata.creation_date if leitor.metadata else None
        if data and 1990 <= data.year <= 2030:
            return data.year
    except Exception:
        pass
    return None


def gravar_recorte(conteudo: bytes, fonte: FonteDocumento) -> Dict[str, object]:
    """Grava `documentos/<id>.pdf` com as páginas selecionadas."""
    leitor = PdfReader(io.BytesIO(conteudo))
    paginas = selecionar_paginas(leitor, fonte.palavras_chave)

    escritor = PdfWriter()
    for indice in paginas:
        escritor.add_page(leitor.pages[indice])
    destino = PASTA_DOCUMENTOS / f"{fonte.id}.pdf"
    with destino.open("wb") as arquivo:
        escritor.write(arquivo)

    metadados = fonte.metadados()
    metadados["ano"] = fonte.ano or detectar_ano(leitor) or 0
    metadados["arquivo"] = destino.name
    metadados["paginas_originais"] = [i + 1 for i in paginas]
    metadados["total_paginas_original"] = len(leitor.pages)
    return metadados


def carregar_registro() -> Dict[str, Dict[str, object]]:
    """Lê o `fontes.json` já existente (vazio se ainda não houver)."""
    if not ARQUIVO_FONTES.exists():
        return {}
    return {item["id"]: item for item in json.loads(ARQUIVO_FONTES.read_text("utf-8"))}


def salvar_registro(registro: Dict[str, Dict[str, object]]) -> None:
    ARQUIVO_FONTES.write_text(
        json.dumps(list(registro.values()), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def baixar_todas(forcar: bool = False) -> Dict[str, Dict[str, object]]:
    """Baixa e recorta todas as fontes do catálogo."""
    PASTA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    registro = carregar_registro()

    for fonte in FONTES:
        destino = PASTA_DOCUMENTOS / f"{fonte.id}.pdf"
        if destino.exists() and fonte.id in registro and not forcar:
            print(f"✔ {fonte.id}: já existe, pulando")
            continue
        try:
            conteudo = baixar_pdf((fonte.url, *fonte.urls_alternativas))
            registro[fonte.id] = gravar_recorte(conteudo, fonte)
            paginas = registro[fonte.id]["paginas_originais"]
            print(f"✔ {fonte.id}: páginas {paginas} do original → {destino.name}")
        except Exception as erro:
            print(f"✘ {fonte.id}: {erro}", file=sys.stderr)
            print(
                f"  Baixe manualmente de {fonte.url} e rode:\n"
                f"  python -m app.rag.baixar_documentos --arquivo {fonte.id} caminho/do/arquivo.pdf",
                file=sys.stderr,
            )

    salvar_registro(registro)
    return registro


def importar_arquivo_local(identificador: str, caminho: str) -> Dict[str, object]:
    """Recorta um PDF baixado à mão para uma fonte do catálogo."""
    fonte = next((f for f in FONTES if f.id == identificador), None)
    if fonte is None:
        raise SystemExit(f"Fonte desconhecida: {identificador}")
    with open(caminho, "rb") as arquivo:
        conteudo = arquivo.read()
    PASTA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)
    registro = carregar_registro()
    registro[fonte.id] = gravar_recorte(conteudo, fonte)
    salvar_registro(registro)
    print(f"✔ {fonte.id}: páginas {registro[fonte.id]['paginas_originais']} gravadas")
    return registro[fonte.id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa as fontes da base do DocMind.")
    parser.add_argument("--forcar", action="store_true", help="Baixa tudo de novo.")
    parser.add_argument(
        "--arquivo",
        nargs=2,
        metavar=("ID_FONTE", "CAMINHO_PDF"),
        help="Usa um PDF já baixado em vez de baixar da internet.",
    )
    args = parser.parse_args()
    if args.arquivo:
        importar_arquivo_local(*args.arquivo)
    else:
        baixar_todas(forcar=args.forcar)


if __name__ == "__main__":
    main()
