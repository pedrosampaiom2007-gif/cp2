"""Catálogo das fontes reais que alimentam a base de conhecimento do DocMind.

Cada fonte é um PDF público do domínio de treino e atividade física. Do
arquivo original são extraídas no máximo `MAX_PAGINAS` páginas — as que mais
concentram os termos de `palavras_chave` — e o recorte é salvo em
`documentos/` junto com os metadados usados no metadata filtering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

MAX_PAGINAS = 5


@dataclass(frozen=True)
class FonteDocumento:
    """Uma fonte da base: de onde vem, como se chama e como é classificada."""

    id: str
    titulo: str
    autoria: str
    tipo: str
    categoria: str
    publico: str
    url: str
    palavras_chave: Tuple[str, ...]
    ano: Optional[int] = None
    urls_alternativas: Tuple[str, ...] = field(default_factory=tuple)

    def metadados(self) -> Dict[str, object]:
        """Metadados gravados junto de cada chunk no ChromaDB."""
        dados = asdict(self)
        dados.pop("palavras_chave")
        dados.pop("urls_alternativas")
        dados["ano"] = self.ano or 0
        return dados


FONTES: List[FonteDocumento] = [
    FonteDocumento(
        id="ms_guia_atividade_fisica",
        titulo="Guia de Atividade Física para a População Brasileira",
        autoria="Ministério da Saúde — Secretaria de Atenção Primária à Saúde",
        tipo="guia_oficial",
        categoria="recomendacao_atividade_fisica",
        publico="adultos",
        ano=2021,
        url="https://bvsms.saude.gov.br/bvs/publicacoes/guia_atividade_fisica_populacao_brasileira.pdf",
        urls_alternativas=(
            "http://bvsms.saude.gov.br/bvs/publicacoes/guia_atividade_fisica_populacao_brasileira.pdf",
        ),
        palavras_chave=(
            "adulto", "fortalecimento", "muscular", "musculação", "semana",
            "minutos", "moderada", "vigorosa", "intensidade", "força",
        ),
    ),
    FonteDocumento(
        id="oms_diretrizes_atividade_fisica",
        titulo="Diretrizes da OMS para atividade física e comportamento sedentário: num piscar de olhos",
        autoria="Organização Mundial da Saúde (OMS)",
        tipo="guia_oficial",
        categoria="recomendacao_atividade_fisica",
        publico="geral",
        ano=2020,
        url="https://iris.who.int/bitstream/handle/10665/337001/9789240014886-por.pdf",
        urls_alternativas=(
            "https://iris.who.int/server/api/core/bitstreams/9e776de6-adc7-46c1-936f-6dd2bb4f7373/content",
            "https://apps.who.int/iris/bitstream/handle/10665/337001/9789240014886-por.pdf",
        ),
        palavras_chave=(
            "adultos", "idosos", "fortalecimento", "muscular", "semana",
            "minutos", "moderada", "vigorosa", "sedentário", "multicomponente",
        ),
    ),
    FonteDocumento(
        id="rbme_recuperacao_entre_series",
        titulo="Recuperação entre séries no treino de força: revisão sistemática e meta-análise",
        autoria="Revista Brasileira de Medicina do Esporte (SciELO)",
        tipo="artigo_cientifico",
        categoria="prescricao_treino_forca",
        publico="adultos",
        url="https://www.scielo.br/j/rbme/a/Y9vYkwhHhbzKcKNSG9Ft85s/?format=pdf&lang=pt",
        palavras_chave=(
            "intervalo", "recuperação", "séries", "força", "hipertrofia",
            "repetições", "minutos", "volume", "conclusão", "resultados",
        ),
    ),
    FonteDocumento(
        id="rbme_variaveis_treino_idosos",
        titulo=(
            "Influência de variáveis do treinamento contra-resistência sobre a força "
            "muscular de idosos: uma revisão sistemática com ênfase nas relações dose-resposta"
        ),
        autoria="Revista Brasileira de Medicina do Esporte (SciELO)",
        tipo="artigo_cientifico",
        categoria="prescricao_treino_forca",
        publico="idosos",
        url="https://www.scielo.br/j/rbme/a/8z4PZxrP4fPvJgfccndzx8M/?format=pdf&lang=pt",
        palavras_chave=(
            "idosos", "intensidade", "volume", "séries", "frequência",
            "força", "repetições", "dose", "resposta", "conclusão",
        ),
    ),
    FonteDocumento(
        id="jpe_treino_supervisionado",
        titulo=(
            "Sessão de treinamento de força supervisionada aumenta a carga total "
            "levantada e as respostas subjetivas em sujeitos treinados"
        ),
        autoria="Journal of Physical Education (SciELO)",
        tipo="artigo_cientifico",
        categoria="supervisao_treino",
        publico="adultos",
        url="https://www.scielo.br/j/jpe/a/5fnPtNjMh8Swt3g8kcHTgkt/?format=pdf&lang=pt",
        palavras_chave=(
            "supervisão", "supervisionada", "carga", "repetições", "esforço",
            "percepção", "séries", "força", "resultados", "conclusão",
        ),
    ),
]


def fonte_por_id(identificador: str) -> Optional[FonteDocumento]:
    """Procura uma fonte do catálogo pelo id."""
    return next((f for f in FONTES if f.id == identificador), None)


def tabela_fontes_markdown() -> str:
    """Tabela com título, autoria, classificação e link de cada fonte."""
    linhas = [
        "| # | Documento | Autoria | Tipo | Público | Link |",
        "|---:|---|---|---|---|---|",
    ]
    for indice, fonte in enumerate(FONTES, start=1):
        linhas.append(
            f"| {indice} | {fonte.titulo} | {fonte.autoria} | {fonte.tipo} | "
            f"{fonte.publico} | [PDF]({fonte.url}) |"
        )
    return "\n".join(linhas)
