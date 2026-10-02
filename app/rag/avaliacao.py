"""Comparação de chunking com RAGAS (faithfulness + answer_relevancy).

    python -m app.rag.avaliacao                     # 256, 512 e 1024
    python -m app.rag.avaliacao --chunks 512 1024

Para cada chunk_size, o mesmo conjunto de perguntas passa pelo pipeline
inteiro e o RAGAS mede:

- faithfulness: fração das afirmações da resposta que são sustentadas pelos
  trechos recuperados (mede alucinação);
- answer_relevancy: o quanto a resposta trata do que foi perguntado.

O resultado sai em `resultados/` como CSV por pergunta, CSV de médias e um
relatório em Markdown com a configuração vencedora.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence

import pandas as pd

from app.config import PASTA_RESULTADOS, Config, carregar_config
from app.rag.carregador import carregar_documentos
from app.rag.divisor import dividir_documentos, estatisticas_chunks
from app.rag.pipeline import DocMindRAG
from app.rag.reranker import Reranker
from app.rag.vetores import criar_embeddings

CHUNKS_PADRAO = (256, 512, 1024)
META_FAITHFULNESS = 0.7

PERGUNTAS_TESTE: List[str] = [
    "Quantos minutos por semana de atividade física aeróbica de intensidade moderada são recomendados para adultos?",
    "Quantas vezes por semana um adulto deve fazer atividades de fortalecimento muscular?",
    "O que é recomendado para idosos além das atividades aeróbicas e de fortalecimento?",
    "O que as diretrizes recomendam sobre o tempo que passamos sentados?",
    "Qual intervalo de recuperação entre séries favorece o ganho de força e de hipertrofia?",
    "Quais variáveis do treinamento contra-resistência influenciam o ganho de força em idosos?",
    "A supervisão do treino muda a carga total levantada pelos praticantes?",
    "Atividade física de intensidade vigorosa pode substituir a moderada? Em que proporção?",
]


@dataclass
class ResultadoConfiguracao:
    """Notas RAGAS de uma configuração de chunking."""

    chunk_size: int
    total_chunks: int
    tabela: pd.DataFrame

    @property
    def faithfulness(self) -> float:
        """Faithfulness médio das perguntas."""
        return float(self.tabela["faithfulness"].mean())

    @property
    def answer_relevancy(self) -> float:
        """Answer relevancy médio das perguntas."""
        return float(self.tabela["answer_relevancy"].mean())


def _wrappers_ragas(config: Config):
    """LLM juiz e embeddings do RAGAS: gemma4:cloud e nomic-embed-text."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    from app.chain import construir_llm

    llm = LangchainLLMWrapper(construir_llm(config, temperatura=0.0))
    embeddings = LangchainEmbeddingsWrapper(criar_embeddings(config))
    return llm, embeddings


def avaliar_configuracao(
    chunk_size: int,
    perguntas: Sequence[str] = PERGUNTAS_TESTE,
    config: Optional[Config] = None,
    reranker: Optional[Reranker] = None,
    documentos=None,
) -> ResultadoConfiguracao:
    """Roda as perguntas no pipeline de um chunk_size e mede com RAGAS."""
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import Faithfulness, ResponseRelevancy
    from ragas.run_config import RunConfig

    config = config or carregar_config()
    rag = DocMindRAG(chunk_size=chunk_size, config=config, reranker=reranker, documentos=documentos)
    rag.preparar()

    respostas = []
    for pergunta in perguntas:
        resposta = rag.responder(pergunta)
        print(f"  [{chunk_size}] {pergunta[:60]}... → {len(resposta.trechos)} trechos")
        respostas.append(resposta)

    amostras = [
        {
            "user_input": r.pergunta,
            "response": r.resposta,
            "retrieved_contexts": r.contextos or [""],
        }
        for r in respostas
    ]
    llm, embeddings = _wrappers_ragas(config)
    avaliacao = evaluate(
        dataset=EvaluationDataset.from_list(amostras),
        metrics=[Faithfulness(), ResponseRelevancy()],
        llm=llm,
        embeddings=embeddings,
        run_config=RunConfig(timeout=240, max_workers=4, max_retries=5),
        show_progress=True,
    )

    notas = avaliacao.to_pandas()
    tabela = pd.DataFrame(
        {
            "chunk_size": chunk_size,
            "pergunta": [r.pergunta for r in respostas],
            "resposta": [r.resposta for r in respostas],
            "fontes": ["; ".join(t.citacao() for t in r.trechos) for r in respostas],
            "faithfulness": notas["faithfulness"].astype(float).round(3),
            "answer_relevancy": notas["answer_relevancy"].astype(float).round(3),
        }
    )
    return ResultadoConfiguracao(chunk_size, len(rag.chunks), tabela)


def escolher_vencedor(resultados: Sequence[ResultadoConfiguracao]) -> ResultadoConfiguracao:
    """Maior faithfulness médio; answer_relevancy desempata."""
    return max(resultados, key=lambda r: (round(r.faithfulness, 3), round(r.answer_relevancy, 3)))


def tabela_resumo(resultados: Sequence[ResultadoConfiguracao]) -> pd.DataFrame:
    """Médias de faithfulness e answer_relevancy por configuração."""
    return pd.DataFrame(
        [
            {
                "chunk_size": r.chunk_size,
                "chunk_overlap": int(r.chunk_size * 0.125),
                "total_chunks": r.total_chunks,
                "faithfulness_medio": round(r.faithfulness, 3),
                "answer_relevancy_medio": round(r.answer_relevancy, 3),
                "atinge_meta_0_7": r.faithfulness >= META_FAITHFULNESS,
            }
            for r in resultados
        ]
    )


def tabela_por_pergunta(resultados: Sequence[ResultadoConfiguracao]) -> pd.DataFrame:
    """Uma linha por pergunta, com as duas métricas de cada configuração lado a lado."""
    base = None
    for r in resultados:
        parte = r.tabela[["pergunta", "faithfulness", "answer_relevancy"]].rename(
            columns={
                "faithfulness": f"faithfulness_{r.chunk_size}",
                "answer_relevancy": f"answer_relevancy_{r.chunk_size}",
            }
        )
        base = parte if base is None else base.merge(parte, on="pergunta")
    return base


def justificativa(resultados: Sequence[ResultadoConfiguracao]) -> str:
    """Texto com a escolha final apoiada nos números medidos."""
    vencedor = escolher_vencedor(resultados)
    outros = [r for r in resultados if r is not vencedor]
    comparacoes = "; ".join(
        f"chunk {r.chunk_size}: faithfulness {r.faithfulness:.3f}, "
        f"answer_relevancy {r.answer_relevancy:.3f}, {r.total_chunks} chunks"
        for r in outros
    )
    meta = "atinge" if vencedor.faithfulness >= META_FAITHFULNESS else "NÃO atinge"
    return (
        f"Vencedor: chunk_size={vencedor.chunk_size} (overlap {int(vencedor.chunk_size * 0.125)}), "
        f"com faithfulness médio {vencedor.faithfulness:.3f} e answer_relevancy médio "
        f"{vencedor.answer_relevancy:.3f} em {len(vencedor.tabela)} perguntas, sobre "
        f"{vencedor.total_chunks} chunks. Demais configurações — {comparacoes}. "
        f"O faithfulness do vencedor {meta} a meta de {META_FAITHFULNESS}."
    )


def salvar_resultados(resultados: Sequence[ResultadoConfiguracao]) -> str:
    """Grava CSVs e o relatório Markdown em `resultados/`."""
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    detalhado = pd.concat([r.tabela for r in resultados], ignore_index=True)
    detalhado.to_csv(PASTA_RESULTADOS / "ragas_por_pergunta.csv", index=False)
    resumo = tabela_resumo(resultados)
    resumo.to_csv(PASTA_RESULTADOS / "ragas_resumo.csv", index=False)

    relatorio = (
        "# Comparação de chunking — RAGAS\n\n"
        "## Médias por configuração\n\n"
        + resumo.to_markdown(index=False)
        + "\n\n## Por pergunta\n\n"
        + tabela_por_pergunta(resultados).to_markdown(index=False)
        + "\n\n## Escolha final\n\n"
        + justificativa(resultados)
        + "\n"
    )
    (PASTA_RESULTADOS / "ragas_relatorio.md").write_text(relatorio, encoding="utf-8")
    return relatorio


def comparar_chunking(
    chunk_sizes: Sequence[int] = CHUNKS_PADRAO,
    perguntas: Sequence[str] = PERGUNTAS_TESTE,
) -> List[ResultadoConfiguracao]:
    """Avalia cada chunk_size com o mesmo reranker e as mesmas perguntas."""
    config = carregar_config()
    documentos = carregar_documentos()
    reranker = Reranker()
    resultados = []
    for tamanho in chunk_sizes:
        stats = estatisticas_chunks(dividir_documentos(documentos, tamanho))
        print(f"\n=== chunk_size={tamanho} · {stats}")
        resultados.append(
            avaliar_configuracao(tamanho, perguntas, config, reranker=reranker, documentos=documentos)
        )
    return resultados


def main() -> None:
    """Entry point: python -m app.rag.avaliacao"""
    parser = argparse.ArgumentParser(description="Compara configurações de chunking com RAGAS.")
    parser.add_argument("--chunks", nargs="+", type=int, default=list(CHUNKS_PADRAO))
    args = parser.parse_args()

    resultados = comparar_chunking(args.chunks)
    print("\n" + salvar_resultados(resultados))


if __name__ == "__main__":
    main()
