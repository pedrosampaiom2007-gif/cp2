"""DocMind RAG — base de conhecimento de treino e atividade física (CKP02).

    fontes             -> catálogo dos documentos reais e seus links
    baixar_documentos  -> baixa as fontes e recorta até 5 páginas de cada
    carregador         -> LOAD: PDF/TXT/MD com metadados
    divisor            -> SPLIT: RecursiveCharacterTextSplitter
    vetores            -> EMBED + STORE: nomic-embed-text + ChromaDB
    reranker           -> cross-encoder que reordena os candidatos
    pipeline           -> RETRIEVE + GENERATE e a função buscar(consulta)
    avaliacao          -> comparação de chunking com RAGAS
"""

__all__ = [
    "fontes",
    "baixar_documentos",
    "carregador",
    "divisor",
    "vetores",
    "reranker",
    "pipeline",
    "avaliacao",
]
