"""Interface Gradio + entry point do Halter (CKP01) com o DocMind RAG (CKP02).

    python -m app.main          # sobe a interface em http://localhost:7860
    python -m app.main --share  # link público temporário do Gradio
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import Dict, List, Optional, Tuple

from app.chain import RESPOSTA_BLOQUEIO, ChatbotTreino, detectar_tentativa_injecao
from app.config import ConfiguracaoInvalida, carregar_config
from app.memory_manager import contar_turnos_usuario, resumo_estrategia
from app.prompts import NOME_ASSISTENTE
from app.rag.fontes import FONTES, tabela_fontes_markdown

TITULO = f"{NOME_ASSISTENTE} — assistente de treino de academia"

EXEMPLOS_RAG = [
    "Quantos minutos por semana de atividade moderada um adulto deve fazer?",
    "Quantas vezes por semana devo fazer fortalecimento muscular?",
    "Qual intervalo entre séries favorece a hipertrofia?",
    "O que é recomendado para idosos além do treino de força?",
    "A supervisão do treino muda a carga levantada?",
]

TODOS = "(todos)"

MENSAGEM_ABERTURA = (
    f"Fala! Eu sou o {NOME_ASSISTENTE}, seu assistente de treino. "
    "Me conta seu objetivo, há quanto tempo você treina, quantos dias por "
    "semana você tem disponíveis e se existe alguma lesão ou limitação."
)

EXEMPLOS = [
    "Quero ganhar massa, treino há 3 meses e tenho 4 dias por semana.",
    "Monta um treino A de peito e tríceps pra mim.",
    "Sinto dor no ombro direito quando faço supino inclinado.",
    "Quantas séries por semana eu preciso fazer de costas?",
    "Qual anabolizante você recomenda?",
    "Qual é a capital da Austrália?",
]


# ---------------------------------------------------------------------------
# Estado da aplicação
# ---------------------------------------------------------------------------
class Aplicacao:
    """Guarda o chatbot e a última análise, compartilhados pelas abas."""

    def __init__(self, verbose: bool = False) -> None:
        self.bot = ChatbotTreino(verbose=verbose)
        self.ultima_analise = None
        self.bases_rag: Dict[int, object] = {}

    # -- chat --------------------------------------------------------------
    def conversar(self, mensagem: str) -> Tuple[str, str]:
        """Roda as 2 chains e devolve (resposta, análise em Markdown)."""
        mensagem = (mensagem or "").strip()
        if not mensagem:
            return "Manda a sua dúvida que eu respondo.", self.analise_markdown()

        try:
            resposta, analise = self.bot.responder_com_analise(mensagem)
            self.ultima_analise = analise
            return resposta, self.analise_markdown()
        except Exception as erro:  # rede fora, chave inválida, JSON quebrado...
            traceback.print_exc()
            return (
                f"⚠️ Não consegui falar com o modelo agora: `{erro}`\n\n"
                "Confira a `OLLAMA_API_KEY` no `.env` e a sua conexão.",
                self.analise_markdown(),
            )

    def analise_markdown(self) -> str:
        """Renderiza a última saída validada pelo Pydantic."""
        a = self.ultima_analise
        if a is None:
            return "_Nenhuma mensagem analisada ainda._"
        grupos = ", ".join(a.grupos_musculares) if a.grupos_musculares else "—"
        alerta = "🚨 **Encaminhar a profissional de saúde**" if a.exige_encaminhamento() else "—"
        escopo = "❌ fora do escopo" if a.fora_do_escopo else "✅ dentro do escopo"
        manipulacao = "🛑 **sim — bloqueado**" if a.tentativa_manipulacao else "—"
        return (
            "### `AnaliseConsulta` — saída validada por Pydantic v2\n\n"
            f"| Campo | Valor |\n|---|---|\n"
            f"| `objetivo_treino` | {a.objetivo_treino} |\n"
            f"| `nivel_experiencia` | {a.nivel_experiencia} |\n"
            f"| `grupos_musculares` | {grupos} |\n"
            f"| `risco_seguranca` | {a.risco_seguranca}/5 |\n"
            f"| `fora_do_escopo` | {escopo} |\n"
            f"| `tentativa_manipulacao` | {manipulacao} |\n"
            f"| `resumo_intencao` | {a.resumo_intencao} |\n\n"
            f"{alerta}\n\n"
            "<details><summary>JSON bruto</summary>\n\n```json\n"
            f"{a.model_dump_json(indent=2)}\n```\n</details>"
        )

    # -- relatório ---------------------------------------------------------
    def relatorio(self) -> str:
        """Gera o RelatorioSessao a partir do histórico em memória."""
        if contar_turnos_usuario(self.bot.memoria) == 0:
            return "_Converse primeiro — não há sessão para relatar._"
        try:
            return self.bot.gerar_relatorio().para_markdown()
        except Exception as erro:
            traceback.print_exc()
            return f"⚠️ Não consegui gerar o relatório: `{erro}`"

    # -- memória -----------------------------------------------------------
    def painel_memoria(self) -> str:
        """Mostra a justificativa da memória e o estado atual do histórico."""
        cfg = self.bot.config
        turnos = contar_turnos_usuario(self.bot.memoria)
        base = resumo_estrategia(cfg.memoria_estrategia, cfg.memoria_max_tokens)
        marca = "✅" if turnos >= 5 else "⏳"
        return (
            base
            + f"\n---\n\n**Turnos do usuário nesta sessão:** {turnos} "
            f"{marca} (o checkpoint pede a demonstração em 5 ou mais turnos)\n"
        )

    def limpar(self) -> Tuple[List, str, str]:
        """Zera a conversa e devolve o estado inicial das abas."""
        self.bot.limpar_memoria()
        self.ultima_analise = None
        return [], self.analise_markdown(), self.painel_memoria()

    # -- DocMind RAG -------------------------------------------------------
    def base_rag(self, chunk_size: int):
        """Base RAG do chunk_size pedido, indexada na primeira chamada."""
        from app.rag.pipeline import DocMindRAG

        if chunk_size not in self.bases_rag:
            rag = DocMindRAG(chunk_size=chunk_size, config=self.bot.config)
            rag.preparar()
            self.bases_rag[chunk_size] = rag
        return self.bases_rag[chunk_size]

    def indexar_rag(self, chunk_size: int) -> str:
        """Carrega, divide e indexa os documentos no ChromaDB."""
        from app.rag.carregador import resumo_documentos
        from app.rag.divisor import estatisticas_chunks

        try:
            self.bases_rag.pop(int(chunk_size), None)
            rag = self.base_rag(int(chunk_size))
            linhas = "\n".join(
                f"| {d['titulo']} | {d['tipo']} | {d['paginas']} | {d['caracteres']} |"
                for d in resumo_documentos(rag.documentos)
            )
            stats = estatisticas_chunks(rag.chunks)
            return (
                f"**Coleção `{rag.base.nome_colecao}`** com {rag.base.total()} chunks "
                f"(média de {stats['media_caracteres']} caracteres).\n\n"
                "| Documento | Tipo | Páginas | Caracteres |\n|---|---|---:|---:|\n" + linhas
            )
        except Exception as erro:
            traceback.print_exc()
            return f"⚠️ Não consegui indexar: `{erro}`"

    def conversar_rag(
        self,
        pergunta: str,
        chunk_size: int,
        tipo: str,
        publico: str,
        reranking: bool,
    ) -> Tuple[str, str]:
        """Resposta do RAG e o painel de fontes citadas."""
        from app.rag.vetores import montar_filtro

        pergunta = (pergunta or "").strip()
        if not pergunta:
            return "Faça uma pergunta sobre treino ou atividade física.", ""
        if detectar_tentativa_injecao(pergunta):
            return RESPOSTA_BLOQUEIO, ""
        try:
            rag = self.base_rag(int(chunk_size))
            filtros = montar_filtro(
                tipo=None if tipo == TODOS else tipo,
                publico=None if publico == TODOS else publico,
            )
            resposta = rag.responder(pergunta, filtros=filtros, reranking=reranking)
            return resposta.para_markdown(), resposta.fontes_markdown()
        except Exception as erro:
            traceback.print_exc()
            return (
                f"⚠️ Falha no RAG: `{erro}`\n\nConfira se os documentos foram baixados "
                "(`python -m app.rag.baixar_documentos`) e a `OLLAMA_API_KEY`.",
                "",
            )

    # -- context rot -------------------------------------------------------
    def rodar_context_rot(self) -> str:
        """Executa a demonstração de degradação e devolve a tabela."""
        from app.context_rot import relatorio_completo, rodar_experimento

        try:
            resultados = rodar_experimento()
            return relatorio_completo(resultados)
        except Exception as erro:
            traceback.print_exc()
            return f"⚠️ Falha ao rodar o experimento: `{erro}`"


# ---------------------------------------------------------------------------
# Interface Gradio
# ---------------------------------------------------------------------------
def _versao_gradio(gr) -> int:
    """Versão maior do Gradio instalado (usada para pequenos ajustes de API)."""
    try:
        return int(str(gr.__version__).split(".")[0])
    except Exception:
        return 4


def construir_interface(app: Aplicacao):
    """Monta as abas da interface local."""
    import gradio as gr

    major = _versao_gradio(gr)

    blocks_kwargs = {"title": TITULO}
    if major < 6:
        blocks_kwargs["theme"] = gr.themes.Soft()

    chatbot_kwargs = {} if major >= 6 else {"type": "messages"}

    with gr.Blocks(**blocks_kwargs) as demo:
        gr.Markdown(
            f"# 🏋️ {TITULO}\n"
            "FIAP · Prompt Engineering & AI · 2º Semestre — "
            "LangChain LCEL, memória gerenciada, Pydantic v2 e RAG com ChromaDB."
        )

        with gr.Tab("💬 Conversa"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        value=[{"role": "assistant", "content": MENSAGEM_ABERTURA}],
                        height=460,
                        label=NOME_ASSISTENTE,
                        **chatbot_kwargs,
                    )
                    entrada = gr.Textbox(
                        placeholder="Escreva aqui e aperte Enter...",
                        label="Sua mensagem",
                        lines=2,
                    )
                    with gr.Row():
                        enviar = gr.Button("Enviar", variant="primary")
                        limpar = gr.Button("Nova sessão")
                    gr.Examples(examples=EXEMPLOS, inputs=entrada, label="Exemplos")

                with gr.Column(scale=2):
                    gr.Markdown("### Saída estruturada")
                    painel_analise = gr.Markdown(app.analise_markdown())

        with gr.Tab("📚 DocMind RAG"):
            gr.Markdown(
                "Perguntas respondidas **somente** com base nos documentos da base, "
                "com o trecho de origem citado em cada resposta."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    chat_rag = gr.Chatbot(height=460, label="DocMind", **chatbot_kwargs)
                    entrada_rag = gr.Textbox(
                        placeholder="Pergunte sobre treino ou atividade física...",
                        label="Pergunta",
                        lines=2,
                    )
                    with gr.Row():
                        enviar_rag = gr.Button("Perguntar", variant="primary")
                        limpar_rag = gr.Button("Limpar")
                    gr.Examples(examples=EXEMPLOS_RAG, inputs=entrada_rag, label="Exemplos")
                with gr.Column(scale=2):
                    chunk_rag = gr.Dropdown(
                        choices=[256, 512, 1024],
                        value=app.bot.config.rag_chunk_size,
                        label="chunk_size",
                    )
                    tipo_rag = gr.Dropdown(
                        choices=[TODOS] + sorted({f.tipo for f in FONTES}),
                        value=TODOS,
                        label="Filtro: tipo de documento",
                    )
                    publico_rag = gr.Dropdown(
                        choices=[TODOS] + sorted({f.publico for f in FONTES}),
                        value=TODOS,
                        label="Filtro: público",
                    )
                    rerank_rag = gr.Checkbox(
                        value=app.bot.config.rag_reranking, label="Reranking com cross-encoder"
                    )
                    indexar_rag = gr.Button("Indexar documentos")
                    painel_fontes = gr.Markdown("### Fontes citadas\n_Faça uma pergunta._")
            with gr.Accordion("Documentos da base", open=False):
                gr.Markdown(tabela_fontes_markdown())
                painel_indice = gr.Markdown()

        with gr.Tab("🧠 Memória"):
            gr.Markdown(
                "A memória é gerenciada pela `ConversationChain` (chain 1 da "
                "arquitetura da Aula 03). A justificativa completa da escolha "
                "está no `README.md`."
            )
            painel_memoria = gr.Markdown(app.painel_memoria())
            atualizar_memoria = gr.Button("Atualizar")

        with gr.Tab("📋 Relatório da sessão"):
            gr.Markdown(
                "Fecha o atendimento com a segunda saída validada por Pydantic "
                "(`RelatorioSessao`), gerada por um pipeline LCEL separado."
            )
            botao_relatorio = gr.Button("Gerar relatório", variant="primary")
            painel_relatorio = gr.Markdown()

        with gr.Tab("📉 Context rot"):
            gr.Markdown(
                "Mesma pergunta de controle, janelas de contexto de 0, 5, 10, "
                "15 e 20 turnos. O experimento faz 5 chamadas ao modelo e "
                "leva cerca de 1 minuto."
            )
            botao_rot = gr.Button("Rodar experimento", variant="primary")
            painel_rot = gr.Markdown()

        # -- ligações --------------------------------------------------------
        def _turno(mensagem: str, historico: List[Dict[str, str]]):
            """Callback de um turno: atualiza chat, análise e painel de memória."""
            mensagem = (mensagem or "").strip()
            if not mensagem:
                return historico, "", app.analise_markdown(), app.painel_memoria()
            resposta, analise_md = app.conversar(mensagem)
            historico = list(historico) + [
                {"role": "user", "content": mensagem},
                {"role": "assistant", "content": resposta},
            ]
            return historico, "", analise_md, app.painel_memoria()

        saidas = [chatbot, entrada, painel_analise, painel_memoria]
        enviar.click(_turno, [entrada, chatbot], saidas)
        entrada.submit(_turno, [entrada, chatbot], saidas)

        def _limpar():
            _, analise_md, memoria_md = app.limpar()
            return (
                [{"role": "assistant", "content": MENSAGEM_ABERTURA}],
                "",
                analise_md,
                memoria_md,
            )

        limpar.click(_limpar, None, saidas)

        def _turno_rag(pergunta, historico, chunk_size, tipo, publico, reranking):
            pergunta = (pergunta or "").strip()
            if not pergunta:
                return historico, "", gr.update()
            resposta, fontes = app.conversar_rag(pergunta, chunk_size, tipo, publico, reranking)
            historico = list(historico or []) + [
                {"role": "user", "content": pergunta},
                {"role": "assistant", "content": resposta},
            ]
            return historico, "", "### Fontes citadas\n\n" + (fontes or "—")

        entradas_rag = [entrada_rag, chat_rag, chunk_rag, tipo_rag, publico_rag, rerank_rag]
        saidas_rag = [chat_rag, entrada_rag, painel_fontes]
        enviar_rag.click(_turno_rag, entradas_rag, saidas_rag)
        entrada_rag.submit(_turno_rag, entradas_rag, saidas_rag)
        limpar_rag.click(lambda: ([], "", "### Fontes citadas\n_Faça uma pergunta._"), None, saidas_rag)
        indexar_rag.click(app.indexar_rag, chunk_rag, painel_indice)
        atualizar_memoria.click(app.painel_memoria, None, painel_memoria)
        botao_relatorio.click(app.relatorio, None, painel_relatorio)
        botao_rot.click(app.rodar_context_rot, None, painel_rot)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=TITULO)
    parser.add_argument("--share", action="store_true", help="Link público do Gradio.")
    parser.add_argument("--porta", type=int, default=7860, help="Porta do Gradio.")
    parser.add_argument("--verbose", action="store_true", help="Loga as chains.")
    args = parser.parse_args(argv)

    # Falha cedo e com mensagem clara quando o .env não está pronto.
    try:
        carregar_config()
    except ConfiguracaoInvalida as erro:
        print(f"\n❌ Configuração inválida:\n\n{erro}\n", file=sys.stderr)
        return 1

    app = Aplicacao(verbose=args.verbose)

    try:
        demo = construir_interface(app)
    except ImportError:
        print(
            "\n❌ Gradio não está instalado. Rode:  "
            "pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 1

    import gradio as gr

    launch_kwargs = {
        "server_name": "127.0.0.1",
        "server_port": args.porta,
        "share": args.share,
    }
    if _versao_gradio(gr) >= 6:
        launch_kwargs["theme"] = gr.themes.Soft()

    demo.launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
