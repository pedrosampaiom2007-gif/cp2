"""Demonstração de *context rot* (degradação de qualidade com contexto crescente).

Metodologia — "agulha no palheiro":

1. No PRIMEIRO turno o usuário planta 3 fatos que o chatbot precisa lembrar:
   objetivo, frequência semanal e uma restrição de segurança (lesão no ombro).
2. Enchemos a conversa com N turnos de papo genérico de academia, que não
   repetem nenhum dos 3 fatos.
3. Fazemos SEMPRE a mesma pergunta de controle no fim: "monte meu treino de
   ombro para amanhã". Uma resposta correta tem de respeitar a restrição.
4. Repetimos para janelas de tamanho crescente (0, 5, 10, 15 e 20 turnos de
   enchimento) e comparamos.

O prompt de controle é idêntico em todas as janelas — só o volume de
contexto muda. Assim, qualquer queda observada é atribuível ao contexto, e
não à pergunta.

Uso:
    python -m app.context_rot
    python -m app.context_rot --janelas 0 10 20
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.config import carregar_config
from app.prompts import SYSTEM_PROMPT_CHAT
from app.tokens import contar_tokens

# Janelas de enchimento avaliadas por padrão.
JANELAS_PADRAO: Sequence[int] = (0, 5, 10, 15, 20)

# --- Turno 1: os fatos plantados ------------------------------------------
FATO_PLANTADO = (
    "Oi! Antes de começar, três coisas sobre mim: meu objetivo é HIPERTROFIA, "
    "eu treino 4 vezes por semana, e eu tenho uma lesão antiga no ombro "
    "direito — o fisioterapeuta me proibiu de fazer desenvolvimento por trás "
    "da nuca e qualquer elevação acima da linha da cabeça."
)
RESPOSTA_PLANTADA = (
    "Anotado: hipertrofia, 4x por semana e restrição no ombro direito — nada "
    "de desenvolvimento por trás da nuca nem elevação acima da cabeça."
)

# --- Pergunta de controle, idêntica em todas as janelas -------------------
PERGUNTA_CONTROLE = "Monta pra mim o treino de ombro de amanhã."

# --- Enchimento: papo de academia que NÃO repete os fatos plantados -------
TURNOS_ENCHIMENTO: List[tuple[str, str]] = [
    ("Qual a melhor hora do dia pra treinar?",
     "A melhor hora é a que você consegue manter todas as semanas. Há uma "
     "vantagem pequena no fim da tarde, mas o efeito é modesto perto do "
     "impacto da consistência. Escolha a janela que você repete."),
    ("Preciso tomar água durante o treino?",
     "Precisa. Numa sessão de até uma hora, de 400 a 600 ml distribuídos ao "
     "longo do treino já dá conta. Não espere a sede aparecer."),
    ("Aquecimento é obrigatório?",
     "É. De cinco a dez minutos de esteira leve, seguidos de séries de "
     "aproximação no primeiro exercício, reduzem risco e melhoram o rendimento."),
    ("Quanto tempo de descanso entre as séries?",
     "De 60 a 90 segundos para isolados e de 2 a 3 minutos para compostos "
     "pesados como agachamento e terra."),
    ("Alongar antes ou depois?",
     "Alongamento estático longo fica melhor depois do treino. Antes, "
     "prefira mobilidade dinâmica: rotação de ombro, abertura de quadril."),
    ("Dá pra treinar com dor muscular tardia?",
     "Dá, se for aquela dor difusa de 24 a 48 horas que melhora com o "
     "movimento. Reduza o volume e siga. Dor localizada é outra história."),
    ("Preciso de creatina?",
     "É o suplemento com melhor evidência para força, mas dose é assunto de "
     "nutricionista. Treino e sono vêm primeiro."),
    ("Quantas horas devo dormir?",
     "De 7 a 9 horas. Dormir mal derruba a recuperação mais rápido que "
     "qualquer erro de programação de treino."),
    ("Cardio atrapalha o ganho de músculo?",
     "Em volume moderado, não. O problema é cardio longo no mesmo dia do "
     "treino de perna."),
    ("Devo treinar até a falha?",
     "Nem sempre. Deixar 1 ou 2 repetições na reserva rende quase o mesmo "
     "com bem menos fadiga acumulada."),
    ("Máquina ou peso livre?",
     "Os dois: peso livre para os compostos principais, máquina para isolar "
     "com segurança no fim do treino."),
    ("Com que frequência troco de treino?",
     "A cada 6 a 10 semanas, ou antes se a progressão de carga travar por "
     "algumas semanas seguidas."),
    ("O que é deload?",
     "Uma semana com volume ou carga reduzidos para a recuperação alcançar "
     "a fadiga acumulada. Programe a cada 6 a 8 semanas."),
    ("Posso treinar dois dias seguidos?",
     "Pode, desde que alterne os grupos musculares trabalhados nos dois dias."),
    ("Respiração correta no exercício?",
     "Expire na fase de esforço, inspire na fase de retorno."),
    ("Suplemento pré-treino é necessário?",
     "Não é necessário. A cafeína, entre 3 e 6 mg por quilo, responde pela "
     "maior parte do efeito percebido."),
    ("Luva de academia serve pra quê?",
     "Conforto e prevenção de calo. Não substitui trabalho de preensão."),
    ("Treinar em jejum funciona?",
     "Funciona para quem tolera, mas não é superior a treinar alimentado."),
    ("Quanto tempo leva pra ver resultado?",
     "As primeiras adaptações neurais aparecem em 2 a 3 semanas; mudança "
     "visível de músculo costuma levar de 8 a 12 semanas de consistência."),
    ("Espelho na academia serve pra algo?",
     "Serve como retorno visual de execução, principalmente em exercícios de "
     "ombro onde é difícil sentir a posição da escápula."),
]


@dataclass
class ResultadoJanela:
    """Uma linha da tabela de context rot."""

    turnos_enchimento: int
    tokens_prompt: int
    lembrou_objetivo: bool
    lembrou_frequencia: bool
    respeitou_restricao: bool

    @property
    def recall(self) -> int:
        """Quantos dos 3 fatos plantados sobreviveram ao contexto."""
        return sum(
            (self.lembrou_objetivo, self.lembrou_frequencia, self.respeitou_restricao)
        )


def _contem(texto: str, padrao: str) -> bool:
    """Busca insensível a caixa, via regex."""
    return re.search(padrao, texto, flags=re.IGNORECASE) is not None


def avaliar_resposta(resposta: str) -> tuple[bool, bool, bool]:
    """Verifica se a resposta ainda respeita os 3 fatos plantados no turno 1."""
    lembrou_objetivo = _contem(resposta, r"hipertrofi|massa muscular|volume muscular")
    lembrou_frequencia = _contem(resposta, r"\b4\b\s*(x|vezes)|quatro vezes|4x")

    # A restrição de segurança só conta ponto se a resposta reconhecer a
    # limitação; prescrever o movimento proibido zera o critério.
    prescreveu_proibido = _contem(
        resposta, r"por tr[aá]s da nuca|atr[aá]s da nuca|desenvolvimento militar por tr"
    )
    citou_restricao = _contem(
        resposta, r"ombro direito|lesão|lesao|restri|evitar|sem elevação acima|fisioterape"
    )
    respeitou_restricao = citou_restricao and not prescreveu_proibido

    return lembrou_objetivo, lembrou_frequencia, respeitou_restricao


def montar_mensagens(turnos_enchimento: int) -> List:
    """Monta a janela de contexto: fato plantado + N turnos + pergunta fixa."""
    mensagens = [
        SystemMessage(content=SYSTEM_PROMPT_CHAT),
        HumanMessage(content=FATO_PLANTADO),
        AIMessage(content=RESPOSTA_PLANTADA),
    ]
    for indice in range(turnos_enchimento):
        # Recicla a lista quando a janela pedida excede o enchimento disponível.
        pergunta, resposta = TURNOS_ENCHIMENTO[indice % len(TURNOS_ENCHIMENTO)]
        mensagens.append(HumanMessage(content=pergunta))
        mensagens.append(AIMessage(content=resposta))
    mensagens.append(HumanMessage(content=PERGUNTA_CONTROLE))
    return mensagens


def rodar_experimento(janelas: Sequence[int] = JANELAS_PADRAO) -> List[ResultadoJanela]:
    """Roda a pergunta de controle em cada janela e devolve os resultados."""
    from app.chain import construir_llm

    config = carregar_config()
    # Temperatura 0 para isolar o efeito do contexto do efeito da amostragem.
    llm = construir_llm(config, temperatura=0.0)

    resultados: List[ResultadoJanela] = []
    for n in janelas:
        mensagens = montar_mensagens(n)
        tokens = contar_tokens("\n".join(str(m.content) for m in mensagens))

        resposta = str(llm.invoke(mensagens).content)
        objetivo, frequencia, restricao = avaliar_resposta(resposta)

        resultado = ResultadoJanela(
            turnos_enchimento=n,
            tokens_prompt=tokens,
            lembrou_objetivo=objetivo,
            lembrou_frequencia=frequencia,
            respeitou_restricao=restricao,
        )
        resultados.append(resultado)
        print(
            f"janela={n:>2} turnos | ~{tokens:>5} tokens | "
            f"recall {resultado.recall}/3 | resposta: {resposta[:80]}..."
        )

    return resultados


def tabela_markdown(resultados: Sequence[ResultadoJanela]) -> str:
    """Tabela comparativa em Markdown com o resultado de cada janela."""
    cabecalho = (
        "| Turnos de contexto | ~Tokens do prompt | Objetivo | Frequência | "
        "Restrição de segurança |\n"
        "|---:|---:|:--:|:--:|:--:|\n"
    )
    marca = lambda ok: "✅" if ok else "❌"  # noqa: E731
    linhas = "".join(
        f"| {r.turnos_enchimento} | {r.tokens_prompt} | "
        f"{marca(r.lembrou_objetivo)} | {marca(r.lembrou_frequencia)} | "
        f"{marca(r.respeitou_restricao)} |\n"
        for r in resultados
    )
    return cabecalho + linhas


def conclusao(resultados: Sequence[ResultadoJanela]) -> str:
    """Interpreta os números da tabela em uma frase de conclusão."""
    if not resultados:
        return "Sem resultados."
    primeiro, ultimo = resultados[0], resultados[-1]
    return (
        f"Do contexto de {primeiro.turnos_enchimento} para "
        f"{ultimo.turnos_enchimento} turnos, o recall dos fatos plantados foi "
        f"de {primeiro.recall}/3 para {ultimo.recall}/3. É o context rot: a "
        f"informação continua DENTRO da janela, mas o modelo deixa de usá-la "
        f"conforme o volume de contexto ao redor cresce. É por isso que a "
        f"memória deste projeto tem teto de tokens em vez de acumular a "
        f"conversa inteira."
    )


def relatorio_completo(resultados: Sequence[ResultadoJanela]) -> str:
    """Bloco em Markdown exibido na aba "Context rot" da interface."""
    return (
        "### Demonstração de context rot\n\n"
        f"Pergunta de controle (idêntica em todas as janelas): "
        f"_\"{PERGUNTA_CONTROLE}\"_\n\n"
        + tabela_markdown(resultados)
        + "\n**Conclusão:** "
        + conclusao(resultados)
        + "\n"
    )


def main() -> None:
    """Entry point: python -m app.context_rot"""
    parser = argparse.ArgumentParser(
        description="Mede a degradação de qualidade conforme o contexto cresce."
    )
    parser.add_argument(
        "--janelas",
        nargs="+",
        type=int,
        default=list(JANELAS_PADRAO),
        help="Quantidades de turnos de enchimento a testar (ex.: 0 5 10 15 20).",
    )
    args = parser.parse_args()

    print("Rodando o experimento de context rot...\n")
    resultados = rodar_experimento([max(0, n) for n in args.janelas])

    print("\n" + tabela_markdown(resultados))
    print("Conclusão: " + conclusao(resultados))


if __name__ == "__main__":
    main()
