"""Mapa GABINETE (DataJud) -> ministro relator, apurado da propria base de sessoes.

Por que existe
--------------
O relator extraido do video e fragil: quando o nome nao e dito com clareza no trecho, o
modelo pega um nome da lista de "Composição da sessão" que o proprio prompt injeta no
contexto do bloco -- e essa lista vem de uma VOTACAO entre os chunks do scan, cujo corte
(25% das leituras) tem uma faixa de empate tecnico onde o ruido decide. Em 26/08/2026 o
mesmo julgamento (lista triplice do TRE-AM, janela 1565-1713s, identica nas tres rodadas)
saiu com tres relatores diferentes -- Estela Aranha, "Stella Aranha" e Isabel Gallotti --
porque Isabel Gallotti cruzou o limiar da composicao na terceira rodada.

O DataJud publica o ORGAO JULGADOR de cada processo, que no TSE e o gabinete
("GABINETE STF1", "GABINETE JURISTA 1"...). O gabinete e a CADEIRA do relator: e um dado
oficial, estavel dentro de uma composicao e imune ao ruido do audio.

Por que o mapa e apurado, e nao escrito a mao
--------------------------------------------
A ocupacao dos gabinetes muda quando um ministro sai. Uma tabela fixa no codigo passaria a
"corrigir" relatores para o ministro ERRADO, de forma sistematica e silenciosa, no dia
seguinte a uma posse. Aqui o mapa e derivado da propria base a cada apuracao, restrito a
uma janela recente, e so vale para o gabinete cuja ocupacao a base indica com dominancia
clara e amostra suficiente -- gabinete em transicao de ocupante fica dividido, nao alcanca o
limiar e nao corrige nada.

Apuracao de 26/08/2026 (150 processos, 6 meses). Conclusivos: STF1=Nunes Marques,
STF2 fica de fora (76%), STF3=Andre Mendonca, STJ1 fica de fora (Cueva -> Isabel Gallotti,
transicao em curso), STJ2=Antonio Carlos Ferreira, JURISTA 1=Estela Aranha,
JURISTA 2=Floriano de Azevedo Marques.

CUIDADO ao ler esta base: "Min. Isabel Gallotti" NAO e nome alucinado -- ela e ministra do
TSE e ocupa o STJ1. O erro de 26/08/2026 nao foi inventa-la, foi atribuir-lhe um processo
do JURISTA 1 (cadeira de jurista, que ela nao ocupa por ser ministra do STJ).
"""
from __future__ import annotations

import collections
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

STATE_DIR = Path(__file__).resolve().parent / "artifacts" / "state"
CACHE_FILE = STATE_DIR / "gabinete_relator.json"
# TTL do cache. Curto o bastante para acompanhar uma mudanca de composicao em poucos dias,
# longo o bastante para nao pagar a apuracao (dezenas de consultas ao DataJud) a cada lote.
CACHE_TTL_SECONDS = 7 * 24 * 3600
# Um gabinete so entra no mapa com pelo menos estes processos E com a dominancia abaixo.
# Calibrado na apuracao de 26/08/2026 (150 processos, janela de 6 meses):
#   STF1 96% Nunes Marques | STJ2 96% Antonio Carlos Ferreira | STF3 91% Andre Mendonca
#   JURISTA 2 92% Floriano | JURISTA 1 91% Estela Aranha          <- entram
#   STF2 76% (Toffoli, com 4 leituras "Nunes Marques" do presidente falando)
#   STJ1 74% (Cueva x Isabel Gallotti: TRANSICAO REAL de ocupante)  <- ficam de fora
#   CGE n=4 e "JUIZ AUXILIAR 2" n=1 (nao sao cadeira de relator titular) <- fora pelo n
# Nao se exige unanimidade: nenhum gabinete e 100%, porque as impurezas SAO os erros de
# extracao que este mapa existe para corrigir. Mas 85% deixa fora o gabinete em transicao,
# que e o unico caso em que corrigir automaticamente causaria dano sistematico.
MIN_PROCESSOS = 8
MIN_DOMINANCIA = 0.85
# Janela da apuracao: composicao vigente. Curta de proposito -- com 18 meses, um gabinete que
# trocou de ocupante no meio do periodo aparece dividido e some do mapa por tempo demais.
JANELA_MESES = 6


def _norm_gabinete(valor: str) -> str:
    texto = re.sub(r"\s+", " ", str(valor or "")).strip().upper()
    return texto


def _carregar_cache() -> Optional[dict[str, Any]]:
    try:
        dados = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - cache ausente/corrompido nao e erro
        return None
    if not isinstance(dados, dict) or not isinstance(dados.get("mapa"), dict):
        return None
    if time.time() - float(dados.get("apurado_em") or 0) > CACHE_TTL_SECONDS:
        return None
    return dados


def _gravar_cache(mapa: dict[str, str], amostra: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps({"apurado_em": time.time(), "mapa": mapa, "amostra": amostra},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")


def apurar_mapa(
    paginas: list[dict[str, Any]],
    lookup_process,
    *,
    session=None,
    logger: Optional[logging.Logger] = None,
    limite_consultas: int = 160,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Apura {gabinete: ministro} cruzando a base com o orgao julgador do DataJud.

    `paginas` e uma lista de {"numero": <CNJ-20>, "relator": "Min. Fulano"} ja filtrada
    pela janela de tempo. So devolve gabinete com >= MIN_PROCESSOS e dominancia do ocupante
    >= MIN_DOMINANCIA.
    """
    log = logger or logging.getLogger(__name__)
    votos: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    consultas = 0
    for item in paginas:
        if consultas >= limite_consultas:
            break
        numero = str(item.get("numero") or "")
        relator = str(item.get("relator") or "").strip()
        if not relator or len(re.sub(r"\D", "", numero)) != 20:
            continue
        try:
            info = lookup_process(numero, session=session)
        except Exception as exc:  # noqa: BLE001 - rede
            log.debug("gabinete: lookup falhou para %s: %s", numero, exc)
            continue
        consultas += 1
        gabinete = _norm_gabinete(getattr(info, "orgao_julgador", "") if info else "")
        if not gabinete:
            continue
        votos[gabinete][relator] += 1

    mapa: dict[str, str] = {}
    amostra: dict[str, Any] = {}
    for gabinete, contagem in votos.items():
        total = sum(contagem.values())
        nome, n = contagem.most_common(1)[0]
        amostra[gabinete] = {"total": total, "distribuicao": dict(contagem)}
        # Gabinete entra no mapa so com amostra suficiente E dominancia clara. Gabinete em
        # transicao de ocupante fica dividido, nao alcanca o limiar e nao corrige nada --
        # quando a transicao se completa, a proxima apuracao o traz de volta sozinha.
        amostra[gabinete]["dominancia"] = round(n / total, 3) if total else 0.0
        if total >= MIN_PROCESSOS and n / total >= MIN_DOMINANCIA:
            mapa[gabinete] = nome
    log.info("gabinete->relator: %s gabinete(s) conclusivos em %s consultas ao DataJud (%s vistos).",
             len(mapa), consultas, len(votos))
    return mapa, amostra


def carregar_mapa(
    apurador=None,
    *,
    logger: Optional[logging.Logger] = None,
    forcar: bool = False,
) -> dict[str, str]:
    """Mapa do cache; reapura via `apurador()` quando vencido/ausente.

    `apurador` deve devolver (mapa, amostra). Sem apurador e sem cache valido, devolve {}
    -- e a resposta segura: sem mapa, nada e corrigido.
    """
    log = logger or logging.getLogger(__name__)
    if not forcar:
        cache = _carregar_cache()
        if cache:
            return dict(cache["mapa"])
    if apurador is None:
        return {}
    try:
        mapa, amostra = apurador()
    except Exception as exc:  # noqa: BLE001
        log.warning("gabinete->relator: apuracao falhou (%s); seguindo sem o mapa.", exc)
        return {}
    if mapa:
        _gravar_cache(mapa, amostra)
    return mapa


def relator_do_gabinete(mapa: dict[str, str], orgao_julgador: str) -> str:
    return mapa.get(_norm_gabinete(orgao_julgador), "")


# ---------------------------------------------------------------------------
# Quem integra a Corte: apurado por RELATORIA, nao pela coluna `composicao`.
#
# A coluna `composicao` e justamente o campo contaminado (o relator e a composicao
# saem do mesmo ruido do scan), entao usa-la para validar a si mesma nao funciona.
# Relatoria e prova: ministro que relata um processo naquela sessao estava la.
# Foi assim que se datou a troca do STJ1 -- Isabel Gallotti relatou ate nov/2025,
# Cueva assumiu e relata desde fev/2026 -- e assim que se identificou
# "Min. Alexandre de Moraes": ZERO relatorias em 13 meses e ~290 julgamentos,
# contra 10 aparicoes na coluna composicao. Nome que nunca relata nao e membro.
# ---------------------------------------------------------------------------
MIN_MEMBROS_PLAUSIVEL = 7
# O plenario do TSE tem 7 cadeiras (3 do STF, 2 do STJ, 2 juristas).
CADEIRAS_TSE = 7
# Janela para apurar o ELENCO vigente. Curta: 3 meses bastam para os 7 titulares
# aparecerem relatando e deixam a cauda de quem saiu para tras (em 26/08/2026, com 3
# meses, os 7 mais frequentes eram exatamente o colegiado, e Isabel Gallotti -- que
# deixou a Corte -- ficava em 8o com uma unica relatoria).
JANELA_ELENCO_MESES = 3
# O 7o colocado precisa de pelo menos isto para o elenco ser considerado apurado.
MIN_RELATORIAS_TITULAR = 3


def apurar_membros(paginas: list[dict[str, Any]]) -> set[str]:
    """Ministros com ao menos uma RELATORIA na janela -- prova de pertencimento.

    Deliberadamente generoso: inclui quem ja saiu da Corte no meio da janela. Serve
    para reprovar o nome que NUNCA relatou (ruido do scan), nao para datar mandato.
    """
    return {
        str(item.get("relator") or "").strip()
        for item in paginas
        if str(item.get("relator") or "").strip()
    }


def nao_e_membro(membros: set[str], nome: str) -> bool:
    """True so quando ha base para afirmar que o nome nao integra a Corte.

    Exige um conjunto de membros de tamanho plausivel: com poucos relatores apurados
    (base recem-criada, janela vazia, falha de rede) nada e reprovado -- e a resposta
    segura, porque um falso positivo aqui APAGA ministro legitimo da composicao.
    """
    if len(membros) < MIN_MEMBROS_PLAUSIVEL:
        return False
    return bool(nome) and nome not in membros


def apurar_elenco(paginas: list[dict[str, Any]]) -> list[str]:
    """Os titulares vigentes: os CADEIRAS_TSE ministros que mais relataram na janela.

    A composicao de uma sessao e, por padrao, o colegiado COMPLETO -- ausencia e a
    excecao, nao a regra. Em 25/08/2026 os sete estavam presentes e a extracao registrou
    cinco; completar pelo elenco acerta o caso comum e deixa a ausencia para a vistoria.

    Devolve [] quando a apuracao nao sustenta um elenco (poucos relatores, ou o 7o
    colocado com relatorias de menos para ser titular) -- sem elenco, nada e completado.
    """
    contagem = collections.Counter(
        str(item.get("relator") or "").strip()
        for item in paginas
        if str(item.get("relator") or "").strip()
    )
    if len(contagem) < CADEIRAS_TSE:
        return []
    top = contagem.most_common(CADEIRAS_TSE)
    if top[-1][1] < MIN_RELATORIAS_TITULAR:
        return []
    return [nome for nome, _ in top]
