"""Cliente da API pública do CNJ DataJud (https://datajud-wiki.cnj.jus.br/api-publica)
para a Justiça Eleitoral (TSE e TREs).

A API pública entrega: numeroProcesso (CNJ completo), classe (codigo+nome), assuntos,
movimentos, orgaoJulgador, grau, tribunal, dataAjuizamento. NÃO entrega partes nem
advogados (omitidos por LGPD).

Uso típico:
    info = lookup_process("0601309-60", tribunal="TRE-PA", year="2024")
    # info.numero_completo, info.classe_sigla, info.has_vista, info.has_julgamento
"""

from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

API_BASE = "https://api-publica.datajud.cnj.jus.br"
API_KEY = "APIKey cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw=="
REQUEST_TIMEOUT = 40
THROTTLE_SECONDS = 0.12

# Nome de classe do CNJ (normalizado) -> sigla do projeto. Só mapeia para SIGLAS que
# já existem na base; classes administrativas/sem sigla correspondente retornam "".
CNJ_CLASSE_TO_SIGLA = {
    "agravo em recurso especial eleitoral": "AREspe",
    "recurso especial eleitoral": "REspe",
    "recurso ordinario eleitoral": "RO",
    "recurso ordinario": "RO",
    "recurso contra expedicao de diploma": "RCED",
    "prestacao de contas": "PC",
    "prestacao de contas anual": "PC",
    "prestacao de contas eleitorais": "PC",
    "tutela cautelar antecedente": "TutCautAnt",
    "tutela antecipada antecedente": "TutCautAnt",
    "acao cautelar": "AC",
    "mandado de seguranca civel": "MS",
    "mandado de seguranca criminal": "MS",
    "representacao": "Rp",
    "representacao especial": "Rp",
    "processo administrativo": "PA",
    "lista triplice": "Lista Tríplice",
    "recurso em mandado de seguranca": "RMS",
    "consulta": "CTA",
    "recurso em habeas corpus": "RHC",
    "habeas corpus criminal": "HC",
    "habeas corpus": "HC",
    "registro de partido politico": "RPP",
    "cancelamento de registro de partido politico": "RPP",
    "registro de federacao partidaria": "RPP",
    "acao rescisoria eleitoral": "AR",
    "registro de candidatura": "RCand",
    "criacao de zona eleitoral ou remanejamento": "Czer",
    "revisao de eleitorado": "RvE",
    "acao de investigacao judicial eleitoral": "AIJE",
    "peticao civel": "PetCiv",
    "reclamacao": "Rcl",
    "agravo de instrumento": "AI",
    "acao de impugnacao de mandato eletivo": "AIME",
    "direito de resposta": "DRp",
    "mandado de injuncao": "MI",
}

# Só "pedido de vista" (movimento que de fato suspende o julgamento) — NÃO vista trivial
# ("vista dos autos", "vista ao MP", "carga/vista"), que aparece em quase todo processo.
_VISTA_RE = re.compile(r"pedido de vista|vista regimental", re.IGNORECASE)
# códigos/termos de movimentos que indicam julgamento/decisão de mérito
_JULGAMENTO_TERMS = (
    "julgamento",
    "negacao de seguimento",
    "negativa de seguimento",
    "provimento",
    "nao provimento",
    "homologacao",
    "decisao",
    "acordao",
    "improcedencia",
    "procedencia",
)


def format_cnj_number(digits: str) -> str:
    """Formata 20 dígitos no padrão CNJ NNNNNNN-DD.AAAA.J.TR.OOOO (ou '' se inválido)."""
    digits = re.sub(r"\D", "", str(digits or ""))
    if len(digits) != 20:
        return ""
    return f"{digits[0:7]}-{digits[7:9]}.{digits[9:13]}.{digits[13:14]}.{digits[14:16]}.{digits[16:20]}"


def _fold(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return re.sub(r"\s+", " ", text).strip()


def cnj_classe_sigla(nome: str) -> str:
    """Mapeia o nome de classe do CNJ para a sigla do projeto (ou '' se não houver
    sigla correspondente)."""
    return CNJ_CLASSE_TO_SIGLA.get(_fold(nome), "")


@dataclass
class CnjProcess:
    numero_completo: str = ""
    classe_nome: str = ""
    classe_sigla: str = ""
    grau: str = ""
    tribunal: str = ""
    orgao_julgador: str = ""
    has_vista: bool = False
    has_julgamento: bool = False
    movimentos_count: int = 0
    assuntos: list[str] = field(default_factory=list)
    source_index: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "numero_completo": self.numero_completo,
            "classe_nome": self.classe_nome,
            "classe_sigla": self.classe_sigla,
            "grau": self.grau,
            "tribunal": self.tribunal,
            "orgao_julgador": self.orgao_julgador,
            "has_vista": self.has_vista,
            "has_julgamento": self.has_julgamento,
            "movimentos_count": self.movimentos_count,
            "assuntos": self.assuntos,
            "source_index": self.source_index,
        }


def _tre_alias(tribunal: str) -> str:
    match = re.search(r"TRE[-/ ]?([A-Za-z]{2})", str(tribunal or ""), flags=re.IGNORECASE)
    return f"api_publica_tre-{match.group(1).lower()}" if match else ""


def _indices_for(tribunal: str) -> list[str]:
    indices = ["api_publica_tse"]
    tre = _tre_alias(tribunal)
    if tre:
        indices.append(tre)
    return indices


def _search(alias: str, body: dict[str, Any], session: Optional[requests.Session] = None) -> list[dict[str, Any]]:
    getter = session or requests
    url = f"{API_BASE}/{alias}/_search"
    try:
        response = getter.post(url, headers={"Authorization": API_KEY, "Content-Type": "application/json"}, json=body, timeout=REQUEST_TIMEOUT)
    except Exception:
        return []
    if response.status_code >= 400:
        return []
    if THROTTLE_SECONDS:
        time.sleep(THROTTLE_SECONDS)
    return response.json().get("hits", {}).get("hits", []) or []


def _build_process(source: dict[str, Any], alias: str) -> CnjProcess:
    classe = source.get("classe", {}) or {}
    movimentos = source.get("movimentos", []) or []
    nomes = [_fold(m.get("nome", "")) for m in movimentos]
    has_vista = any(_VISTA_RE.search(n) for n in nomes)
    has_julg = any(any(term in n for term in _JULGAMENTO_TERMS) for n in nomes)
    orgao = (source.get("orgaoJulgador", {}) or {}).get("nome", "")
    return CnjProcess(
        numero_completo=str(source.get("numeroProcesso", "")),
        classe_nome=str(classe.get("nome", "")),
        classe_sigla=cnj_classe_sigla(classe.get("nome", "")),
        grau=str(source.get("grau", "")),
        tribunal=str(source.get("tribunal", "")),
        orgao_julgador=str(orgao or ""),
        has_vista=has_vista,
        has_julgamento=has_julg,
        movimentos_count=len(movimentos),
        assuntos=[str(a.get("nome", "")) for a in (source.get("assuntos", []) or []) if isinstance(a, dict) and a.get("nome")],
        source_index=alias,
    )


def lookup_process(
    numero_display: str,
    tribunal: str = "",
    year: str = "",
    session: Optional[requests.Session] = None,
) -> Optional[CnjProcess]:
    """Localiza o processo no DataJud a partir do número (completo ou curto), do
    tribunal e do ano. Para números curtos usa busca por prefixo (NNNNNNN+DD) e
    desambigua pelo ano. Retorna None se não houver correspondência inequívoca."""
    digits = re.sub(r"\D", "", str(numero_display or ""))
    if len(digits) < 7:
        return None
    year = re.sub(r"\D", "", str(year or ""))[:4]
    for alias in _indices_for(tribunal):
        if len(digits) >= 20:
            hits = _search(alias, {"query": {"match": {"numeroProcesso": digits[:20]}}, "size": 3}, session)
        else:
            prefix = digits[:9] if len(digits) >= 9 else digits
            hits = _search(alias, {"query": {"prefix": {"numeroProcesso": prefix}}, "size": 25}, session)
        sources = [h.get("_source", {}) for h in hits if h.get("_source")]
        if not sources:
            continue
        # desambigua por ano (posições 9..12 do numeroProcesso) quando houver
        if year and len(digits) < 20:
            same_year = [s for s in sources if str(s.get("numeroProcesso", ""))[9:13] == year]
            if same_year:
                sources = same_year
        if len(sources) == 1:
            return _build_process(sources[0], alias)
        # múltiplos: só aceita se todos convergirem para a mesma classe-sigla e número
        numeros = {str(s.get("numeroProcesso", "")) for s in sources}
        if len(numeros) == 1:
            return _build_process(sources[0], alias)
    return None
