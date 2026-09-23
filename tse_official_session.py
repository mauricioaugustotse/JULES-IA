"""Read-only reconciliation against the public TSE session inventory.

This module neither publishes nor changes extracted rows.  A successful HTTP
request is not a coverage guarantee: unavailable sources and unknown process
statuses remain explicit issues.  Published files are evidence snapshots, never
an implicit fallback for a different date or an outdated session.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timezone
import re
import unicodedata
from typing import Any

import requests


API_URL = "https://plenario-virtual-api.tse.jus.br/plenario-virtual/rest/v1/sessao/data"
INVENTORY_FILENAME = "00_official_session_inventory.json"
_NAME_CANONICAL = {
    "nunes marques": "Min. Nunes Marques",
    "kassio nunes marques": "Min. Nunes Marques",
    "andre mendonca": "Min. André Mendonça",
    "antonio carlos ferreira": "Min. Antônio Carlos Ferreira",
    "ricardo villas boas cueva": "Min. Ricardo Villas Bôas Cueva",
    "floriano de azevedo marques": "Min. Floriano de Azevedo Marques",
    "estela aranha": "Min. Estela Aranha",
    "dias toffoli": "Min. Dias Toffoli",
    "carmen lucia": "Min. Cármen Lúcia",
    "sebastiao reis junior": "Min. Sebastião Reis Júnior",
}
_CLASS_ALIASES = {
    "lt": "lt", "lista triplice": "lt",
    "rve": "rve", "revisao de eleitorado": "rve",
    "pa": "pa", "processo administrativo": "pa",
    "respel": "respe", "respe": "respe", "recurso especial eleitoral": "respe",
    "arespe": "arespe", "arespel": "arespe", "agravo em recurso especial eleitoral": "arespe",
    "tutantant": "tutantant", "tutela antecipada antecedente": "tutantant",
    "tutcautant": "tutcautant", "tutela cautelar antecedente": "tutcautant",
    "rot": "rot", "ro el": "rot", "recurso ordinario eleitoral": "rot", "recurso ordinario": "rot",
    "pc": "pc", "prestacao de contas": "pc",
    "pet": "pet", "peticao": "pet",
    "aije": "aije", "acao de investigacao judicial eleitoral": "aije",
    "ed": "ed", "embargos de declaracao": "ed",
    "cst": "cst", "consulta": "cst",
}


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def _name(value: Any) -> str:
    text = _norm(value)
    text = re.sub(r"^(?:(?:min|ministro|ministra|senhor|senhora)\s+)+", "", text)
    text = re.sub(r"\s+presidente$", "", text)
    return "nunes marques" if text == "kassio nunes marques" else text


def _class(value: Any) -> str | None:
    # Incidentes mantêm a classe de origem no cadastro oficial. Só removemos
    # prefixos processuais conhecidos; AREspE continua distinto de REspE.
    text = _norm(value)
    text = re.sub(r"^(?:(?:ref|agr|agrg|ed)\s+)+", "", text)
    return _CLASS_ALIASES.get(text)


def _cnj(value: Any) -> tuple[str, str]:
    """Return (short core, full digits); do not infer missing year/tribunal."""
    raw = str(value or "").strip()
    full = re.fullmatch(r"(\d{1,7})-(\d{2})\.(\d{4})\.(\d)\.(\d{2})\.(\d{4})", raw)
    if full:
        core = full[1].zfill(7) + full[2]
        return core, core + "".join(full.groups()[2:])
    short = re.fullmatch(r"(\d{1,7})-(\d{2})", raw)
    if short:
        return short[1].zfill(7) + short[2], ""
    if raw.isdigit() and len(raw) == 20:
        return raw[:9], raw
    if raw.isdigit() and len(raw) == 9:
        return raw, ""
    return "", ""


def _date(value: Any) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()


def _issue(code: str, message: str, severity: str = "error", **extra: Any) -> dict[str, Any]:
    return {"code": code, "kind": code, "source": "tse_official_session", "severity": severity, "message": message, **extra}


def _classification(process: dict[str, Any]) -> str:
    status = _norm(process.get("situacaoProcesso"))
    if status in {"retirado de julgamento", "retirado de pauta"}:
        return "withdrawn"
    # The class Lista Tríplice is an individual case, not a voting list.
    if status == "julgado" and (process.get("blocoJulgamento") or process.get("agrupadorOrgaoJulgador") is True):
        return "list"
    if status == "julgado":
        return "judged"
    return "unknown"


def normalize_official_session(raw: Any, session_date: Any) -> dict[str, Any]:
    """Validate/filter an API response, also usable with an explicit offline snapshot.

    Filtering is exact: TSE, requested calendar date and explicitly nonvirtual.
    An absent matching session is unavailable, never a confirmed empty inventory.
    """
    requested = _date(session_date)
    out: dict[str, Any] = {
        "schema_version": 1, "status": "unavailable", "session_date": requested,
        "source_url": f"{API_URL}?sgTribunal=TSE&dataSessao={requested}",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "sessions": [], "processes": [], "excluded": [], "expected_count": None,
        "counts": {"judged": 0, "withdrawn": 0, "list": 0, "unknown": 0},
    }
    if not isinstance(raw, list):
        out["error"] = "Resposta oficial inválida: esperada lista de sessões."
        return out
    matches = []
    for session in raw:
        if not isinstance(session, dict):
            continue
        if str(session.get("tribunal", "")).upper() != "TSE":
            continue
        if str(session.get("dataSessao", ""))[:10] != requested:
            continue
        virtual = session.get("virtual")
        if virtual is not False and _norm(virtual) not in {"nao", "n", "false"}:
            continue
        matches.append(session)
    if not matches:
        out["error"] = "Nenhuma sessão presencial do TSE confirmada para a data solicitada."
        return out
    out["sessions"] = matches
    if any(not isinstance(s.get("processos"), list) or any(not isinstance(p, dict) for p in s["processos"]) for s in matches):
        out["error"] = "Sessão oficial sem inventário processual válido."
        return out
    for session in matches:
        for p in session["processos"]:
            item = dict(p)
            item["session_id"] = session.get("id")
            item["classification"] = _classification(p)
            out["processes"].append(item)
            out["counts"][item["classification"]] += 1
            if item["classification"] in {"withdrawn", "list"}:
                out["excluded"].append({
                    "numero_processo": p.get("numeroProcesso", ""),
                    "classification": item["classification"],
                    "official_status": p.get("situacaoProcesso"),
                    "block": p.get("blocoJulgamento"),
                    "session_id": session.get("id"),
                })
    out["status"] = "available"
    # Unique expected identities; duplicate official records stay in processes and
    # will trigger an ambiguous-inventory issue in comparison, not silent selection.
    out["expected_count"] = len({p.get("numeroProcesso") for p in out["processes"] if p["classification"] == "judged"})
    return out


def fetch_official_session(session_date: Any, artifact_store: Any, http: Any = None, *, timeout: float = 25) -> dict[str, Any]:
    """Fetch fresh public data and persist one audit snapshot via write_json().

    ``http`` may be requests, a Session or a test double providing get().  Failed
    refreshes are unavailable; an older on-disk snapshot is never silently reused.
    Transport error messages are not persisted because clients can include secrets.
    """
    requested = _date(session_date)
    client = http if http is not None else requests
    try:
        response = client.get(API_URL, params={"sgTribunal": "TSE", "dataSessao": requested}, timeout=timeout)
        response.raise_for_status()
        payload = normalize_official_session(response.json(), requested)
    except Exception as exc:
        payload = normalize_official_session(None, requested)
        payload["error"] = f"Falha ao consultar inventário oficial ({type(exc).__name__})."
    artifact_store.write_json(INVENTORY_FILENAME, payload)
    return payload


def _result(value: Any) -> str:
    text = _norm(value)
    if not text:
        return ""
    if text in {"indeferido", "indeferida", "indefiro"}:
        return "indeferido"
    if text in {"deferido", "deferida", "defiro"}:
        return "deferido"
    if text in {"aprovada", "aprovado", "aprovo"}:
        return "aprovado"
    if text in {"desprovido", "nao provido", "negado provimento", "nego provimento"}:
        return "desprovido"
    if text in {"provido", "dou provimento"}:
        return "provido"
    if text in {"nao conhecido", "nao conhecida", "nao conheco"}:
        return "nao conhecido"
    if text in {"referendada", "referendado", "referendo"}:
        return "referendada"
    if text in {"improcedente", "julgo improcedente"}:
        return "improcedente"
    if text in {"procedente", "julgo procedente"}:
        return "procedente"
    if text in {"devolvida", "devolvido", "devolucao", "retorno a origem", "devolucao para recomposicao", "determino o retorno do processo"}:
        return "devolvido"
    return text


def _official_decision(process: dict[str, Any]) -> tuple[str, str]:
    """Only unambiguous collective dispositions; never substitute a dissent vote."""
    raw = str(process.get("proclamacaoDecisao") or "")
    text = _norm(raw.split("\n\n", 1)[0])
    voting = ""
    if "por unanimidade" in text or "a unanimidade" in text:
        voting = "unanime"
    elif "por maioria" in text:
        voting = "por maioria"
    # When the court grants the agravo precisely to reject the special appeal,
    # the outcome label follows the appeal. Keep other compound dispositions
    # unresolved: partial knowledge or several merits outcomes need review.
    if (
        "agravo" in text and "recurso especial" in text
        and re.search(r"\b(?:deu provimento ao agravo|proveu o agravo)\b", text)
        and re.search(r"\b(?:para|e)\s+(?:negar|negou)\s+provimento\s+ao\s+recurso especial\b", text)
        and not re.search(r"\b(?:parcialmente|em parte|parte conhecida)\b", text)
    ):
        return "desprovido", voting
    # A process may have distinct interlocutory and merits outcomes.
    if ("agravo" in text and "recurso especial" in text) or "no merito" in text or "parcialmente" in text or "em parte" in text:
        return "", voting
    if re.search(r"\b(?:acolheu|rejeitou).+\b(?:indeferiu|deferiu|negou|deu provimento)\b", text):
        return "", voting
    choices = [
        (r"\bindeferiu\b", "indeferido"), (r"\bdeferiu\b", "deferido"),
        (r"\baprovou\b", "aprovado"), (r"\bnegou provimento\b", "desprovido"),
        (r"\bdeu provimento\b", "provido"),
        (r"\bnao conheceu\b", "nao conhecido"),
        (r"\breferendou\b", "referendada"),
        (r"\bjulgou improcedente\b", "improcedente"),
        (r"\bjulgou procedente\b", "procedente"),
        (r"\bdeterminou (?:a devolucao|o retorno)\b", "devolvido"),
    ]
    # Main finite verb in the court's own dispositive; subordinate reports such
    # as 'acórdão por meio do qual foi deferido' do not override 'aprovou'.
    hits = [(m.start(), outcome) for pattern, outcome in choices if (m := re.search(pattern, text))]
    return (min(hits)[1] if hits else ""), voting


def _official_composition(process: dict[str, Any]) -> list[str] | None:
    if process.get("segredoJustica"):
        return None
    text = str(process.get("proclamacaoDecisao") or "")
    match = re.search(r"composi[çc][ãa]o(?: do julgamento)?\s*:\s*(.+)", text, flags=re.I | re.S)
    if match:
        # Explicit composition includes preserved historical votes that can be
        # absent from votantes (e.g. Cármen Lúcia on 17/09/2026).
        tail = re.sub(r"^ministros?\s*(?:\(as\))?\s*", "", match[1].strip(), flags=re.I)
        tail = re.split(r"[.\n]", tail, maxsplit=1)[0]
        parts = re.split(r",|\s+e\s+", tail)
        names = [_name(p) for p in parts if p.strip()]
        if len(names) >= 2 and all(n in _NAME_CANONICAL for n in names):
            return sorted(set(names))
        return None
    voters = process.get("votantes")
    if not isinstance(voters, list) or len(voters) not in {6, 7}:
        return None
    if any(not isinstance(v, dict) or v.get("impedido") or v.get("omisso") for v in voters):
        return None
    names = [_name(v.get("nome")) for v in voters]
    if len(set(names)) != len(names) or not all(n in _NAME_CANONICAL for n in names):
        return None
    return sorted(names)


def compare_official_rows(inventory: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report coverage/identity/semantic conflicts without modifying either input.

    Errors identify publish blockers; warnings require review; informational
    ``official_exclusion`` entries account for normal nonindividual/withdrawn
    records even when no extracted row exists. Unknown official statuses remain
    warnings and are never treated as an exclusion or proof of no missing case.
    """
    if inventory.get("status") != "available":
        return [_issue("official_inventory_unavailable", inventory.get("error") or "Inventário oficial indisponível; cobertura não confirmada.", "warning", coverage_status="unknown")]
    requested = str(inventory.get("session_date") or "")
    if not isinstance(inventory.get("processes"), list) or not requested:
        return [_issue("official_inventory_unavailable", "Snapshot oficial incompleto; cobertura não confirmada.", "warning", coverage_status="unknown")]
    issues: list[dict[str, Any]] = []
    by_core: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in inventory["processes"]:
        core, full = _cnj(p.get("numeroProcesso"))
        if not core or not full:
            issues.append(_issue("official_invalid_identity", "Inventário oficial contém processo sem número CNJ completo utilizável.", "warning", numero_processo=p.get("numeroProcesso"), coverage_status="unknown"))
            continue
        by_core[core].append(p)
        kind = p.get("classification", _classification(p))
        if kind in {"withdrawn", "list"}:
            issues.append(_issue("official_exclusion", "Processo oficialmente retirado de julgamento." if kind == "withdrawn" else "Processo julgado em lista, fora dos julgamentos individuais.", "info", numero_processo=p["numeroProcesso"], classification=kind))
        elif kind == "unknown":
            issues.append(_issue("official_unknown_status", "Situação oficial ainda não permite confirmar julgamento nem exclusão.", "warning", numero_processo=p["numeroProcesso"], actual=p.get("situacaoProcesso"), coverage_status="unknown"))
    seen: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        numero = str(row.get("numero_processo") or "")
        core, full = _cnj(numero)
        row_date = row.get("data_sessao")
        if isinstance(row_date, dict):
            row_date = row_date.get("start")
        if row_date and str(row_date)[:10] != requested:
            issues.append(_issue("official_session_date_mismatch", "Data da linha diverge da sessão oficial consultada.", row_index=index, numero_processo=numero, field="data_sessao", expected=requested, actual=row_date))
            # A row from another session never satisfies coverage of this one.
            continue
        candidates = by_core.get(core, []) if core else []
        if not candidates:
            issues.append(_issue("official_unexpected_row", "Linha não corresponde a processo do inventário oficial desta sessão.", row_index=index, numero_processo=numero, field="numero_processo", expected=None, actual=numero))
            continue
        exact = [p for p in candidates if full and _cnj(p["numeroProcesso"])[1] == full]
        matches = exact or candidates
        # The same short prefix can belong to different years/courts. Without a
        # unique full identity this is unresolved, not an arbitrary first match.
        if len(matches) != 1:
            issues.append(_issue("official_ambiguous_identity", "Mais de um registro oficial corresponde ao identificador da linha.", row_index=index, numero_processo=numero, field="numero_processo", expected=[p["numeroProcesso"] for p in matches], actual=numero))
            continue
        p = matches[0]
        canonical = p["numeroProcesso"]
        seen[canonical].append(index)
        ctx = {"row_index": index, "numero_processo": canonical}
        kind = p.get("classification", _classification(p))
        if kind in {"withdrawn", "list"}:
            issues.append(_issue("official_excluded_row", "Linha trata como julgamento individual um processo oficialmente retirado ou julgado em lista.", **ctx, classification=kind, field="disposition", expected=kind, actual="individual_row"))
            continue
        if full and full != _cnj(canonical)[1]:
            issues.append(_issue("official_process_number_mismatch", "Ano, tribunal ou origem numérica divergem do CNJ oficial.", **ctx, field="numero_processo", expected=canonical, actual=numero))
        elif not full:
            issues.append(_issue("official_incomplete_process_number", "Número curto identificado; o inventário oficial fornece o CNJ completo.", "warning", **ctx, field="numero_processo", expected=canonical, actual=numero))
        if kind != "judged":
            continue
        def mismatch(field: str, expected: Any, actual: Any) -> None:
            issues.append(_issue("official_field_mismatch", f"Campo {field} diverge do registro oficial do julgamento.", **ctx, field=field, expected=expected, actual=actual))
        official_class = _class(p.get("siglaClasseJudicial")) or _class(p.get("classeJudicial"))
        if official_class and _class(row.get("classe_processo")) != official_class:
            mismatch("classe_processo", p.get("siglaClasseJudicial") or p.get("classeJudicial"), row.get("classe_processo"))
        if p.get("origem") and _norm(row.get("origem")) != _norm(p["origem"]):
            mismatch("origem", p["origem"], row.get("origem"))
        if p.get("relator") and _name(row.get("relator")) != _name(p["relator"]):
            mismatch("relator", p["relator"], row.get("relator"))
        expected_result, expected_vote = _official_decision(p)
        if expected_result and _result(row.get("resultado")) != expected_result:
            mismatch("resultado", expected_result, row.get("resultado"))
        actual_vote = _norm(row.get("votacao"))
        if actual_vote in {"unanimidade", "por unanimidade"}:
            actual_vote = "unanime"
        if actual_vote == "maioria":
            actual_vote = "por maioria"
        if expected_vote and actual_vote != expected_vote:
            mismatch("votacao", expected_vote, row.get("votacao"))
        composition = _official_composition(p)
        actual = row.get("composicao")
        if composition is not None:
            parsed = sorted(set(_name(n) for n in actual)) if isinstance(actual, list) else []
            if parsed != composition:
                mismatch("composicao", [_NAME_CANONICAL[n] for n in composition], actual)
    for canonical, indexes in seen.items():
        if len(indexes) > 1:
            issues.append(_issue("official_duplicate_row", "Mais de uma linha representa o mesmo processo da sessão.", numero_processo=canonical, row_indices=indexes, expected=1, actual=len(indexes)))
    for candidates in by_core.values():
        for p in candidates:
            if p.get("classification", _classification(p)) == "judged" and p["numeroProcesso"] not in seen:
                issues.append(_issue("official_missing_judgment", "Processo julgado individualmente não aparece nas linhas extraídas.", numero_processo=p["numeroProcesso"], field="numero_processo", expected=p["numeroProcesso"], actual=None))
    return issues
