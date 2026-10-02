"""Deterministic repairs backed by the inventory of this exact TSE session.

Inputs are copied. Every changed field and every excluded citation is returned
in an audit suitable for a JSON artifact. Ambiguous identities stay untouched.
The optional evidence is the original ``SessionAnalysis.model_dump()`` payload.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from tse_official_session import (
    _NAME_CANONICAL, _class, _classification, _cnj, _name, _norm,
    _official_composition, _official_decision, compare_official_rows,
)


_RESULTS = {
    "indeferido": "Indeferido", "deferido": "Deferido", "aprovado": "Aprovada",
    "desprovido": "Desprovido", "provido": "Provido", "nao conhecido": "Não conhecido",
    "referendada": "Referendada", "improcedente": "Improcedente", "procedente": "Procedente",
    "devolvido": "Devolvida",
}
_CLASSES = {
    "lt": "Lista Tríplice", "pa": "PA", "rot": "RO", "rve": "RvE",
    "respe": "REspEl", "arespe": "AREspE", "pc": "PC", "pet": "Pet",
    "aije": "AIJE", "cst": "CTA", "ed": "ED",
}
_SUSPENSION_NOTE = "O julgamento não foi concluído nesta sessão; o voto do relator não constitui decisão final do colegiado."


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return deepcopy(value)
    return value.model_dump(mode="json")


def _valid_full_cnj(full: str) -> bool:
    return len(full) == 20 and (int(full[:7] + full[9:] + full[7:9]) % 97 == 1)


def _row_date(row: dict[str, Any]) -> str:
    value = row.get("data_sessao")
    return str(value.get("start") if isinstance(value, dict) else value or "")[:10]


def _bundle_evidence(row: dict[str, Any], evidence: dict[str, Any]) -> tuple[dict, dict]:
    index = row.get("source_bundle_index")
    bundles = evidence.get("bundles") or []
    if not isinstance(index, int) or not 1 <= index <= len(bundles):
        return {}, {}
    bundle = bundles[index - 1]
    if not isinstance(bundle, dict) or bundle.get("start_seconds") != row.get("source_start_seconds"):
        return {}, {}
    scans = [s for s in (evidence.get("session") or {}).get("judgments", [])
             if isinstance(s, dict) and s.get("start_seconds") == row.get("source_start_seconds")]
    return bundle, scans[0] if len(scans) == 1 else {}


def _identity_support(row: dict[str, Any], process: dict[str, Any]) -> list[str] | None:
    """OCR repair needs an explicit matching relator; other known fields must agree."""
    if not _name(row.get("relator")) or _name(row.get("relator")) != _name(process.get("relator")):
        return None
    support = ["mesma sessão", "relator coincidente"]
    official_class = _class(process.get("siglaClasseJudicial")) or _class(process.get("classeJudicial"))
    if official_class:
        if _class(row.get("classe_processo")) != official_class:
            return None
        support.append("classe coincidente")
    if process.get("origem"):
        if _norm(row.get("origem")) != _norm(process["origem"]):
            return None
        support.append("origem coincidente")
    return support


def _scan_identity_support(row: dict, process: dict, evidence: dict) -> list[str] | None:
    """An independently scanned short number can repair an invalid suffix/relator."""
    bundle, scan = _bundle_evidence(row, evidence)
    core = _cnj(process.get("numeroProcesso"))[0]
    short_numbers = [_cnj(n) for n in scan.get("mentioned_process_numbers", [])]
    official_class = _class(process.get("siglaClasseJudicial")) or _class(process.get("classeJudicial"))
    if (not bundle or not scan or (core, "") not in short_numbers
            or not official_class or _class(row.get("classe_processo")) != official_class
            or not process.get("origem") or _norm(row.get("origem")) != _norm(process["origem"])):
        return None
    return ["mesma sessão", "núcleo CNJ confirmado independentemente na varredura",
            "classe coincidente", "origem coincidente"]


def _match_identity(row: dict[str, Any], processes: list[dict], evidence: dict) -> tuple[dict | None, str, list[str]]:
    core, full = _cnj(row.get("numero_processo"))
    exact = [p for p in processes if full and _cnj(p.get("numeroProcesso"))[1] == full]
    if exact:
        return (exact[0], "exact_cnj", ["CNJ completo idêntico"]) if len(exact) == 1 else (None, "ambiguous", [])
    # A valid complete number can identify a different year/court. Never repair it
    # merely because a similar number happens to occur in this inventory.
    if full and _valid_full_cnj(full):
        return None, "valid_cnj_not_in_session", []
    same_core = [p for p in processes if core and _cnj(p.get("numeroProcesso"))[0] == core]
    if same_core:
        if len(same_core) != 1:
            return None, "ambiguous", []
        if not full:
            return same_core[0], "unique_short_cnj", ["núcleo CNJ único na sessão"]
        support = _identity_support(row, same_core[0])
        if not support:
            support = _scan_identity_support(row, same_core[0], evidence)
        if support:
            return same_core[0], "invalid_cnj_suffix", ["CNJ extraído reprova no dígito verificador", *support]
        return None, "insufficient_evidence", []
    bundle, scan = _bundle_evidence(row, evidence)
    values = [row.get("numero_origem_video"), row.get("numero_processo"), bundle.get("title_hint"), scan.get("title_hint")]
    repaired_cores = set()
    for value in values:
        raw = str(value or "").strip()
        if not re.fullmatch(r"\d{10}", raw):
            continue
        # Only one extra zero, with the two check digits preserved. This is not
        # edit-distance matching against arbitrary numbers.
        repaired_cores.update(raw[:i] + raw[i + 1:] for i, digit in enumerate(raw[:-2]) if digit == "0")
    candidates = []
    for p in processes:
        if _cnj(p.get("numeroProcesso"))[0] not in repaired_cores:
            continue
        support = _identity_support(row, p)
        if support:
            candidates.append((p, support))
    if len(candidates) == 1:
        p, support = candidates[0]
        return p, "single_extra_zero", ["um zero excedente no número original do vídeo", *support]
    return None, "ambiguous" if len(candidates) > 1 else "unmatched", []


def _extracted_composition(row: dict, evidence: dict, processes: list[dict]) -> list[str] | None:
    """Restore the recorded session panel only with independent official agreement."""
    bundle, _ = _bundle_evidence(row, evidence)
    item_index = row.get("source_item_index")
    items = bundle.get("items") or []
    item = items[item_index - 1] if isinstance(item_index, int) and 1 <= item_index <= len(items) else {}
    extracted = item.get("composicao") or (evidence.get("session") or {}).get("composicao") or []
    names = sorted(set(_name(n) for n in extracted))
    if len(names) not in {6, 7} or any(n not in _NAME_CANONICAL for n in names):
        return None
    panels = [_official_composition(p) for p in processes]
    panels = [p for p in panels if p is not None]
    # A change in the actual panel anywhere in the session makes session-wide
    # completion unsafe; retain the original row for review in that situation.
    if not panels or any(p != names for p in panels):
        return None
    return [_NAME_CANONICAL[n] for n in names]


def _vista_requester(process: dict) -> str:
    direct = _name(process.get("orgaoJulgadorVista"))
    if direct in _NAME_CANONICAL:
        return _NAME_CANONICAL[direct]
    text = _norm(process.get("proclamacaoDecisao"))
    candidates = [canonical for name, canonical in _NAME_CANONICAL.items()
                  if re.search(r"(?:pediu vista|pedido de vista)(?:\s+(?:o|a|ministro|ministra))*\s+" + re.escape(name) + r"\b", text)]
    return candidates[0] if len(set(candidates)) == 1 else ""


def _suspension_text(text: str, official_vote: str) -> str:
    # A generated assertion that the TSE decided the merits must not survive as
    # legal analysis under a suspended label. Preserve the exact prior text in
    # the audit; the official account supplies the replacement in that case.
    norm = _norm(text)
    definitive = re.search(r"\b(?:o tse|o tribunal superior eleitoral|o colegiado|o tribunal)\s+(?:deu|negou|manteve|indeferiu|deferiu|decidiu|concluiu|reconheceu|julgou)\b", norm)
    if definitive:
        text = official_vote
    text = str(text or "").strip()
    if _SUSPENSION_NOTE not in text:
        text = (text + "\n\n" + _SUSPENSION_NOTE).strip()
    return text


def _relator_reference_text(text: str, old: str, new: str) -> str:
    """Correct an explicit main-relator attribution, preserving other mentions."""
    old_canonical = _NAME_CANONICAL.get(_name(old))
    if not old_canonical or old_canonical == new:
        return text
    old_name = re.sub(r"^Min\.\s+", "", old_canonical)
    pattern = r"(\b(?:o|a)\s+relator(?:a)?[\s,]+)(?:Min(?:istro|istra)?\.?\s+)?" + re.escape(old_name) + r"\b"
    return re.sub(pattern, lambda m: m[1] + new, text, flags=re.IGNORECASE)


def _institutional_evidence(index: int, row: dict, evidence: dict) -> dict | None:
    """Exclude only a positively identified internal election without a case."""
    if any(row.get(k) for k in ("numero_processo", "numero_origem_video", "classe_processo", "partes", "relator")):
        return None
    bundle, scan = _bundle_evidence(row, evidence)
    item_index = row.get("source_item_index")
    items = bundle.get("items") or []
    if not isinstance(item_index, int) or not 1 <= item_index <= len(items):
        return None
    item = items[item_index - 1]
    if any(item.get(k) for k in ("numero_processo", "classe_processo", "partes", "relator")):
        return None
    titles = [_norm(b.get("title_hint")) for b in (bundle, scan)]
    original = _norm(item.get("analise_do_conteudo_juridico"))
    if (not scan or scan.get("mentioned_process_numbers")
            or not all("eleicao" in title and "corregedor geral eleitoral" in title for title in titles)
            or "eleicao interna" not in original or "cargo de corregedor geral eleitoral" not in original):
        return None
    return {"row_index": index, "original_row_index": index, "numero_processo": "",
            "code": "institutional_act", "status": "automatic_exclusion", "classification": "institutional",
            "reason": "Eleição interna de Corregedor-Geral Eleitoral, sem processo ou julgamento individual.",
            "evidence": ["mesma sessão e trecho do vídeo", "varredura e detalhe identificam eleição interna",
                         "texto original descreve eleição para cargo do Tribunal", "nenhum número, classe, parte ou relator processual"],
            "row": deepcopy(row)}


def _citation_evidence(index: int, rows: list[dict], matches: dict[int, dict], evidence: dict, official_cores: set[str]) -> dict | None:
    row = rows[index]
    core = _cnj(row.get("numero_processo"))[0]
    if not core or core in official_cores or row.get("source_item_index", 0) <= 1:
        return None
    bundle, scan = _bundle_evidence(row, evidence)
    anchor = _cnj(bundle.get("title_hint"))[0]
    if not anchor or anchor != _cnj(scan.get("title_hint"))[0]:
        return None
    mentioned = {_cnj(n)[0] for n in scan.get("mentioned_process_numbers", [])}
    if core not in mentioned:
        return None
    parents = [i for i, p in matches.items() if _cnj(p.get("numeroProcesso"))[0] == anchor
               and rows[i].get("source_bundle_index") == row.get("source_bundle_index")
               and rows[i].get("source_start_seconds") == row.get("source_start_seconds")]
    if len(parents) != 1:
        return None
    parent_index = parents[0]
    parent = rows[parent_index]
    shared_parties = set(_norm(p) for p in row.get("partes", [])) & set(_norm(p) for p in parent.get("partes", []))
    if not shared_parties:
        return None
    text = _norm(row.get("analise_do_conteudo_juridico"))
    # Positive evidence of an antecedent conviction forming the basis for this
    # same registration appeal, not just temporal proximity or official absence.
    antecedent = re.search(r"\b(?:fundamentad[oa]|basead[oa]|fundad[oa])\s+em\s+condenacao\b", text)
    if not antecedent or "registro de candidatura" not in text or "condenac" not in _norm(parent.get("analise_do_conteudo_juridico")):
        return None
    return {
        "row_index": index, "original_row_index": index,
        "numero_processo": row.get("numero_processo"), "code": "cited_process_number",
        "status": "automatic_exclusion", "classification": "citation",
        "parent_row_index": parent_index, "parent_numero_processo": parent.get("numero_processo"),
        "reason": "Número de condenação citado como fundamento de outro julgamento, sem julgamento individual nesta sessão.",
        "evidence": ["mesmo bloco e trecho do vídeo", "título do scan e do detalhe identifica o processo principal oficial",
                     "número secundário consta das menções do scan", "mesma parte", "texto descreve condenação antecedente como fundamento do registro"],
        "row": deepcopy(row),
    }


def reconcile_session_rows(rows: list[Any], inventory: dict[str, Any], *, evidence: Any = None) -> tuple[list[Any], dict[str, Any]]:
    """Return copied rows and a JSON audit; indexes refer to the original inputs.

    Both dictionaries and PublishPreviewRow objects are accepted. ``matches``
    identifies rows suitable for skipping weaker external identity enrichment.
    Excluded citations are retained verbatim under audit["exclusions"].
    """
    data = [_as_dict(r) for r in rows]
    evidence = _as_dict(evidence) if evidence is not None else {}
    audit = {"schema_version": 1, "session_date": inventory.get("session_date"),
             "source_url": inventory.get("source_url"), "status": "unavailable",
             "matches": [], "corrections": [], "exclusions": [], "unresolved": []}
    processes = inventory.get("processes")
    requested = str(inventory.get("session_date") or "")
    if inventory.get("status") != "available" or not isinstance(processes, list) or not requested:
        return [deepcopy(r) for r in rows], audit
    evidence_date = str((evidence.get("session") or {}).get("data_sessao") or "")[:10]
    if evidence and evidence_date != requested:
        evidence = {}  # Never borrow scan identities or a panel from another session.
    processes = [p for p in processes if isinstance(p, dict) and _cnj(p.get("numeroProcesso"))[1]]
    audit["status"] = "complete"
    matched = {}

    def change(index: int, field: str, value: Any, reason: str) -> None:
        before = data[index].get(field)
        if before == value:
            return
        data[index][field] = deepcopy(value)
        audit["corrections"].append({"row_index": index, "numero_processo": data[index].get("numero_processo"),
                                     "field": field, "before": before, "after": deepcopy(value), "reason": reason})

    for index, row in enumerate(data):
        if _row_date(row) != requested:
            audit["unresolved"].append({"row_index": index, "reason": "session_date_mismatch"})
            continue
        process, method, support = _match_identity(row, processes, evidence)
        if process is None:
            audit["unresolved"].append({"row_index": index, "reason": method})
            continue
        matched[index] = process
        bundle, scan = _bundle_evidence(row, evidence)
        aliases = list(dict.fromkeys(str(value) for value in
                       (row.get("numero_processo"), row.get("numero_origem_video"),
                        bundle.get("title_hint"), scan.get("title_hint"), process["numeroProcesso"])
                       if value))
        confirmed_fields = ["numero_processo", "data_sessao"]
        match_audit = {"row_index": index, "numero_processo": process["numeroProcesso"], "method": method,
                       "evidence": support, "aliases": aliases, "confirmed_fields": confirmed_fields,
                       "source_bundle_index": row.get("source_bundle_index"),
                       "source_start_seconds": row.get("source_start_seconds"),
                       "original_numero_processo": row.get("numero_processo"),
                       "numero_origem_video": row.get("numero_origem_video")}
        audit["matches"].append(match_audit)
        change(index, "numero_processo", process["numeroProcesso"], "Identidade única no inventário oficial desta sessão.")
        if _classification(process) in {"list", "withdrawn", "deferred"}:
            continue  # The existing official gate accounts for these exclusions.
        canonical_relator = _NAME_CANONICAL.get(_name(process.get("relator")))
        if canonical_relator:
            confirmed_fields.append("relator")
            old_relator = row.get("relator")
            for field in ("analise_do_conteudo_juridico", "raciocinio_juridico", "punchline"):
                original = str(row.get(field) or "")
                corrected = _relator_reference_text(original, old_relator, canonical_relator)
                if original != corrected:
                    change(index, field, corrected, "Referência explícita ao relator corrigida pelo registro oficial.")
            change(index, "relator", canonical_relator, "Relator do processo na sessão oficial.")
        official_class = _class(process.get("siglaClasseJudicial")) or _class(process.get("classeJudicial"))
        if official_class in _CLASSES and _class(row.get("classe_processo")) != official_class:
            change(index, "classe_processo", _CLASSES[official_class], "Classe expressa no inventário oficial.")
        if official_class in _CLASSES:
            confirmed_fields.append("classe_processo")
        if process.get("origem") and _norm(row.get("origem")) != _norm(process["origem"]):
            change(index, "origem", re.sub(r"\s+-\s+([A-Z]{2})$", r"/\1", process["origem"]), "Origem expressa no inventário oficial.")
        if process.get("origem"):
            confirmed_fields.append("origem")
        composition = _official_composition(process)
        canonical_panel = [_NAME_CANONICAL[n] for n in composition] if composition else _extracted_composition(row, evidence, processes)
        if canonical_panel:
            confirmed_fields.append("composicao")
            change(index, "composicao", canonical_panel, "Composição efetiva da sessão, incluindo ministros substitutos.")
        if _norm(process.get("situacaoProcesso")) == "nao julgado" and _norm(process.get("motivoRetiradaPauta")) == "pedido de vista":
            requester = _vista_requester(process)
            confirmed_fields.extend(["resultado", "votacao", "punchline", "analise_do_conteudo_juridico", "raciocinio_juridico"])
            change(index, "resultado", "Suspenso por vista", "Pedido de vista sem conclusão do julgamento nesta sessão.")
            change(index, "votacao", "Suspenso", "Não houve votação final do colegiado.")
            if requester:
                confirmed_fields.append("pedido_vista")
                change(index, "pedido_vista", requester, "Ministro que pediu vista no registro oficial.")
            summary = "O julgamento foi suspenso por pedido de vista" + (f" de {requester}" if requester else "") + ", após o voto do relator. " + _SUSPENSION_NOTE
            change(index, "punchline", summary, "A síntese distingue voto do relator e decisão do colegiado.")
            official_vote = str(process.get("proclamacaoDecisao") or "").split("\n\n", 1)[0].strip()
            for field in ("analise_do_conteudo_juridico", "raciocinio_juridico"):
                change(index, field, _suspension_text(str(row.get(field) or ""), official_vote), "Análise limitada ao voto proferido; julgamento permanece suspenso.")
        elif _classification(process) == "judged":
            result, voting = _official_decision(process)
            disposition = _norm(str(process.get("proclamacaoDecisao") or "").split("\n\n", 1)[0])
            if "por unanimidade" in disposition and "por maioria" in disposition:
                # More than one appellant/disposition: preserve the extracted
                # main appeal rather than protecting a result from another one.
                result, voting = "", ""
            elif re.search(r"\bparcial (?:provimento|procedencia)\b", disposition):
                result = ""
            if result in _RESULTS:
                confirmed_fields.append("resultado")
                change(index, "resultado", _RESULTS[result], "Dispositivo inequívoco da proclamação oficial.")
            if voting:
                confirmed_fields.append("votacao")
                change(index, "votacao", {"unanime": "Unânime", "por maioria": "Por maioria"}[voting], "Votação inequívoca da proclamação oficial.")

    official_cores = {_cnj(p.get("numeroProcesso"))[0] for p in processes}
    excluded = set()
    for index in range(len(data)):
        if index in matched or _row_date(data[index]) != requested:
            continue
        exclusion = (_institutional_evidence(index, data[index], evidence)
                     or _citation_evidence(index, data, matched, evidence, official_cores))
        if exclusion:
            excluded.add(index)
            audit["exclusions"].append(exclusion)
    audit["unresolved"] = [i for i in audit["unresolved"] if i["row_index"] not in excluded]

    # Only proven exclusions close gaps; blocked judgments retain their position.
    removed_numbers = {int(m[1]) for i in excluded
                       if (m := re.fullmatch(r"Julgamento\s+(\d+)", str(data[i].get("tipo_registro") or "")))}
    for index, row in enumerate(data):
        match = re.fullmatch(r"Julgamento\s+(\d+)", str(row.get("tipo_registro") or ""))
        if index not in excluded and _row_date(row) == requested and match:
            number = int(match[1])
            shift = sum(n < number for n in removed_numbers)
            if shift:
                change(index, "tipo_registro", f"Julgamento {number - shift}", "Sequência recomposta após exclusões comprovadas.")

    # Only discard official blockers that a fresh comparison proves resolved.
    issues = compare_official_rows(inventory, data)
    for index in matched:
        active = [i for i in issues if index == i.get("row_index") or index in i.get("row_indices", [])]
        old_errors = data[index].get("errors") or []
        kept_errors = []
        for message in old_errors:
            code = re.match(r"\[oficial:([^]]+)\]", message)
            field = re.search(r"Campo (\w+) diverge", message)
            if not code or any(i.get("code") == code[1] and (not field or i.get("field") == field[1]) for i in active):
                kept_errors.append(message)
        if kept_errors != old_errors:
            change(index, "errors", kept_errors, "Bloqueios anteriores reavaliados contra o inventário oficial.")
        changed = {c["field"] for c in audit["corrections"] if c["row_index"] == index}
        kept_warnings = []
        for warning in data[index].get("warnings") or []:
            norm = _norm(warning)
            stale_number = "numero_processo" in changed and (norm.startswith("numero do processo textual invalido") or norm.startswith("numero do processo nao identificado") or norm.startswith("digito verificador do cnj reprova") or norm.startswith("cnj indica uf do tse"))
            stale_panel = "composicao" in changed and (norm.startswith("composicao extraida do video trazia nome fora") or norm.startswith("composicao completada com o colegiado vigente"))
            stale_relator = "relator" in changed and norm.startswith("relator ") and "datajud" in norm
            if not (stale_number or stale_panel or stale_relator):
                kept_warnings.append(warning)
        if kept_warnings != (data[index].get("warnings") or []):
            change(index, "warnings", kept_warnings, "Avisos superados pela reconciliação oficial.")

    output = [data[i] if isinstance(rows[i], dict) else rows[i].model_copy(update=data[i], deep=True)
              for i in range(len(rows)) if i not in excluded]
    audit["counts"] = {"input": len(rows), "output": len(output), "matched": len(matched),
                       "corrected_fields": len(audit["corrections"]),
                       "excluded_citations": sum(e["code"] == "cited_process_number" for e in audit["exclusions"]),
                       "excluded_institutional": sum(e["code"] == "institutional_act" for e in audit["exclusions"]),
                       "unresolved": len(audit["unresolved"])}
    return output, audit
