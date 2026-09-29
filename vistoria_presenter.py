"""Apresentação da vistoria: um caso, suas evidências e um próximo passo legível.

Lê apenas os artefatos locais. Agrupar a exibição nunca altera o estado da fila.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any


FIELD_LABELS = {
    "numero_processo": "Número do processo", "classe_processo": "Classe",
    "origem": "Origem", "relator": "Relator", "resultado": "Resultado",
    "votacao": "Votação", "pedido_vista": "Pedido de vista",
    "composicao": "Composição", "data_sessao": "Data da sessão",
}


def _text(value: Any) -> str:
    if value is None or value == "":
        return "Não informado"
    if isinstance(value, list):
        return "; ".join(_text(v) for v in value) or "Não informado"
    return html.unescape(re.sub(r"<[^>]*>", "", str(value))).strip()


def coverage_issues(item: dict[str, Any]) -> list[dict[str, Any]]:
    extra = item.get("extra") or {}
    issues = extra.get("coverage_issues") or [extra.get("coverage_issue")]
    return [v for v in issues if isinstance(v, dict)]


def process_number(item: dict[str, Any]) -> str:
    row = item.get("row") or item.get("display_row") or {}
    dje = (item.get("extra") or {}).get("dje") or {}
    number = row.get("numero_processo") or dje.get("numeroUnico") or item.get("numero_hint")
    if not number:
        number = next((i.get("numero_processo") for i in coverage_issues(item) if i.get("numero_processo")), "")
    if not number:
        match = re.search(r"\b\d{7}-\d{2}(?:\.\d{4}\.\d\.\d{2}\.\d{4})?\b|\b\d{20}\b", " ".join(item.get("reasons") or []))
        number = match.group(0) if match else ""
    digits = re.sub(r"\D", "", str(number or ""))
    if len(digits) == 20:
        return f"{digits[:7]}-{digits[7:9]}.{digits[9:13]}.{digits[13]}.{digits[14:16]}.{digits[16:]}"
    return str(number or "")


def timestamp_seconds(item: dict[str, Any]) -> int | None:
    row = item.get("row") or item.get("display_row") or {}
    seconds = row.get("source_start_seconds")
    if isinstance(seconds, (int, float)) and seconds >= 0:
        return int(seconds)
    link = str(row.get("youtube_link") or item.get("youtube_url") or "")
    match = re.search(r"[?&]t=(\d+)", link)
    return int(match.group(1)) if match else None


def video_url(item: dict[str, Any]) -> str:
    row = item.get("row") or item.get("display_row") or {}
    url = str(row.get("youtube_link") or item.get("youtube_url") or "")
    if not url and item.get("video_id"):
        url = f"https://www.youtube.com/watch?v={item['video_id']}"
    seconds = timestamp_seconds(item)
    if url and seconds is not None and not re.search(r"[?&]t=", url):
        url += ("&" if "?" in url else "?") + f"t={seconds}"
    return url


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def enrich_evidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Anexa provas para exibição; não promove uma linha antiga a publicável."""
    cache: dict[str, tuple[dict, list]] = {}
    enriched = []
    for source in items:
        item = dict(source)
        folder = str(item.get("artifact_dir") or "")
        if folder and folder not in cache:
            directory = Path(folder)
            inventory = _read(directory / "00_official_session_inventory.json")
            rows = _read(directory / "04h_publish_preview_rows.json")
            cache[folder] = (inventory if isinstance(inventory, dict) else {}, rows if isinstance(rows, list) else [])
        inventory, rows = cache.get(folder, ({}, []))
        if not item.get("row"):
            index = next((i.get("row_index") for i in coverage_issues(item) if isinstance(i.get("row_index"), int)), None)
            if index is not None and 0 <= index < len(rows):
                item["display_row"] = rows[index]
        item["data_sessao"] = item.get("data_sessao") or inventory.get("session_date", "")
        item["official_url"] = inventory.get("source_url", "")
        digits = re.sub(r"\D", "", process_number(item))
        candidates = [p for p in inventory.get("processes", []) if digits and (
            re.sub(r"\D", "", str(p.get("numeroProcesso", ""))) == digits
            or (len(digits) == 9 and re.sub(r"\D", "", str(p.get("numeroProcesso", "")))[:9] == digits)
        )]
        item["official_process"] = candidates[0] if len(candidates) == 1 else {}
        enriched.append(item)
    return enriched


def group_review_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Agrupa só identidades iguais no mesmo vídeo e estado; preserva todos os ids."""
    items = enrich_evidence(items)
    full: dict[tuple[str, str], set[str]] = {}
    for item in items:
        digits = re.sub(r"\D", "", process_number(item))
        if len(digits) == 20:
            full.setdefault((str(item.get("video_id")), digits[:9]), set()).add(digits)
    groups: dict[tuple, list] = {}
    for item in items:
        digits = re.sub(r"\D", "", process_number(item))
        matches = full.get((str(item.get("video_id")), digits), set())
        if len(digits) == 9 and len(matches) == 1:
            digits = next(iter(matches))
        key = (item.get("video_id"), item.get("status", "pending"), digits or item.get("id"))
        groups.setdefault(key, []).append(item)
    result = []
    for members in groups.values():
        representative = max(members, key=lambda x: (bool(x.get("row")), len(str(x.get("row") or ""))))
        item = dict(representative)
        item["member_ids"] = [str(m["id"]) for m in members]
        item["members"] = members
        item["reasons"] = list(dict.fromkeys(str(r) for m in members for r in (m.get("reasons") or [])))
        issues = [issue for m in members for issue in coverage_issues(m)]
        item["extra"] = {**(item.get("extra") or {}), "coverage_issues": issues}
        item["alert_count"] = max(len(issues), len(members))
        result.append(item)
    return sorted(result, key=lambda i: (i.get("data_sessao", ""), process_number(i)), reverse=True)


def is_notice(item: dict[str, Any]) -> bool:
    issues = coverage_issues(item)
    return bool(issues) and all(i.get("severity") in {"warning", "info"} for i in issues) and not item.get("row")


def problem_label(item: dict[str, Any]) -> str:
    if item.get("status") == "published":
        return "Publicado no Notion"
    if item.get("status") == "resolved":
        return "Resolvido automaticamente"
    if item.get("status") == "rejected":
        return "Descartado"
    codes = {i.get("code") or i.get("kind") for i in coverage_issues(item)}
    if "official_missing_judgment" in codes:
        return "Julgamento ainda não recuperado"
    if "notion_final_unverified" in codes or "notion_unverified" in codes:
        return "Publicação precisa de conferência"
    if "official_field_mismatch" in codes:
        return "Dados divergem da sessão oficial"
    if "official_unexpected_row" in codes:
        return "Identidade do processo incerta"
    if is_notice(item):
        return "Aviso da sessão"
    disposition = item.get("disposition", "")
    return {
        "blocked": "Dados incompletos ou incoerentes",
        "skipped": "Conferir se houve julgamento próprio",
        "duplicata_numero": "Possível página duplicada",
        "faltante_dje": "Julgamento localizado no DJe",
        "contagem_rito": "Conferir cobertura da sessão",
        "cobertura": "Conferir evidências da sessão",
    }.get(disposition, "Revisão necessária")


def case_title(item: dict[str, Any]) -> str:
    row = item.get("row") or item.get("display_row") or {}
    official = item.get("official_process") or {}
    parts = [row.get("classe_processo") or official.get("siglaClasseJudicial"), row.get("origem") or official.get("origem")]
    if any(parts):
        return " · ".join(str(v) for v in parts if v)
    return str(row.get("tema") or item.get("tema_hint") or official.get("assuntoPrincipal") or "Conferência da sessão")


def next_step(item: dict[str, Any], approval_reason: str = "") -> str:
    status = item.get("status")
    if status in {"published", "resolved", "rejected"}:
        return {"published": "Abra a página no Notion para consultar o resultado.", "resolved": "Nenhuma decisão pendente. A correção ficou registrada no histórico.", "rejected": "Caso encerrado sem publicação. Restaure somente se a decisão precisar ser revista."}[status]
    if is_notice(item):
        return "Consulte a situação oficial abaixo. Este aviso não contém uma publicação para aprovar."
    if approval_reason:
        return ("Use Corrigir dados. " if item.get("row") else "") + approval_reason
    if item.get("row"):
        return "Confira o resumo e o trecho do vídeo. Publique quando os dados estiverem confirmados."
    return "Consulte o trecho e a fonte oficial. É preciso recuperar ou corrigir os dados do julgamento antes de publicar."


def detail_text(item: dict[str, Any], approval_reason: str = "") -> str:
    row = item.get("row") or item.get("display_row") or {}
    official = item.get("official_process") or {}
    number = process_number(item)
    lines = [f"{number or 'Conferência da sessão'} — {case_title(item)}", f"Sessão: {item.get('data_sessao') or 'Não informada'}", "", "O QUE PRECISA DE ATENÇÃO", problem_label(item), "", "PRÓXIMO PASSO", next_step(item, approval_reason)]
    comparisons = [issue for issue in coverage_issues(item) if issue.get("field") and ("expected" in issue or "actual" in issue)]
    if comparisons:
        lines += ["", "DIVERGÊNCIAS IDENTIFICADAS"]
        for issue in comparisons:
            lines += [f"{FIELD_LABELS.get(issue['field'], issue['field'])}:", f"  Extração registrada: {_text(issue.get('actual'))}", f"  Registro oficial: {_text(issue.get('expected'))}"]
    lines += ["", "EVIDÊNCIA DO VÍDEO"]
    if row:
        for field in ("numero_processo", "classe_processo", "origem", "relator", "resultado", "votacao", "pedido_vista", "composicao"):
            if row.get(field):
                lines.append(f"{FIELD_LABELS[field]}: {_text(row[field])}")
        for label, field in (("Resumo", "punchline"), ("Análise do julgamento", "analise_do_conteudo_juridico")):
            if row.get(field):
                lines += ["", f"{label}: {_text(row[field])}"]
    else:
        lines.append("Ainda não há texto de julgamento recuperado para este alerta.")
    seconds = timestamp_seconds(item)
    if seconds is not None:
        lines.append(f"Trecho: {seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02} — botão Abrir trecho")
    if video_url(item):
        lines.append(video_url(item))
    lines += ["", "REGISTRO OFICIAL DA SESSÃO"]
    if official:
        for label, field in (("Processo", "numeroProcesso"), ("Classe", "classeJudicial"), ("Origem", "origem"), ("Relator", "relator"), ("Situação", "situacaoProcesso"), ("Motivo", "motivoRetiradaPauta"), ("Proclamação", "proclamacaoDecisao")):
            if official.get(field):
                lines.append(f"{label}: {_text(official[field])}")
        if official.get("segredoJustica"):
            lines.append("O registro oficial omite parte dos dados deste processo. Campos vazios não provam ausência de julgamento.")
    else:
        lines.append("Nenhum registro oficial foi associado com segurança a este item.")
    dje = (item.get("extra") or {}).get("dje") or {}
    if dje.get("ementa"):
        lines += ["", f"Ementa do DJe: {_text(dje['ementa'])}"]
    lines += ["", "MOTIVOS REGISTRADOS"] + [f"• {r}" for r in item.get("reasons") or ["Sem motivo adicional."]]
    if item.get("triagem") or item.get("resolution"):
        lines += ["", f"Tratamento: {_text(item.get('triagem') or item.get('resolution'))}"]
    edits = (item.get("extra") or {}).get("manual_edits") or []
    if edits:
        lines += ["", "ÚLTIMA CORREÇÃO NA TELA"]
        for change in edits[-1].get("changes") or []:
            lines.append(f"{FIELD_LABELS.get(change['field'], change['field'])}: {_text(change.get('before'))} → {_text(change.get('after'))}")
            if change.get("requested") != change.get("after"):
                lines.append(f"  Valor digitado: {_text(change.get('requested'))}. A conferência oficial ajustou o valor final.")
    if item.get("artifact_dir"):
        lines += ["", f"Pasta de evidências: {item['artifact_dir']}"]
    return "\n".join(lines)
