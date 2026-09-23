"""Monitor local de cobertura e confirmacao de publicacao do lote TSE.

Os sinais do scan/rito sao candidatos a conferencia, nao prova de julgamento.
Nunca cria processos nem aprova automaticamente itens bloqueados.
"""
from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any
from tse_official_session import compare_official_rows

from tse_normalization import (
    canonicalize_numero_processo,
    extract_youtube_video_id,
    normalize_numero_processo_display,
    normalize_youtube_link,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def process_key(value: Any) -> str:
    return canonicalize_numero_processo(str(value or ""))


def reconcile_video(
    analysis: dict[str, Any], rows: list[dict[str, Any]],
    results: list[dict[str, Any]] | None = None, *,
    rito: dict[str, Any] | None = None,
    scan: dict[str, Any] | None = None,
    detail: dict[str, Any] | None = None,
    verification: list[dict[str, Any]] | None = None,
    published: bool = False,
    official: dict[str, Any] | None = None,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    official_comparison = compare_official_rows(official, rows) if official is not None else []
    information = [i for i in official_comparison if i.get("severity") == "info"]
    issues.extend(i for i in official_comparison if i.get("severity") != "info")
    official_exclusions = {process_key(i.get("numero_processo")) for i in information
                           if i.get("code") == "official_exclusion"} - {""}

    def issue(code: str, message: str, **context: Any) -> None:
        issues.append({"code": code, "message": message, **context})

    if not scan or scan.get("status") != "complete":
        issue("scan_unverified", "Cobertura integral da varredura nao confirmada.",
              uncovered_intervals=(scan or {}).get("uncovered_intervals", []))
    for block in (detail or {}).get("blocks", []):
        if block.get("status") == "pending":
            issue("detail_unresolved", "Releitura do bloco ainda apresenta inconsistencia.",
                  bundle_index=block.get("index"), start_seconds=block.get("start_seconds"),
                  expected_process_numbers=block.get("expected_process_numbers", []),
                  observed_process_numbers=block.get("observed_process_numbers", []),
                  reasons=block.get("issues", []))
    bundles = analysis.get("bundles") or []
    row_keys = {process_key(r.get("numero_processo")) for r in rows} - {""}
    for window in (analysis.get("session") or {}).get("judgments", []):
        expected = {process_key(n) for n in window.get("mentioned_process_numbers", [])} - {""}
        if window.get("should_ignore"):
            excluded.append({"stage": "scan", **window})
            if not window.get("ignore_reason") and not (expected and expected <= official_exclusions):
                issue("unexplained_exclusion", "Bloco ignorado sem justificativa.", window=window)
            continue
        for number in sorted(expected - row_keys - official_exclusions):
            issue("scan_process_missing", "Processo mencionado na varredura sem linha final; conferir se era julgamento ou citacao.",
                  numero_processo=number, start_seconds=window.get("start_seconds"),
                  end_seconds=window.get("end_seconds"), title_hint=window.get("title_hint"))
        if not expected and not any(
            abs(int(b.get("start_seconds", -10000)) - int(window.get("start_seconds", 0))) <= 90
            for b in bundles if not b.get("should_ignore") and b.get("items")
        ):
            issue("window_without_detail", "Bloco identificado sem detalhamento correspondente.", window=window)
    for index, bundle in enumerate(bundles, 1):
        if bundle.get("should_ignore"):
            excluded.append({"stage": "detail", **bundle})
            mentioned = {process_key(n) for n in bundle.get("mentioned_process_numbers", [])}
            mentioned.update(process_key(i.get("numero_processo")) for i in bundle.get("items", []))
            mentioned.discard("")
            if mentioned and mentioned <= official_exclusions:
                continue
            # A primeira varredura o identificou como candidato: a decisao posterior
            # precisa continuar visivel mesmo quando o modelo apresenta um motivo.
            issue("detail_excluded", "Bloco descartado no detalhamento; conferir exclusao.",
                  bundle_index=index, reason=bundle.get("ignore_reason", ""))
            continue
        if not bundle.get("items"):
            issue("empty_detail", "Bloco de julgamento sem itens extraidos.", bundle_index=index)
        for item in bundle.get("items") or []:
            number = process_key(item.get("numero_processo"))
            if number and number not in row_keys and number not in official_exclusions:
                issue("detail_process_missing", "Processo detalhado desapareceu da previa final.",
                      numero_processo=number, bundle_index=index)
    calls = (rito or {}).get("apregoamentos_individuais")
    if (rito or {}).get("transcript_available") and isinstance(calls, int) and calls > len(rows):
        issue("rito_gap", f"Rito indica {calls} apregoamentos para {len(rows)} linhas extraidas.",
              expected=calls, extracted=len(rows))
    outcomes = [r for r in (results or []) if r.get("status") in {"created", "updated", "blocked", "skipped", "error"}]
    if published:
        if len(outcomes) != len(rows):
            issue("publication_gap", "Numero de resultados da publicacao difere da previa.",
                  extracted=len(rows), outcomes=len(outcomes))
        for index, result in enumerate(outcomes):
            if result.get("status") not in {"created", "updated"}:
                issue("unpublished_row", "Linha nao publicada; revisar motivo.", row_index=index,
                      numero_processo=result.get("numero_processo"), status=result.get("status"),
                      reasons=list(result.get("errors") or []) + list(result.get("warnings") or []))
        verified = {v.get("row_index") for v in (verification or []) if v.get("status") == "verified"}
        for index, result in enumerate(outcomes):
            if result.get("status") in {"created", "updated"} and index not in verified:
                issue("notion_unverified", "Gravacao sem confirmacao por leitura no Notion.",
                      row_index=index, numero_processo=result.get("numero_processo"), page_id=result.get("page_id"))
    if not rows and not (official and official.get("status") == "available"
                        and official.get("expected_count") == 0
                        and not official.get("counts", {}).get("unknown")):
        issue("no_rows", "Nenhum julgado extraido; sessao exige conferencia.")
    return {
        "status": "pending" if issues else ("verified" if published else "preview"),
        "counts": {"extracted": len(rows), "created": sum(r.get("status") == "created" for r in outcomes),
                   "updated": sum(r.get("status") == "updated" for r in outcomes),
                   "verified": sum(v.get("status") == "verified" for v in (verification or [])),
                   "issues": len(issues)},
        "issues": issues, "information": information, "excluded_windows": excluded,
        "evidence": {"whole_video_scan": "complete" if scan and scan.get("status") == "complete" else "unknown",
                     "official_inventory": (official or {}).get("status", "not_checked"),
                     "official_session_date": (official or {}).get("session_date"),
                     "official_expected": (official or {}).get("expected_count"),
                     "official_counts": (official or {}).get("counts", {})},
        "official_comparison": official_comparison,
        "limitation": "Confere os sinais disponiveis; ausencia de alerta nao prova que o modelo identificou todos os julgados.",
    }


def _property_value(prop: dict[str, Any], kind: str) -> Any:
    """Compare values sent by our writer, ignoring Notion IDs and display colors."""
    if kind not in prop:
        raise ValueError(f"Propriedade sem valor {kind}")
    value = prop[kind]
    if kind in {"title", "rich_text"}:
        return "".join(part.get("plain_text", part.get("text", {}).get("content", ""))
                       for part in value or [])
    if kind in {"select", "status"}:
        return (value or {}).get("name")
    if kind == "multi_select":
        return sorted(v.get("name") for v in value or [])
    if kind == "relation":
        if prop.get("has_more"):
            raise ValueError("Propriedade relation incompleta na releitura")
        return sorted(v.get("id", "").replace("-", "") for v in value or [])
    if kind == "date":
        return None if not value else tuple(value.get(k) for k in ("start", "end", "time_zone"))
    return value


def _field_matches(name: str, expected: Any, actual: Any) -> bool:
    if name == "numero_processo" and expected:
        return normalize_numero_processo_display(str(actual)) == normalize_numero_processo_display(str(expected))
    if name == "youtube_link" and expected:
        return (bool(extract_youtube_video_id(str(expected)))
                and normalize_youtube_link(str(actual)) == normalize_youtube_link(str(expected)))
    return actual == expected


def verify_notion_rows(rows: list[Any], results: list[dict[str, Any]], client: Any, schema: Any,
                       checkpoint=None, *, fields: set[str] | None = None) -> list[dict[str, Any]]:
    """Le cada pagina e confere todas as propriedades escritas, inclusive limpezas.

    ``fields`` limita a reconferencia final aos dados do julgamento: tratamentos
    posteriores podem enriquecer noticias, relacoes e texto deliberadamente.
    """
    checks: list[dict[str, Any]] = []
    for index, (row, result) in enumerate(zip(rows, results)):
        if result.get("status") not in {"created", "updated"}:
            continue
        check = {"row_index": index, "page_id": result.get("page_id", ""), "status": "unverified"}
        try:
            if not check["page_id"]:
                raise ValueError("API nao retornou id da pagina")
            page = client._request("GET", f"/pages/{check['page_id']}")
            if page.get("archived") or page.get("in_trash"):
                raise ValueError("Pagina arquivada ou na lixeira")
            if page.get("id", "").replace("-", "") != check["page_id"].replace("-", ""):
                raise ValueError("Identificador da pagina diverge")
            builder = getattr(client, "build_properties_payload", None)
            if callable(builder):
                payload = builder(schema, row)
                names = [name for name in payload if fields is None or name in fields]
                if not names:
                    raise ValueError("Nenhuma propriedade escrita disponivel para conferencia")
                for name in names:
                    expected_prop = payload[name]
                    kind = next(iter(expected_prop))
                    expected = _property_value(expected_prop, kind)
                    actual = _property_value(page.get("properties", {}).get(name, {}), kind)
                    if not _field_matches(name, expected, actual):
                        raise ValueError(f"Campo {name} diverge da previa publicada")
                check["checked_fields"] = names
                check["status"] = "verified"
                checks.append(check)
                if checkpoint:
                    checkpoint(checks)
                continue
            # Compatibilidade com clientes de teste/legados sem construtor.
            for name in ("numero_processo", "data_sessao", "youtube_link", "tema",
                         "resultado", "votacao", "classe_processo", "origem", "relator", "composicao"):
                if fields is not None and name not in fields:
                    continue
                expected = str(getattr(row, name, "") or "")
                if not expected:
                    continue
                actual = client._extract_property_text(page, schema, name)
                if name == "numero_processo":
                    # Na releitura a API deve preservar o CNJ inteiro enviado;
                    # comparar apenas o nucleo ocultaria ano/tribunal divergentes.
                    normalized_expected = normalize_numero_processo_display(expected)
                    match = bool(normalized_expected) and normalize_numero_processo_display(actual) == normalized_expected
                elif name == "data_sessao":
                    match = actual[:10] == expected[:10]
                elif name == "youtube_link":
                    # O timestamp distingue julgados da mesma sessao, sobretudo
                    # quando nao ha numero de processo identificado.
                    match = (bool(extract_youtube_video_id(expected))
                             and normalize_youtube_link(actual) == normalize_youtube_link(expected))
                else:
                    match = actual.strip() == expected.strip()
                if not match:
                    raise ValueError(f"Campo {name} diverge da previa publicada")
            check["status"] = "verified"
        except Exception as exc:
            check["error"] = str(exc)[:1000]
        checks.append(check)
        if checkpoint:
            checkpoint(checks)
    return checks


class BatchMonitor:
    """Estado persistente e painel HTML atualizado a cada etapa do fluxo."""
    def __init__(self, root: Path, videos: list[Any]):
        self.root = root
        self.state = {
            "status": "running", "pid": os.getpid(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "videos": {v.video_id: {"url": v.url, "status": "queued", "stage": "Aguardando"} for v in videos},
        }
        self.save()

    def event(self, video_id: str, stage: str, **fields: Any) -> None:
        self.state["stage"] = stage
        if video_id in self.state["videos"]:
            self.state["videos"][video_id].update(stage=stage, **fields)
        else:
            self.state.update(fields)
        self.save()

    def finish(self, summary: dict[str, Any]) -> None:
        incomplete = (summary.get("total_pending", 0) or summary.get("total_error", 0)
                      or summary.get("total_unprocessed", 0) or summary.get("total_stopped", 0)
                      or (summary.get("post_publish") or {}).get("falhas"))
        status = "pending" if incomplete else ("preview" if not summary.get("publish_requested", True) else "verified")
        self.state.update(status=status, stage="Finalizado", summary=summary)
        self.save()

    def save(self) -> None:
        self.state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_json(self.root / "monitor_status.json", self.state)
        items = []
        for video_id, item in self.state["videos"].items():
            report = item.get("coverage") or {}
            messages = "".join("<li>" + html.escape(i["message"] + (" " + str(i.get("numero_processo")) if i.get("numero_processo") else "")) + "</li>" for i in report.get("issues", []))
            evidence = report.get("evidence") or {}
            source_summary = ("Varredura integral: " + str(evidence.get("whole_video_scan", "unknown"))
                              + "; inventario oficial: " + str(evidence.get("official_inventory", "not_checked"))
                              + "; julgados individuais esperados: " + str(evidence.get("official_expected", "?")))
            exclusions = "".join("<li>" + html.escape(i["message"] + " " + str(i.get("numero_processo", ""))) + "</li>"
                                 for i in report.get("information", []) if i.get("code") == "official_exclusion")
            info = "<details><summary>Retiradas e julgamentos em lista confirmados</summary><ul>" + exclusions + "</ul></details>" if exclusions else ""
            items.append(f"<tr><td>{html.escape(video_id)}</td><td>{html.escape(item['status'])}</td><td>{html.escape(item['stage'])}<p>{html.escape(source_summary)}</p><ul>{messages}</ul>{info}</td></tr>")
        page = ("<!doctype html><html lang='pt-BR'><meta charset='utf-8'><meta http-equiv='refresh' content='10'>"
                "<title>Monitor TSE YouTube Notion</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#f4f6fa;color:#172738}td,th{padding:15px;border-bottom:1px solid #ccc;text-align:left}table{width:100%;background:white}li{margin:8px 0}</style>"
                f"<h1>Monitor TSE YouTube → Notion</h1><p>Estado: <strong>{html.escape(self.state['status'])}</strong> · Atualizado: {self.state['updated_at']}</p>"
                f"<p>{html.escape(self.state.get('stage', 'Iniciando'))}</p>"
                "<p>O painel reflete a ultima etapa registrada. Se o processo for encerrado, os horarios param de avancar. Pendencias exigem conferencia; nao sao aprovadas automaticamente.</p>"
                "<table><tr><th>Video</th><th>Estado</th><th>Etapa e inconsistencias</th></tr>" + "".join(items) + "</table></html>")
        temporary = self.root / "monitor.html.tmp"
        temporary.write_text(page, encoding="utf-8")
        os.replace(temporary, self.root / "monitor.html")
