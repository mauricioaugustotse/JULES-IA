"""Reduz um CSV BRUTO/GIGANTE de jurisprudencia do TSE (ex.: o consolidado de ~664 MB com TODAS
as decisoes colegiadas e monocraticas) para um CSV pequeno contendo SOMENTE as linhas relevantes
para a base de sessoes do Notion. Faz UMA passada streaming (nao carrega o arquivo na memoria),
preservando TODAS as colunas/ordem, para que os pipelines a jusante
(complete_cnj/classe/fill_composicao/fill_partes/fill_metadata) rodem rapido e sem estourar memoria.

Mantem a linha quando:
  (a) digits(numeroUnico)[:20] esta entre os CNJ-20 ja presentes na base  -> casa partes/classe/composicao/metadata; OU
  (b) (digits(numeroProcesso), iso(dataDecisao)) bate com uma pagina SEM CNJ-20 -> alimenta complete_cnj (proc+data); OU
  (c) iso(dataDecisao) e uma data de alguma pagina sem CNJ-20 -> alimenta a corroboracao por partes do complete_cnj.

Uso:
  python prefilter_dje_csv.py --input "C:\\Users\\mauri\\Downloads\\TSE_decisoes_consolidado_2026-06-28.csv" \
                              --out   "C:\\Users\\mauri\\ProjetoConversor\\dje_consolidado"
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime
from pathlib import Path

from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)  # textoDecisao pode passar do default (128 KB) e abortar o DictReader
LOGGER = logging.getLogger("prefilter_dje_csv")


def digits(s) -> str:
    return re.sub(r"\D", "", str(s or ""))


def iso(d) -> str:
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def build_base_index(data_source_id: str):
    """Le a base do Notion e devolve (cnj_set, incomplete_keys, datas_incompletas, all_session_dates)."""
    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    cnj_set: set[str] = set()
    incomplete_keys: set[tuple[str, str]] = set()
    datas_incompletas: set[str] = set()
    all_session_dates: set[str] = set()
    n_completas = n_incompletas = 0
    for p in pages:
        bd = digits(t(p, "numero_processo"))
        data = (t(p, "data_sessao") or "")[:10]
        if data:
            all_session_dates.add(data)
        if len(bd) >= 20:
            cnj_set.add(bd[:20])
            n_completas += 1
            continue
        n_incompletas += 1
        if not data:
            continue
        datas_incompletas.add(data)
        if bd:
            incomplete_keys.add((bd, data))
    LOGGER.info(
        "Base: %s paginas | CNJ-20: %s | sem CNJ-20: %s | chaves(proc,data): %s | datas sem CNJ: %s | datas de sessao: %s",
        len(pages), n_completas, n_incompletas, len(incomplete_keys), len(datas_incompletas), len(all_session_dates),
    )
    return cnj_set, incomplete_keys, datas_incompletas, all_session_dates


# Faltantes: decisao no CSV oficial cuja data bate com uma sessao da base mas cujo
# CNJ-20 nao tem pagina. Julgamentos "em lista" sao IGNORADOS por decisao de
# projeto — por isso o relatorio NAO alimenta a fila de vistoria automaticamente:
# so via --queue-from-artifacts, que exige evidencia de tratamento INDIVIDUAL do
# processo no video da sessao (numero mencionado nos artifacts).
MISSING_ROW_CAP = 20000
_MISSING_FIELDS = (
    "numeroUnico",
    "cnj20",
    "numeroProcesso",
    "siglaClasse",
    "descricaoClasse",
    "dataDecisao",
    "relatores",
    "partes",
    "ementa",
)


def _is_tse_colegiada(row: dict, fieldnames: list[str]) -> bool:
    """Filtro best-effort de decisao colegiada do TSE; sem coluna, nao filtra."""
    sigla = str(row.get("siglaTribunalJE", "") or "").strip().upper()
    if "siglaTribunalJE" in fieldnames and sigla and sigla != "TSE":
        return False
    for column in ("tipoDecisao", "descricaoTipoDecisao"):
        if column in fieldnames:
            value = str(row.get(column, "") or "").lower()
            if value and "monocr" in value:
                return False
    return True


def _missing_record(row: dict, cnj20: str) -> dict:
    ementa = str(row.get("textoEmenta", "") or row.get("textoDecisao", "") or "")
    return {
        "numeroUnico": str(row.get("numeroUnico", "") or ""),
        "cnj20": cnj20,
        "numeroProcesso": str(row.get("numeroProcesso", "") or ""),
        "siglaClasse": str(row.get("siglaClasse", "") or ""),
        "descricaoClasse": str(row.get("descricaoClasse", "") or ""),
        "dataDecisao": str(row.get("dataDecisao", "") or ""),
        "relatores": str(row.get("relatores", "") or row.get("relator", "") or ""),
        "partes": str(row.get("partes", "") or "")[:150],
        "ementa": re.sub(r"\s+", " ", ementa)[:200],
    }


def _queue_missing_with_artifact_evidence(
    missing_rows: list[dict],
    batch_root: Path,
    backlog_root: Path | None = None,
) -> tuple[int, int]:
    """Cruza faltantes com os artifacts dos vídeos: só entra na fila quem tem o
    número citado no vídeo da MESMA data (evidência de julgamento individual);
    o resto é presumido "em lista" e fica só no relatório."""
    import json as _json

    import vistoria_queue
    from tse_normalization import infer_session_date_from_video_title

    videos_by_date: dict[str, list[Path]] = {}
    for rows_file in batch_root.glob("*/*/04b_enriched_preview_rows.json"):
        try:
            rows = _json.loads(rows_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for data in {str(r.get("data_sessao", "") or "")[:10] for r in rows if isinstance(r, dict)}:
            if data:
                videos_by_date.setdefault(data, []).append(rows_file.parent)
    if backlog_root is not None and backlog_root.exists():
        # Backlog de playlists (Drive): data da sessão vem do título no 07_backfill_summary.
        for summary_file in backlog_root.glob("[0-9][0-9][0-9][0-9]_PL*/*/07_backfill_summary.json"):
            try:
                title = str(_json.loads(summary_file.read_text(encoding="utf-8")).get("title", "") or "")
            except Exception:
                continue
            data = infer_session_date_from_video_title(title) or ""
            if data:
                videos_by_date.setdefault(data, []).append(summary_file.parent)

    digits_cache: dict[Path, str] = {}

    def video_digits(video_dir: Path) -> str:
        if video_dir not in digits_cache:
            chunks: list[str] = []
            for pattern in ("raw_global_response_chunk_*.txt", "raw_detail_*.txt", "04b_enriched_preview_rows.json"):
                for path in sorted(video_dir.glob(pattern)):
                    try:
                        chunks.append(re.sub(r"\D", "", path.read_text(encoding="utf-8", errors="ignore")))
                    except Exception:
                        continue
            digits_cache[video_dir] = "|".join(chunks)
        return digits_cache[video_dir]

    queued = 0
    presumed_lista = 0
    items = []
    for record in missing_rows:
        session_date = iso(record["dataDecisao"])
        core_digits = record["cnj20"][:9]  # NNNNNNN + DV
        candidates = videos_by_date.get(session_date, [])
        mentioned = any(core_digits and core_digits in video_digits(d) for d in candidates)
        if not mentioned:
            presumed_lista += 1
            continue
        video_dir = next(d for d in candidates if core_digits in video_digits(d))
        items.append(
            vistoria_queue.make_vistoria_item(
                source="dje",
                video_id=video_dir.name.split("_", 1)[-1],
                youtube_url="",
                disposition="faltante_dje",
                reasons=[
                    f"CNJ {record['numeroUnico']} consta do CSV oficial do DJE em {record['dataDecisao']} "
                    "e é mencionado individualmente no vídeo da sessão, mas não tem página no Notion."
                ],
                row=None,
                artifact_dir=str(video_dir),
                data_sessao=session_date,
                extra={"dje": record},
                dedupe_key=record["cnj20"],
            )
        )
    if items:
        queued = vistoria_queue.append_items(items)
    return queued, presumed_lista


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV bruto/gigante de jurisprudencia do TSE.")
    ap.add_argument("--out", required=True, help="Pasta de saida do CSV reduzido.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument(
        "--missing-out",
        default="",
        help="Pasta do relatorio de FALTANTES (decisoes do CSV sem pagina no Notion em data de sessao conhecida). Default: mesma pasta de --out.",
    )
    ap.add_argument(
        "--no-missing",
        action="store_true",
        help="Desliga o relatorio de faltantes.",
    )
    ap.add_argument(
        "--queue-from-artifacts",
        default="",
        help=(
            "Raiz dos artifacts do batch (ex.: artifacts/tse_youtube_notion/batch_gui). "
            "Faltantes com o numero mencionado no video da mesma data entram na fila de vistoria; "
            "os demais sao presumidos 'julgados em lista' e ficam so no relatorio."
        ),
    )
    ap.add_argument(
        "--queue-from-backlog",
        default="",
        help="Raiz adicional do backlog de playlists (ex.: H:\\Meu Drive\\TSE_YOUTUBE_NOTION_BACKLOG\\backfill_2025).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    src = Path(args.input)
    if not src.exists():
        LOGGER.error("CSV de entrada nao existe: %s", src)
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"reduzido_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    cnj_set, incomplete_keys, datas_incompletas, all_session_dates = build_base_index(args.data_source_id)

    missing_enabled = not args.no_missing
    missing_rows: list[dict] = []
    missing_capped = False
    seen_missing_cnj: set[str] = set()

    lidas = mantidas = por_cnj = por_proc_data = por_data = 0
    with open(src, encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames or []
        if "numeroUnico" not in fieldnames:
            LOGGER.error("CSV de entrada sem coluna numeroUnico (cabecalho: %s)", fieldnames[:5])
            return 1
        with open(dest, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                lidas += 1
                if lidas % 100000 == 0:
                    LOGGER.info("  ... %s linhas lidas, %s mantidas", lidas, mantidas)
                cnj20 = digits(row.get("numeroUnico"))[:20]
                dt = iso(row.get("dataDecisao"))
                keep = False
                if len(cnj20) >= 20 and cnj20 in cnj_set:
                    keep = True
                    por_cnj += 1
                elif dt:
                    proc = digits(row.get("numeroProcesso"))
                    if proc and (proc, dt) in incomplete_keys:
                        keep = True
                        por_proc_data += 1
                    elif dt in datas_incompletas:
                        keep = True
                        por_data += 1
                if keep:
                    writer.writerow(row)
                    mantidas += 1
                elif (
                    missing_enabled
                    and len(cnj20) >= 20
                    and dt
                    and dt in all_session_dates
                    and cnj20 not in seen_missing_cnj
                    and _is_tse_colegiada(row, fieldnames)
                ):
                    seen_missing_cnj.add(cnj20)
                    if len(missing_rows) < MISSING_ROW_CAP:
                        missing_rows.append(_missing_record(row, cnj20))
                    else:
                        missing_capped = True

    LOGGER.info(
        "OK -> %s | lidas=%s mantidas=%s (cnj=%s proc+data=%s data=%s)",
        dest, lidas, mantidas, por_cnj, por_proc_data, por_data,
    )

    if missing_enabled:
        import json

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        missing_dir = Path(args.missing_out) if args.missing_out else out_dir
        missing_dir.mkdir(parents=True, exist_ok=True)
        missing_csv = missing_dir / f"missing_{stamp}.csv"
        with open(missing_csv, "w", encoding="utf-8", newline="") as fout:
            missing_writer = csv.DictWriter(fout, fieldnames=list(_MISSING_FIELDS))
            missing_writer.writeheader()
            missing_writer.writerows(missing_rows)
        summary = {
            "input": str(src),
            "missing_total": len(missing_rows),
            "missing_capped": missing_capped,
            "queued": 0,
            "presumed_lista": None,
        }
        if args.queue_from_artifacts and missing_rows:
            batch_root = Path(args.queue_from_artifacts)
            backlog_root = Path(args.queue_from_backlog) if args.queue_from_backlog else None
            if batch_root.exists():
                queued, presumed_lista = _queue_missing_with_artifact_evidence(missing_rows, batch_root, backlog_root)
                summary["queued"] = queued
                summary["presumed_lista"] = presumed_lista
                LOGGER.info(
                    "Faltantes: %s no relatorio | %s na fila de vistoria | %s presumidos 'em lista' (ignorados)",
                    len(missing_rows), queued, presumed_lista,
                )
            else:
                LOGGER.warning("--queue-from-artifacts nao existe: %s", batch_root)
        (missing_dir / f"missing_{stamp}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        LOGGER.info("Relatorio de faltantes -> %s (%s linha(s))", missing_csv, len(missing_rows))

    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
