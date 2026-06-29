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
    """Le a base do Notion e devolve (cnj_set, incomplete_keys, datas_incompletas)."""
    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    cnj_set: set[str] = set()
    incomplete_keys: set[tuple[str, str]] = set()
    datas_incompletas: set[str] = set()
    n_completas = n_incompletas = 0
    for p in pages:
        bd = digits(t(p, "numero_processo"))
        if len(bd) >= 20:
            cnj_set.add(bd[:20])
            n_completas += 1
            continue
        n_incompletas += 1
        data = (t(p, "data_sessao") or "")[:10]
        if not data:
            continue
        datas_incompletas.add(data)
        if bd:
            incomplete_keys.add((bd, data))
    LOGGER.info(
        "Base: %s paginas | CNJ-20: %s | sem CNJ-20: %s | chaves(proc,data): %s | datas sem CNJ: %s",
        len(pages), n_completas, n_incompletas, len(incomplete_keys), len(datas_incompletas),
    )
    return cnj_set, incomplete_keys, datas_incompletas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="CSV bruto/gigante de jurisprudencia do TSE.")
    ap.add_argument("--out", required=True, help="Pasta de saida do CSV reduzido.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
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

    cnj_set, incomplete_keys, datas_incompletas = build_base_index(args.data_source_id)

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

    LOGGER.info(
        "OK -> %s | lidas=%s mantidas=%s (cnj=%s proc+data=%s data=%s)",
        dest, lidas, mantidas, por_cnj, por_proc_data, por_data,
    )
    print(str(dest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
