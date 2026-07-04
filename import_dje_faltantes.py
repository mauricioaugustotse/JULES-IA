# -*- coding: utf-8 -*-
"""Importa para o Notion os FALTANTES do DJE aprovados na vistoria (source="dje").

Fontes por linha: dados OFICIAIS do CSV bruto do DJE (numeroUnico, classe,
relator, ementa completa, dispositivo, município/UF, ano da eleição) + link do
vídeo da sessão no trecho em que o processo é mencionado (artifacts). Resultado
e votação são inferidos do dispositivo oficial ("ACORDAM ... por unanimidade,
negar provimento"). Tema/punchline vêm do enricher Gemini do próprio pipeline
(fallback determinístico). Partes/advogados/composição ficam vazios de
propósito: os pipelines do watcher DJE (fill_partes/fill_composicao/...)
completam com o registro oficial no próximo confronto, casando por CNJ-20+data.

tipo_registro recebe o próximo "Julgamento N" livre da data (posição real na
pauta não verificada — warning registrado na linha).

Uso:
  python import_dje_faltantes.py --csv "<CSV bruto>"            # dry-run
  python import_dje_faltantes.py --csv "<CSV bruto>" --apply
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path

from local_secrets import get_secret
from tse_normalization import CNJ_ELECTORAL_UF_BY_CODE
from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_NOTION_DATA_SOURCE_ID,
    NotionSessoesClient,
    PublishPreviewRow,
    RunArtifacts,
    assess_row_publishability,
    enrich_preview_rows_with_theme_punchline,
    publish_preview_rows,
    validate_preview_row,
)

import vistoria_queue

LOGGER = logging.getLogger("import_dje_faltantes")
csv.field_size_limit(50 * 1024 * 1024)

UF_CAPITAL = {
    "AC": "Rio Branco", "AL": "Maceió", "AP": "Macapá", "AM": "Manaus", "BA": "Salvador",
    "CE": "Fortaleza", "DF": "Brasília", "ES": "Vitória", "GO": "Goiânia", "MA": "São Luís",
    "MT": "Cuiabá", "MS": "Campo Grande", "MG": "Belo Horizonte", "PA": "Belém", "PB": "João Pessoa",
    "PR": "Curitiba", "PE": "Recife", "PI": "Teresina", "RJ": "Rio de Janeiro", "RN": "Natal",
    "RS": "Porto Alegre", "RO": "Porto Velho", "RR": "Boa Vista", "SC": "Florianópolis",
    "SE": "Aracaju", "SP": "São Paulo", "TO": "Palmas",
}


def digits(value: str) -> str:
    return re.sub(r"\D", "", str(value or ""))


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def iso(value: str) -> str:
    match = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(value or ""))
    return f"{match.group(3)}-{int(match.group(2)):02d}-{int(match.group(1)):02d}" if match else ""


def format_cnj20(cnj20: str) -> str:
    d = digits(cnj20)
    if len(d) != 20:
        return cnj20
    return f"{d[:7]}-{d[7:9]}.{d[9:13]}.{d[13]}.{d[14:16]}.{d[16:20]}"


# Início do comando decisório colegiado. Deliberadamente estrito ("O Tribunal, por..."):
# "O TRIBUNAL REGIONAL" no meio de ementa/relatório não pode disparar.
DISPOSITIVO_START_RE = re.compile(r"ACORDAM|O Tribunal, por|A Corte, por")

# siglaClasse do CSV do DJE -> classe canônica da base
CLASSE_MAP = {
    "REspEl": "REspe",
    "AREspEl": "AREspe",
    "RO-El": "RO",
    "ROEl": "RO",
    "AgR-REspEl": "AgRg-REspe",
    "AgR-AREspEl": "AgRg-AREspe",
    "AgR-RO-El": "AgRg-RO",
    "ED-REspEl": "ED-REspe",
    "CtaEl": "CTA",
    "Cta": "CTA",
    "RCEd": "RCED",
    "LT": "Lista Tríplice",
}
RESULTADO_POR_COMANDO: tuple[tuple[str, str], ...] = (
    (r"negar[a]?m? provimento|negou provimento|desprove", "Desprovido"),
    (r"(?:dar|deu|d[aã]o) parcial provimento|provimento parcial|prove[ur]?\w* em parte", "Provido em parte"),
    (r"(?:dar|deu|d[aã]o) provimento|proveu", "Provido"),
    (r"n[aã]o conhec", "Não conhecido"),
    (r"rejeitar\w* os embargos|rejeitou os embargos|rejeit\w+ os embargos", "Rejeitados"),
    (r"acolher\w* (?:em parte )?os embargos|acolheu", "Acolhidos"),
    (r"julgar\w* parcialmente procedente|procedente em parte", "Procedente em parte"),
    (r"julg\w+ improcedente|improced[eê]ncia", "Improcedente"),
    (r"julg\w+ procedente|proced[eê]ncia do pedido", "Procedente"),
    (r"aprovar?\w* (?:as contas )?com ressalvas|aprovou .{0,40}com ressalvas", "Aprovada com ressalvas"),
    (r"desaprovar?\w*|desaprovou", "Desaprovada"),
    (r"aprovar?\w*|aprovou", "Aprovada"),
    (r"referend", "Referendada"),
    (r"indefer", "Indeferido"),
    (r"defer", "Deferido"),
    (r"denegar|denegou", "Denegado"),
    (r"conceder|concedeu", "Concedido"),
    (r"prejudicad", "Prejudicado"),
    (r"encaminhamento da lista|encaminhar a lista", "Aprovada"),
)


def extract_dispositivo(texto_decisao: str) -> str:
    """Trecho decisório do textoDecisao (a partir de 'ACORDAM'/'O Tribunal')."""
    text = clean_text(texto_decisao)
    match = DISPOSITIVO_START_RE.search(text)
    if match:
        text = text[match.start():]
    return text[:1500]


def resultado_votacao_from_dispositivo(dispositivo: str) -> tuple[str, str]:
    lowered = dispositivo.lower()
    votacao = ""
    if "unanimidade" in lowered or "unânime" in lowered or "unanime" in lowered:
        votacao = "Unânime"
    elif "por maioria" in lowered:
        votacao = "Por maioria"
    resultado = ""
    for pattern, value in RESULTADO_POR_COMANDO:
        if re.search(pattern, lowered):
            resultado = value
            break
    return resultado, votacao


def parse_relator(text: str) -> str:
    """Último "Min. Nome" da coluna (relator designado prevalece sobre o original)."""
    names = re.findall(r"Min(?:istr[oa])?\.?\s+([A-ZÁÂÃÉÊÍÓÔÕÚÇ][^;,]*)", clean_text(text))
    if not names:
        return ""
    return "Min. " + names[-1].strip()


def tribunal_origem_from_cnj(cnj20: str, municipio: str, uf: str) -> tuple[str, str]:
    tr_code = cnj20[14:16] if len(cnj20) >= 16 else ""
    if tr_code == "00":
        return "TSE", "Brasília/DF"
    tr_uf = CNJ_ELECTORAL_UF_BY_CODE.get(tr_code, "")
    uf = (uf or tr_uf or "").strip().upper()
    municipio = clean_text(municipio)
    if not municipio and uf:
        municipio = UF_CAPITAL.get(uf, "")
    origem = f"{municipio}/{uf}" if municipio and uf else ""
    tribunal = f"TRE-{tr_uf}" if tr_uf else ""
    return tribunal, origem


def find_mention_timestamp(video_dir: Path, cnj20: str) -> int | None:
    """Menor start_seconds dos bundles cujo texto menciona o núcleo do processo."""
    core_digits = cnj20[:9]
    best: int | None = None
    for bundle_path in sorted(video_dir.glob("02_judgment_*.json")):
        try:
            payload = json.loads(bundle_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if core_digits in digits(json.dumps(payload, ensure_ascii=False)):
            start = int(payload.get("start_seconds", 0) or 0)
            if best is None or start < best:
                best = start
    return best


def load_full_rows(csv_path: Path, wanted_cnj20: set[str]) -> dict[str, dict]:
    """Uma passada streaming pelo CSV bruto coletando as linhas completas."""
    found: dict[str, dict] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cnj20 = digits(row.get("numeroUnico"))[:20]
            if cnj20 in wanted_cnj20 and cnj20 not in found:
                found[cnj20] = dict(row)
                if len(found) == len(wanted_cnj20):
                    break
    return found


def next_judgment_numbers(client: NotionSessoesClient, schema, dates: set[str]) -> dict[str, int]:
    """Maior N de "Julgamento N" por data (varredura única da base)."""
    max_by_date: dict[str, int] = {data: 0 for data in dates}
    for page in client.query_data_source():
        data = (client._extract_property_text(page, schema, "data_sessao") or "")[:10]
        if data not in max_by_date:
            continue
        tipo = client._extract_property_text(page, schema, "tipo_registro") or ""
        match = re.match(r"Julgamento\s+(\d+)", tipo)
        if match:
            max_by_date[data] = max(max_by_date[data], int(match.group(1)))
    return max_by_date


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="CSV BRUTO consolidado do DJE.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--skip-theme-enricher", action="store_true", help="Não chama o Gemini (tema determinístico).")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    items = [i for i in vistoria_queue.load_items("pending") if i.get("source") == "dje"]
    if not items:
        LOGGER.info("Nenhum faltante DJE pendente na fila.")
        return 0
    LOGGER.info("Faltantes DJE pendentes: %s", len(items))

    wanted = {str((i.get("extra") or {}).get("dje", {}).get("cnj20", "")) for i in items}
    wanted.discard("")
    LOGGER.info("Relendo o CSV bruto para %s CNJs...", len(wanted))
    full_rows = load_full_rows(Path(args.csv), wanted)
    LOGGER.info("Linhas completas localizadas: %s", len(full_rows))

    client = NotionSessoesClient(
        api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
        data_source_id=args.data_source_id,
        normalize_multiselect_colors_post_write=False,
    )
    schema = client.fetch_schema()

    dates = {i["data_sessao"] for i in items if i.get("data_sessao")}
    max_by_date = next_judgment_numbers(client, schema, dates)

    artifact_store = RunArtifacts(ARTIFACT_ROOT / "dje_import" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    artifact_store.root_dir.mkdir(parents=True, exist_ok=True)

    rows: list[PublishPreviewRow] = []
    row_items: list[dict] = []
    counters: dict[str, int] = {}
    for item in sorted(items, key=lambda x: (x.get("data_sessao", ""), x["id"])):
        record = (item.get("extra") or {}).get("dje", {})
        cnj20 = str(record.get("cnj20", ""))
        full = full_rows.get(cnj20)
        if not full:
            LOGGER.warning("Sem linha completa no CSV para %s; pulando.", record.get("numeroUnico"))
            continue
        data_sessao = item.get("data_sessao", "") or iso(full.get("dataDecisao", ""))
        ementa = clean_text(full.get("textoEmenta", ""))
        dispositivo = extract_dispositivo(full.get("textoDecisao", ""))
        resultado, votacao = resultado_votacao_from_dispositivo(dispositivo)
        tribunal, origem = tribunal_origem_from_cnj(cnj20, full.get("nomeMunicipio", ""), full.get("siglaUF", ""))
        counters[data_sessao] = counters.get(data_sessao, 0) + 1
        numero_julgamento = max_by_date.get(data_sessao, 0) + counters[data_sessao]

        video_dir = Path(item.get("artifact_dir", ""))
        video_id = item.get("video_id", "")
        timestamp = find_mention_timestamp(video_dir, cnj20) if video_dir.exists() else None
        youtube_link = f"https://www.youtube.com/watch?v={video_id}" + (f"&t={timestamp}" if timestamp else "")

        sigla_classe = clean_text(full.get("siglaClasse", ""))
        row = PublishPreviewRow(
            tema="",
            classe_processo=CLASSE_MAP.get(sigla_classe, sigla_classe or clean_text(full.get("descricaoClasse", ""))),
            tipo_registro=f"Julgamento {numero_julgamento}",
            eleicao=str(full.get("anoEleicao", "") or ""),
            origem=origem,
            tribunal=tribunal,
            numero_processo=format_cnj20(cnj20),
            youtube_link=youtube_link,
            relator=parse_relator(full.get("relatores", "") or full.get("relator", "")),
            resultado=resultado,
            votacao=votacao,
            data_sessao=data_sessao,
            analise_do_conteudo_juridico=ementa[:1900],
            raciocinio_juridico=dispositivo,
        )
        row.add_warning(
            "Importado do confronto DJE (CSV oficial); posição na pauta não verificada; "
            "partes/advogados/composição serão preenchidos pelo confronto oficial."
        )
        rows.append(row)
        row_items.append(item)

    if not args.skip_theme_enricher and rows:
        try:
            rows = enrich_preview_rows_with_theme_punchline(
                rows,
                api_key=get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY"),
                model=args.model,
                artifact_store=artifact_store,
                logger=LOGGER,
                notion_schema=schema,
            )
        except Exception as exc:
            LOGGER.warning("Enricher de tema falhou (%s); seguindo com fallback determinístico.", exc)

    rows = [validate_preview_row(row, schema) for row in rows]

    print()
    for row in rows:
        disposition, reasons = assess_row_publishability(row)
        print(
            f"{row.data_sessao} {row.tipo_registro:<14} {row.numero_processo:<28} "
            f"{row.classe_processo:<12} {row.relator:<30} {row.resultado:<20} "
            f"{row.votacao:<12} -> {disposition} {reasons if disposition != 'publish' else ''}"
        )
        print(f"    tema: {row.tema[:100]}")
    artifact_store.write_json("import_preview.json", [row.model_dump(mode="json") for row in rows])

    if not args.apply:
        print(f"\n[dry-run] Nada publicado. Preview em {artifact_store.root_dir}. Rode com --apply.")
        return 0

    results = publish_preview_rows(rows, client, schema)
    artifact_store.write_json("import_results.json", results)
    published_ids = []
    for item, result in zip(row_items, results[: len(row_items)]):
        status = result.get("status")
        print(f"[{status}] {result.get('numero_processo')} -> {result.get('url', '')}")
        if status in {"created", "updated"}:
            published_ids.append(str(item["id"]))
    if published_ids:
        vistoria_queue.update_status(published_ids, "published")
    print(f"\nPublicados: {len(published_ids)} de {len(row_items)}. Artifacts: {artifact_store.root_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
