"""BACKFILL (nao faz parte do fluxo normal): recupera os julgamentos SEM numero do backlog
de artefatos do backfill_2025, cujos vIdeos usam o formato de PLAYLIST (02_judgment_NN.json
= JudgmentBundleExtraction, sem 04_preview_rows). Para cada video: assembla o AnalysisResult
(bundles = os 02_judgment; session = data/composicao dos itens), chama build_preview_rows
(reusa TODA a normalizacao do pipeline), filtra SEM numero + assess 'publish', casa por URL
EXATA (idempotente) e publica via publish_preview_rows.

Uso:
  python recover_numberless_from_backlog.py --backlog-root "H:\\Meu Drive\\TSE_YOUTUBE_NOTION_BACKLOG\\backfill_2025"            # dry-run
  python recover_numberless_from_backlog.py --backlog-root "..." --apply
  (--limit N processa só os N primeiros vIdeos; util p/ teste rapido)
"""
from __future__ import annotations

import argparse, glob, json, logging, os, re
from datetime import datetime
from pathlib import Path

from local_secrets import get_secret
from tse_youtube_notion_core import (
    DEFAULT_NOTION_DATA_SOURCE_ID, AnalysisResult, JudgmentBundleExtraction, NotionSessoesClient,
    PublishPreviewRow, SessionExtraction, assess_row_publishability, build_preview_rows,
    canonicalize_numero_processo, publish_preview_rows,
)

LOGGER = logging.getLogger("recover_numberless_from_backlog")
ARTIFACT_ROOT = Path("artifacts") / "recover_numberless_backlog"


def video_id_from_dir(name: str) -> str:
    m = re.match(r"\d+_(.+)$", name)
    return m.group(1) if m else name


def assemble_analysis(vdir: str) -> AnalysisResult | None:
    """Assembla SO os itens SEM numero na fonte (numero_processo sem digitos). Os itens com
    numero curto (ex.: 'REspe 47736') TEM numero e ja foram publicados -> NAO recuperar
    (seria duplicata). A composicao/data da sessao e derivada de QUALQUER item (p/ fallback)."""
    raw: list[JudgmentBundleExtraction] = []
    for bp in sorted(glob.glob(os.path.join(vdir, "02_judgment_*.json"))):
        try:
            raw.append(JudgmentBundleExtraction.model_validate(json.load(open(bp, encoding="utf-8"))))
        except Exception:
            continue
    if not raw:
        return None
    data_sessao, composicao = "", []
    for b in raw:
        for it in b.items:
            if not data_sessao and it.data_sessao:
                data_sessao = it.data_sessao
            if not composicao and it.composicao:
                composicao = list(it.composicao)
    bundles: list[JudgmentBundleExtraction] = []
    for b in raw:
        nb = b.model_copy(update={"items": [it for it in b.items if not re.sub(r"\D", "", it.numero_processo or "")]})
        if nb.items:
            bundles.append(nb)
    if not bundles:
        return None
    return AnalysisResult(session=SessionExtraction(data_sessao=data_sessao, composicao=composicao), bundles=bundles)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backlog-root", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude-suspenso", action="store_true",
                    help="Nao publica 'Suspenso por vista' sem numero; guarda em held_suspenso.json p/ a fase Suspenso.")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"),
                                 data_source_id=args.data_source_id,
                                 normalize_multiselect_colors_post_write=False)
    schema = client.fetch_schema()

    root = args.backlog_root
    playlists = sorted(d for d in os.listdir(root) if re.match(r"\d{4}_PL", d) and os.path.isdir(os.path.join(root, d)))
    all_rows: list[PublishPreviewRow] = []
    nvid = 0
    for pl in playlists:
        pld = os.path.join(root, pl)
        for vname in sorted(os.listdir(pld)):
            vdir = os.path.join(pld, vname)
            if not os.path.isdir(vdir) or not glob.glob(os.path.join(vdir, "02_judgment_*.json")):
                continue
            analysis = assemble_analysis(vdir)
            if analysis is None:
                continue
            nvid += 1
            url = f"https://www.youtube.com/watch?v={video_id_from_dir(vname)}"
            try:
                all_rows.extend(build_preview_rows(analysis, url, schema, None))  # client=None: sem query por video
            except Exception as exc:
                LOGGER.warning("falha em %s: %s", vname, exc)
            if args.limit and nvid >= args.limit:
                break
        if args.limit and nvid >= args.limit:
            break

    candidatos: list[PublishPreviewRow] = []
    for row in all_rows:
        if canonicalize_numero_processo(row.numero_processo):
            continue
        disp, _ = assess_row_publishability(row)
        if disp == "publish" and row.youtube_link:
            candidatos.append(row)
    LOGGER.info("videos=%s rows=%s | SEM numero publicaveis=%s", nvid, len(all_rows), len(candidatos))

    suspenso = [r for r in candidatos if (r.resultado or "") == "Suspenso por vista" or (r.votacao or "") in {"Suspenso", "Suspenso*"}]
    if args.exclude_suspenso:
        held_ids = {id(r) for r in suspenso}
        candidatos = [r for r in candidatos if id(r) not in held_ids]
        LOGGER.info("Suspenso sem numero GUARDADOS: %s | publicaveis agora: %s", len(suspenso), len(candidatos))

    cria = atualiza = 0
    for row in candidatos:
        match = client.find_existing_row(schema, youtube_link=row.youtube_link, numero_processo="")
        if match:
            row.page_id = match.page_id; row.action = "update"; atualiza += 1
        else:
            row.action = "create"; cria += 1
    LOGGER.info("idempotencia: %s CRIA, %s ATUALIZA", cria, atualiza)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    sample = [{"acao": r.action, "classe": r.classe_processo, "data": r.data_sessao, "tema": r.tema[:60],
               "res": r.resultado, "vot": r.votacao, "link": r.youtube_link} for r in candidatos]
    (run_dir / "candidatos.json").write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "held_suspenso.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in suspenso], ensure_ascii=False, indent=2), encoding="utf-8")

    if not args.apply:
        summary = {"mode": "dry-run", "videos": nvid, "rows": len(all_rows), "candidatos": len(candidatos),
                   "cria": cria, "atualiza": atualiza}
        (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("DRY-RUN. RESUMO: %s | %s", json.dumps(summary, ensure_ascii=False), run_dir)
        return 0

    results = publish_preview_rows(candidatos, client, schema)
    cr = sum(1 for r in results if r.get("status") == "created")
    up = sum(1 for r in results if r.get("status") == "updated")
    bl = sum(1 for r in results if r.get("status") == "blocked")
    sk = sum(1 for r in results if r.get("status") == "skipped")
    summary = {"mode": "apply", "videos": nvid, "candidatos": len(candidatos),
               "created": cr, "updated": up, "blocked": bl, "skipped": sk}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("APLICADO: %s", json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
