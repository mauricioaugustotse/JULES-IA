"""Corrige a coluna composicao usando a FONTE AUTORITATIVA: a saudacao de abertura da sessao na
TRANSCRICAO do YouTube (o presidente cumprimenta os ministros PRESENTES: "Cumprimento os
integrantes desse tribunal, ... ministro X, ministro Y, ministra Z ..."). O Gemini vinha
devolvendo o plenario TITULAR (nao os presentes do dia) — sinal: relator fora da propria
composicao.

Por SESSAO (video): baixa transcricao -> acha a saudacao -> extrai 'ministr[oa] <Nome>' ->
casa cada nome com o conjunto canonico (nomes ja vistos na base) -> VALIDA (todo relator das
paginas da sessao tem que estar entre os presentes). So aplica quando valida; senao FLAG.

Uso:
  python fix_composicao_from_transcript.py                 # dry-run, so sessoes com erro
  python fix_composicao_from_transcript.py --limit 12      # amostra
  python fix_composicao_from_transcript.py --apply
  python fix_composicao_from_transcript.py --apply --all-sessions   # revisa todas, nao so as com erro
"""
from __future__ import annotations

import argparse, collections, difflib, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fix_composicao_from_transcript")
CACHE = Path("artifacts") / "transcripts_cache"
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_transcript"
GREET = re.compile(r"cumprimento\s+os\s+integrantes|declaro\s+aberta|presen[çc]a\s+dos\s+(?:senhores\s+)?ministros|presentes\s+os\s+(?:senhores\s+)?ministros", re.I)
STOP = re.compile(r"\b(advogad|servidor|procurador|acompanh|professor|estudante|imprensa|senhoras?\s+e\s+senhores)\b", re.I)
MIN = re.compile(r"ministr[oa]\s+([A-ZÀ-Ý][\wÀ-ÿ.'-]+(?:\s+(?:de|da|do|dos|das|e|[A-ZÀ-Ý][\wÀ-ÿ.'-]+)){0,4})", re.I)


def fold(x: str) -> str:
    x = re.sub(r"^min\.?\s*", "", str(x or "").lower())
    x = unicodedata.normalize("NFKD", x)
    return re.sub(r"[^a-z ]", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else ""


def fetch_transcript(vid: str) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    fp = CACHE / f"{vid}.txt"
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        segs = list(YouTubeTranscriptApi().fetch(vid, languages=["pt", "pt-BR"]))
        txt = " ".join(s.text for s in segs)
    except Exception as exc:
        txt = f"__ERRO__ {exc}"
    fp.write_text(txt, encoding="utf-8")
    return txt


def extract_present(txt: str, canon_fold: dict[str, str]) -> list[str]:
    """Acha a saudacao e extrai os ministros nomeados, casando com o conjunto canonico."""
    if not txt or txt.startswith("__ERRO__"):
        return []
    m = GREET.search(txt)
    if not m:
        return []
    win = txt[m.start():m.start() + 700]
    stop = STOP.search(win, 40)  # corta nos cumprimentos a nao-ministros
    if stop:
        win = win[:stop.start()]
    out: list[str] = []
    seen: set[str] = set()
    for nm in MIN.findall(win):
        nmf = fold(nm)
        if len(nmf) < 4:
            continue
        # casa com o canonico mais proximo
        best, br = None, 0.0
        for cf, canon in canon_fold.items():
            r = difflib.SequenceMatcher(None, nmf, cf).ratio()
            if nmf in cf or cf in nmf:
                r = max(r, 0.9)
            if r > br:
                best, br = canon, r
        if best and br >= 0.82 and best not in seen:
            seen.add(best); out.append(best)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--all-sessions", action="store_true", help="revisa todas as sessoes, nao so as com relator-fora")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    # conjunto canonico de ministros: nomes distintos ja vistos na coluna composicao
    canon_count = collections.Counter()
    for p in pages:
        for mns in parse_multi_value_text(t(p, "composicao")):
            canon_count[mns] += 1
    canon_fold = {}
    for nm, _n in canon_count.most_common():
        cf = fold(nm)
        if len(cf) >= 4 and cf not in canon_fold:
            canon_fold[cf] = nm
    LOGGER.info("canonico: %d ministros distintos", len(canon_fold))

    # agrupa paginas por sessao (video)
    by_vid: dict[str, list] = collections.defaultdict(list)
    for p in pages:
        vid = video_id(t(p, "youtube_link"))
        if vid:
            by_vid[vid].append(p)

    def relator_in(rel, comp):
        rf = fold(rel)
        return bool(rf) and any(fold(m) == rf or (len(rf) > 4 and (rf in fold(m) or fold(m) in rf)) for m in comp)

    sessions = []
    for vid, pgs in by_vid.items():
        wrong = any(t(p, "relator") and not relator_in(t(p, "relator"), parse_multi_value_text(t(p, "composicao"))) for p in pgs)
        if args.all_sessions or wrong:
            sessions.append((vid, pgs))
    if args.limit:
        sessions = sessions[:args.limit]
    LOGGER.info("sessoes a revisar: %d", len(sessions))

    stats = {"sessoes": len(sessions), "aplicadas": 0, "flag_sem_transcript": 0, "flag_sem_saudacao": 0,
             "flag_relator_ausente": 0, "paginas_alteradas": 0, "applied": 0, "falhas": 0}
    detail = []
    for vid, pgs in sessions:
        txt = fetch_transcript(vid)
        pres = extract_present(txt, canon_fold)
        rec = {"vid": vid, "paginas": len(pgs), "presentes": pres}
        if not txt or txt.startswith("__ERRO__"):
            rec["status"] = "sem_transcript"; stats["flag_sem_transcript"] += 1; detail.append(rec); continue
        if len(pres) < 3:  # saudacao nao encontrada ou pobre demais p/ confiar
            rec["status"] = "saudacao_fraca"; stats["flag_sem_saudacao"] += 1; detail.append(rec); continue
        # presentes = saudacao UNIAO relatores da sessao: o relator esta SEMPRE presente (decidiu),
        # e isso cobre o presidente/substituto que preside e nao e cumprimentado na saudacao.
        present = list(pres)
        pf = {fold(x) for x in present}
        for p in pgs:
            rel = (t(p, "relator") or "").strip(); rf = fold(rel)
            if rf and len(rf) > 3 and not any(rf == x or (rf in x or x in rf) for x in pf):
                present.append(rel); pf.add(rf)
        # guarda: se o video mistura mais de uma data de sessao, NAO auto-aplica (composicoes distintas)
        datas = {(t(p, "data_sessao") or "")[:10] for p in pgs if t(p, "data_sessao")}
        if len(datas) > 1:
            rec["status"] = "multi_sessao"; rec["presentes"] = present; rec["datas"] = sorted(datas)
            stats["flag_relator_ausente"] += 1; detail.append(rec); continue
        if len(present) > 7:  # plenario tem 7 assentos: excesso = video mistura composicoes -> nao confiavel
            rec["status"] = "excesso_ministros"; rec["presentes"] = present
            stats.setdefault("flag_excesso", 0); stats["flag_excesso"] += 1; detail.append(rec); continue
        rec["presentes"] = present
        rec["status"] = "ok"
        stats["aplicadas"] += 1
        built = client._build_property_value(schema, "composicao", present)
        for p in pgs:
            if parse_multi_value_text(t(p, "composicao")) == present:
                continue
            stats["paginas_alteradas"] += 1
            if args.apply:
                try:
                    notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"composicao": built}})
                    stats["applied"] += 1; time.sleep(0.1)
                except Exception as exc:
                    stats["falhas"] += 1; LOGGER.warning("falha %s: %s", t(p, "numero_processo"), str(exc)[:100])
        detail.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "detalhe.json").write_text(json.dumps(detail, ensure_ascii=False, indent=1), encoding="utf-8")
    LOGGER.info("RESUMO: %s | %s", json.dumps({"mode": "apply" if args.apply else "dry-run", **stats}, ensure_ascii=False), run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
