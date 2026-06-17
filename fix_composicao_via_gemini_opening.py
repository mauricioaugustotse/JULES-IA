"""FALLBACK ECONOMICO: corrige composicao dos casos sem CSV-na-data-certa e sem transcricao,
re-rodando o Gemini APENAS no trecho de ABERTURA da sessao (o presidente apregoa/cumprimenta
nominalmente os ministros presentes — chamada de presenca informal).

ECONOMIA: clipa so [0, --end-seconds] do video (videoMetadata), --fps baixissimo (a chamada e
falada), modelo flash-lite, CACHE por video (re-rodar nao re-cobra), --max-videos p/ limitar.
Casa por sessao (1 video = 1 sessao = 1 data) e so toca paginas onde o relator esta na lista
extraida (validacao). Aplica via page-value (sem Playwright).

Uso (sempre medir custo num lote pequeno primeiro):
  python fix_composicao_via_gemini_opening.py --max-videos 3            # dry-run + custo
  python fix_composicao_via_gemini_opening.py --max-videos 3 --apply
  python fix_composicao_via_gemini_opening.py --apply                   # todos os pendentes
"""
from __future__ import annotations

import argparse, collections, difflib, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import (DEFAULT_GEMINI_MODEL, DEFAULT_NOTION_DATA_SOURCE_ID,
                                     NotionSessoesClient, call_gemini_generate_content_rest)

LOGGER = logging.getLogger("fix_composicao_via_gemini_opening")
CACHE = Path("artifacts") / "gemini_composicao_opening"
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_gemini"
SYS = ("Voce analisa os minutos INICIAIS de uma sessao plenaria do TSE. Na abertura o presidente "
       "declara aberta a sessao e CUMPRIMENTA nominalmente os ministros PRESENTES naquele dia (uma "
       "chamada de presenca). Extraia APENAS os nomes dos ministros presentes citados nessa saudacao "
       "inicial (use tambem as placas/legendas com os nomes que aparecem na tela). Ignore vinhetas/"
       "propaganda institucional e ministros citados dentro de processos/votos. Se nao houver saudacao "
       "no trecho, retorne lista vazia.")
PROMPT = "Liste os nomes dos ministros presentes citados na saudacao de abertura desta sessao."


class _Resp(BaseModel):
    ministros_presentes: list[str] = []


def fold(x: str) -> str:
    x = re.sub(r"^min\.?\s*", "", str(x or "").lower())
    x = unicodedata.normalize("NFKD", x)
    return re.sub(r"[^a-z ]", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def video_id(url: str) -> str:
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", url or "")
    return m.group(1) if m else ""


def gemini_opening(url: str, key: str, start: int, end: int, fps: float, model: str) -> tuple[list[str], int]:
    import requests
    from tse_youtube_notion_core import GEMINI_REST_BASE_URL, _extract_generate_content_text
    part = {"fileData": {"fileUri": url, "mimeType": "video/*"},
            "videoMetadata": {"startOffset": f"{int(start)}s", "endOffset": f"{int(end)}s", "fps": fps}}
    payload = {"contents": [{"parts": [part, {"text": PROMPT}]}],
               "systemInstruction": {"parts": [{"text": SYS}]},
               "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"}}
    r = requests.post(f"{GEMINI_REST_BASE_URL}/models/{model}:generateContent?key={key}",
                      json=payload, timeout=(10, 240))
    if r.status_code >= 400:
        raise RuntimeError(f"REST {r.status_code}: {r.text[:200]}")
    pj = r.json()
    text = _extract_generate_content_text(pj)
    toks = int((pj.get("usageMetadata") or {}).get("totalTokenCount", 0) or 0)
    names: list = []
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            names = obj.get("ministros_presentes") or obj.get("ministros") or next((v for v in obj.values() if isinstance(v, list)), [])
        elif isinstance(obj, list):
            names = obj
    except Exception:
        names = re.findall(r'"([^"]{4,40})"', text)
    out = []
    for n in names:
        if isinstance(n, dict):
            n = n.get("nome") or n.get("name") or ""
        if isinstance(n, str) and len(n.strip()) >= 4:
            out.append(n.strip())
    return out, toks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--start-seconds", type=int, default=0)
    ap.add_argument("--end-seconds", type=int, default=480, help="cobre saudacao apos as vinhetas")
    ap.add_argument("--fps", type=float, default=0.25, help="capta as placas com os nomes (visual)")
    ap.add_argument("--retries", type=int, default=2, help="Gemini-video e nao-deterministico: re-tenta no vazio")
    ap.add_argument("--model", default=DEFAULT_GEMINI_MODEL, help="flash-lite acerta a saudacao (validado jun/2026, 6/6); p/ maxima limpeza em caso dificil use gemini-2.5-pro (NAO existe gemini-3.1-flash)")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    CACHE.mkdir(parents=True, exist_ok=True)
    key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    canon_count = collections.Counter()
    for p in pages:
        for mns in parse_multi_value_text(t(p, "composicao")):
            canon_count[mns] += 1
    canon_fold = {}
    for nm, _n in canon_count.most_common():
        cf = fold(nm)
        if len(cf) >= 4 and cf not in canon_fold:
            canon_fold[cf] = nm

    def match_canon(nm: str) -> str | None:
        nmf = fold(nm)
        if len(nmf) < 4:
            return None
        best, br = None, 0.0
        for cf, canon in canon_fold.items():
            r = difflib.SequenceMatcher(None, nmf, cf).ratio()
            if nmf in cf or cf in nmf:
                r = max(r, 0.92)
            if r > br:
                best, br = canon, r
        return best if br >= 0.82 else ("Min. " + nm.strip().title())

    def relator_in(rel, comp):
        rf = fold(rel)
        return bool(rf) and any(fold(m) == rf or (len(rf) > 4 and (rf in fold(m) or fold(m) in rf)) for m in comp)

    # so as sessoes (videos) com >=1 pagina relator-fora
    by_vid: dict[str, list] = collections.defaultdict(list)
    for p in pages:
        vid = video_id(t(p, "youtube_link"))
        if vid:
            by_vid[vid].append(p)
    pend = [(vid, pgs) for vid, pgs in by_vid.items()
            if any(t(p, "relator") and not relator_in(t(p, "relator"), parse_multi_value_text(t(p, "composicao"))) for p in pgs)]
    if args.max_videos:
        pend = pend[:args.max_videos]
    LOGGER.info("videos pendentes a processar: %d (end=%ss fps=%s modelo=%s)", len(pend), args.end_seconds, args.fps, DEFAULT_GEMINI_MODEL)

    stats = {"videos": len(pend), "ok": 0, "vazio": 0, "relator_ausente": 0, "multi_sessao": 0,
             "paginas": 0, "applied": 0, "falhas": 0, "tokens": 0, "chamadas_gemini": 0}
    detail = []
    for vid, pgs in pend:
        url = f"https://www.youtube.com/watch?v={vid}"  # URL limpa (sem &t=, que quebra o fetch do Gemini)
        cf = CACHE / f"{vid}_{args.start_seconds}_{args.end_seconds}.json"
        if cf.exists():
            roll = json.loads(cf.read_text(encoding="utf-8")).get("roll", [])
        else:
            names: list = []
            for _att in range(max(1, args.retries)):  # nao-determinismo do Gemini-video: re-tenta no vazio
                try:
                    names, toks = gemini_opening(url, key, args.start_seconds, args.end_seconds, args.fps, args.model)
                    stats["tokens"] += toks; stats["chamadas_gemini"] += 1
                except Exception as exc:
                    LOGGER.warning("Gemini falhou %s: %s", vid, str(exc)[:160]); stats["falhas"] += 1; names = []; break
                if names:
                    break
                time.sleep(0.4)
            roll = []
            for nm in names:
                c = match_canon(nm)
                if c and c not in roll:
                    roll.append(c)
            if roll:  # so cacheia ACERTO; vazio fica sem cache p/ re-tentar em runs futuros
                cf.write_text(json.dumps({"vid": vid, "names": names, "roll": roll}, ensure_ascii=False), encoding="utf-8")
            time.sleep(0.2)
        rec = {"vid": vid, "roll": roll}
        if len(roll) < 2:
            rec["status"] = "vazio"; stats["vazio"] += 1; detail.append(rec); continue
        datas = {(t(p, "data_sessao") or "")[:10] for p in pgs if t(p, "data_sessao")}
        if len(datas) > 1:
            rec["status"] = "multi_sessao"; stats["multi_sessao"] += 1; detail.append(rec); continue
        present = list(roll)
        pf = {fold(x) for x in present}
        for p in pgs:
            rel = (t(p, "relator") or "").strip(); rf = fold(rel)
            if rf and len(rf) > 3 and not any(rf == x or rf in x or x in rf for x in pf):
                present.append(rel); pf.add(rf)
        if len(present) > 7:
            rec["status"] = "excesso"; rec["present"] = present; stats["relator_ausente"] += 1; detail.append(rec); continue
        rec["status"] = "ok"; rec["present"] = present; stats["ok"] += 1
        built = client._build_property_value(schema, "composicao", present)
        for p in pgs:
            if parse_multi_value_text(t(p, "composicao")) == present:
                continue
            stats["paginas"] += 1
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
    if stats["chamadas_gemini"]:
        LOGGER.info("CUSTO: %d tokens em %d chamadas (~%d tok/video)", stats["tokens"], stats["chamadas_gemini"],
                    stats["tokens"] // max(1, stats["chamadas_gemini"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
