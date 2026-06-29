"""Grava o INTEIRO TEOR do acordao (ementa + textoDecisao do CSV do DJE) no CORPO da pagina do
Notion, APENAS no registro do julgamento CONCLUSIVO de cada processo (votacao != Suspenso; havendo
varias, a de data_sessao mais recente).

Formatacao: heading_2 marcador + heading_3 "Ementa"/"Decisão / Acórdão" + paragrafos quebrados em
fim de FRASE (sem cortar palavras; <=1900 chars/bloco), espacos normalizados.

PARALELO (varios workers; 1 client por thread) e RETOMAVEL: por padrao PULA paginas que ja tem a
formatacao NOVA (heading_3 "Ementa"/"Decisão"). Sem --regravar, pula qualquer pagina com o marcador.

Uso:
  python fill_inteiro_teor.py --input-dir "<dir>"                                  # dry-run
  python fill_inteiro_teor.py --input-dir "<dir>" --apply                          # grava quem nao tem
  python fill_inteiro_teor.py --input-dir "<dir>" --apply --regravar               # regrava antigas (pula novas)
  python fill_inteiro_teor.py --input-dir "<dir>" --apply --regravar --workers 8
"""
from __future__ import annotations

import argparse, collections, csv, glob, json, logging, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import normalize_votacao
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)
LOGGER = logging.getLogger("fill_inteiro_teor")
ARTIFACT_ROOT = Path("artifacts") / "notion_inteiro_teor"
MARKER = "Inteiro teor (acórdão — DJE)"
MARKER_KEY = "inteiro teor"
HARDMAX = 1900
_tl = threading.local()
_progress = {"done": 0}
_lock = threading.Lock()


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def iso(d):
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


def norm_ws(s: str) -> str:
    s = str(s or "").replace("\r", " ").replace("\n", " ").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" +([,.;:)\]])", r"\1", s)
    s = re.sub(r"([(\[]) +", r"\1", s)
    return s.strip()


def to_paragraphs(text: str, target: int = 1000, hardmax: int = HARDMAX) -> list[str]:
    text = norm_ws(text)
    paras: list[str] = []
    while len(text) > hardmax:
        window = text[:hardmax]
        cut = window.rfind(". ")
        if cut < int(target * 0.5):
            cut = window.rfind(" ")
            cut = cut if cut > 0 else hardmax
        else:
            cut += 1
        paras.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        paras.append(text)
    return paras


def _p(text: str) -> dict:
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]}}


def _h(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {"object": "block", "type": key, key: {"rich_text": [{"type": "text", "text": {"content": text}}]}}


def build_blocks(ementa: str, decisao: str) -> list[dict]:
    blocks = [_h(2, MARKER)]
    if (ementa or "").strip():
        blocks.append(_h(3, "Ementa"))
        blocks += [_p(p) for p in to_paragraphs(ementa)]
    if (decisao or "").strip():
        blocks.append(_h(3, "Decisão / Acórdão"))
        blocks += [_p(p) for p in to_paragraphs(decisao)]
    return blocks


def _heading_text(b: dict) -> str:
    bt = b.get("type", "")
    rt = b.get(bt, {}).get("rich_text", [])
    return "".join(x.get("plain_text", "") or x.get("text", {}).get("content", "") for x in rt)


def marker_index(children: list[dict]):
    for i, b in enumerate(children):
        if b.get("type", "").startswith("heading") and MARKER_KEY in _heading_text(b).lower():
            return i
    return None


def tem_formato_novo(children: list[dict], idx: int) -> bool:
    for b in children[idx + 1:]:
        if b.get("type") == "heading_3":
            txt = _heading_text(b).lower()
            if "ementa" in txt or "decis" in txt or "acórd" in txt or "acord" in txt:
                return True
    return False


def get_client(api_key: str, dsid: str) -> NotionSessoesClient:
    if not hasattr(_tl, "client"):
        _tl.client = NotionSessoesClient(api_key=api_key, data_source_id=dsid)
    return _tl.client


def get_all_children(client, page_id: str) -> list[dict]:
    out: list[dict] = []
    cursor = None
    while True:
        path = f"/blocks/{page_id}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        resp = notion_request_with_retry(client, "GET", path)
        out.extend(resp.get("results", []))
        if not resp.get("has_more"):
            break
        cursor = resp.get("next_cursor")
    return out


def append_blocks(client, page_id: str, blocks: list[dict]) -> None:
    for i in range(0, len(blocks), 100):
        notion_request_with_retry(client, "PATCH", f"/blocks/{page_id}/children", json={"children": blocks[i:i + 100]})


def montar(ementa: str, decisao: str) -> tuple[str, str]:
    ementa = (ementa or "").strip()
    decisao = (decisao or "").strip()
    if ementa and ementa in decisao:
        ementa = ""
    return ementa, decisao


def processar(task: dict, api_key: str, dsid: str, regravar: bool, total: int) -> dict:
    cl = get_client(api_key, dsid)
    pid = task["page_id"]
    try:
        children = get_all_children(cl, pid)
        idx = marker_index(children)
        if idx is not None:
            if tem_formato_novo(children, idx):
                status = "ja_novo"
            elif not regravar:
                status = "ja_existia"
            else:
                for b in children[idx:]:
                    notion_request_with_retry(cl, "DELETE", f"/blocks/{b['id']}")
                append_blocks(cl, pid, build_blocks(task["ementa"], task["decisao"]))
                status = "regravado"
        else:
            append_blocks(cl, pid, build_blocks(task["ementa"], task["decisao"]))
            status = "gravado"
    except Exception as exc:
        status = "failed"; task["error"] = str(exc)
    with _lock:
        _progress["done"] += 1
        if _progress["done"] % 100 == 0:
            LOGGER.info("progresso: %s/%s", _progress["done"], total)
    task["status"] = status
    return task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", action="append", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--regravar", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    input_dirs = args.input_dir or [r"C:\Users\mauri\ProjetoConversor\dje_consolidado"]
    api_key = get_secret("NOTION_API_KEY", "NOTION_TOKEN")

    client = NotionSessoesClient(api_key=api_key, data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    by_cnj_date: dict[tuple, tuple] = {}
    best_by_cnj: dict[str, tuple] = {}
    files = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for fp in files:
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            cnj = digits(r.get("numeroUnico"))[:20]
            dt = iso(r.get("dataDecisao"))
            dec = r.get("textoDecisao") or ""
            ementa = r.get("textoEmenta") or ""
            if len(cnj) < 20 or not dec:
                continue
            if dt and len(dec) > len(by_cnj_date.get((cnj, dt), ("", ""))[1]):
                by_cnj_date[(cnj, dt)] = (ementa, dec)
            is_ac = 1 if re.search(r"ac[óo]rd[ãa]o", str(r.get("descricaoTipoDecisao") or ""), re.I) else 0
            rank = (is_ac, dt or "")
            if cnj not in best_by_cnj or rank > best_by_cnj[cnj][0]:
                best_by_cnj[cnj] = (rank, dt, ementa, dec)

    by_cnj: dict[str, list] = collections.defaultdict(list)
    for p in pages:
        cnj = digits(t(p, "numero_processo"))[:20]
        if len(cnj) >= 20:
            by_cnj[cnj].append(p)

    tasks: list[dict] = []
    stats = collections.Counter()
    for cnj, grp in by_cnj.items():
        nao_susp = [p for p in grp if normalize_votacao(t(p, "votacao")) not in ("Suspenso", "Suspenso*")]
        if not nao_susp:
            stats["sem_conclusivo"] += 1
            continue
        conclusivo = max(nao_susp, key=lambda p: (t(p, "data_sessao") or "")[:10])
        data = (t(conclusivo, "data_sessao") or "")[:10]
        fonte = by_cnj_date.get((cnj, data)); via = "cnj+data"
        if not fonte:
            b = best_by_cnj.get(cnj)
            if b:
                fonte = (b[2], b[3]); via = "acordao_final_cnj"
        if not fonte:
            stats["sem_acordao_no_csv"] += 1
            continue
        ementa, decisao = montar(fonte[0], fonte[1])
        if not decisao and not ementa:
            continue
        tasks.append({"page_id": conclusivo["id"], "numero": t(conclusivo, "numero_processo"), "data": data, "via": via, "ementa": ementa, "decisao": decisao})

    LOGGER.info("CSVs: %s | conclusivos com acordao: %s | workers: %s", len(files), len(tasks), args.workers)
    results: list[dict] = []
    if args.apply:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(lambda tk: processar(tk, api_key, args.data_source_id, args.regravar, len(tasks)), tasks))
    else:
        stats["a_gravar"] = len(tasks)
        results = tasks

    for r in results:
        if r.get("status"):
            stats[r["status"]] += 1
    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps([{k: v for k, v in r.items() if k not in ("ementa", "decisao")} for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "regravar": args.regravar, **dict(stats)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
