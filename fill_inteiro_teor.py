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


# abreviações jurídicas que NÃO encerram frase — nunca cortar parágrafo após elas
_ABREV = {
    "n", "nº", "no", "ns", "art", "arts", "inc", "incs", "al", "als", "par", "fl", "fls",
    "p", "pp", "pag", "pags", "min", "dr", "dra", "rel", "j", "c", "cf", "cfr", "obs",
    "ex", "vol", "ed", "op", "cit", "et", "ss", "seg", "segs", "id", "ids", "res",
}


def _corte_valido(text: str, cut: int) -> bool:
    """cut aponta para o '.' de um '. ' candidato; válido se o token anterior não é abreviação."""
    prev = re.search(r"([A-Za-zÀ-ÿº§]+)\.$", text[: cut + 1])
    if prev and prev.group(1).lower().rstrip("º") in _ABREV:
        return False
    # nunca deixar marcador de item numerado órfão no fim do bloco ("... 4.")
    if re.search(r"(?:^|\s)\d{1,2}\.$", text[: cut + 1]):
        return False
    seguinte = text[cut + 2: cut + 3]
    if seguinte and seguinte.islower():
        return False
    return True


def segmenta_semantico(text: str) -> list[str]:
    """Divide o teor em parágrafos: um por item numerado ('1. Trata-se...').

    Quebra antes de 'N. Maiúscula' quando precedido de fim de sentença — inclusive
    grudado ('APROVAÇÃO.1. Trata-se'). Não confunde com 'n. 23.585' (exige espaço
    após o ponto do item) nem com datas/artigos.
    """
    text = norm_ws(text)
    # RÓTULO de seção grudado: "eleições.CONCLUSÃOPedido indeferido" ->
    #   "eleições." / "CONCLUSÃO" / "Pedido indeferido".
    # Guarda: só quando o rótulo é seguido de texto em caixa MISTA (Maiúscula+minúscula).
    # Isso impede fragmentar ementas, que são inteiras em CAIXA ALTA
    # ("... RECURSO ESPECIAL. AÇÃO DE INVESTIGAÇÃO ... DECISÃO. PROVA ...").
    _ROT = (r"CONCLUS[ÃA]O|DISPOSITIVO|RELAT[ÓO]RIO|EXTRATO DA ATA|"
            r"VOTO(?:\s*-?\s*VISTA|\s+VENCIDO)?|TESES? FIXADAS?")
    text = re.sub(rf"(?<=[\wà-ÿÀ-Ú)\]])\.\s*(?=(?:{_ROT})[A-ZÀ-Ú][a-zà-ÿ])", ".\n\n", text)
    text = re.sub(rf"\b({_ROT})(?=[A-ZÀ-Ú][a-zà-ÿ])", r"\1\n\n", text)
    # metadados do acórdão em parágrafo próprio
    text = re.sub(r"(?<=[.;!?])\s+(?=Composi[çc][ãa]o(?: do julgamento)?:)", "\n\n", text)
    # SUBitem grudado no fim da frase: "R$ 124.913,03.6.1. O fato" -> quebra antes de "6.1."
    text = re.sub(r"(?<=[\wà-ÿÀ-Ú)\]])\.(?=\d{1,2}\.\d{1,2}\.\s+[A-ZÀ-Ú])", ". ", text)
    text = re.sub(r"(?<=[.;:!?])\s+(?=\d{1,2}\.\d{1,2}\.\s+[A-ZÀ-Ú])", "\n\n", text)
    # ementa estruturada: seção romana grudada "DIPLOMAÇÃO.I. QUESTÃO" -> quebra antes de "I. TÍTULO"
    text = re.sub(r"(?<=[\wà-ÿÀ-Ú)\]])\.(?=[IVX]{1,4}\.\s*[A-ZÀ-Ú]{2,})", ". ", text)
    text = re.sub(r"(?<=[.;:!?])\s+(?=[IVX]{1,4}\.\s*[A-ZÀ-Ú]{2,})", "\n\n", text)
    # título em CAIXA grudado no item numerado: "ORDEM1. Questão" -> quebra antes de "1. Questão"
    text = re.sub(r"(?<=[A-ZÀ-Ú])(?=\d{1,2}\.\s+[A-ZÀ-Ú])", "\n\n", text)
    # separa o caso grudado "APROVAÇÃO.1. Trata" -> "APROVAÇÃO. 1. Trata"
    text = re.sub(r"(?<=[A-Za-zÀ-ÿ)\]”\"])\.(?=\d{1,2}\.\s+[A-ZÀ-Ú])", ". ", text)
    # quebra de parágrafo antes de item numerado precedido de fim de sentença
    text = re.sub(r"(?<=[.;:!?])\s+(?=\d{1,2}\.\s+[A-ZÀ-Ú])", "\n\n", text)
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def to_paragraphs(text: str, target: int = 1000, hardmax: int = HARDMAX) -> list[str]:
    text = norm_ws(text)
    paras: list[str] = []
    while len(text) > hardmax:
        window = text[:hardmax]
        cut = window.rfind(". ")
        while cut >= int(target * 0.5) and not _corte_valido(text, cut):
            cut = window.rfind(". ", 0, cut)
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


def _paragrafos_finais(texto: str) -> list[str]:
    """Segmentação semântica (itens numerados) + subdivisão por tamanho."""
    out: list[str] = []
    for seg in segmenta_semantico(texto):
        out.extend(to_paragraphs(seg))
    return out


def build_blocks(ementa: str, decisao: str, marker: str = MARKER) -> list[dict]:
    blocks = [_h(2, marker)]
    if (ementa or "").strip():
        blocks.append(_h(3, "Ementa"))
        blocks += [_p(p) for p in _paragrafos_finais(ementa)]
    if (decisao or "").strip():
        blocks.append(_h(3, "Decisão / Acórdão"))
        blocks += [_p(p) for p in _paragrafos_finais(decisao)]
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


_ABREV_FIM = re.compile(r"\b(n|nº|no|ns|arts?|incs?|als?|fls?|p|pp|min|dr|dra|rel|j|c|cf|res|id|ids)\.$", re.I)
_DATA_ASSINATURA = re.compile(r"Bras[íi]lia(?:/DF)?,?\s+(\d{1,2})[ºo]?\s+de\s+(\w+)\s+de\s+(\d{4})", re.I)
_MESES = {"janeiro": 1, "fevereiro": 2, "março": 3, "marco": 3, "abril": 4, "maio": 5, "junho": 6,
          "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}


def _paragrafos_da_secao(children: list[dict], idx: int) -> list[str]:
    """Textos dos parágrafos após o marker (até o fim ou próximo heading_2)."""
    out = []
    for b in children[idx + 1:]:
        bt = b.get("type", "")
        if bt == "heading_2":
            break
        if bt == "paragraph":
            out.append(_heading_text(b))
    return out


def _data_assinatura_iso(texto: str) -> str:
    m = None
    for m in _DATA_ASSINATURA.finditer(texto):
        pass
    if not m:
        return ""
    mes = _MESES.get(m.group(2).lower())
    if not mes:
        return ""
    return f"{m.group(3)}-{mes:02d}-{int(m.group(1)):02d}"


def sanear(task: dict, api_key: str, dsid: str, total: int) -> dict:
    """Regrava seções com fonte trocada ou corte em abreviação; flaga suspeitos sem fonte."""
    cl = get_client(api_key, dsid)
    pid = task["page_id"]
    try:
        children = get_all_children(cl, pid)
        idx = marker_index(children)
        tem_fonte = bool((task.get("decisao") or "").strip() or (task.get("ementa") or "").strip())
        if idx is None:
            if tem_fonte:
                append_blocks(cl, pid, build_blocks(task["ementa"], task["decisao"]))
                status = "gravado_faltante"
            else:
                status = "sem_teor_sem_fonte"  # fila SJUR
        else:
            paras = _paragrafos_da_secao(children, idx)
            gravado = norm_ws(" ".join(paras))
            if tem_fonte:
                fonte_txt = norm_ws(" ".join(x for x in (task["ementa"], task["decisao"]) if x))
                mesmo_conteudo = gravado[:300] == fonte_txt[:300] and gravado[-200:] == fonte_txt[-200:]
                paras_novos = _paragrafos_finais(task["ementa"]) + _paragrafos_finais(task["decisao"])
                mesma_forma = [p.strip() for p in paras if p.strip()] == paras_novos
                if not mesmo_conteudo or not mesma_forma:
                    for b in children[idx:]:
                        notion_request_with_retry(cl, "DELETE", f"/blocks/{b['id']}")
                    append_blocks(cl, pid, build_blocks(task["ementa"], task["decisao"]))
                    status = "regravado_fonte" if not mesmo_conteudo else "regravado_formato"
                else:
                    status = "ja_ok"
            else:
                assin = _data_assinatura_iso(gravado)
                ds = task.get("data") or ""
                if assin and ds:
                    try:
                        delta = abs((datetime.strptime(assin, "%Y-%m-%d") - datetime.strptime(ds, "%Y-%m-%d")).days)
                    except ValueError:
                        delta = 0
                    status = "teor_suspeito_sem_fonte" if delta > 90 else "sem_fonte_data_ok"
                    task["data_assinatura"] = assin
                else:
                    status = "sem_fonte_sem_assinatura"
    except Exception as exc:
        status = "failed"; task["error"] = str(exc)
    with _lock:
        _progress["done"] += 1
        if _progress["done"] % 100 == 0:
            LOGGER.info("progresso: %s/%s", _progress["done"], total)
    task["status"] = status
    return task


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
    ap.add_argument("--sanear", action="store_true", help="regrava fonte trocada/quebras em abreviação; flaga teor suspeito sem fonte")
    ap.add_argument("--max", type=int, default=0, help="limita o número de tasks (piloto)")
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

    cnjs_da_base = {digits(t(p, "numero_processo"))[:20] for p in pages}
    cnjs_da_base = {c for c in cnjs_da_base if len(c) >= 20}

    by_cnj_date: dict[tuple, tuple] = {}
    all_by_cnj: dict[str, list] = collections.defaultdict(list)
    files = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for fp in files:
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            cnj = digits(r.get("numeroUnico"))[:20]
            if cnj not in cnjs_da_base:
                continue
            dt = iso(r.get("dataDecisao"))
            dec = r.get("textoDecisao") or ""
            ementa = r.get("textoEmenta") or ""
            if len(cnj) < 20 or not dec:
                continue
            if dt and len(dec) > len(by_cnj_date.get((cnj, dt), ("", ""))[1]):
                by_cnj_date[(cnj, dt)] = (ementa, dec)
            is_ac = 1 if re.search(r"ac[óo]rd[ãa]o", str(r.get("descricaoTipoDecisao") or ""), re.I) else 0
            all_by_cnj[cnj].append((dt, is_ac, ementa, dec))


    def fonte_compativel(cnj: str, data_sessao: str):
        """Decisão do CNJ mais próxima da sessão dentro de [-5, +60] dias; preferir acórdão.

        NUNCA cai na decisão mais recente do processo: decisão de outra fase
        (cumprimento de sentença etc.) NÃO é o acórdão da sessão retratada.
        """
        exato = by_cnj_date.get((cnj, data_sessao))
        if exato:
            return exato, "cnj+data"
        if not data_sessao:
            return None, ""
        try:
            base = datetime.strptime(data_sessao, "%Y-%m-%d")
        except ValueError:
            return None, ""
        melhor = None
        for dt, is_ac, ementa, dec in all_by_cnj.get(cnj, []):
            if not dt:
                continue
            try:
                delta = (datetime.strptime(dt, "%Y-%m-%d") - base).days
            except ValueError:
                continue
            if not (-5 <= delta <= 60):
                continue
            rank = (is_ac, -abs(delta), len(dec))
            if melhor is None or rank > melhor[0]:
                melhor = (rank, (ementa, dec))
        if melhor:
            return melhor[1], "cnj+data_proxima"
        return None, ""

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
        fonte, via = fonte_compativel(cnj, data)
        if not fonte:
            stats["sem_acordao_compativel_no_csv"] += 1
            continue
        ementa, decisao = montar(fonte[0], fonte[1])
        if not decisao and not ementa:
            continue
        tasks.append({"page_id": conclusivo["id"], "numero": t(conclusivo, "numero_processo"), "data": data, "via": via, "ementa": ementa, "decisao": decisao})

    if args.sanear:
        # inclui também conclusivos SEM fonte compatível (para flagar teor suspeito/faltante)
        com_fonte = {tk["page_id"] for tk in tasks}
        for cnj, grp in by_cnj.items():
            nao_susp = [p for p in grp if normalize_votacao(t(p, "votacao")) not in ("Suspenso", "Suspenso*")]
            if not nao_susp:
                continue
            conclusivo = max(nao_susp, key=lambda p: (t(p, "data_sessao") or "")[:10])
            if conclusivo["id"] in com_fonte:
                continue
            tasks.append({"page_id": conclusivo["id"], "numero": t(conclusivo, "numero_processo"),
                          "data": (t(conclusivo, "data_sessao") or "")[:10], "via": "sem_fonte",
                          "ementa": "", "decisao": ""})

    if args.max:
        tasks = tasks[: args.max]
    LOGGER.info("CSVs: %s | tasks: %s | workers: %s | modo: %s", len(files), len(tasks), args.workers,
                "sanear" if args.sanear else ("apply" if args.apply else "dry-run"))
    results: list[dict] = []
    if args.sanear:
        if args.apply:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(lambda tk: sanear(tk, api_key, args.data_source_id, len(tasks)), tasks))
        else:
            stats["a_sanear"] = len(tasks)
            results = tasks
    elif args.apply:
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
