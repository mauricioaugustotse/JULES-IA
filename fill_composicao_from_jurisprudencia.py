"""Preenche/corrige a coluna composicao a partir da FONTE OFICIAL: o campo "Composicao:
Ministros(as) X (presidente), Y, Z ... e W" no textoDecisao dos CSVs de jurisprudencia do TSE
(os mesmos que o watchdog ja processa para partes/advogados). Por PROCESSO (casa por CNJ-20),
entao reflete a composicao real que decidiu aquele caso — inclui o presidente e substitutos,
o que o Gemini (que devolvia o plenario titular) errava.

Casa cada nome com o conjunto canonico de ministros (ja vistos na base) por similaridade.
page-value PATCH (sem Playwright).

Uso:
  python fill_composicao_from_jurisprudencia.py --input-dir artifacts/jurisprudencia_csv          # dry-run
  python fill_composicao_from_jurisprudencia.py --input-dir artifacts/jurisprudencia_csv --apply
"""
from __future__ import annotations

import argparse, collections, csv, difflib, glob, json, logging, re, time, unicodedata
from datetime import datetime
from pathlib import Path

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import parse_multi_value_text
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

LOGGER = logging.getLogger("fill_composicao_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_composicao_csv"
# "Composicao [do julgamento]: Ministros (as) <lista>" ate o fim da frase. Captura tanto o formato
# antigo ("Composicao: Ministros...") quanto o moderno do TSE ("Composicao do julgamento: Ministros
# (as) ..."). NAO casa o trecho "Acompanharam o Relator os Ministros ..." (esse nao tem "composicao"),
# que lista so quem votou com o relator -> usar a COMPOSICAO OFICIAL evita a hiperinflacao do video.
COMP = re.compile(r"composi[çc][aã]o\s*(?:d[oa]\s+julgamento)?\s*:?\s*ministr[oa]s?\s*(?:\(as\)|\(os\))?\s*(.+?)(?:\.\s|\bsala\b|\bbras[ií]lia\b|$)", re.I)
csv.field_size_limit(10 * 1024 * 1024)


def fold(x: str) -> str:
    x = re.sub(r"^min\.?\s*", "", str(x or "").lower())
    x = unicodedata.normalize("NFKD", x)
    return re.sub(r"[^a-z ]", " ", "".join(c for c in x if not unicodedata.combining(c))).strip()


def digits(s: str) -> str:
    return re.sub(r"\D", "", str(s or ""))


def iso_date(s: str) -> str:
    s = str(s or "").strip()[:10]
    m = re.match(r"(\d{2})/(\d{2})/(\d{4})", s)  # DD/MM/YYYY (CSV)
    if m:
        return f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)  # YYYY-MM-DD (Notion)
    return s[:10] if m else ""


_CONN = {"de", "da", "do", "dos", "das", "e"}


def _toks(s: str) -> list[str]:
    return [t for t in re.sub(r"[^a-z ]", " ", fold(s)).split() if len(t) > 1 and t not in _CONN]


def _subseq(a: list[str], b: list[str]) -> bool:
    it = iter(b)
    return all(t in it for t in a)


def parse_ministros(blob: str, canon_fold: dict[str, str]) -> list[str]:
    out, seen = [], set()
    for raw in re.split(r",|\be\b", blob, flags=re.I):
        nm = re.sub(r"\(.*?\)", "", raw).strip(" .;-")  # tira (presidente)/(as)
        nmf = fold(nm)
        # rejeita lixo de texto de decisao que vaza ('em nao conhecer do recurso' etc.)
        if len(nmf) < 4 or re.search(r"\b(recurso|conhec|negar|prover|provim|agravo|embarg|voto|termos|unanim|maioria|julgam|sess|relator|presente|composi|provid|impro)\b", nmf):
            continue
        nm_tk = _toks(nm)
        best, br = None, 0.0
        for cf, canon in canon_fold.items():
            r = difflib.SequenceMatcher(None, nmf, cf).ratio()
            if nmf in cf or cf in nmf:
                r = max(r, 0.92)
            # canonico CURTO que e subsequencia do nome completo do CSV (Carlos Horbach <- Carlos
            # Bastide Horbach; Gilmar Mendes <- Gilmar Ferreira Mendes) -> casa, NAO cria opcao nova
            ctk = _toks(canon)
            if len(ctk) >= 2 and _subseq(ctk, nm_tk):
                r = max(r, 0.95)
            if r > br:
                best, br = canon, r
        chosen = best if (best and br >= 0.82) else ("Min. " + nm.title())
        if fold(chosen) not in seen:
            seen.add(fold(chosen)); out.append(chosen)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", default="artifacts/jurisprudencia_csv")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")

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

    # CSV: (CNJ-20, DATA) -> lista de ministros (UM processo tem varias decisoes em datas distintas;
    # so a decisao da MESMA data da sessao tem a composicao certa).
    csv_comp: dict[tuple, list[str]] = {}
    files = sorted(glob.glob(str(Path(args.input_dir) / "*.csv")))
    for fp in files:
        try:
            for r in csv.DictReader(open(fp, encoding="utf-8")):
                m = COMP.search(r.get("textoDecisao") or "")
                if not m:
                    continue
                cnj = digits(r.get("numeroUnico") or r.get("numeroProcesso"))
                iso = iso_date(r.get("dataDecisao"))
                if len(cnj) < 18 or not iso:
                    continue
                mins = parse_ministros(m.group(1), canon_fold)
                if 3 <= len(mins) <= 7:
                    csv_comp[(cnj[:20], iso)] = mins
        except Exception as exc:
            LOGGER.warning("erro CSV %s: %s", fp, str(exc)[:100])
    LOGGER.info("CSVs: %d | (CNJ,data) com composicao oficial: %d", len(files), len(csv_comp))

    # correcoes anteriores (p/ REVERTER as aplicadas com data errada na 1a passada por-CNJ)
    prior = {}
    prev = sorted(glob.glob(str(ARTIFACT_ROOT / "*" / "changes.json")))
    for fp in prev:
        try:
            for c in json.loads(Path(fp).read_text(encoding="utf-8")):
                prior.setdefault(c["cnj"], c)  # earliest = original 'old'
        except Exception:
            pass

    changes, applied, failed, iguais, revert = [], 0, 0, 0, 0
    for p in pages:
        cnj = digits(t(p, "numero_processo"))[:20]
        if len(cnj) < 18:
            continue
        iso = iso_date(t(p, "data_sessao"))
        cur = parse_multi_value_text(t(p, "composicao"))
        dm = csv_comp.get((cnj, iso)) if iso else None
        if dm:
            nova, acao = dm, "csv_data"
        elif cnj in prior and cur == prior[cnj]["new"] and prior[cnj]["new"] != prior[cnj]["old"]:
            nova, acao = prior[cnj]["old"], "reverter"; revert += 1  # desfaz a aplicacao com data errada
        else:
            continue
        if cur == nova:
            iguais += 1; continue
        changes.append({"cnj": cnj, "data": iso, "acao": acao, "old": cur, "new": nova})
        if args.apply:
            try:
                built = client._build_property_value(schema, "composicao", nova) or client._build_empty_property_value(schema, "composicao")
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": {"composicao": built}})
                applied += 1; time.sleep(0.1)
            except Exception as exc:
                failed += 1; LOGGER.warning("falha %s: %s", cnj, str(exc)[:100])

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=1), encoding="utf-8")
    csvd = sum(1 for c in changes if c["acao"] == "csv_data")
    LOGGER.info("RESUMO: %s | %s", json.dumps({"mode": "apply" if args.apply else "dry-run",
                "cnj_data_csv": len(csv_comp), "mudancas": len(changes), "por_csv_data": csvd,
                "revertidos": revert, "ja_iguais": iguais, "applied": applied, "falhas": failed}, ensure_ascii=False), run_dir)
    for c in changes[:6]:
        LOGGER.info("  [%s] %s: %s", c["acao"], c["cnj"], ", ".join(c["new"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
