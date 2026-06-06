"""Diagnostico de UF da origem usando o TR (codigo do tribunal regional) embutido no CNJ-20.
O CNJ NNNNNNN-DD.AAAA.J.TR.OOOO tem TR = d[14:16] = o TRE de origem. Construimos o consenso
TR->UF a partir da PROPRIA base (a maioria das origens esta certa) e flagamos os casos cuja
UF da origem diverge da UF-consenso do seu TR -> provavel erro de UF do Gemini (ex.: 'Brasilia/DF'
num CNJ TR=25=SE). NAO corrige (precisa do municipio certo de outra fonte) — so DIAGNOSTICA.

Uso: python origem_uf_check.py
"""
from __future__ import annotations

import collections, json, re
from datetime import datetime
from pathlib import Path

from local_secrets import get_secret
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

UF_RE = re.compile(r"/([A-Z]{2})\s*$")


def main() -> int:
    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=DEFAULT_NOTION_DATA_SOURCE_ID)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    by_tr: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    rows = []
    for p in pages:
        d = re.sub(r"\D", "", t(p, "numero_processo") or "")
        if len(d) < 20:
            continue
        tr = d[14:16]
        origem = (t(p, "origem") or "").strip()
        m = UF_RE.search(origem.upper())
        uf = m.group(1) if m else ""
        if not uf:
            continue
        by_tr[tr][uf] += 1
        rows.append({"page_id": p["id"], "cnj": t(p, "numero_processo"), "tr": tr, "uf": uf, "origem": origem})

    # consenso TR->UF (so confiavel: >=5 casos e consenso >=70%)
    tr_uf = {}
    for tr, ctr in by_tr.items():
        total = sum(ctr.values())
        top, n = ctr.most_common(1)[0]
        if total >= 5 and n / total >= 0.70:
            tr_uf[tr] = (top, total, n)

    flag = [r for r in rows if r["tr"] in tr_uf and r["uf"] != tr_uf[r["tr"]][0]]
    print(f"CNJ-20 com UF: {len(rows)} | TRs com consenso: {len(tr_uf)} | FLAGADOS (UF != consenso TR): {len(flag)}")
    print("\nconsenso TR->UF (amostra):")
    for tr, (uf, tot, n) in sorted(tr_uf.items())[:12]:
        print(f"  TR={tr} -> {uf} ({n}/{tot})")
    print("\nFLAGADOS (origem UF diverge do TR) — amostra:")
    by_pair = collections.Counter((r["uf"], tr_uf[r["tr"]][0]) for r in flag)
    for (uf_base, uf_tr), c in by_pair.most_common(15):
        print(f"  {c}x  origem-UF={uf_base}  mas TR indica {uf_tr}")

    run_dir = Path("artifacts") / "notion_origem_uf_check" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "flagados.json").write_text(json.dumps(flag, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "tr_uf.json").write_text(json.dumps({k: v[0] for k, v in tr_uf.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nartefato: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
