# -*- coding: utf-8 -*-
"""Motor de casamento parte+data contra a jurisprudência do TSE (via UI headless).

Para cada caso do núcleo duro com nome de parte: navega na rota de pesquisa
(expressãoLivre = nome), captura o JSON do POST /public/pesquisa (o hCaptcha
invisível emite token por consulta) e salva hits brutos em JSONL retomável.

Uso: python sjur_ui_motor.py [--limit N] [--offset K]
"""
import json, os, re, sys, time, urllib.parse
from playwright.sync_api import sync_playwright

ART = r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria"
CAND = os.path.join(ART, "nucleo_duro_casamento_candidatos_20260821.json")
OUT = os.path.join(ART, "nucleo_duro_sjur_hits_20260821.jsonl")
MODO = "partes"  # "partes" | "tema"

limit = None
offset = 0
args = sys.argv[1:]
for i, a in enumerate(args):
    if a == "--limit":
        limit = int(args[i + 1])
    if a == "--offset":
        offset = int(args[i + 1])
    if a == "--cand":
        CAND = args[i + 1]
    if a == "--out":
        OUT = args[i + 1]
    if a == "--modo":
        MODO = args[i + 1]

casos = json.load(open(CAND, encoding="utf-8"))

feitos = set()
if os.path.exists(OUT):
    for ln in open(OUT, encoding="utf-8"):
        try:
            feitos.add(json.loads(ln)["page_id"])
        except Exception:
            pass

fila = [c for c in casos if c["page_id"] not in feitos][offset:]
if limit:
    fila = fila[:limit]
print(f"{len(casos)} casos, {len(feitos)} feitos, fila desta rodada: {len(fila)}", flush=True)
if not fila:
    sys.exit(0)


STOP = {"de", "da", "do", "das", "dos", "e", "em", "a", "o", "por", "para", "sobre",
        "com", "sem", "no", "na", "nos", "nas", "ao", "aos"}


def variacoes(caso):
    """Termos de pesquisa em ordem de precisão."""
    vs = []
    if MODO == "tema":
        tema = (caso.get("tema") or "").strip()
        # temas da base podem vir truncados no meio da palavra — descarta o último token
        toks = [t for t in re.findall(r"\w+", tema) if t.lower() not in STOP]
        if tema and len(tema) >= 55 and toks:
            toks = toks[:-1]
        if len(toks) >= 3:
            vs.append(" ".join(toks[:6]))
        if len(toks) >= 5:
            vs.append(" ".join(toks[:4]))
        return vs[:2]
    for p in caso["partes_uteis"][:2]:
        nome = re.sub(r"\s*\(.*?\)\s*", " ", p).strip()  # tira apelido entre parenteses
        nome = re.sub(r"\s+", " ", nome)
        toks = nome.split()
        if len(toks) >= 2:
            vs.append(nome)
        if len(toks) >= 4:
            vs.append(f"{toks[0]} {toks[-2]} {toks[-1]}")
    # dedup preservando ordem
    seen = set()
    out = []
    for v in vs:
        if v.lower() not in seen:
            seen.add(v.lower())
            out.append(v)
    return out[:3]


def compacta_hit(c):
    return {k: c.get(k) for k in (
        "siglaClasse", "numeroProcesso", "numeroUnicoFormatado", "numeroUnico",
        "dataDecisao", "anoEleicao", "siglaUF", "nomeMunicipio", "siglaTribunalJE",
        "descricaoTipoDecisao", "origemDecisao")} | {
        "partes": [f"{p.get('tipoParte','')}: {p.get('nomeParte','')}"
                   for p in (c.get("partes") or [])[:8]],
        "relatores": [r.get("nome") for r in (c.get("relatores") or [])[:3]],
    }


with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, channel="msedge")
    ctx = browser.new_context(locale="pt-BR")
    page = ctx.new_page()
    page.goto("https://jurisprudencia.tse.jus.br/", wait_until="networkidle", timeout=90000)
    time.sleep(2)

    falhas_robo_seguidas = 0
    fout = open(OUT, "a", encoding="utf-8")

    for idx, caso in enumerate(fila):
        registro = {"page_id": caso["page_id"], "data": caso["data"], "numero_base": caso["numero"],
                    "classe_base": caso.get("classe"), "origem_video": caso.get("origem"),
                    "partes_uteis": caso.get("partes_uteis"), "tema": caso.get("tema"),
                    "modo": MODO, "consultas": []}
        import datetime as _dt
        try:
            d = _dt.date.fromisoformat(caso["data"])
            dex = d.isoformat()
            d0, d1 = (d - _dt.timedelta(days=10)).isoformat(), (d + _dt.timedelta(days=10)).isoformat()
        except Exception:
            dex = d0 = d1 = None
        vs = variacoes(caso)
        plano = []  # (termo, janela) com janela None | (ini, fim)
        if vs:
            if MODO == "tema":
                # tema exige data: dia exato primeiro, depois ±10; nunca sem data
                if dex:
                    plano.append((vs[0], (dex, dex)))
                    plano.append((vs[0], (d0, d1)))
                    if len(vs) > 1:
                        plano.append((vs[1], (d0, d1)))
            else:
                if d0:
                    plano.append((vs[0], (d0, d1)))
                plano.append((vs[0], None))
                if len(vs) > 1 and d0:
                    plano.append((vs[1], (d0, d1)))
        for pi, (termo, janela) in enumerate(plano):
            com_data = janela is not None
            q = urllib.parse.quote(termo)
            url = (f"https://jurisprudencia.tse.jus.br/#/jurisprudencia/pesquisa?"
                   f"expressaoLivre={q}&params=s&cb={idx}x{pi}")
            if com_data:
                url += "&datas=" + urllib.parse.quote(f"Julgamento_{janela[0]}_{janela[1]}_")
            resposta = None
            try:
                with page.expect_response(
                        lambda r: "public/pesquisa" in r.url and r.request.method == "POST",
                        timeout=25000) as ev:
                    page.goto(url, timeout=40000)
                resposta = ev.value.json()
            except Exception as e:
                registro["consultas"].append({"termo": termo, "janela_datas": janela,
                                              "erro": str(e)[:200]})
                continue
            msg = resposta.get("mensagem")
            total = resposta.get("totalRegistros")
            if msg and "antirrob" in str(msg).lower():
                falhas_robo_seguidas += 1
                registro["consultas"].append({"termo": termo, "janela_datas": janela,
                                              "erro": "antirrobo"})
                print(f"  ANTIRROBO ({falhas_robo_seguidas} seguidas)", flush=True)
                if falhas_robo_seguidas >= 3:
                    print("3 falhas antirrobo seguidas — abortando rodada.", flush=True)
                    fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
                    fout.close()
                    browser.close()
                    sys.exit(5)
                time.sleep(20)
                continue
            falhas_robo_seguidas = 0
            hits = [compacta_hit(c) for c in (resposta.get("content") or [])[:25]]
            registro["consultas"].append({"termo": termo, "janela_datas": janela,
                                          "total": total, "hits": hits})
            if total and 0 < total <= 25:
                break  # resultado enxuto — suficiente p/ o casamento
            time.sleep(0.8)
        fout.write(json.dumps(registro, ensure_ascii=False) + "\n")
        fout.flush()
        tot_txt = "; ".join(f"{c.get('termo','?')[:35]}→{c.get('total', c.get('erro'))}"
                            for c in registro["consultas"])
        print(f"[{idx+1}/{len(fila)}] {caso['data']} {caso['numero'][:22]}: {tot_txt}", flush=True)
        time.sleep(0.5)

    fout.close()
    browser.close()
print("FIM", flush=True)


