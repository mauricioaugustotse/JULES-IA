# -*- coding: utf-8 -*-
"""Campos analíticos que só copiam o acórdão -> reescritos a partir do inteiro teor.

Até 27/09/2026 o import_dje_faltantes gravava a ementa crua em analise_do_conteudo_juridico
(ementa[:1900]) e o dispositivo cru em raciocinio_juridico. Este script acha essas páginas
na base sessões e as reescreve com o MESMO redator do import corrigido
(core.TeorAnaliseEnricher, OpenAI gpt-6-luna), que recusa cópia.

Etapas (cada uma grava um arquivo e a seguinte o lê — prévia obrigatória antes de aplicar):
  reescrever_analise_copiada.py --detectar [--retrato DIR] [--sem-baixar]
      baixa o retrato da base (properties + corpo; retomável) e grava candidatos_<STAMP>.json
  reescrever_analise_copiada.py --gerar --candidatos ARQ.json
      redige os campos copiados (gpt-6-luna) e grava propostas_<STAMP>.json — nada vai ao Notion
  reescrever_analise_copiada.py --aplicar --propostas ARQ.json
      grava no Notion as propostas com "aprovada" != false. Antes de cada PATCH relê a página
      e só grava se o campo ainda for o valor ANTES (backup em aplicar_<STAMP>.jsonl; retomável).

Fonte do teor: o corpo da página (heading "Inteiro teor ..."), desde que CONFIRA com a cópia —
há páginas cujo corpo traz o teor de OUTRO julgamento do mesmo processo (LT 0600662-03: página da
sessão de 13/08/2024 com o teor dos embargos de 26/05/2026). Sem teor no corpo, ou com teor
divergente, a própria cópia (ementa/dispositivo desta sessão) é o texto oficial e serve de fonte.
Campo analítico VAZIO em página com teor também entra (import cuja redação falhou).

A punchline entra junto: com a ementa no lugar da análise, o fallback do enricher de tema
montava "<1ª frase da ementa>. O desfecho registrado foi X." (às vezes um nome truncado). A
nova sai do enricher de tema/punchline do pipeline (core.ThemePunchlineEnricher, gpt-6-luna), já
alimentado com a análise redigida; tema bom é mantido.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, r"C:\Users\mauri\JULES-IA")
import tse_youtube_notion_core as core  # noqa: E402
from audit_notion_sessoes_round2 import notion_request_with_retry  # noqa: E402

ART = Path(r"C:\Users\mauri\JULES-IA\artifacts\notion_sessoes_auditoria")
STAMP = time.strftime("%Y%m%d_%H%M%S")
CAMPOS_RETRATO = [
    "numero_processo", "data_sessao", "tipo_registro", "classe_processo", "tema", "punchline",
    "analise_do_conteudo_juridico", "raciocinio_juridico", "resultado", "votacao", "relator", "origem",
    "tribunal", "partes", "pedido_vista", "eleicao",
]
CAMPOS = core.TEOR_ANALISE_CAMPOS
CAMPOS_REESCRITOS = (*CAMPOS, "punchline", "tema")
# --- retrato ------------------------------------------------------------------------------

class _Ritmo:
    """~2,6 req/s somados entre as threads: o teto do Notion é por INTEGRAÇÃO (~3 req/s)."""

    def __init__(self, por_segundo=2.6):
        self.intervalo = 1 / por_segundo
        self.ultimo = 0.0
        self.lock = threading.Lock()

    def aguarda(self):
        with self.lock:
            espera = self.ultimo + self.intervalo - time.monotonic()
            if espera > 0:
                time.sleep(espera)
            self.ultimo = time.monotonic()


def _texto_bloco(bloco):
    rich = (bloco.get(bloco.get("type")) or {}).get("rich_text")
    return "".join(r.get("plain_text", "") for r in rich) if rich else ""


def baixar_retrato(client, schema, pasta: Path):
    """props.json + corpos.jsonl (append-only: páginas já baixadas são puladas)."""
    pasta.mkdir(parents=True, exist_ok=True)
    pages = client.query_data_source()
    props = []
    for p in pages:
        rec = {"page_id": p["id"], "url": p.get("url"), "created_time": p.get("created_time")}
        for campo in CAMPOS_RETRATO:
            rec[campo] = client._extract_property_text(p, schema, campo) or ""
        props.append(rec)
    (pasta / "props.json").write_text(json.dumps(props, ensure_ascii=False, indent=1), encoding="utf-8")

    corpos = pasta / "corpos.jsonl"
    feitos = set()
    if corpos.exists():
        for ln in corpos.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            if "blocks" in rec:  # registro de erro não conta como baixado
                feitos.add(rec["page_id"])
    fila = [p["id"] for p in pages if p["id"] not in feitos]
    print(f"{len(pages)} páginas; corpos a baixar: {len(fila)}", flush=True)
    ritmo, lock = _Ritmo(), threading.Lock()

    def filhos(bid, prof=0):
        out, cursor = [], None
        while True:
            ritmo.aguarda()
            q = f"/blocks/{bid}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
            r = notion_request_with_retry(client, "GET", q)
            for b in r.get("results", []):
                out.append({"type": b["type"], "text": _texto_bloco(b), "depth": prof})
                if b.get("has_children") and prof < 2 and b["type"] not in ("child_page", "child_database"):
                    out.extend(filhos(b["id"], prof + 1))
            if not r.get("has_more"):
                return out
            cursor = r.get("next_cursor")

    def job(pid):
        try:
            rec = {"page_id": pid, "blocks": filhos(pid)}
        except Exception as exc:  # página que falhar fica fora de corpos.jsonl e volta na próxima
            print(f"  falha {pid}: {exc}", flush=True)
            return
        with lock, corpos.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(job, fila))


def ler_retrato(pasta: Path):
    props = json.loads((pasta / "props.json").read_text(encoding="utf-8"))
    corpos = {}
    for ln in (pasta / "corpos.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        if "blocks" in rec:
            corpos[rec["page_id"]] = rec["blocks"]
    return props, corpos


def teor_do_corpo(blocos) -> tuple[str, str]:
    """(ementa, decisao) da seção "Inteiro teor ..." do corpo; ("", "") se não houver."""
    ementa, decisao = [], []
    dentro, secao = False, "ementa"
    for bloco in blocos or []:
        texto = (bloco.get("text") or "").strip()
        tipo = bloco.get("type")
        if tipo in ("heading_1", "heading_2"):
            dentro = core.fold_text_for_match(texto).startswith("inteiro teor")
            secao = "ementa"
            continue
        if not dentro:
            continue
        if tipo == "heading_3":
            rotulo = core.fold_text_for_match(texto)
            if "decis" in rotulo or "acord" in rotulo:
                secao = "decisao"
            elif "ementa" in rotulo:
                secao = "ementa"
            continue
        if texto:
            (ementa if secao == "ementa" else decisao).append(texto)
    return " ".join(ementa), " ".join(decisao)


# --- detectar -----------------------------------------------------------------------------

def detectar(pasta: Path) -> Path:
    props, corpos = ler_retrato(pasta)
    candidatos, sem_corpo = [], 0
    for p in props:
        if p["page_id"] not in corpos:
            sem_corpo += 1
        ementa, decisao = teor_do_corpo(corpos.get(p["page_id"]))
        teor = f"{ementa} {decisao}".strip()
        diag = {}
        for campo in (*CAMPOS, "punchline"):
            valor = p.get(campo, "")
            diag[campo] = {
                "chars": len(valor),
                "forma": core.parece_copia_do_acordao(valor),
                "fracao": round(core.fracao_copiada_do_teor(valor, teor), 3) if teor else None,
                "copia": core.campo_analitico_copia_teor(valor, teor),
                "vazio": campo in CAMPOS and not valor.strip() and bool(teor),
            }
        diag["punchline"]["copia"] = core.punchline_copia_teor(
            p.get("punchline", ""), teor, diag["analise_do_conteudo_juridico"]["copia"])
        diag["tema"] = {"copia": core.tema_quebrado(p.get("tema", "")), "vazio": False}
        for campo in CAMPOS_REESCRITOS:
            diag[campo]["reescrever"] = diag[campo]["copia"] or diag[campo]["vazio"]
        if not any(diag[c]["reescrever"] for c in CAMPOS_REESCRITOS):
            continue
        # O teor do corpo confere se ao menos uma cópia analítica está nele (as do import ficam
        # em ~0,82-1,0); cópia só pela FORMA com fração baixa = corpo com teor de outro julgamento.
        copias = [diag[c]["fracao"] or 0.0 for c in CAMPOS if diag[c]["copia"]]
        teor_confere = bool(teor) and (not copias or max(copias) >= core.TEOR_COPIA_LIMIAR)
        candidatos.append({**{k: p.get(k, "") for k in ("page_id", "url", "created_time", *CAMPOS_RETRATO)},
                           "teor_ementa": ementa, "teor_decisao": decisao, "teor_confere": teor_confere,
                           "diag": diag})
    saida = ART / f"analise_copiada_candidatos_{STAMP}.json"
    saida.write_text(json.dumps(candidatos, ensure_ascii=False, indent=1), encoding="utf-8")
    por_campo = {c: sum(x["diag"][c]["copia"] for x in candidatos) for c in CAMPOS_REESCRITOS}
    vazios = {c: sum(x["diag"][c]["vazio"] for x in candidatos) for c in CAMPOS}
    divergentes = sum(1 for x in candidatos
                      if (x["teor_ementa"] + x["teor_decisao"]).strip() and not x["teor_confere"])
    print(f"{len(props)} páginas ({sem_corpo} sem corpo no retrato); candidatos: {len(candidatos)}; "
          f"cópias {por_campo}; vazios com teor {vazios}; corpo com teor de outro julgamento: {divergentes}")
    print(f"-> {saida}")
    return saida


# --- gerar --------------------------------------------------------------------------------

def _row(c) -> core.PublishPreviewRow:
    return core.PublishPreviewRow(
        numero_processo=c["numero_processo"], classe_processo=c["classe_processo"],
        data_sessao=c["data_sessao"][:10], tribunal=c["tribunal"], origem=c["origem"],
        eleicao=c["eleicao"], relator=c["relator"], pedido_vista=c["pedido_vista"],
        resultado=c["resultado"], votacao=c["votacao"], tema=c["tema"],
        partes=[x.strip() for x in c["partes"].split(",") if x.strip()],
        analise_do_conteudo_juridico=c["analise_do_conteudo_juridico"],
        raciocinio_juridico=c["raciocinio_juridico"],
    )


def _fonte(c) -> tuple[str, str]:
    if c["teor_confere"]:
        return c["teor_ementa"], c["teor_decisao"]
    # Sem teor no corpo, ou teor de outro julgamento: a cópia É o texto oficial desta sessão
    # (ementa/dispositivo) e vira a fonte.
    return tuple(c[campo] if c["diag"][campo]["copia"] else "" for campo in CAMPOS)


# Moldes do build_editorial_punchline_fallback: não servem de punchline nova.
_MOLDE_FALLBACK_RE = re.compile(r"\bO desfecho registrado foi\b|^A controvérsia levou o TSE a examinar\b")


def _tema_punchline(candidatos, novos, fontes, pasta: Path) -> dict[tuple[int, str], str]:
    """Tema/punchline novos pelo enricher do pipeline, sobre a análise JÁ redigida.

    Só o campo marcado é aproveitado (tema bom fica como está)."""
    indices = [i for i, c in enumerate(candidatos)
               if c["diag"]["punchline"]["reescrever"] or c["diag"]["tema"]["reescrever"]]
    if not indices:
        return {}
    base = []
    for i in indices:
        row = novos[i].model_copy(deep=True)
        for campo in ("punchline", "tema"):  # a cópia não volta como contexto
            if candidatos[i]["diag"][campo]["reescrever"]:
                setattr(row, campo, "")
        base.append(row)
    enricher = core.ThemePunchlineEnricher(artifact_store=core.RunArtifacts(pasta / "tema_punchline"))
    saida = {}
    fallback = {"punchline": core.PUNCHLINE_FALLBACK_WARNING, "tema": core.TEMA_FALLBACK_WARNING}
    for i, row in zip(indices, enricher.enrich_rows(base)):
        teor = " ".join(fontes[i])
        for campo in ("punchline", "tema"):
            valor = core.normalize_model_text(getattr(row, campo))
            # Só o texto que veio do MODELO serve: o fallback local remonta uma frase da análise.
            if (not candidatos[i]["diag"][campo]["reescrever"] or not valor
                    or fallback[campo] in row.warnings or _MOLDE_FALLBACK_RE.search(valor)):
                continue
            if campo == "punchline" and core.campo_analitico_copia_teor(valor, teor, proposta=True):
                continue
            if campo == "tema" and core.tema_quebrado(valor):
                continue
            saida[(i, campo)] = valor
    return saida


FATIA = 20  # páginas por fatia; cada fatia tem a sua pasta de cache (retomada estável)


def _gerar_fatia(fatia, pasta: Path, lote: int):
    enricher = core.TeorAnaliseEnricher(artifact_store=core.RunArtifacts(pasta), batch_size=lote,
                                        api_key=core.get_openai_api_key())
    rows = [_row(c) for c in fatia]
    fontes = [_fonte(c) for c in fatia]
    novos = enricher.enrich_rows(rows, fontes)
    return rows, fontes, novos, _tema_punchline(fatia, novos, fontes, pasta)


def gerar(arq_candidatos: Path, lote: int, workers: int) -> Path:
    candidatos = json.loads(arq_candidatos.read_text(encoding="utf-8"))
    # Pasta fixa por arquivo de candidatos: rodar --gerar de novo reaproveita o cache do modelo.
    pasta = ART / arq_candidatos.stem.replace("_candidatos_", "_redacao_")
    fatias = [candidatos[i:i + FATIA] for i in range(0, len(candidatos), FATIA)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        partes = list(ex.map(lambda kf: _gerar_fatia(kf[1], pasta / f"fatia_{kf[0]:02d}", lote), enumerate(fatias)))
    rows = [r for parte in partes for r in parte[0]]
    fontes = [f for parte in partes for f in parte[1]]
    novos = [n for parte in partes for n in parte[2]]
    tema_punchline = {(k * FATIA + i, campo): valor
                      for k, parte in enumerate(partes) for (i, campo), valor in parte[3].items()}
    propostas = []
    for i, (c, antes, depois, (ementa, decisao)) in enumerate(zip(candidatos, rows, novos, fontes)):
        teor = f"{ementa} {decisao}"
        for campo in CAMPOS_REESCRITOS:
            if not c["diag"][campo]["reescrever"]:
                continue
            if campo in ("punchline", "tema"):
                novo, antes_valor = tema_punchline.get((i, campo), ""), c[campo]
            else:
                novo, antes_valor = getattr(depois, campo), getattr(antes, campo)
            propostas.append({
                "page_id": c["page_id"], "url": c["url"], "numero_processo": c["numero_processo"],
                "data_sessao": c["data_sessao"][:10], "classe_processo": c["classe_processo"],
                "campo": campo, "antes": antes_valor, "depois": novo, "tema": c["tema"],
                "motivo": "copia" if c["diag"][campo]["copia"] else "vazio",
                "teor_divergente": bool((c["teor_ementa"] + c["teor_decisao"]).strip()) and not c["teor_confere"],
                **{k: c[k] for k in ("resultado", "votacao", "relator", "origem", "tribunal", "partes")},
                "fracao_depois": round(core.fracao_copiada_do_teor(novo, teor), 3) if novo else None,
                "avisos": [w for w in depois.warnings if campo in w or "fonte insuficiente" in w],
                "teor_ementa": ementa, "teor_decisao": decisao,
                "aprovada": bool(novo),
            })
    saida = ART / f"analise_copiada_propostas_{STAMP}.json"
    saida.write_text(json.dumps(propostas, ensure_ascii=False, indent=1), encoding="utf-8")
    vazias = sum(not p["depois"] for p in propostas)
    print(f"{len(propostas)} propostas ({vazias} sem texto aceito) -> {saida}")
    return saida


# --- aplicar ------------------------------------------------------------------------------

def aplicar(arq_propostas: Path, client, schema) -> None:
    propostas = [p for p in json.loads(arq_propostas.read_text(encoding="utf-8"))
                 if p.get("aprovada", True) and p.get("depois")]
    feitos = set()
    for log in ART.glob("analise_copiada_aplicar_*.jsonl"):
        for ln in log.read_text(encoding="utf-8").splitlines():
            rec = json.loads(ln)
            if rec.get("status") == "ok":
                feitos.add((rec["page_id"], rec["campo"]))
    por_pagina = {}
    for p in propostas:
        if (p["page_id"], p["campo"]) not in feitos:
            por_pagina.setdefault(p["page_id"], []).append(p)
    log = ART / f"analise_copiada_aplicar_{STAMP}.jsonl"
    print(f"{sum(map(len, por_pagina.values()))} campos em {len(por_pagina)} páginas a gravar -> {log}")
    ok = divergentes = falhas = 0
    for n, (page_id, itens) in enumerate(por_pagina.items(), 1):
        try:
            pagina = notion_request_with_retry(client, "GET", f"/pages/{page_id}")
            props = {}
            for p in itens:
                atual = client._extract_property_text(pagina, schema, p["campo"]) or ""
                if atual.strip() != p["antes"].strip():
                    divergentes += 1
                    _registra(log, p, "divergente", atual=atual)
                    continue
                props[p["campo"]] = client._build_property_value(schema, p["campo"], p["depois"])
            if props:
                notion_request_with_retry(client, "PATCH", f"/pages/{page_id}", json={"properties": props})
                for p in itens:
                    if p["campo"] in props:
                        ok += 1
                        _registra(log, p, "ok")
        except Exception as exc:
            falhas += 1
            for p in itens:
                _registra(log, p, "falha", erro=str(exc)[:300])
        if n % 50 == 0:
            print(f"  {n}/{len(por_pagina)} páginas", flush=True)
        time.sleep(0.35)
    print(f"gravados {ok}; divergentes {divergentes}; páginas com falha {falhas}")


def _registra(log: Path, p, status, **extra):
    rec = {"page_id": p["page_id"], "campo": p["campo"], "numero_processo": p["numero_processo"],
           "status": status, "antes": p["antes"], "depois": p["depois"], **extra}
    with log.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    etapa = ap.add_mutually_exclusive_group(required=True)
    etapa.add_argument("--detectar", action="store_true")
    etapa.add_argument("--gerar", action="store_true")
    etapa.add_argument("--aplicar", action="store_true")
    ap.add_argument("--retrato", type=Path, help="Pasta do retrato (retomada se já existir).")
    ap.add_argument("--sem-baixar", action="store_true", help="Detecta sobre o retrato como está, sem ir ao Notion.")
    ap.add_argument("--candidatos", type=Path)
    ap.add_argument("--propostas", type=Path)
    ap.add_argument("--lote", type=int, default=5, help="Itens por chamada ao modelo.")
    ap.add_argument("--workers", type=int, default=6, help="Fatias redigidas em paralelo (--gerar).")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.gerar:
        if not args.candidatos:
            ap.error("--gerar exige --candidatos")
        gerar(args.candidatos, args.lote, args.workers)
        return 0
    client = core.NotionSessoesClient(core.get_notion_api_key())
    schema = client.fetch_schema()
    if args.detectar:
        pasta = args.retrato or ART / f"analise_copiada_retrato_{STAMP}"
        if not args.sem_baixar:
            baixar_retrato(client, schema, pasta)  # retoma: só baixa corpos que faltam
        detectar(pasta)
        return 0
    if not args.propostas:
        ap.error("--aplicar exige --propostas")
    aplicar(args.propostas, client, schema)
    return 0


if __name__ == "__main__":
    sys.exit(main())
