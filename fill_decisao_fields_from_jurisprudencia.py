"""Confronta/corrige `votacao` e `resultado` extraindo do DISPOSITIVO do acordao oficial
(textoDecisao do CSV do DJE), casando por (CNJ-20, dataDecisao == data_sessao) — a decisao
DAQUELE dia (um processo tem varios julgamentos em datas distintas).

Dispositivo: "O Tribunal, por unanimidade/por maioria, desproveu/negou provimento/deu provimento/
deu parcial provimento/nao conheceu ..., nos termos do voto...".

votacao: se ha "maioria" em qualquer ponto do dispositivo -> Por maioria; senao "unanim" -> Unânime.
         NAO sobrescreve base 'Suspenso'/'Suspenso*' (marca de suspensao).
resultado: 1o verbo-desfecho do dispositivo -> termo canonico (CTA tratada a parte). Classifica:
  - vazio_preenchido: base vazia -> seguro.
  - erro_classe: o resultado ATUAL da base nao pertence ao vocabulario valido da classe (erro
    inequivoco) -> seguro corrigir.
  - divergente: base valida p/ a classe mas != CSV (ambos plausiveis; decisoes com varias partes
    podem ter nuance) -> SO aplica com --apply-divergentes; senao fica em relatorio p/ revisao.

pedido_vista NAO entra: o acordao final nao traz o pedido de vista do TSE de forma extraivel
(mencoes a "vista" sao de TRE/historico) -> confronto via CSV e inocuo.

Uso:
  python fill_decisao_fields_from_jurisprudencia.py --input-dir "<dir>"                       # dry-run
  python fill_decisao_fields_from_jurisprudencia.py --input-dir "<dir>" --apply               # vazios + erro_classe
  python fill_decisao_fields_from_jurisprudencia.py --input-dir "<dir>" --apply --apply-divergentes
"""
from __future__ import annotations

import argparse, collections, csv, glob, json, logging, re, time
from datetime import datetime
from pathlib import Path
from typing import Any

from audit_notion_sessoes_round2 import notion_request_with_retry
from local_secrets import get_secret
from tse_normalization import (
    normalize_class_text,
    normalize_classe_processo,
    normalize_resultado_final,
    normalize_votacao,
)
from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID, NotionSessoesClient

csv.field_size_limit(50 * 1024 * 1024)
LOGGER = logging.getLogger("fill_decisao_fields_from_jurisprudencia")
ARTIFACT_ROOT = Path("artifacts") / "notion_decisao_fields"
COLS = ("votacao", "resultado")

DISP_RE = re.compile(
    r"O Tribunal,?\s+(por unanimidade|por maioria|à unanimidade|por aclama\w+)\s*,?\s*(.*?)"
    r"(?:\bnos termos\b|\bvencid|\bComposi[çc][aã]o\b|\bRedigir|\bSala\b|\bBras[ií]lia\b|\.\s|$)",
    re.IGNORECASE | re.DOTALL,
)


def digits(s):
    return re.sub(r"\D", "", str(s or ""))


def iso(d):
    m = re.match(r"\s*(\d{1,2})/(\d{1,2})/(\d{4})", str(d or ""))
    return f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}" if m else ""


# vocabulario de resultado por classe (o CSV de canonizacao do projeto nao esta presente).
_RES_RECURSO = {"Provido", "Desprovido", "Provido em parte", "Não conhecido", "Prejudicado"}
_RES_ACAO = {"Procedente", "Procedente em parte", "Improcedente", "Não conhecido", "Prejudicado"}
_RES_ORDEM = {"Concedido", "Denegado", "Não conhecido", "Prejudicado", "Provido", "Desprovido", "Provido em parte"}
_RES_PET = {"Deferido", "Indeferido", "Prejudicado", "Não conhecido", "Procedente", "Improcedente", "Provido", "Desprovido"}
_RES_BY_CLASSE = {
    "CTA": {"Aprovada", "Não conhecido"},
    "PC": {"Aprovada", "Aprovada com ressalvas", "Rejeitada", "Não conhecido", "Prejudicado"},
    "Rp": _RES_ACAO,
    "Representação": _RES_ACAO,
    "AIJE": _RES_ACAO,
    "AC": _RES_ACAO | {"Deferido", "Indeferido", "Referendada", "Provido", "Desprovido"},
    "AR": _RES_ACAO,
    "Rcl": _RES_ACAO,
    "RCED": _RES_ACAO | {"Provido", "Desprovido", "Provido em parte"},
    "RPP": {"Deferido", "Indeferido", "Prejudicado", "Não conhecido"},
    "RCand": {"Deferido", "Indeferido", "Prejudicado", "Não conhecido"},
    "HC": _RES_ORDEM,
    "AgR-HC": _RES_ORDEM,
    "AgRg-HC": _RES_ORDEM,
    "MS": _RES_ORDEM,
    "AgRg-MS": _RES_ORDEM,
    "Lista Tríplice": {"Aprovada", "Acolhidos", "Deferido", "Indeferido", "Prejudicado", "Não conhecido"},
    "PetCiv": _RES_PET,
    "PET": _RES_PET,
    "Petição": _RES_PET,
    "PA": {"Aprovada", "Deferido", "Indeferido", "Referendada", "Prejudicado", "Procedente", "Improcedente", "Não conhecido", "Homologada"},
    "Czer": {"Aprovada", "Deferido", "Homologada"},
    "QO": {"Acolhidos", "Prejudicado", "Provido", "Desprovido", "Provido em parte", "Não conhecido"},
    "Ref-TutCautAnt": {"Referendada", "Deferido", "Indeferido", "Prejudicado"},
    "Ref.-MS": {"Referendada", "Deferido", "Indeferido", "Prejudicado"},
    "TutCautAnt": {"Deferido", "Indeferido", "Referendada", "Prejudicado", "Procedente", "Improcedente"},
}


def _allowed_resultados(classe_canon: str):
    if classe_canon in _RES_BY_CLASSE:
        return _RES_BY_CLASSE[classe_canon]
    if classe_canon.startswith(("AgRg-", "AgR-", "ED-")) or classe_canon in {"REspe", "AREspe", "RO", "RHC", "RMS", "RvE"}:
        return _RES_RECURSO
    return None


def _erro_de_classe(cur: str, classe_canon: str) -> bool:
    allowed = _allowed_resultados(classe_canon)
    if not allowed or not cur:
        return False
    return normalize_class_text(cur) not in {normalize_class_text(a) for a in allowed}


# (padrao regex, resultado canonico) — escolhe o PRIMEIRO desfecho na ORDEM DO TEXTO (nao por
# prioridade): assim "julgou improcedente a representacao E prejudicado o agravo" -> Improcedente
# (o 'prejudicado' acessorio do agravo fica por ultimo). 'preliminar' nunca e desfecho.
_RESULT_PATTERNS = [
    (r"parcial provimento|provimento parcial|deu(?:-lhe)? parcial provimento|prove\w* parcial|provid\w* em parte|parcialmente provid", "Provido em parte"),
    (r"negou(?:-lhe)? provimento|negar provimento|desprove\w|desprovid|nao prove\w", "Desprovido"),
    (r"deu(?:-lhe)? provimento|dar provimento", "Provido"),
    (r"nao conhec", "Não conhecido"),
    (r"procedente em parte|parcialmente procedente|proced[eê]ncia parcial", "Procedente em parte"),
    (r"improcedente", "Improcedente"),
    (r"\bproceden", "Procedente"),
    (r"concedeu (?:a )?(?:ordem|seguranca)|concedeu de oficio|concedida a (?:ordem|seguranca)|concedeu parcial", "Concedido"),
    (r"denegou|denegada a seguranca|denegou a (?:ordem|seguranca)", "Denegado"),
    (r"referend", "Referendada"),
    (r"homolog", "Homologada"),
    (r"aprov\w+ com ressalv", "Aprovada com ressalvas"),
    (r"desaprov|rejeit\w+ (?:as contas|o recurso|os embargos|o agravo|o pedido|a representacao|a a[cç][aã]o)", "Rejeitada"),
    (r"\baprov", "Aprovada"),
    (r"acolhi", "Acolhidos"),
    (r"indeferi", "Indeferido"),
    (r"\bdeferi", "Deferido"),
    (r"prejudicad", "Prejudicado"),
]


def _fam(s: str) -> str:
    """Forma de comparacao insensivel a genero/numero (rejeitados==rejeitada, nao conhecida==nao
    conhecido) para nao gerar mudancas puramente cosmeticas."""
    n = normalize_class_text(s)
    return re.sub(r"(rejeitad|acolhid|provid|deferid|indeferid|conhecid|aprovad|prejudicad|desaprovad)[oa]s?", r"\1", n)


def votacao_de(disp_full: str) -> str:
    dn = normalize_class_text(disp_full)
    if "maioria" in dn or re.search(r"\bvencid", dn):
        return "Por maioria"
    if "unanim" in dn:
        return "Unânime"
    return ""


def resultado_de(disp: str, classe: str) -> str:
    d = normalize_class_text(disp)
    classe_canon = normalize_classe_processo(classe) if classe else ""
    if classe_canon == "CTA":
        # consulta: respondida -> Aprovada; so nao conhecida -> Não conhecido
        if re.search(r"respond|respondeu|respondida", d):
            base = "Aprovada"
        elif re.search(r"nao conhec", d):
            base = "Não conhecido"
        else:
            base = ""
        return (normalize_resultado_final(base, classe) or base) if base else ""
    # contas aprovadas COM RESSALVAS (ressalva pode nao ser adjacente a "aprovou")
    if re.search(r"ressalv", d) and re.search(r"aprov", d):
        return normalize_resultado_final("Aprovada com ressalvas", classe) or "Aprovada com ressalvas"
    # demais classes: PRIMEIRO desfecho na ORDEM DO TEXTO, IGNORANDO 'prejudicado' quando ha outro
    # desfecho (o 'prejudicado' costuma ser de agravo/incidente acessorio). So vira Prejudicado
    # quando e o UNICO desfecho ("julgou prejudicada a acao cautelar").
    best_pos, best = 10 ** 9, ""
    for pat, val in _RESULT_PATTERNS:
        if val == "Prejudicado":
            continue
        m = re.search(pat, d)
        if m and m.start() < best_pos:
            best_pos, best = m.start(), val
    if not best and re.search(r"prejudicad", d):
        best = "Prejudicado"
    if not best:
        return ""
    nf = normalize_resultado_final(best, classe)
    return nf or best


def _eh_claro(disp: str, classe_canon: str) -> bool:
    """Dispositivo 'claro/seguro' p/ aplicar divergência: NÃO é CTA (consulta parcial é ambígua) e
    tem UM ÚNICO desfecho de mérito (exclui 'Prejudicado' acessório da contagem). Múltiplos
    desfechos (várias partes / agravo acessório provido + recurso) => NÃO é claro."""
    if classe_canon == "CTA":
        return False
    d = normalize_class_text(disp)
    # acessorios que enganam o desfecho principal -> deixa p/ revisao manual
    if re.search(r"ressalv|homolog|amicus|ingresso|assistente|agravo interno|agravo regimental|embarg", d):
        return False
    vals = set()
    for pat, val in _RESULT_PATTERNS:
        if val == "Prejudicado":
            continue
        if re.search(pat, d):
            vals.add(val)
    return len(vals) <= 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", action="append", default=None)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--apply-divergentes", action="store_true", help="Aplica tambem as divergencias plausiveis (base valida p/ a classe).")
    ap.add_argument("--apply-claros", action="store_true", help="Aplica os divergentes CLAROS (dispositivo com 1 unico desfecho, exceto CTA).")
    ap.add_argument("--data-source-id", default=DEFAULT_NOTION_DATA_SOURCE_ID)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    input_dirs = args.input_dir or [r"C:\Users\mauri\ProjetoConversor\dje_consolidado"]

    client = NotionSessoesClient(api_key=get_secret("NOTION_API_KEY", "NOTION_TOKEN"), data_source_id=args.data_source_id)
    schema = client.fetch_schema()
    pages = client.query_data_source()

    def t(p, f):
        return client._extract_property_text(p, schema, f)

    decis: dict[tuple, str] = {}
    files = []
    for d in input_dirs:
        files.extend(glob.glob(str(Path(d) / "*.csv")))
    for fp in files:
        for r in csv.DictReader(open(fp, encoding="utf-8-sig")):
            cnj = digits(r.get("numeroUnico"))[:20]
            dt = iso(r.get("dataDecisao"))
            td = r.get("textoDecisao") or ""
            if len(cnj) < 20 or not dt or not td:
                continue
            if len(td) > len(decis.get((cnj, dt), "")):
                decis[(cnj, dt)] = td
    LOGGER.info("CSVs: %s | (cnj,data) com textoDecisao: %s", len(files), len(decis))

    changes: list[dict[str, Any]] = []
    stats: dict[str, Any] = {"match": 0, "applied": 0, "failed": 0}
    for c in COLS:
        stats[c] = {"iguais": 0, "vazio_preenchido": 0, "erro_classe": 0, "divergente_valido": 0, "divergente_incerto": 0, "suspenso_mantido": 0}

    for p in pages:
        cnj = digits(t(p, "numero_processo"))[:20]
        data = (t(p, "data_sessao") or "")[:10]
        if len(cnj) < 20 or not data:
            continue
        td = decis.get((cnj, data))
        if not td:
            continue
        m = DISP_RE.search(td)
        if not m:
            continue
        stats["match"] += 1
        classe = t(p, "classe_processo")
        classe_canon = normalize_classe_processo(classe) if classe else ""
        disp = m.group(2) or ""
        novo = {"votacao": votacao_de((m.group(1) or "") + " " + disp),
                "resultado": resultado_de(disp, classe)}

        props: dict[str, Any] = {}
        detail: dict[str, Any] = {}
        for col in COLS:
            val = novo[col]
            if not val:
                continue
            cur_raw = t(p, col)
            cur = normalize_votacao(cur_raw) if col == "votacao" else normalize_resultado_final(cur_raw, classe)
            if _fam(cur) == _fam(val):
                stats[col]["iguais"] += 1
                continue
            # nao sobrescreve marca de suspensao da base
            if re.search(r"suspens", normalize_class_text(cur_raw)):
                stats[col]["suspenso_mantido"] += 1
                continue
            if not cur_raw.strip():
                kind = "vazio_preenchido"
            elif col == "votacao":
                kind = "erro_classe"  # votacao e binaria (Unânime/Maioria); divergencia = erro
            elif col == "resultado" and _erro_de_classe(cur, classe_canon):
                kind = "erro_classe"
            elif col == "resultado" and normalize_class_text(cur) == "prejudicado" and normalize_class_text(val) != "prejudicado":
                kind = "erro_classe"  # 'Prejudicado' anterior era de agravo/incidente acessorio (bug corrigido)
            else:
                allowed = _allowed_resultados(classe_canon)
                if allowed and normalize_class_text(val) in {normalize_class_text(a) for a in allowed}:
                    kind = "divergente_valido"   # valor do CSV coerente c/ a classe -> provavel correcao
                else:
                    kind = "divergente_incerto"  # CSV pegou outro aspecto / classe sem mapa -> revisao
            stats[col][kind] += 1
            claro = (col == "resultado" and kind == "divergente_valido" and _eh_claro(disp, classe_canon))
            if claro:
                stats["claros"] = stats.get("claros", 0) + 1
            aplica = (kind in ("vazio_preenchido", "erro_classe")
                      or (args.apply_divergentes and kind == "divergente_valido")
                      or (args.apply_claros and claro))
            detail[col] = {"old": cur_raw, "new": val, "kind": kind, "claro": claro, "aplica": aplica}
            if aplica:
                props[col] = client._build_property_value(schema, col, val)
        if not detail:
            continue
        rec = {"page_id": p["id"], "numero": t(p, "numero_processo"), "data": data, "classe": classe, "detail": detail}
        if args.apply and props:
            try:
                notion_request_with_retry(client, "PATCH", f"/pages/{p['id']}", json={"properties": props})
                rec["status"] = "updated"; stats["applied"] += 1
            except Exception as exc:
                rec["status"] = "failed"; rec["error"] = str(exc); stats["failed"] += 1
            time.sleep(0.12)
        changes.append(rec)

    run_dir = ARTIFACT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "changes.json").write_text(json.dumps(changes, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"mode": "apply" if args.apply else "dry-run", "apply_divergentes": args.apply_divergentes, **stats, "paginas_no_relatorio": len(changes)}
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("RESUMO: %s", json.dumps(summary, ensure_ascii=False))
    for c in changes[:20]:
        for col, d in c["detail"].items():
            LOGGER.info("  [%s/%s] %s: %r -> %r (%s, aplica=%s)", c["numero"], c["classe"], col, d["old"], d["new"], d["kind"], d["aplica"])
    LOGGER.info("Relatorios em %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
