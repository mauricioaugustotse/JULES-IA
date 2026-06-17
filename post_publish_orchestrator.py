"""Orquestrador GOING-FORWARD de tratamentos POS-PUBLICACAO (TSE->Notion).
Encadeia, por SUBPROCESS (isolado + idempotente — cada script re-consulta o estado vivo e e
no-downgrade), os tratamentos que deixam a base limpa/padronizada/validada/relacionada DEPOIS
que a GUI publica um lote. Chamado por run_post_publish_treatments(...) (da GUI/app) ou via CLI.

PASSOS DEFAULT a cada lote (Notion-only / pouca rede, exceto pedido_vista/composicao que usam IA p/ poucos casos):
  materia       -> materia_semelhante_update.py   (relations: mesmo CNJ liga registros)
  suspenso      -> recheck_suspenso_via_datajud.py (so os ~Suspenso, rapido)
  pedido_vista  -> fill_pedido_vista_via_grounding.py (QUEM pediu vista, via noticia TSE; so "Suspenso por vista" vazios)
  composicao    -> fix_composicao_from_transcript.py + fix_composicao_via_gemini_opening.py (fallback p/ sessao recente)
  classe_nomes  -> fix_classe_nomes.py            (nome-por-extenso -> sigla)
  sanear        -> sanear_coluna.py (advogados+composicao) (gemeos acento/junk/multi)

PASSOS CAROS (SADP/DataJud base-inteira, lentos) — OPT-IN (rodar periodicamente):
  backfill_cnj  -> backfill_notion_cnj.py
  sadp_cnj      -> sadp_complete_cnj.py + sadp_correct_cnj20.py
  sadp_origem   -> sadp_validate_origem.py
  origem_uf     -> origem_uf_check.py (diag) + fix_origem_via_datajud.py
  classe_valida -> classe_validate_datajud.py

Uso CLI:
  python post_publish_orchestrator.py --steps materia,suspenso,classe_nomes,sanear            # dry-run
  python post_publish_orchestrator.py --steps default --apply
  python post_publish_orchestrator.py --steps all --apply
"""
from __future__ import annotations

import argparse, json, logging, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

LOGGER = logging.getLogger("post_publish_orchestrator")
SCRIPT_DIR = Path(__file__).resolve().parent

# step -> [(script, extra_args, accepts_apply, accepts_dsid)]
STEPS: dict[str, list[tuple[str, list[str], bool, bool]]] = {
    "materia": [("materia_semelhante_update.py", [], True, True)],
    "suspenso": [("recheck_suspenso_via_datajud.py", [], True, True)],
    "pedido_vista": [("fill_pedido_vista_via_grounding.py", [], True, True)],
    # composicao: transcript (gratis, sessao antiga) + opening como FALLBACK p/ sessao RECENTE
    # (transcript indisponivel por dias); opening prioriza os videos mais recentes e limita custo.
    "composicao": [("fix_composicao_from_transcript.py", [], True, True),
                   ("fix_composicao_via_gemini_opening.py", ["--max-videos", "8", "--since-days", "25"], True, True)],
    "classe_nomes": [("fix_classe_nomes.py", [], True, True)],
    "sanear": [("sanear_advogados.py", [], True, True),
               ("sanear_coluna.py", ["--column", "composicao"], True, True)],
    "backfill_cnj": [("backfill_notion_cnj.py", [], True, True)],
    "sadp_cnj": [("sadp_complete_cnj.py", [], True, True), ("sadp_correct_cnj20.py", [], True, True)],
    "sadp_origem": [("sadp_validate_origem.py", [], True, True),
                    ("apply_origem_corrections.py", [], True, True)],
    "origem_uf": [("origem_uf_check.py", [], False, False),
                  ("fix_origem_via_datajud.py", [], True, False)],
    "classe_valida": [("classe_validate_datajud.py", [], True, True)],
}
DEFAULT_STEPS = ["materia", "suspenso", "pedido_vista", "composicao", "classe_nomes", "sanear"]
EXPENSIVE_STEPS = ["backfill_cnj", "sadp_cnj", "sadp_origem", "origem_uf", "classe_valida"]
ALL_STEPS = DEFAULT_STEPS + EXPENSIVE_STEPS


def _resolve_steps(steps: list[str]) -> list[str]:
    out: list[str] = []
    for s in steps:
        if s == "default":
            out += DEFAULT_STEPS
        elif s == "all":
            out += ALL_STEPS
        elif s == "expensive":
            out += EXPENSIVE_STEPS
        elif s in STEPS:
            out.append(s)
    seen, uniq = set(), []
    for s in out:  # ordem canonica de ALL_STEPS, sem duplicar
        if s not in seen:
            seen.add(s)
    return [s for s in ALL_STEPS if s in seen]


def run_post_publish_treatments(
    *,
    data_source_id: str,
    apply: bool,
    steps: Optional[list[str]] = None,
    logger: Optional[logging.Logger] = None,
    artifact_dir: Optional[Path] = None,
    log_line: Optional[Callable[[str], None]] = None,
    timeout_per_script: int = 5400,
) -> dict[str, Any]:
    """Executa os passos pos-publicacao. Retorna metrics por script. Falha isolada (um script
    que quebra nao aborta os demais). `log_line` (opcional) recebe cada linha de stdout p/ a GUI."""
    log = logger or LOGGER
    resolved = _resolve_steps(steps or DEFAULT_STEPS)
    log.info("Pos-publicacao: passos=%s | apply=%s", resolved, apply)
    metrics: dict[str, Any] = {"mode": "apply" if apply else "dry-run", "steps": resolved, "results": {}}
    for step in resolved:
        for script, extra, acc_apply, acc_dsid in STEPS[step]:
            cmd = [sys.executable, str(SCRIPT_DIR / script), *extra]
            if apply and acc_apply:
                cmd.append("--apply")
            if acc_dsid:
                cmd += ["--data-source-id", data_source_id]
            key = f"{step}:{script} {' '.join(extra)}".strip()
            log.info("  -> %s", " ".join(cmd[1:]))
            t0 = time.time()
            try:
                proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True,
                                      encoding="utf-8", errors="replace", timeout=timeout_per_script)
                out = (proc.stdout or "") + (proc.stderr or "")
                if log_line:
                    for ln in out.splitlines():
                        if ln.strip():
                            log_line(f"[{step}] {ln}")
                resumo = next((ln for ln in reversed(out.splitlines()) if "RESUMO" in ln), "")
                metrics["results"][key] = {"returncode": proc.returncode, "secs": round(time.time() - t0, 1),
                                           "resumo": resumo[-400:], "status": "ok" if proc.returncode == 0 else "failed"}
            except Exception as exc:
                metrics["results"][key] = {"status": "failed", "error": str(exc), "secs": round(time.time() - t0, 1)}
                log.warning("  FALHA %s: %s", key, exc)
    if artifact_dir:
        try:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)
            (Path(artifact_dir) / "05b_post_publish_treatments.json").write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    ok = sum(1 for r in metrics["results"].values() if r.get("status") == "ok")
    log.info("Pos-publicacao concluida: %s/%s scripts ok", ok, len(metrics["results"]))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default="default", help="csv de passos OU default|expensive|all")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--data-source-id", default="")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    dsid = args.data_source_id
    if not dsid:
        from tse_youtube_notion_core import DEFAULT_NOTION_DATA_SOURCE_ID
        dsid = DEFAULT_NOTION_DATA_SOURCE_ID
    run_dir = SCRIPT_DIR / "artifacts" / "post_publish" / datetime.now().strftime("%Y%m%d_%H%M%S")
    m = run_post_publish_treatments(data_source_id=dsid, apply=args.apply,
                                    steps=[s.strip() for s in args.steps.split(",") if s.strip()],
                                    artifact_dir=run_dir)
    print(json.dumps(m, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
