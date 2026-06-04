"""Orquestrador de SANEAMENTO da base de sessões no Notion: encadeia, NA ORDEM SEGURA,
os tratamentos JÁ EXISTENTES e SEGUROS (escrita page-value, dry-run por padrão). Serve
tanto para o saneamento RETROATIVO da base inteira quanto como passo padrão após uma
leva de vídeos.

Ordem (cada passo só escreve com --apply):
  1. backfill_notion_cnj.py            -> numero CNJ-20 + classe (vazio/PA) + Suspenso*
  2. reconcile_notion_votacao_suspenso -> Suspenso -> Suspenso* (mesmo processo julgado)
  3. recheck_suspenso_via_datajud.py   -> 'Suspenso por vista' julgado depois (via DataJud)
  4. cleanup_notion_sessoes_advogados  -> consolida MESMA PESSOA (advogados) + Dr./Dra.
  5. cleanup_notion_person_labels.py   -> normalização por-valor (partes/advogados/composicao)

NÃO inclui (precisam de passo MANUAL ou CÓDIGO NOVO — ver o .txt no Desktop):
  - partes/advogados via CSV (download manual por HCaptcha) -> watch_jurisprudencia_csv.py
  - composição 3+2+2 in-place (corretor ainda não existe)
  - etiquetas via Playwright (recolor default + excluir órfãs) -> SEMPRE por último, Edge :9222

Uso:
  python sanear_base.py                 # dry-run de TODOS os passos (não escreve nada)
  python sanear_base.py --apply         # aplica todos os passos, na ordem
  python sanear_base.py --only 1,2      # só os passos 1 e 2 (dry-run)
  python sanear_base.py --apply --only 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# (n, rótulo, script, args_extra) — args_extra acrescentados sempre; --apply é adicionado quando pedido
STEPS = [
    (1, "CNJ DataJud (numero/classe/Suspenso*)", "backfill_notion_cnj.py", []),
    (2, "Reconcilia Suspenso -> Suspenso*", "reconcile_notion_votacao_suspenso.py", []),
    (3, "Re-check 'Suspenso por vista' via DataJud", "recheck_suspenso_via_datajud.py", []),
    (4, "Consolida advogados (mesma pessoa) + Dr./Dra.", "cleanup_notion_sessoes_advogados.py", []),
    (5, "Normaliza partes/advogados/composicao (por-valor)", "cleanup_notion_person_labels.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="Aplica (sem ela: dry-run em todos os passos).")
    ap.add_argument("--only", default="", help="Lista de passos a rodar, ex.: 1,2,4 (default: todos).")
    ap.add_argument("--data-source-id", default="", help="Repassa --data-source-id aos passos.")
    ap.add_argument("--pause-secs", type=float, default=8.0, help="Pausa entre passos (lag do índice Notion).")
    ap.add_argument("--continue-on-error", action="store_true", help="Segue para o próximo passo mesmo se um falhar.")
    args = ap.parse_args()

    only = {int(x) for x in args.only.split(",") if x.strip().isdigit()} if args.only else None
    steps = [s for s in STEPS if (only is None or s[0] in only)]

    mode = "APLICAR" if args.apply else "DRY-RUN"
    print(f"=== SANEAMENTO DA BASE — modo {mode} — passos: {[s[0] for s in steps]} ===", flush=True)
    if args.apply:
        print("    (escrevendo no Notion; cada script já roda escrita SEGURA page-value)", flush=True)

    results = []
    for i, (n, label, script, extra) in enumerate(steps):
        path = SCRIPT_DIR / script
        if not path.exists():
            print(f"\n[passo {n}] PULADO — script não encontrado: {script}", flush=True)
            results.append((n, "ausente"))
            continue
        cmd = [sys.executable, str(path), *extra, "--log-level", "INFO"]
        if args.apply:
            cmd.append("--apply")
        if args.data_source_id:
            cmd += ["--data-source-id", args.data_source_id]
        print(f"\n{'='*70}\n[passo {n}] {label}\n  $ {' '.join(cmd[1:])}\n{'='*70}", flush=True)
        proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
        status = "ok" if proc.returncode == 0 else f"ERRO({proc.returncode})"
        results.append((n, status))
        if proc.returncode != 0 and not args.continue_on_error:
            print(f"\n!! passo {n} falhou ({status}). Parando (use --continue-on-error para seguir).", flush=True)
            break
        if i < len(steps) - 1 and args.pause_secs:
            time.sleep(args.pause_secs)

    print(f"\n=== RESUMO ({mode}): " + " | ".join(f"passo {n}:{st}" for n, st in results) + " ===", flush=True)
    print("Lembrete: partes/advogados via CSV, composição 3+2+2 e etiquetas Playwright são à parte (ver .txt no Desktop).", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
