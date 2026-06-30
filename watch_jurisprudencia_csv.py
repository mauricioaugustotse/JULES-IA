"""Vigia uma pasta e, a cada CSV NOVO de jurisprudência do TSE que aparecer, dispara
a extração/preenchimento de partes+advogados no Notion (reusando o pipeline
fill_partes_advogados_from_jurisprudencia.py, que escreve com segurança via page-values).

Pensado para o fluxo manual: você baixa os CSVs no navegador NORMAL (sem captcha),
aos poucos; este watcher detecta cada arquivo novo e processa sozinho.

Modos da pasta vigiada (--watch-dir):
  (a) artifacts/jurisprudencia_csv  -> aponte o download do navegador para cá (default)
  (b) C:/Users/<voce>/Downloads     -> baixe normalmente; o watcher reconhece os CSVs
                                       do TSE pelo CONTEUDO (content-sniff) e ignora o resto

Seguranca:
  - DRY-RUN por padrao (nada e escrito no Notion). Use --apply para gravar de fato.
  - A escrita do pipeline e page-value multi_select (cria etiquetas com seguranca);
    NUNCA faz PATCH em options/schema do data_source.
  - Idempotente: so grava colunas que mudaram; arquivo ja processado nao repete (hash).

Uso:
  python watch_jurisprudencia_csv.py                         # vigia a pasta do projeto, dry-run
  python watch_jurisprudencia_csv.py --watch-dir "C:/Users/mauri/Downloads"
  python watch_jurisprudencia_csv.py --apply                 # grava no Notion
  python watch_jurisprudencia_csv.py --once                  # processa o que ja existe e sai
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PERM_DIR = SCRIPT_DIR / "artifacts" / "jurisprudencia_csv"   # acervo permanente dos CSVs
STATE_FILE = PERM_DIR / "_watch_state.json"
LOCK_FILE = PERM_DIR / "_watch.lock"   # guarda de instancia unica (so no modo continuo)
DEFAULT_WATCH_DIR = r"C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
REPORTS_DIR = SCRIPT_DIR / "artifacts" / "jurisprudencia_partes_advogados"
PIPELINE = SCRIPT_DIR / "fill_partes_advogados_from_jurisprudencia.py"
PIPELINE_COMP = SCRIPT_DIR / "fill_composicao_from_jurisprudencia.py"  # composicao oficial do acordao
PIPELINE_CNJ = SCRIPT_DIR / "complete_cnj_from_jurisprudencia.py"      # completa CNJ-20 das paginas incompletas
PIPELINE_CLASSE = SCRIPT_DIR / "classe_from_jurisprudencia.py"         # classe canonica (anti-downgrade)
PIPELINE_META = SCRIPT_DIR / "fill_metadata_from_jurisprudencia.py"    # eleicao + origem oficiais

TSE_SIGNATURE_COLS = ("siglaTribunalJE", "textoDecisao", "partes", "relatores", "numeroProcesso")
CNJ20_RE = re.compile(r"\d{20}")


def log(msg: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} | {msg}", flush=True)


def _setup_logging(log_file: str) -> None:
    """Redireciona stdout/stderr para um arquivo quando rodando sem console.
    Necessario sob pythonw.exe (Tarefa Agendada oculta): la sys.stdout pode ser None
    e os print() do watcher quebrariam. Se --log-file vier vazio mas nao houver console,
    cai num log default ao lado do estado."""
    target = log_file
    if not target and (sys.stdout is None or sys.stderr is None):
        target = str(PERM_DIR / "watch_dje.log")
    if not target:
        return
    try:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        f = open(target, "a", encoding="utf-8", errors="replace", buffering=1)
        sys.stdout = f
        sys.stderr = f
    except Exception:
        pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data.setdefault("applied", [])    # hashes processados COM --apply (gravados no Notion)
    data.setdefault("dry_run", [])    # hashes processados em dry-run
    data.setdefault("skip", [])       # hashes de arquivos que NAO sao jurisprudencia do TSE
    data.setdefault("files", {})      # hash -> nome original (para log)
    return data


def save_state(state: dict) -> None:
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    """True se o processo `pid` ainda existe (Windows e POSIX).

    Atencao: no Windows, os.kill(pid, 0) NAO e uma checagem inocua de vida -- o sinal 0
    e interpretado como CTRL_C_EVENT, falha para um PID arbitrario e daria 'morto' para
    processo vivo. Por isso usamos a API Win32 (OpenProcess + WaitForSingleObject)."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.OpenProcess.argtypes = [ctypes.c_uint, ctypes.c_int, ctypes.c_uint]
            kernel32.WaitForSingleObject.restype = ctypes.c_uint
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            SYNCHRONIZE = 0x00100000
            WAIT_TIMEOUT = 0x00000102
            ERROR_ACCESS_DENIED = 5
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if not handle:
                # sem handle: ACCESS_DENIED => existe (outro contexto); demais => nao existe
                return ctypes.get_last_error() == ERROR_ACCESS_DENIED
            try:
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True  # erro inesperado: assume vivo (conservador, evita 2 watchers)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except Exception:
        return True
    return True


def acquire_lock() -> bool:
    """Guarda de instancia unica para o modo continuo. Retorna False se ja ha um watcher
    vivo (e o chamador deve sair). Lock orfao (PID morto) e sobrescrito."""
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            other = int((LOCK_FILE.read_text(encoding="utf-8").strip() or "0"))
        except Exception:
            other = 0
        if other and other != os.getpid() and _pid_alive(other):
            log(f"Ja existe um watcher rodando (PID {other}). Saindo.")
            return False
        # lock orfao (processo morto ou ilegivel): sobrescreve
    LOCK_FILE.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        if LOCK_FILE.exists():
            try:
                owner = int((LOCK_FILE.read_text(encoding="utf-8").strip() or "0"))
            except Exception:
                owner = os.getpid()
            if owner == os.getpid():
                LOCK_FILE.unlink()
    except Exception:
        pass


def sniff_is_tse_csv(path: Path) -> tuple[bool, str]:
    """Confirma pelo CONTEUDO que e um export de jurisprudencia do TSE: header com
    'numeroUnico' + alguma coluna-assinatura, e ao menos uma linha com CNJ de 20 digitos."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            cols = reader.fieldnames or []
            if "numeroUnico" not in cols:
                return False, "sem coluna numeroUnico"
            if not any(c in cols for c in TSE_SIGNATURE_COLS):
                return False, "sem colunas-assinatura do TSE"
            for i, row in enumerate(reader):
                num = re.sub(r"\D", "", str(row.get("numeroUnico") or row.get("numeroProcesso") or ""))
                if len(num) >= 20:
                    return True, "ok"
                if i > 50:
                    break
            return False, "nenhum CNJ de 20 digitos nas primeiras linhas"
    except Exception as exc:
        return False, f"erro ao ler: {exc}"


def stable_csvs(watch_dir: Path, sizes: dict) -> list[Path]:
    """Retorna os *.csv cujo tamanho ficou ESTAVEL desde o poll anterior (debounce contra
    arquivo ainda sendo gravado). Atualiza `sizes` in-place."""
    ready: list[Path] = []
    current = {}
    for p in sorted(watch_dir.glob("*.csv")):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        current[str(p)] = sz
        if sz > 0 and sizes.get(str(p)) == sz:
            ready.append(p)
    sizes.clear()
    sizes.update(current)
    return ready


def unique_dest(name: str, content_hash: str) -> Path:
    """Caminho no acervo permanente, evitando sobrescrever arquivo de conteudo diferente."""
    dest = PERM_DIR / name
    if dest.exists() and sha256_of(dest) != content_hash:
        stem, suf = dest.stem, dest.suffix
        k = 2
        while True:
            alt = PERM_DIR / f"{stem}__{k}{suf}"
            if not alt.exists() or sha256_of(alt) == content_hash:
                return alt
            k += 1
    return dest


def newest_report_summary() -> dict:
    if not REPORTS_DIR.exists():
        return {}
    dirs = sorted([d for d in REPORTS_DIR.iterdir() if d.is_dir()])
    if not dirs:
        return {}
    sm = dirs[-1] / "summary.json"
    try:
        return json.loads(sm.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _run_one(path: Path, label: str, staging: Path, apply: bool, data_source_id: str | None, env: dict) -> None:
    """Roda um pipeline (fill_*/classe/complete_cnj) sobre o staging; loga se retornar erro."""
    cmd = [sys.executable, str(path), "--input-dir", str(staging), "--log-level", "WARNING"]
    if apply:
        cmd.append("--apply")
    if data_source_id:
        cmd += ["--data-source-id", data_source_id]
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  ! {label} retornou {proc.returncode}: {(proc.stderr or proc.stdout or '').strip()[-400:]}")


def run_pipeline(staging: Path, apply: bool, data_source_id: str | None) -> dict:
    """Confronta o(s) CSV(s) do lote com a base de sessoes, na ordem:
    1) completa o CNJ-20 das paginas incompletas (amplia o match dos demais);
    2) partes+advogados; 3) composicao oficial; 4) classe canonica (anti-downgrade);
    5) eleicao+origem oficiais. Cada um e seguro/idempotente (page-values)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    _run_one(PIPELINE_CNJ, "cnj", staging, apply, data_source_id, env)
    _run_one(PIPELINE, "partes/advogados", staging, apply, data_source_id, env)
    _run_one(PIPELINE_COMP, "composicao", staging, apply, data_source_id, env)
    _run_one(PIPELINE_CLASSE, "classe", staging, apply, data_source_id, env)
    _run_one(PIPELINE_META, "metadata", staging, apply, data_source_id, env)
    return newest_report_summary()


def process_batch(files: list[Path], state: dict, args) -> None:
    """Copia o lote para o acervo + staging isolado e roda o pipeline UMA vez (amortiza o
    full-scan do Notion). Marca todos como processados no modo atual."""
    staging = Path(tempfile.mkdtemp(prefix="tse_stage_"))
    staged_hashes: list[tuple[str, str]] = []
    try:
        for p in files:
            h = sha256_of(p)
            perm = unique_dest(p.name, h)
            if perm.resolve() != p.resolve():
                shutil.copy2(p, perm)          # arquiva no acervo permanente
            shutil.copy2(p, staging / perm.name)
            staged_hashes.append((h, perm.name))
        names = ", ".join(n for _, n in staged_hashes)
        log(f"Processando {len(staged_hashes)} arquivo(s): {names} [{'APLICAR' if args.apply else 'dry-run'}]")
        summary = run_pipeline(staging, args.apply, args.data_source_id)
        if summary:
            log("  RESUMO: match={match} partes±={muda_partes} advogados±={muda_advogados} "
                "paginas_mudanca={paginas_com_mudanca} applied={applied} failed={failed}".format(
                    match=summary.get("match"), muda_partes=summary.get("muda_partes"),
                    muda_advogados=summary.get("muda_advogados"),
                    paginas_com_mudanca=summary.get("paginas_com_mudanca"),
                    applied=summary.get("applied"), failed=summary.get("failed")))
        bucket = state["applied"] if args.apply else state["dry_run"]
        for h, n in staged_hashes:
            if h not in bucket:
                bucket.append(h)
            state["files"][h] = n
        save_state(state)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def already_done(h: str, state: dict, apply: bool) -> bool:
    if h in state["skip"]:
        return True
    if apply:
        return h in state["applied"]
    return h in state["dry_run"] or h in state["applied"]


def scan_once(watch_dir: Path, sizes: dict, state: dict, args) -> int:
    ready = stable_csvs(watch_dir, sizes)
    batch: list[Path] = []
    for p in ready:
        try:
            h = sha256_of(p)
        except OSError:
            continue
        if already_done(h, state, args.apply):
            continue
        ok, why = sniff_is_tse_csv(p)
        if not ok:
            if h not in state["skip"]:
                state["skip"].append(h)
                state["files"][h] = p.name
                log(f"Ignorado (nao e CSV de jurisprudencia TSE): {p.name} -> {why}")
            continue
        batch.append(p)
    if batch:
        process_batch(batch, state, args)
    return len(batch)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch-dir", default=os.environ.get("DJE_WATCH_DIR", DEFAULT_WATCH_DIR),
                    help=rf"Pasta a vigiar. Default: {DEFAULT_WATCH_DIR} (env DJE_WATCH_DIR).")
    ap.add_argument("--apply", action="store_true",
                    help="Grava no Notion. Sem ela: dry-run (so relatorios, nao escreve).")
    ap.add_argument("--once", action="store_true",
                    help="Processa os CSVs ja presentes e sai (nao fica vigiando).")
    ap.add_argument("--poll-secs", type=float, default=3.0)
    ap.add_argument("--data-source-id", default=None)
    ap.add_argument("--log-file", default="",
                    help="Redireciona a saida para este arquivo (uso da Tarefa Agendada oculta com pythonw).")
    args = ap.parse_args()
    _setup_logging(args.log_file)

    watch_dir = Path(args.watch_dir).resolve()
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    if not PIPELINE.exists():
        log(f"ERRO: pipeline nao encontrado: {PIPELINE}")
        return 1

    if not watch_dir.exists():
        if args.once:
            log(f"ERRO: pasta nao existe: {watch_dir}")
            return 1
        # Modo continuo: a pasta pode ainda nao ter sincronizado (OneDrive no logon).
        # Espera ela aparecer em vez de abortar. NAO cria a pasta.
        log(f"Pasta ainda nao existe: {watch_dir} -- aguardando aparecer (poll={args.poll_secs}s)...")
        while not watch_dir.exists():
            time.sleep(max(args.poll_secs, 1.0))
        log(f"Pasta encontrada: {watch_dir}")

    state = load_state()
    log(f"Vigiando: {watch_dir}")
    log(f"Acervo permanente: {PERM_DIR}")
    log(f"Modo: {'APLICAR no Notion' if args.apply else 'DRY-RUN (nada escrito)'} | poll={args.poll_secs}s")

    # No modo --once, considera tudo 'estavel' de imediato (sem esperar 2 polls).
    sizes: dict = {}
    if args.once:
        for p in watch_dir.glob("*.csv"):
            try:
                sizes[str(p)] = p.stat().st_size
            except OSError:
                pass
        n = scan_once(watch_dir, sizes, state, args)
        log(f"--once: {n} arquivo(s) processado(s). Saindo.")
        return 0

    # Modo continuo: guarda de instancia unica (evita 2 watchers na mesma pasta/estado).
    if not acquire_lock():
        return 0
    log("Baixe os CSVs no navegador normal; cada arquivo novo sera processado. Ctrl+C para parar.")
    try:
        while True:
            try:
                scan_once(watch_dir, sizes, state, args)
            except Exception as exc:
                log(f"  ! erro no scan (continuo): {exc}")
            time.sleep(args.poll_secs)
    except KeyboardInterrupt:
        log("Encerrado pelo usuario.")
        save_state(state)
    finally:
        release_lock()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
