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
# Quantas vezes retentar um CSV cujo pipeline falhou antes de desistir. Sem teto, um arquivo
# permanentemente quebrado viraria loop infinito a cada poll de 5 s.
MAX_TENTATIVAS = 3
# Ledger compartilhado com o ProjetoConversor (stdlib puro, importavel deste venv). Opcional:
# se sumir, o watcher segue funcionando exatamente como antes dele existir.
PROJETO_CONVERSOR = Path(r"C:\Users\mauri\ProjetoConversor")
try:
    if str(PROJETO_CONVERSOR) not in sys.path:
        sys.path.append(str(PROJETO_CONVERSOR))
    import dje_etapas  # type: ignore
except Exception:  # noqa: BLE001
    dje_etapas = None  # type: ignore


def _geracao_agora() -> str | None:
    """Watermark de paginas no instante da chamada (None se o ledger nao estiver disponivel)."""
    if not dje_etapas:
        return None
    try:
        return dje_etapas.geracao_paginas()
    except Exception:  # noqa: BLE001
        return None


def _marcar_no_ledger(hashes: list[str], geracao: str | None = None) -> None:
    """Registra no ledger que estes CSVs foram confrontados COM a geracao de paginas vigente.

    E este carimbo que permite responder depois: "nasceram paginas de sessao desde o ultimo
    confronto?". Sem ele, o hash em `applied` dizia apenas "ja passou por aqui um dia" -- e foi
    por isso que as 6 paginas de 19/08/2026 ficaram sem partes: o delta constava aplicado desde
    as 09:57, e elas nasceram as 11:23.
    """
    if not dje_etapas or not hashes:
        return
    try:
        dje_etapas.marcar("sessoes_enrich", hashes=hashes, por="watcher", geracao=geracao)
    except Exception as exc:  # noqa: BLE001
        log(f"  (aviso: nao consegui registrar no ledger: {exc})")


def _hashes_desatualizados(watch_dir: Path) -> set[str]:
    """Hashes de CSVs ja aplicados que precisam de nova passada porque nasceram paginas depois."""
    if not dje_etapas:
        return set()
    try:
        pend = dje_etapas.pendencias(watch_dir)
        aplicados = dje_etapas.hashes_enriquecidos()
        alvo = set()
        for p in pend["sessoes_enrich"]:
            h = dje_etapas.sha256(p)
            if h in aplicados:      # so os JA aplicados; os ineditos o fluxo normal pega
                alvo.add(h)
        return alvo
    except Exception as exc:  # noqa: BLE001
        log(f"  (aviso: nao consegui consultar o ledger: {exc})")
        return set()
REPORTS_DIR = SCRIPT_DIR / "artifacts" / "jurisprudencia_partes_advogados"
PIPELINE = SCRIPT_DIR / "fill_partes_advogados_from_jurisprudencia.py"
PIPELINE_COMP = SCRIPT_DIR / "fill_composicao_from_jurisprudencia.py"  # composicao oficial do acordao
PIPELINE_CNJ = SCRIPT_DIR / "complete_cnj_from_jurisprudencia.py"      # completa CNJ-20 das paginas incompletas
PIPELINE_CLASSE = SCRIPT_DIR / "classe_from_jurisprudencia.py"         # classe canonica (anti-downgrade)
PIPELINE_META = SCRIPT_DIR / "fill_metadata_from_jurisprudencia.py"    # eleicao + origem oficiais
# 30/07/2026 — os dois passos que fecham o ciclo "pedido de vista -> acordao publicado".
# Existiam prontos e ficavam FORA do pipeline, dependendo de alguem lembrar de roda-los:
# o crosscheck nunca funcionou de verdade (apontava para a pasta Downloads) e o inteiro
# teor nunca entrou no fluxo automatico. Como a coleta virou automatica (tse_coletor
# deposita o delta e este watcher o consome), a integracao tambem tem de ser.
PIPELINE_VISTA = SCRIPT_DIR / "suspenso_crosscheck_csv.py"             # fecha "Suspenso por vista"
PIPELINE_TEOR = SCRIPT_DIR / "fill_inteiro_teor.py"                    # inteiro teor no CORPO da pagina
# 24/08/2026 — campanha da cadeia do julgado: o crosscheck acima cobre so quem estava
# marcado "Suspenso por vista". Linhas ANTERIORES gravadas com desfecho conclusivo
# indevido (fase de julgamento interrompido) exigem juizo, entao o watcher NAO corrige:
# apenas DETECTA pares novos e alimenta a fila cadeia_fila_pendente.json (rotina
# manutencao_sessoes\cadeia_julgado.py; painel decide, aplicar_cadeia.py aplica).
PIPELINE_CADEIA = SCRIPT_DIR / "manutencao_sessoes" / "cadeia_julgado.py"
# 25/08/2026 — relation `sessoes` <-> `DJe` e o teor que ela destrava. Cada CSV novo cria
# paginas no DJe, e a ligacao delas com a sessao correspondente ficava dependendo de alguem
# lembrar de rodar o `DJE_relations.py --modo cross` (que le as 188 mil paginas do DJe e
# grava sem comparar com o estado atual -- uma rodada que nao conclui e indistinguivel de
# uma sem trabalho, e foi assim que 391 pares disponiveis passaram tres rodadas soltos).
# Aqui a varredura parte das SESSOES sem relation: uma consulta por processo, minutos.
# Em seguida, so nas paginas ligadas AGORA, o acordao pareado vira fonte de inteiro teor
# para quem o CSV do dia nao cobriu. Os dois sao advisory: nunca reprovam a rodada.
PIPELINE_RELATION = SCRIPT_DIR / "manutencao_sessoes" / "auditar_relation_dje.py"
PIPELINE_TEOR_DJE = SCRIPT_DIR / "manutencao_sessoes" / "preencher_teor_do_dje.py"
FILA_RELATION = SCRIPT_DIR / "artifacts" / "notion_sessoes_auditoria" / "relation_fila_novas.json"
# 24/08/2026 — conferencia do vocabulario de ministros. Vive no ProjetoConversor porque e
# la que mora `_ministros_canonico`, a fonte da verdade das DUAS bases; roda com o Python
# daquele projeto (dependencias proprias). So LE os dois schemas, entao e barato e nunca
# reprova a rodada: rc 2 vira aviso no log. Existe porque teste nenhum enxerga uma opcao
# recriada a mao pela UI do Notion -- e foi assim que as bases divergiram (ver a memoria
# `dje-composicao-congelada-lista-curta` e `_ministros_conferir.py`).
CONFERIR_MINISTROS = Path(r"C:\Users\mauri\ProjetoConversor\_ministros_conferir.py")
PY_CONVERSOR = Path(r"C:\Users\mauri\AppData\Local\Programs\Python\Python313\python.exe")

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


def sha256_cached(path: Path, state: dict) -> str:
    """sha256 com cache por (nome, tamanho, mtime).

    Sem isto, scan_once hasheia TODO CSV estavel da pasta a cada poll -- inclusive os
    que ja foram processados, porque o hash e justamente o que decide isso. Com um
    consolidado de centenas de MB na pasta e --poll-secs 3, vira I/O continuo sem fim
    (diagnosticado em 30/07/2026). A tripla muda se o arquivo mudar, entao o cache
    nunca devolve hash de conteudo desatualizado."""
    try:
        st = path.stat()
        chave = f"{path.name}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return sha256_of(path)
    cache = state.setdefault("hash_cache", {})
    if chave in cache:
        return cache[chave]
    h = sha256_of(path)
    cache[chave] = h
    if len(cache) > 2000:  # nao deixar o estado crescer sem limite
        for k in list(cache)[: len(cache) - 1000]:
            cache.pop(k, None)
    return h


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
    data.setdefault("falhou", {})     # hash -> {tentativas, quando, etapas} (retentar)
    data.setdefault("guarda", {})     # hash -> {quando, etapas} (recusado por guarda; nao retenta)
    return data


def save_state(state: dict) -> None:
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_alive(pid: int) -> bool:
    """True se o processo `pid` ainda existe (Windows e POSIX).

    Atencao (1): no Windows, os.kill(pid, 0) NAO e uma checagem inocua de vida -- o sinal 0
    e interpretado como CTRL_C_EVENT, falha para um PID arbitrario e daria 'morto' para
    processo vivo. Por isso usamos a API Win32 (OpenProcess + WaitForSingleObject).

    Atencao (2): ACCESS_DENIED no OpenProcess NAO prova vida. Medido em 20/08/2026: o PID
    48064, morto havia horas, devolvia ACCESS_DENIED em toda chamada -- o lock ficou eterno,
    a Tarefa Agendada saiu na hora por um dia inteiro e nenhum confronto rodou (as paginas
    do lote das 12:46 ficaram sem partes/advogados). Sem handle, a palavra final e do
    tasklist: vivo SO se a linha do PID existir E for de um python -- o mesmo criterio
    anti-PID-reciclado de dje_etapas.trava (ProjetoConversor)."""
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
            handle = kernel32.OpenProcess(SYNCHRONIZE, False, pid)
            if handle:
                try:
                    return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
                finally:
                    kernel32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            pass  # cai para a confirmacao via tasklist
        # Sem handle (ACCESS_DENIED ou PID livre): exigir confirmacao POSITIVA.
        # encoding="oem": o tasklist responde no codepage do console (cp850); em modo UTF-8
        # (-X utf8) o text=True sem encoding estoura no 'Ç' de "INFORMAÇÕES: nenhuma tarefa".
        try:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, encoding="oem",
                errors="replace", timeout=30,
            ).stdout or ""
            linha = next((l for l in out.splitlines() if str(pid) in l), "")
            return "python" in linha.lower()
        except Exception:  # noqa: BLE001
            return True  # tasklist indisponivel: assume vivo (conservador, evita 2 watchers)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except Exception:
        return True
    return True


# Teto de idade do lock, independente da deteccao de PID: o arquivo e reescrito a cada
# aquisicao, entao um lock intocado ha um dia nao pertence a nenhum watcher legitimo.
# E a segunda linha de defesa contra o lock eterno de 19-20/08/2026.
LOCK_MAX_AGE_H = 24.0


def _lock_owner_pid() -> int:
    """PID gravado no lock. Le o formato novo (JSON {pid, iso, argv}) e o antigo (PID cru)."""
    raw = LOCK_FILE.read_text(encoding="utf-8").strip()
    try:
        return int((json.loads(raw) or {}).get("pid") or 0)
    except Exception:  # noqa: BLE001
        try:
            return int(raw or "0")
        except Exception:  # noqa: BLE001
            return 0


def acquire_lock() -> bool:
    """Guarda de instancia unica para o modo continuo. Retorna False se ja ha um watcher
    vivo (e o chamador deve sair). Lock orfao (PID morto ou velho demais) e sobrescrito,
    com registro no log — para o proximo incidente ser legivel."""
    PERM_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        try:
            other = _lock_owner_pid()
        except Exception:  # noqa: BLE001
            other = 0
        try:
            idade_h = (time.time() - LOCK_FILE.stat().st_mtime) / 3600
        except OSError:
            idade_h = 0.0
        if other and other != os.getpid() and _pid_alive(other):
            if idade_h > LOCK_MAX_AGE_H:
                log(f"Lock com {idade_h:.1f}h (PID {other}) passou do teto de "
                    f"{LOCK_MAX_AGE_H:.0f}h; cedendo a vaga.")
            else:
                log(f"Ja existe um watcher rodando (PID {other}). Saindo.")
                return False
        elif other:
            log(f"Lock orfao de PID {other} cedido (processo morto).")
    LOCK_FILE.write_text(json.dumps({
        "pid": os.getpid(),
        "iso": datetime.now().astimezone().isoformat(timespec="seconds"),
        "argv": sys.argv[1:],
    }, ensure_ascii=False), encoding="utf-8")
    return True


def release_lock() -> None:
    try:
        if LOCK_FILE.exists():
            try:
                owner = _lock_owner_pid()
            except Exception:  # noqa: BLE001
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


def _conferir_vocabulario_ministros() -> None:
    """Advisory: as duas bases ainda gravam o mesmo nome para o mesmo ministro?

    Nunca reprova a rodada — o vocabulario divergente nao invalida o que os pipelines
    acabaram de escrever, mas precisa ser VISTO no log do dia em que aparecer, e nao
    meses depois numa auditoria. Conserto no proprio texto do aviso."""
    if not CONFERIR_MINISTROS.exists():
        return
    exe = str(PY_CONVERSOR) if PY_CONVERSOR.exists() else sys.executable
    try:
        proc = subprocess.run([exe, str(CONFERIR_MINISTROS)],
                              cwd=str(CONFERIR_MINISTROS.parent),
                              capture_output=True, text=True, timeout=300)
    except Exception as exc:  # noqa: BLE001
        log(f"  ! conferencia de ministros (advisory) falhou: {exc}")
        return
    if proc.returncode == 0:
        log("  Vocabulario de ministros: DJe e sessoes conferem.")
        return
    if proc.returncode == 2:
        avisos = [ln for ln in (proc.stdout or "").splitlines() if ln.strip().startswith("!")]
        log("  ! VOCABULARIO DE MINISTROS DIVERGENTE entre as bases:")
        for ln in avisos[:10]:
            log(f"    {ln.strip()}")
        log("    conserto: ProjetoConversor\\_ministros_migrar.py --base sessoes --apply "
            "e depois _schema_limpar_orfas.py --apply")
        return
    log(f"  ! conferencia de ministros (advisory) retornou {proc.returncode}: "
        f"{(proc.stderr or '').strip()[-200:]}")


def _run_one(path: Path, label: str, staging: Path, apply: bool, data_source_id: str | None,
             env: dict) -> int:
    """Roda um pipeline (fill_*/classe/complete_cnj) sobre o staging.

    Devolve o returncode. Antes esta funcao engolia a falha (so logava), e process_batch marcava
    o CSV como `applied` de qualquer jeito -- uma etapa quebrada QUEIMAVA o arquivo, que nunca
    mais seria reprocessado. Agora quem chama decide.
    """
    cmd = [sys.executable, str(path), "--input-dir", str(staging), "--log-level", "WARNING"]
    if apply:
        cmd.append("--apply")
    if data_source_id:
        cmd += ["--data-source-id", data_source_id]
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  ! {label} retornou {proc.returncode}: {(proc.stderr or proc.stdout or '').strip()[-400:]}")
    return proc.returncode


# Todo CSV novo passa também pela detecção de FALTANTES (prefilter
# --missing-out), alimentando a fila de vistoria com os que têm menção
# individual nos vídeos (anti-lista). Vale para o consolidado gigante e para
# recortes leves baixados do site do TSE.
PREFILTER = SCRIPT_DIR / "prefilter_dje_csv.py"
BATCH_ARTIFACTS_ROOT = SCRIPT_DIR / "artifacts" / "tse_youtube_notion" / "batch_gui"
MISSING_OUT_DIR = SCRIPT_DIR / "artifacts" / "dje_missing"


def _run_missing_scan(csv_path: Path, data_source_id: str | None, env: dict) -> None:
    log(f"  Varrendo FALTANTES em {csv_path.name} ({max(1, csv_path.stat().st_size // (1024*1024))} MB)...")
    cmd = [
        sys.executable, str(PREFILTER),
        "--input", str(csv_path),
        "--out", str(MISSING_OUT_DIR / "reduzidos"),
        "--missing-out", str(MISSING_OUT_DIR),
        "--queue-from-artifacts", str(BATCH_ARTIFACTS_ROOT),
        "--log-level", "WARNING",
    ]
    if data_source_id:
        cmd += ["--data-source-id", data_source_id]
    proc = subprocess.run(cmd, cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        log(f"  ! faltantes retornou {proc.returncode}: {(proc.stderr or proc.stdout or '').strip()[-400:]}")
    else:
        log(f"  Faltantes: relatorio em {MISSING_OUT_DIR} (fila de vistoria alimentada; importe com import_dje_faltantes.py)")


def _relation_e_teor_do_dje(env: dict) -> None:
    """Etapa 9 (advisory): liga as sessoes as decisoes novas do DJe e, so nas paginas
    ligadas agora, grava o inteiro teor a partir do acordao pareado.

    ADVISORY DE VERDADE: qualquer falha vira aviso no log e nao invalida a rodada nem
    impede o carimbo do CSV -- sao passos de enriquecimento, nao de ingestao. O teor so
    roda se o passo anterior deixou fila; sem `--fila` ele varreria as ~3,2 mil paginas
    com relation relendo blocos (~40 min), o que nao cabe numa rodada automatica."""
    try:
        proc = subprocess.run([sys.executable, str(PIPELINE_RELATION), "--apply"],
                              cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True,
                              timeout=3600)
        linhas = [l for l in (proc.stdout or "").strip().splitlines() if "RESUMO" in l]
        if proc.returncode == 0:
            log(f"  Relation DJe: {linhas[-1] if linhas else 'sem novidades'}")
        else:
            log(f"  ! relation (advisory) retornou {proc.returncode}: "
                f"{(proc.stderr or '').strip()[-200:]}")
            return
    except Exception as exc:
        log(f"  ! relation (advisory) falhou: {exc}")
        return
    if not FILA_RELATION.exists():
        return
    try:
        if not json.loads(FILA_RELATION.read_text(encoding="utf-8")):
            return                      # nenhuma ligacao nova: nada que o teor possa usar
        proc = subprocess.run([sys.executable, str(PIPELINE_TEOR_DJE), "--apply",
                               "--fila", str(FILA_RELATION)],
                              cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True,
                              timeout=3600)
        linhas = [l for l in (proc.stdout or "").strip().splitlines() if "RESUMO" in l]
        if proc.returncode == 0:
            log(f"  Teor pelo acordao pareado: {linhas[-1] if linhas else 'sem novidades'}")
        else:
            log(f"  ! teor-do-dje (advisory) retornou {proc.returncode}: "
                f"{(proc.stderr or '').strip()[-200:]}")
    except Exception as exc:
        log(f"  ! teor-do-dje (advisory) falhou: {exc}")


def run_pipeline(staging: Path, apply: bool, data_source_id: str | None) -> dict:
    """Confronta o(s) CSV(s) do lote com a base de sessoes, na ordem:
    1) completa o CNJ-20 das paginas incompletas (amplia o match dos demais);
    2) partes+advogados; 3) composicao oficial; 4) classe canonica (anti-downgrade);
    5) eleicao+origem oficiais; 6) fecha "Suspenso por vista" cujo acordao ja saiu;
    7) grava o inteiro teor no CORPO da pagina do julgamento conclusivo.
    Cada um e seguro/idempotente (page-values).
    CSVs consolidados (grandes) tambem passam pela deteccao de faltantes.

    A ORDEM DE 6 E 7 IMPORTA: o inteiro teor so deve ser gravado depois que o passo 6
    definiu qual pagina e a do julgamento conclusivo -- senao o texto vai para a pagina
    da sessao em que o julgamento foi SUSPENSO."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    etapas = [
        (PIPELINE_CNJ, "cnj"),
        (PIPELINE, "partes/advogados"),
        (PIPELINE_COMP, "composicao"),
        (PIPELINE_CLASSE, "classe"),
        (PIPELINE_META, "metadata"),
        (PIPELINE_VISTA, "pedido de vista"),
        (PIPELINE_TEOR, "inteiro teor"),
    ]
    falhas: list[str] = []
    guardas: list[str] = []
    for script, label in etapas:
        rc = _run_one(script, label, staging, apply, data_source_id, env)
        # rc 3 = o proprio pipeline recusou a rodada por guarda de seguranca (o freio de massa do
        # fill_composicao). Nao e crash e nao adianta retentar -- mas TAMBEM nao pode ser tratado
        # como sucesso: se o hash entrar em `applied`, o trabalho recusado vira invisivel e a
        # faixa das GUIs passa a dizer "em dia" sobre uma etapa que nao rodou.
        if rc == 3:
            log(f"  ! {label} RECUSOU a rodada (guarda de seguranca). O CSV nao sera marcado como "
                f"processado. Confira o changes.json e, se estiver certo, rode a mao com "
                f"--permitir-massa.")
            guardas.append(label)
        elif rc != 0:
            falhas.append(label)
    if apply:
        for csv_path in sorted(staging.glob("*.csv")):
            _run_missing_scan(csv_path, data_source_id, env)
        # 8) cadeia do julgado: so DETECTA (fila de pendentes p/ painel); nunca aplica.
        #    Advisory: falha aqui nao invalida a rodada nem impede o carimbo do CSV.
        try:
            proc = subprocess.run([sys.executable, str(PIPELINE_CADEIA), "--so-a", "--fila"],
                                  cwd=str(SCRIPT_DIR), env=env, capture_output=True, text=True,
                                  timeout=1800)
            ultima = (proc.stdout or "").strip().splitlines()
            if proc.returncode == 0 and ultima:
                log(f"  Cadeia do julgado: {ultima[-2] if len(ultima) >= 2 else ultima[-1]}")
            else:
                log(f"  ! cadeia (advisory) retornou {proc.returncode}: "
                    f"{(proc.stderr or '').strip()[-200:]}")
        except Exception as exc:
            log(f"  ! cadeia (advisory) falhou: {exc}")
        _relation_e_teor_do_dje(env)
        _conferir_vocabulario_ministros()
    resumo = dict(newest_report_summary() or {})   # copia: nao mutar o dict que veio do disco
    if falhas:
        resumo["_falhas"] = falhas
    if guardas:
        resumo["_guardas"] = guardas
    return resumo


def process_batch(files: list[Path], state: dict, args) -> None:
    """Copia o lote para o acervo + staging isolado e roda o pipeline UMA vez (amortiza o
    full-scan do Notion). Marca todos como processados no modo atual."""
    staging = Path(tempfile.mkdtemp(prefix="tse_stage_"))
    staged_hashes: list[tuple[str, str]] = []
    # Watermark capturado ANTES do trabalho. Se fosse lido no fim, uma pagina nascida DURANTE a
    # passada (a GUI do YouTube publicando em paralelo) entraria no carimbo sem ter sido
    # processada -- e, como so ha reprocessamento quando a geracao avanca, ela nunca mais seria
    # alcancada. Seria reproduzir o buraco de 19/08/2026 por outro caminho.
    ger_inicial = _geracao_agora()
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
        falhas = (summary or {}).get("_falhas") or []
        guardas = (summary or {}).get("_guardas") or []
        bucket = state["applied"] if args.apply else state["dry_run"]
        for h, n in staged_hashes:
            state["files"][h] = n
            if guardas:
                # Recusa por guarda de seguranca: nao e falha (retentar nao adianta) nem sucesso
                # (o trabalho nao foi feito). Bucket proprio, sem contar tentativa e sem entrar
                # em `applied` -- o CSV continua listado como pendente ate alguem decidir.
                state["guarda"][h] = {
                    "quando": datetime.now().isoformat(timespec="seconds"),
                    "etapas": guardas,
                }
                log(f"  ! {n}: recusado por guarda em {guardas}; NAO marcado como processado.")
                continue
            state["guarda"].pop(h, None)
            if falhas:
                # NAO marca como feito: uma etapa quebrada nao pode queimar o CSV para sempre.
                # Retenta no proximo poll; depois de MAX_TENTATIVAS desiste com log explicito,
                # senao um arquivo permanentemente quebrado viraria loop infinito.
                reg = state["falhou"].setdefault(h, {"tentativas": 0})
                reg["tentativas"] = int(reg.get("tentativas", 0)) + 1
                reg["quando"] = datetime.now().isoformat(timespec="seconds")
                reg["etapas"] = falhas
                if reg["tentativas"] >= MAX_TENTATIVAS:
                    log(f"  ! {n}: {reg['tentativas']} tentativas com falha em {falhas}. "
                        f"Desistindo (vai para skip). Corrija e remova o hash de 'skip'.")
                    if h not in state["skip"]:
                        state["skip"].append(h)
                else:
                    log(f"  ! {n}: etapas com falha {falhas}; NAO marcado como processado "
                        f"(tentativa {reg['tentativas']}/{MAX_TENTATIVAS}).")
                continue
            state["falhou"].pop(h, None)
            if h not in bucket:
                bucket.append(h)
        # Carimba a geracao de paginas vigente: e o que permite detectar depois que nasceram
        # paginas novas e o confronto precisa ser refeito (o buraco de 19/08/2026).
        if args.apply and not falhas and not guardas:
            _marcar_no_ledger([h for h, _ in staged_hashes], geracao=ger_inicial)
        save_state(state)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def already_done(h: str, state: dict, apply: bool, forcados: set[str] | None = None) -> bool:
    """`forcados` fura o bloqueio para reprocessar um CSV ja aplicado.

    Existe porque "aplicado" nao significa "terminado": os 7 pipelines refazem a consulta ao
    Notion do zero e se auto-escopam pelos CNJ do CSV, entao reprocessar ALCANCA paginas criadas
    depois. Sem esta valvula, uma pagina que nasce depois do confronto fica vazia para sempre.

    `forcados` vem ANTES de `skip` de proposito. `skip` guarda duas coisas muito diferentes: os
    arquivos que nem sao jurisprudencia do TSE (permanente, correto) e os que desistimos apos
    MAX_TENTATIVAS de falha -- e essas falhas costumam ser transitorias (429/502 do Notion). Se
    `skip` vencesse, 3 erros de rede em minutos queimariam o CSV para sempre, sem nenhum comando
    capaz de resgata-lo a nao ser editar o _watch_state.json a mao.
    """
    if forcados and h in forcados:
        return False
    if h in state["skip"]:
        return True
    if apply:
        return h in state["applied"]
    return h in state["dry_run"] or h in state["applied"]


def scan_once(watch_dir: Path, sizes: dict, state: dict, args, forcados: set[str] | None = None) -> int:
    ready = stable_csvs(watch_dir, sizes)
    batch: list[Path] = []
    dirty = False
    for p in ready:
        try:
            antes = len(state.get("hash_cache", {}))
            h = sha256_cached(p, state)
            dirty = dirty or len(state.get("hash_cache", {})) != antes
        except OSError:
            continue
        if already_done(h, state, args.apply, forcados):
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
    elif dirty:
        save_state(state)  # persiste o cache de hash mesmo sem lote a processar
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
    # --- reprocessamento: alcancar paginas criadas DEPOIS do confronto ------------------------
    ap.add_argument("--force", action="store_true",
                    help="Reprocessa os CSVs mesmo ja aplicados. Exige --once.")
    ap.add_argument("--force-hash", action="append", default=None, metavar="SHA256",
                    help="Reprocessa apenas este(s) hash(es), mesmo ja aplicado(s). Exige --once.")
    ap.add_argument("--alcancar-novas", action="store_true",
                    help="Reprocessa os CSVs cujo confronto ficou para tras de paginas novas "
                         "(consulta o ledger dje_etapas). E o modo que as GUIs usam. Exige --once.")
    args = ap.parse_args()
    _setup_logging(args.log_file)

    # Reprocessar em modo CONTINUO seria um loop: a cada poll de 5 s o mesmo CSV voltaria a ser
    # aplicado, para sempre. Estas flags so fazem sentido numa passada unica.
    if (args.force or args.force_hash or args.alcancar_novas) and not args.once:
        ap.error("--force / --force-hash / --alcancar-novas exigem --once "
                 "(em modo continuo virariam reprocessamento em loop a cada poll).")

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

        forcados: set[str] = set()
        if args.force:
            for p in watch_dir.glob("*.csv"):
                try:
                    forcados.add(sha256_cached(p, state))
                except OSError:
                    pass
            log(f"--force: {len(forcados)} CSV(s) serao reprocessados mesmo ja aplicados.")
        if args.force_hash:
            forcados.update(args.force_hash)
            log(f"--force-hash: {len(args.force_hash)} hash(es) forcado(s).")
        if args.alcancar_novas:
            novos = _hashes_desatualizados(watch_dir)
            forcados.update(novos)
            log(f"--alcancar-novas: {len(novos)} CSV(s) com confronto atrasado em relacao as "
                f"paginas ja criadas.")
            # SEM early-return aqui. `forcados` e apenas a valvula que fura o `applied`; os CSVs
            # INEDITOS sao pegos pelo fluxo normal do scan_once. Sair cedo por "nenhum atrasado"
            # abandonaria um delta recem-coletado sem sequer sniffa-lo -- e o dje_ingerir logo em
            # seguida o levaria para a base DJe, deixando as sessoes sem partes: exatamente a
            # assimetria que este modo existe para evitar.

        # O --once tambem toma o lock: a Tarefa Agendada continua vigiando a pasta e, sem isto,
        # as duas passadas disputariam o mesmo _watch_state.json (lost update) e o mesmo rate
        # limit do Notion.
        if not acquire_lock():
            # rc 4, nao 0: o contínuo NUNCA faz catch-up (scan_once sem `forcados`), entao esta
            # passada era a unica capaz de alcancar paginas nascidas depois do confronto. Sair 0
            # faria a GUI registrar sucesso sobre trabalho que nao aconteceu.
            log("Ha outro watcher ativo nesta pasta; --once nao vai concorrer com ele. "
                "O confronto NAO foi feito -- pare a Tarefa Agendada e repita, se precisar dele.")
            return 4
        try:
            n = scan_once(watch_dir, sizes, state, args, forcados)
        finally:
            release_lock()
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
