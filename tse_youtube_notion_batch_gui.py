from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk
from typing import Any, Callable

from tse_youtube_notion_core import (
    ARTIFACT_ROOT,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS,
    DEFAULT_NEWS_GEMINI_MODEL,
    DEFAULT_NOTION_DATABASE_URL,
    GeminiSessionExtractor,
    NotionSessoesClient,
    RunArtifacts,
    build_preview_rows,
    build_runtime_context,
    dedupe_preview_rows,
    enrich_preview_rows_with_news,
    enrich_preview_rows_with_process_metadata,
    enrich_preview_rows_with_theme_punchline,
    enrich_preview_rows_with_cnj,
    enrich_preview_rows_with_youtube_chapters,
    enrich_preview_rows_with_session_date_from_title,
    extract_youtube_video_id,
    normalize_youtube_link,
    publish_preview_rows,
    require_youtube_transcript_api,
    validate_preview_row,
)
from tse_normalization import infer_session_date_from_video_title

import vistoria_queue


LOGGER = logging.getLogger("tse_youtube_notion_batch_gui")
MAX_LINKS = 10
BATCH_ARTIFACT_ROOT = ARTIFACT_ROOT / "batch_gui"
TERMINAL_STATUSES = {"Concluido", "Erro"}
STAGE_PROGRESS = {
    "Em andamento": 0.02,
    "analisando video": 0.08,
    "montando previa": 0.38,
    "enriquecendo metadados": 0.55,
    "revisando tema e punchline": 0.70,
    "buscando noticias": 0.85,
    "publicando no Notion": 0.95,
}


@dataclass(frozen=True)
class BatchOptions:
    model: str
    news_model: str
    with_news: bool
    publish: bool
    continue_on_error: bool
    # Salvaguarda de qualidade: por padrao EXIGE extracao por video. Se o Gemini nao
    # processar o video (ex.: sessao recem-transmitida ao vivo, ainda nao disponivel como
    # VOD), o video falha e e reprocessado depois — em vez de publicar registros rasos
    # vindos da transcricao (sem numero/resultado, composicao inflada).
    allow_transcript_fallback: bool = False
    # --- GOING-FORWARD: tratamentos pos-publicacao (defaults ligados) ---
    post_publish_steps: tuple = ("materia", "suspenso", "classe_nomes", "sanear")
    recolor_labels: bool = True          # Playwright recolore/exclui orfas (degrada se Edge ausente)
    # Desligado por default: o monitor independente da pasta DJe (Tarefa Agendada
    # WatchDJe_Notion, via CRIAR_TAREFA_WATCH_DJE.ps1) cuida disso continuamente.
    # A caixa segue na GUI para uma rodada --once manual, se desejado.
    watch_dje: bool = False              # processa CSVs ja em DJE (--once)
    dje_apply: bool = True               # grava no Notion (vs dry-run)
    dje_dir: str = r"C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
    cdp_url: str = "http://127.0.0.1:9222"


@dataclass(frozen=True)
class VideoInput:
    position: int
    url: str
    video_id: str


class QueueLogHandler(logging.Handler):
    def __init__(self, output_queue: "queue.Queue[tuple[str, Any]]") -> None:
        super().__init__()
        self.output_queue = output_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.output_queue.put(("log", self.format(record) + "\n"))
        except Exception:
            pass


def split_candidate_urls(text: str) -> list[str]:
    candidates: list[str] = []
    for chunk in re.split(r"[\s,;]+", text.strip()):
        cleaned = chunk.strip()
        if cleaned:
            candidates.append(cleaned)
    return candidates


def normalize_video_input(position: int, raw_url: str) -> VideoInput:
    normalized = normalize_youtube_link(raw_url.strip())
    video_id = extract_youtube_video_id(normalized)
    if not video_id:
        raise ValueError(f"Link do YouTube invalido: {raw_url}")
    return VideoInput(position=position, url=normalized, video_id=video_id)


def open_path(path: Path) -> None:
    if sys.platform.startswith("win"):
        subprocess.Popen(["explorer.exe", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def count_result_status(results: list[dict[str, Any]], status: str) -> int:
    return sum(1 for item in results if item.get("status") == status)


def item_video_link(item: dict[str, Any]) -> str:
    row = item.get("row") or {}
    return str(row.get("youtube_link") or item.get("youtube_url") or "")


def item_timestamp_seconds(item: dict[str, Any]) -> int | None:
    """Timestamp (t=) do link do julgamento, quando registrado no item."""
    match = re.search(r"[?&]t=(\d+)", item_video_link(item))
    return int(match.group(1)) if match else None


def summarize_link_meta(title: str) -> str:
    """Resumo "título — data" exibido na coluna Sessão da lista de links."""
    clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
    session_date = infer_session_date_from_video_title(clean_title) or ""
    display = clean_title[:70] + ("..." if len(clean_title) > 70 else "")
    if session_date:
        return f"{display} — {session_date}" if display else session_date
    return display or "(título indisponível)"


def fetch_video_title_oembed(url: str, timeout: float = 10.0) -> str:
    oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({"url": url, "format": "json"})
    with urllib.request.urlopen(oembed_url, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("title", "") or "")


class Tooltip:
    """Balão de ajuda: aparece ~0,5s depois que o mouse pousa no widget.

    Singleton visual: só UM balão na tela por vez (mostrar um fecha o anterior —
    widgets aninhados não empilham balões) e autoexpira após alguns segundos
    para nunca ficar órfão sobre a interface.
    """

    _active: "Tooltip | None" = None
    AUTO_HIDE_MS = 8000

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 500, wraplength: int = 440) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        self._expire_id: str | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip or not self.text:
            return
        if Tooltip._active is not None and Tooltip._active is not self:
            Tooltip._active._hide()
        Tooltip._active = self
        self._tip = tip_window = tk.Toplevel(self.widget)
        tip_window.wm_overrideredirect(True)
        try:
            tip_window.attributes("-topmost", True)
        except Exception:
            pass
        tk.Label(
            tip_window, text=self.text, justify=tk.LEFT, background="#fffbe6",
            foreground="#333333", relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9), wraplength=self.wraplength, padx=8, pady=6,
        ).pack()
        # Posiciona junto ao CURSOR (não abaixo do widget: em tabelas/abas altas o
        # balão iria parar cortado no rodapé da tela) e mantém dentro da área útil.
        tip_window.update_idletasks()
        tip_w = tip_window.winfo_reqwidth()
        tip_h = tip_window.winfo_reqheight()
        screen_w = self.widget.winfo_screenwidth()
        screen_h = self.widget.winfo_screenheight()
        pointer_x = self.widget.winfo_pointerx()
        pointer_y = self.widget.winfo_pointery()
        x = min(pointer_x + 14, screen_w - tip_w - 8)
        y = pointer_y + 20
        if y + tip_h > screen_h - 56:  # não invade a barra de tarefas: abre ACIMA do cursor
            y = pointer_y - tip_h - 14
        x = max(x, 8)
        y = max(y, 8)
        tip_window.wm_geometry(f"+{x}+{y}")
        self._expire_id = self.widget.after(self.AUTO_HIDE_MS, self._hide)

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._expire_id:
            try:
                self.widget.after_cancel(self._expire_id)
            except Exception:
                pass
            self._expire_id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None
        if Tooltip._active is self:
            Tooltip._active = None


def tip(widget, text: str):
    """Anexa um tooltip e devolve o próprio widget (uso inline na construção da UI)."""
    Tooltip(widget, text)
    return widget


def format_elapsed(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, remainder = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:02d}"
    return f"{minutes:02d}:{remainder:02d}"


def process_single_video(
    video: VideoInput,
    *,
    artifact_store: RunArtifacts,
    notion_client: NotionSessoesClient,
    notion_schema: Any,
    gemini_api_key: str,
    options: BatchOptions,
    progress: Callable[[str], None],
    analysis: Any | None = None,
) -> dict[str, Any]:
    # ``analysis`` pre-extraido (IA bruta vinda dos artefatos) pula a extracao pelo
    # Gemini e segue por TODAS as demais etapas de padronizacao iguais ao fluxo normal
    # (capitulos, data pelo titulo, CNJ, metadados, tema/punchline, noticias, publish).
    if analysis is None:
        progress("analisando video")
        extractor = GeminiSessionExtractor(
            api_key=gemini_api_key,
            model=options.model,
            artifact_store=artifact_store,
            logger=LOGGER,
            allow_transcript_fallback=options.allow_transcript_fallback,
        )
        analysis = extractor.analyze_session(video.url)
    else:
        progress("usando analise pre-extraida dos artefatos (IA bruta)")
    artifact_store.write_json("03_analysis.json", analysis.model_dump(mode="json"))

    progress("montando previa")
    rows = build_preview_rows(
        analysis,
        youtube_url=video.url,
        notion_schema=notion_schema,
        notion_client=notion_client,
    )

    progress("enriquecendo via capítulos do YouTube")
    rows = enrich_preview_rows_with_youtube_chapters(
        rows,
        youtube_url=video.url,
        notion_schema=notion_schema,
        logger=LOGGER,
    )

    progress("corrigindo data da sessão pelo título do vídeo")
    rows = enrich_preview_rows_with_session_date_from_title(
        rows,
        youtube_url=video.url,
        logger=LOGGER,
    )

    progress("enriquecendo via CNJ DataJud")
    rows = enrich_preview_rows_with_cnj(
        rows,
        notion_schema=notion_schema,
        logger=LOGGER,
    )

    progress("enriquecendo metadados")
    rows = enrich_preview_rows_with_process_metadata(
        rows,
        api_key=gemini_api_key,
        model=options.model,
        artifact_store=artifact_store,
        logger=LOGGER,
        notion_schema=notion_schema,
    )
    rows = dedupe_preview_rows(rows, video.url)
    rows = [validate_preview_row(row, notion_schema) for row in rows]

    if rows:
        progress("revisando tema e punchline")
        rows = enrich_preview_rows_with_theme_punchline(
            rows,
            api_key=gemini_api_key,
            model=options.model,
            artifact_store=artifact_store,
            logger=LOGGER,
            notion_schema=notion_schema,
        )
        rows = dedupe_preview_rows(rows, video.url)
        rows = [validate_preview_row(row, notion_schema) for row in rows]

    artifact_store.write_json(
        "04_preview_rows.json",
        [row.model_dump(mode="json") for row in rows],
    )

    if options.with_news and rows:
        progress("buscando noticias")
        rows = enrich_preview_rows_with_news(
            rows,
            api_key=gemini_api_key,
            model=options.news_model,
            artifact_store=artifact_store,
            logger=LOGGER,
        )
        rows = dedupe_preview_rows(rows, video.url)
        rows = [validate_preview_row(row, notion_schema) for row in rows]
        artifact_store.write_json(
            "04b_enriched_preview_rows.json",
            [row.model_dump(mode="json") for row in rows],
        )

    publish_results: list[dict[str, Any]] = []
    if options.publish:
        progress("publicando no Notion")
        publish_results = publish_preview_rows(rows, notion_client, notion_schema)
        artifact_store.write_json("05_publish_results.json", publish_results)

    rito_count_check = _build_rito_count_check(artifact_store, rows)
    if rito_count_check is not None:
        artifact_store.write_json("04c_rito_count_check.json", rito_count_check)

    vistoria_items = vistoria_queue.collect_video_vistoria_items(
        rows,
        publish_results,
        video_id=video.video_id,
        youtube_url=video.url,
        artifact_dir=str(artifact_store.root_dir),
        rito_check=rito_count_check,
        published=bool(options.publish),
    )
    if vistoria_items:
        artifact_store.write_json("04d_vistoria_items.json", vistoria_items)

    summary = {
        "position": video.position,
        "video_id": video.video_id,
        "url": video.url,
        "artifact_dir": str(artifact_store.root_dir),
        "rows_extracted": len(rows),
        "created": count_result_status(publish_results, "created"),
        "updated": count_result_status(publish_results, "updated"),
        "blocked": count_result_status(publish_results, "blocked"),
        "skipped": count_result_status(publish_results, "skipped"),
        "publish_results": publish_results,
    }
    if rito_count_check is not None:
        summary["rito_count_check"] = rito_count_check
    summary["vistoria_items"] = len(vistoria_items)
    artifact_store.write_json("06_batch_video_summary.json", summary)
    return summary


def _build_rito_count_check(artifact_store: RunArtifacts, rows: list[Any]) -> dict[str, Any] | None:
    """Confere o nº de linhas extraídas contra os apregoamentos individuais do rito.

    Divergência não bloqueia nada: vira item de vistoria (a transcrição pode ter
    marcadores perdidos e julgamentos conjuntos geram mais linhas que apregoamentos).
    """
    if not artifact_store.exists("01b_rito_refinement.json"):
        return None
    try:
        rito_report = artifact_store.read_json("01b_rito_refinement.json")
    except Exception:
        return None
    if not rito_report.get("transcript_available"):
        return None
    apregoamentos = rito_report.get("apregoamentos_individuais")
    if not isinstance(apregoamentos, int) or apregoamentos <= 0:
        return None
    delta = apregoamentos - len(rows)
    return {
        "apregoamentos": apregoamentos,
        "rows": len(rows),
        "delta": delta,
        "verdict": "ok" if delta <= 0 else "verificar",
    }


def _gf_dir() -> Path:
    return Path(__file__).resolve().parent


# --- Edge OCULTO para a etapa Playwright do pos-publish -----------------------
# A etapa de etiquetas dirige o Notion via CDP. Para nao "abrir telas", quando o
# CDP nao estiver de pe abrimos um Edge HEADLESS (sem janela) reusando o perfil
# dedicado que ja esta logado no Notion. Os cliques sao injetados via CDP, logo
# nao dependem de foco/janela visivel: a execucao no Notion ocorre plenamente.
# Modo "offscreen" (janela renderizada fora da tela) fica como alternativa caso
# algum dia o headless quebre a automacao por coordenadas.
EDGE_HIDDEN_MODE = "headless"  # "headless" (sem janela) | "offscreen" (fora da tela)
_EDGE_EXE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)
_EDGE_PROFILE_DIR = str(Path(tempfile.gettempdir()) / "notion-cdp-profile")
_EDGE_WARMUP_S = 6.0  # folga pos-CDP para o Notion renderizar a base
_CREATE_NO_WINDOW = 0x08000000


def _cdp_is_up(cdp_url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen((cdp_url or "").rstrip("/") + "/json/version", timeout=timeout):
            return True
    except Exception:
        return False


def _find_edge_exe() -> str | None:
    for p in _EDGE_EXE_CANDIDATES:
        if Path(p).exists():
            return p
    return None


def _gf_ensure_hidden_edge(
    cdp_url: str, notion_url: str, output_queue: "queue.Queue[tuple[str, Any]]"
) -> tuple["subprocess.Popen[Any] | None", bool]:
    """Garante um Edge OCULTO escutando no CDP, reusando o perfil ja logado.
    Retorna (proc, launched_by_us). Se o CDP ja estiver de pe (ex.: 'Preparar
    Edge' aberto manualmente), NAO lanca e NAO assume a posse do processo."""
    if _cdp_is_up(cdp_url):
        return None, False
    if os.name != "nt":
        return None, False
    exe = _find_edge_exe()
    if not exe:
        output_queue.put(("log", "[labels] Edge nao localizado para abrir oculto; "
                                 "etiquetas dependem de um Edge :9222 manual.\n"))
        return None, False
    m = re.search(r":(\d+)", cdp_url or "")
    port = m.group(1) if m else "9222"
    start_url = notion_url or "https://www.notion.so"
    args = [exe,
            f"--remote-debugging-port={port}",
            f"--user-data-dir={_EDGE_PROFILE_DIR}",
            "--no-first-run", "--no-default-browser-check",
            "--disable-features=Translate,msEdgeWelcomePage"]
    if EDGE_HIDDEN_MODE == "offscreen":
        args += ["--window-position=-32000,-32000", "--window-size=1920,1080", "--new-window", start_url]
    else:
        args += ["--headless=new", "--window-size=1920,1080", start_url]
    try:
        proc = subprocess.Popen(args, close_fds=True, creationflags=_CREATE_NO_WINDOW)
    except Exception as exc:
        output_queue.put(("log", f"[labels] falha ao abrir Edge oculto: {exc}\n"))
        return None, False
    try:
        for _ in range(40):  # ~20s para o listener do CDP subir
            if _cdp_is_up(cdp_url):
                output_queue.put(("log", f"[labels] Edge oculto pronto ({EDGE_HIDDEN_MODE}) em {cdp_url}.\n"))
                time.sleep(_EDGE_WARMUP_S)
                return proc, True
            time.sleep(0.5)
    except BaseException:
        # nunca deixa o Edge orfao se a espera for interrompida (ex.: KeyboardInterrupt)
        _gf_close_hidden_edge(proc, True, output_queue)
        raise
    output_queue.put(("log", "[labels] Edge oculto nao respondeu no CDP a tempo "
                             "(perfil pode estar travado por outra instancia do Edge); seguindo.\n"))
    return proc, True


def _gf_close_hidden_edge(
    proc: "subprocess.Popen[Any] | None", launched: bool,
    output_queue: "queue.Queue[tuple[str, Any]]"
) -> None:
    """Encerra o Edge oculto SOMENTE se fomos nos que o lancamos (nunca derruba um
    Edge aberto manualmente pelo usuario)."""
    if not (launched and proc is not None):
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=_CREATE_NO_WINDOW)
        else:
            proc.terminate()
        try:
            proc.wait(timeout=8)
        except Exception:
            proc.kill()
        output_queue.put(("log", "[labels] Edge oculto encerrado.\n"))
    except Exception as exc:
        output_queue.put(("log", f"[labels] falha ao encerrar Edge oculto: {exc}\n"))


def _gf_python_with_playwright() -> str:
    """Interpretador que tenha o pacote `playwright` (exigido por notion_labels_drive.py).
    Prefere o python atual; cai para o venv dedicado de UI (.venv-windows-notion-ui).
    Sem isto, a etapa rodaria no .venv-win (sem playwright) e falharia no import."""
    cands = [sys.executable,
             str(_gf_dir() / ".venv-windows-notion-ui" / "Scripts" / "python.exe"),
             str(_gf_dir() / ".venv-win" / "Scripts" / "python.exe"),
             str(_gf_dir() / ".venv" / "Scripts" / "python.exe")]
    seen: set[str] = set()
    for c in cands:
        if not c or c in seen or not Path(c).exists():
            continue
        seen.add(c)
        try:
            r = subprocess.run(
                [c, "-c", "import importlib.util,sys;"
                          "sys.exit(0 if importlib.util.find_spec('playwright') else 3)"],
                capture_output=True, timeout=20)
            if r.returncode == 0:
                return c
        except Exception:
            continue
    # ultimo recurso: prefere o venv de UI (unico sabidamente com playwright) se existir
    ui = str(_gf_dir() / ".venv-windows-notion-ui" / "Scripts" / "python.exe")
    return ui if Path(ui).exists() else sys.executable


def _gf_run_label_recolor(cdp_url: str, notion_url: str, output_queue: "queue.Queue[tuple[str, Any]]") -> dict[str, Any]:
    """Recolore default + exclui orfas via Playwright. partes e rich_text -> NAO entra.
    Sequencial (mesmo painel CDP). Para no 1o sinal de Edge ausente (returncode 2).

    A etapa roda SEMPRE oculta: se nao houver Edge no CDP, abre um Edge headless
    (sem janela) com o perfil ja logado e o encerra ao final. A execucao no Notion
    e feita plenamente via CDP (cliques nao dependem de foco/janela visivel)."""
    import subprocess
    res: dict[str, Any] = {}
    edge_ok = True
    pyexe = _gf_python_with_playwright()
    edge_proc, edge_launched = _gf_ensure_hidden_edge(cdp_url, notion_url, output_queue)
    try:
        for prop in ("origem",):  # advogados virou rich_text (sem etiquetas); composicao: NAO recolorir
            cmd = [pyexe, str(_gf_dir() / "notion_labels_drive.py"), "--property", prop,
                   "--auto-open", "--apply-deletes", "--cdp-url", cdp_url, "--delay-ms", "110"]
            try:
                p = subprocess.run(cmd, cwd=str(_gf_dir()), capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=7200)
                for ln in (p.stdout or "").splitlines():
                    if any(k in ln for k in ("CONCLUIDO", "FALHOU", "ABORTADO", "Edge nao", "alvos")):
                        output_queue.put(("log", f"[labels:{prop}] {ln}\n"))
                res[prop] = p.returncode
                if p.returncode == 2:  # Edge ausente OU Notion deslogado em :9222
                    output_queue.put(("log", "[labels] Edge ausente ou Notion deslogado (:9222) — "
                                             "etiquetas nao recoloridas. Abra 'Preparar Edge' e faca login.\n"))
                    edge_ok = False; break
            except Exception as exc:
                res[prop] = str(exc); output_queue.put(("log", f"[labels:{prop}] erro: {exc}\n"))
        # ORDEM ALFABETICA (origem + composicao; advogados virou rich_text, sem etiquetas)
        if edge_ok:
            for prop in ("origem", "composicao"):
                cmd = [pyexe, str(_gf_dir() / "notion_labels_drive.py"), "--property", prop,
                       "--auto-open", "--sort-only", "--cdp-url", cdp_url]
                try:
                    p = subprocess.run(cmd, cwd=str(_gf_dir()), capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=900)
                    for ln in (p.stdout or "").splitlines():
                        if any(k in ln for k in ("ordenado", "nao encontr", "Edge nao")):
                            output_queue.put(("log", f"[sort:{prop}] {ln}\n"))
                    res[f"sort_{prop}"] = p.returncode
                except Exception as exc:
                    res[f"sort_{prop}"] = str(exc)
        return res
    finally:
        _gf_close_hidden_edge(edge_proc, edge_launched, output_queue)


def _gf_run_dje_once(dje_dir: str, apply: bool, output_queue: "queue.Queue[tuple[str, Any]]") -> Any:
    """Processa os CSVs de jurisprudencia ja presentes em DJE (modo --once) -> fill partes/advogados."""
    import subprocess
    cmd = [sys.executable, str(_gf_dir() / "watch_jurisprudencia_csv.py"), "--watch-dir", dje_dir, "--once"]
    if apply:
        cmd.append("--apply")
    try:
        p = subprocess.run(cmd, cwd=str(_gf_dir()), capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=3600)
        for ln in (p.stdout or "").splitlines()[-10:]:
            if ln.strip():
                output_queue.put(("log", f"[dje] {ln}\n"))
        return p.returncode
    except Exception as exc:
        output_queue.put(("log", f"[dje] erro: {exc}\n")); return str(exc)


def process_video_batch(
    videos: list[VideoInput],
    options: BatchOptions,
    output_queue: "queue.Queue[tuple[str, Any]]",
    stop_event: threading.Event,
    resume_root: Path | None = None,
    analysis_provider: Callable[[VideoInput], Any] | None = None,
) -> dict[str, Any]:
    runtime = build_runtime_context()
    gemini_key = runtime["gemini_api_key"]
    notion_key = runtime["notion_api_key"]
    if not gemini_key:
        raise RuntimeError("GEMINI_API_KEY/GOOGLE_API_KEY nao encontrado.")
    if not notion_key:
        raise RuntimeError("NOTION_API_KEY/NOTION_TOKEN nao encontrado.")

    run_root = resume_root or (BATCH_ARTIFACT_ROOT / time.strftime("%Y%m%d_%H%M%S"))
    run_root.mkdir(parents=True, exist_ok=True)
    output_queue.put(("batch_artifact_dir", str(run_root)))
    LOGGER.info("Artifacts do lote: %s", run_root)
    if resume_root is not None:
        LOGGER.info("Retomando lote existente a partir dos artifacts.")
    LOGGER.info("Banco Notion: %s", runtime.get("notion_database_url") or DEFAULT_NOTION_DATABASE_URL)
    LOGGER.info("Data source Notion: %s", runtime["notion_data_source_id"])
    LOGGER.info("Modelo Gemini: %s", options.model)
    LOGGER.info("Modelo noticias: %s", options.news_model)
    LOGGER.info("Timeout Gemini por chamada: %ss", DEFAULT_GEMINI_HTTP_TIMEOUT_SECONDS)

    notion_client = NotionSessoesClient(
        api_key=notion_key,
        data_source_id=runtime["notion_data_source_id"],
        logger=LOGGER,
        normalize_multiselect_colors_post_write=False,
    )
    notion_schema = notion_client.fetch_schema()

    summaries: list[dict[str, Any]] = []
    for index, video in enumerate(videos, start=1):
        if stop_event.is_set():
            LOGGER.warning("Execucao interrompida antes do video %s.", video.position)
            break

        artifact_store = RunArtifacts(run_root / f"{video.position:02d}_{video.video_id}")
        output_queue.put(("video_started", video.video_id, index, len(videos)))
        output_queue.put(("status", video.video_id, "Em andamento", ""))
        LOGGER.info("[%s/%s] Iniciando %s", index, len(videos), video.url)

        def _progress(message: str) -> None:
            output_queue.put(("status", video.video_id, message, ""))
            LOGGER.info("[%s] %s", video.video_id, message)

        try:
            injected_analysis = analysis_provider(video) if analysis_provider else None
            summary = process_single_video(
                video,
                artifact_store=artifact_store,
                notion_client=notion_client,
                notion_schema=notion_schema,
                gemini_api_key=gemini_key,
                options=options,
                progress=_progress,
                analysis=injected_analysis,
            )
            summaries.append({"status": "done", **summary})
            final_status = (
                f"OK: {summary['created']} criadas, {summary['updated']} atualizadas, "
                f"{summary['blocked']} bloqueadas, {summary['skipped']} ignoradas"
            )
            output_queue.put(("status", video.video_id, "Concluido", final_status))
            output_queue.put(("video_finished", video.video_id, "done"))
            LOGGER.info("[%s] %s", video.video_id, final_status)
        except Exception as exc:
            error_text = str(exc)
            summaries.append(
                {
                    "status": "error",
                    "position": video.position,
                    "video_id": video.video_id,
                    "url": video.url,
                    "artifact_dir": str(artifact_store.root_dir),
                    "error": error_text,
                    "traceback": traceback.format_exc(),
                }
            )
            artifact_store.write_json("06_batch_video_error.json", summaries[-1])
            output_queue.put(("status", video.video_id, "Erro", error_text))
            output_queue.put(("video_finished", video.video_id, "error"))
            LOGGER.exception("[%s] Falha no processamento", video.video_id)
            if not options.continue_on_error:
                break

    # ===== POS-PUBLICACAO GOING-FORWARD (so se publicou algo e nao interrompido) =====
    post_publish: dict[str, Any] = {}
    publicou = any(item.get("status") == "done" for item in summaries)
    if options.publish and publicou and not stop_event.is_set():
        dsid = runtime["notion_data_source_id"]
        if options.post_publish_steps:
            output_queue.put(("status", "__post__", "Pos-publicacao: tratamentos de dados...", ""))
            try:
                from post_publish_orchestrator import run_post_publish_treatments
                post_publish["treatments"] = run_post_publish_treatments(
                    data_source_id=dsid, apply=True, steps=list(options.post_publish_steps),
                    logger=LOGGER, artifact_dir=run_root,
                    log_line=lambda ln: output_queue.put(("log", ln + "\n")))
            except Exception as exc:
                LOGGER.warning("Pos-publicacao (dados) falhou: %s", exc)
                post_publish["treatments_error"] = str(exc)
        if options.recolor_labels:
            output_queue.put(("status", "__post__", "Pos-publicacao: recolorir/excluir etiquetas (Edge oculto)...", ""))
            post_publish["labels"] = _gf_run_label_recolor(
                options.cdp_url,
                runtime.get("notion_database_url") or DEFAULT_NOTION_DATABASE_URL,
                output_queue)
        if options.watch_dje:
            output_queue.put(("status", "__post__", "Pos-publicacao: CSVs DJE (--once)...", ""))
            post_publish["dje"] = _gf_run_dje_once(options.dje_dir, options.dje_apply, output_queue)
        output_queue.put(("status", "__post__", "Pos-publicacao concluida.", ""))

    # ===== FILA DE VISTORIA: consolida os itens do run e alimenta a fila global =====
    run_vistoria_items: list[dict[str, Any]] = []
    for item in summaries:
        artifact_dir = item.get("artifact_dir", "")
        vistoria_path = Path(artifact_dir) / "04d_vistoria_items.json" if artifact_dir else None
        if vistoria_path and vistoria_path.exists():
            try:
                run_vistoria_items.extend(json.loads(vistoria_path.read_text(encoding="utf-8")))
            except Exception as exc:
                LOGGER.warning("Falha ao ler %s: %s", vistoria_path, exc)
    if run_vistoria_items:
        (run_root / "vistoria_queue.json").write_text(
            json.dumps(run_vistoria_items, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        try:
            added = vistoria_queue.append_items(run_vistoria_items)
            output_queue.put(
                ("log", f"Fila de vistoria: {added} item(ns) novo(s) de {len(run_vistoria_items)} coletado(s).\n")
            )
        except Exception as exc:
            LOGGER.warning("Falha ao alimentar a fila de vistoria: %s", exc)

    summary_payload = {
        "post_publish": post_publish,
        "started_at": run_root.name,
        "artifact_dir": str(run_root),
        "notion_database_url": runtime.get("notion_database_url") or DEFAULT_NOTION_DATABASE_URL,
        "notion_data_source_id": runtime["notion_data_source_id"],
        "total_requested": len(videos),
        "total_done": sum(1 for item in summaries if item.get("status") == "done"),
        "total_error": sum(1 for item in summaries if item.get("status") == "error"),
        "videos": summaries,
    }
    (run_root / "batch_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary_payload


class BatchGuiApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("TSE YouTube > Notion - lote pos-noticias")
        self.root.geometry("1280x840")
        self.root.minsize(1100, 720)

        self.output_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.videos: list[VideoInput] = []
        self.batch_artifact_dir = ""
        self.resume_root: Path | None = None
        self.total_videos = 0
        self.completed_videos = 0
        self.current_video_id = ""
        self.current_video_index = 0
        self.stage_labels: dict[str, str] = {}
        self.stage_started_at: dict[str, float] = {}
        self.link_meta: dict[str, tuple[str, str]] = {}  # video_id -> (sessão, badge CC)

        self.link_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_GEMINI_MODEL)
        self.news_model_var = tk.StringVar(value=DEFAULT_NEWS_GEMINI_MODEL)
        self.with_news_var = tk.BooleanVar(value=True)
        self.publish_var = tk.BooleanVar(value=True)
        self.continue_on_error_var = tk.BooleanVar(value=True)
        self.post_publish_var = tk.BooleanVar(value=True)
        self.recolor_labels_var = tk.BooleanVar(value=True)
        self.watch_dje_var = tk.BooleanVar(value=False)  # monitor independente (Tarefa WatchDJe_Notion) assume; marque p/ rodada --once manual
        self.count_var = tk.StringVar(value=f"0/{MAX_LINKS} links")
        self.target_var = tk.StringVar(value=f"Notion: {DEFAULT_NOTION_DATABASE_URL}")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Pronto")

        self._build_ui()
        self.root.after(200, self._drain_output_queue)
        self.root.after(1000, self._refresh_live_progress)

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("vista")
        except tk.TclError:
            pass
        self.root.option_add("*Font", ("Segoe UI", 10))
        style.configure("TLabelframe.Label", font=("Segoe UI", 10, "bold"), foreground="#1f3a5f")
        style.configure("Treeview", rowheight=24, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=(14, 6))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
        style.configure("Hint.TLabel", foreground="#a15c00", font=("Segoe UI", 10, "bold"))
        style.configure("Muted.TLabel", foreground="#666666", font=("Segoe UI", 9))

        main = ttk.Frame(self.root, padding=(12, 10))
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="TSE YouTube → Notion", font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        notion_link = ttk.Label(
            header, text="abrir base no Notion ↗", foreground="#0b5cad", cursor="hand2",
            font=("Segoe UI", 9, "underline"),
        )
        notion_link.pack(side=tk.RIGHT, pady=(8, 0))
        notion_link.bind("<Button-1>", lambda _e: webbrowser.open(DEFAULT_NOTION_DATABASE_URL))
        Tooltip(notion_link, "Abre no navegador a base de sessões do TSE no Notion, onde as linhas são publicadas.")

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        process_tab = ttk.Frame(self.notebook, padding=10)
        vistoria_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(process_tab, text="  Processar lote  ")
        self.notebook.add(vistoria_tab, text="  Fila de vistoria  ")
        self.vistoria_tab_index = 1

        statusbar = ttk.Frame(main)
        statusbar.pack(fill=tk.X, pady=(8, 0))
        tip(
            ttk.Progressbar(statusbar, variable=self.progress_var, maximum=100, mode="determinate", length=300),
            "Progresso do lote em execução (vídeos concluídos e etapa atual).",
        ).pack(side=tk.LEFT)
        tip(
            ttk.Label(statusbar, textvariable=self.progress_text_var, style="Muted.TLabel"),
            "Etapa em andamento no vídeo atual e tempo decorrido nela.",
        ).pack(side=tk.LEFT, padx=(10, 0))
        self.vistoria_hint_var = tk.StringVar(value="")
        vistoria_hint = ttk.Label(
            statusbar, textvariable=self.vistoria_hint_var, style="Hint.TLabel", cursor="hand2"
        )
        vistoria_hint.pack(side=tk.RIGHT)
        vistoria_hint.bind("<Button-1>", lambda _e: self.notebook.select(self.vistoria_tab_index))
        Tooltip(vistoria_hint, "Atalho: clique para ir direto à Fila de vistoria com os itens que aguardam sua decisão.")

        # ================= ABA 1 — PROCESSAR LOTE =================
        process_tab.columnconfigure(0, weight=1)
        process_tab.rowconfigure(2, weight=3)
        process_tab.rowconfigure(3, weight=1)

        input_frame = ttk.LabelFrame(
            process_tab,
            text="1. Links do YouTube  (título, data da sessão e legenda são verificados ao adicionar)",
            padding=8,
        )
        input_frame.grid(row=0, column=0, sticky="ew")
        input_frame.columnconfigure(0, weight=1)
        tip(
            ttk.Entry(input_frame, textvariable=self.link_var),
            "Cole aqui a URL de um vídeo de sessão plenária do TSE no YouTube. Aceita várias URLs de uma vez "
            "(separadas por espaço, vírgula ou quebra de linha).",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        tip(
            ttk.Button(input_frame, text="Adicionar link", command=self._add_link),
            "Valida a URL digitada e acrescenta o vídeo à lista do lote. Em segundo plano são verificados o "
            "título, a data da sessão e a existência de legenda (colunas Sessão e CC da tabela).",
        ).grid(row=0, column=1, padx=(0, 8))
        tip(
            ttk.Button(input_frame, text="Colar da área", command=self._paste_links),
            "Adiciona de uma vez todos os links de vídeo copiados na área de transferência (Ctrl+C).",
        ).grid(row=0, column=2)
        tip(
            ttk.Label(input_frame, textvariable=self.count_var, style="Muted.TLabel"),
            "Quantos vídeos já estão na lista e o limite por lote.",
        ).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))

        options = ttk.LabelFrame(process_tab, text="2. Opções do fluxo", padding=8)
        options.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)
        tip(
            ttk.Label(options, text="Modelo Gemini"),
            "Modelo de IA que assiste ao vídeo e extrai os julgamentos (número, relator, resultado, análise). "
            "O padrão preenchido é o recomendado; só altere se souber o que está fazendo.",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        tip(
            ttk.Entry(options, textvariable=self.model_var),
            "Nome do modelo Gemini usado na EXTRAÇÃO dos julgamentos do vídeo.",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 14))
        tip(
            ttk.Label(options, text="Modelo noticias"),
            "Modelo de IA usado na etapa de busca de notícias relacionadas a cada julgamento.",
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 8))
        tip(
            ttk.Entry(options, textvariable=self.news_model_var),
            "Nome do modelo Gemini usado na BUSCA/validação de notícias (TSE, TREs e imprensa).",
        ).grid(row=0, column=3, sticky="ew")
        tip(
            ttk.Checkbutton(options, text="Buscar noticias antes de publicar", variable=self.with_news_var),
            "Antes de gravar no Notion, procura notícias oficiais (TSE/TREs) e da imprensa sobre cada julgamento "
            "e anexa os links aprovados à linha. Desmarque para um lote mais rápido e sem notícias.",
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(options, text="Publicar direto no Notion", variable=self.publish_var),
            "Marcado: as linhas extraídas são gravadas na base do Notion ao final. Desmarcado: o lote apenas gera "
            "os arquivos de prévia (artifacts) para conferência, sem publicar nada.",
        ).grid(row=1, column=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(options, text="Continuar se um link falhar", variable=self.continue_on_error_var),
            "Se um vídeo der erro (ex.: transmissão recém-encerrada ainda sem VOD), o lote registra a falha e "
            "segue para o próximo vídeo em vez de parar tudo.",
        ).grid(row=1, column=3, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(options, text="Pos-publicacao: tratar dados (materia/Suspenso/classe/sanear)",
                            variable=self.post_publish_var),
            "Depois de publicar, roda os tratamentos automáticos de qualidade: vincular matéria semelhante, "
            "reconciliar julgamentos 'Suspenso' concluídos depois, normalizar classe processual e sanear dados.",
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(options, text="Recolorir etiquetas (Edge :9222)",
                            variable=self.recolor_labels_var),
            "Ajusta as cores das etiquetas (selects) na base do Notion usando um Edge oculto automatizado. "
            "Puramente estético; requer o Microsoft Edge instalado.",
        ).grid(row=2, column=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(options, text="Processar CSVs DJE",
                            variable=self.watch_dje_var),
            "Roda UMA passada do confronto com os CSVs do Diário da Justiça Eletrônico ao final do lote. "
            "Normalmente desnecessário: o monitor automático (tarefa WatchDJe_Notion) já vigia a pasta DJe "
            "continuamente e processa qualquer CSV novo.",
        ).grid(row=2, column=3, sticky=tk.W, pady=(6, 0))

        exec_frame = ttk.LabelFrame(process_tab, text="3. Execução", padding=8)
        exec_frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        exec_frame.columnconfigure(0, weight=1)
        exec_frame.rowconfigure(1, weight=1)

        actions = ttk.Frame(exec_frame)
        actions.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        self.start_button = tip(
            ttk.Button(actions, text="▶  Processar lote", style="Accent.TButton", command=self._start_batch),
            "Inicia o processamento dos vídeos da lista: extração por IA (corrigida pelos marcadores do rito da "
            "sessão na transcrição), enriquecimentos (CNJ, tema, notícias) e publicação no Notion.",
        )
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = tip(
            ttk.Button(actions, text="Parar após vídeo atual", command=self._request_stop, state=tk.DISABLED),
            "Pede a parada do lote: o vídeo em andamento é concluído normalmente e os seguintes não são iniciados.",
        )
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(actions, text="Remover selecionado", command=self._remove_selected),
            "Remove da lista o(s) vídeo(s) selecionado(s) na tabela abaixo (antes de iniciar o lote).",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(actions, text="Limpar lista", command=self._clear_links),
            "Esvazia a lista de vídeos do lote.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(actions, text="Abrir artifacts", command=self._open_artifacts),
            "Abre no Explorer a pasta dos arquivos intermediários do lote (extrações, prévias e resultados de "
            "publicação) — útil para auditoria e diagnóstico.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(actions, text="Retomar artifacts", command=self._load_resume_root),
            "Reaproveita a extração de IA de um lote anterior (pasta de artifacts) para reprocessar as etapas "
            "seguintes sem gastar nova análise de vídeo.",
        ).pack(side=tk.LEFT, padx=(8, 0))

        columns = ("pos", "video_id", "sessao", "cc", "status", "result", "url")
        self.tree = tip(
            ttk.Treeview(exec_frame, columns=columns, show="headings", height=8),
            "Vídeos do lote e seu andamento. Colunas: # ordem; Video ID código do YouTube; Sessão = título e "
            "data detectados ao adicionar; CC = legenda disponível (CC) ou não (—); Status = etapa atual; "
            "Resultado = resumo ao concluir (criadas/atualizadas/bloqueadas/ignoradas).",
        )
        self.tree.grid(row=1, column=0, sticky="nsew")
        tree_scroll = ttk.Scrollbar(exec_frame, orient=tk.VERTICAL, command=self.tree.yview)
        tree_scroll.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.heading("pos", text="#")
        self.tree.heading("video_id", text="Video ID")
        self.tree.heading("sessao", text="Sessão")
        self.tree.heading("cc", text="CC")
        self.tree.heading("status", text="Status")
        self.tree.heading("result", text="Resultado")
        self.tree.heading("url", text="URL")
        self.tree.column("pos", width=40, stretch=False, anchor=tk.CENTER)
        self.tree.column("video_id", width=110, stretch=False)
        self.tree.column("sessao", width=260, stretch=True)
        self.tree.column("cc", width=36, stretch=False, anchor=tk.CENTER)
        self.tree.column("status", width=140, stretch=False)
        self.tree.column("result", width=300, stretch=True)
        self.tree.column("url", width=300, stretch=True)

        log_frame = ttk.LabelFrame(process_tab, text="Saída", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.output_text = tip(
            tk.Text(log_frame, wrap=tk.WORD, height=7, font=("Consolas", 9), relief=tk.FLAT, background="#f7f7f7"),
            "Log do processamento: mensagens de cada etapa, avisos, erros e o resumo final do lote.",
        )
        self.output_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scroll.set)

        # ================= ABA 2 — FILA DE VISTORIA =================
        vistoria_tab.columnconfigure(0, weight=1)
        vistoria_tab.rowconfigure(2, weight=3)
        vistoria_tab.rowconfigure(3, weight=1)

        ttk.Label(
            vistoria_tab,
            text="Julgamentos que o fluxo NÃO publicou e aguardam a sua decisão: selecione um item para ver os "
            "motivos completos abaixo, depois aprove (publica no Notion) ou descarte.",
            style="Muted.TLabel",
            wraplength=1150,
            justify=tk.LEFT,
        ).grid(row=0, column=0, sticky="w")

        vistoria_actions = ttk.Frame(vistoria_tab)
        vistoria_actions.grid(row=1, column=0, sticky="ew", pady=(8, 6))
        self.vistoria_publish_button = tip(
            ttk.Button(
                vistoria_actions, text="✔  Aprovar e publicar", style="Accent.TButton",
                command=self._approve_selected_vistoria,
            ),
            "Publica no Notion o(s) item(ns) selecionado(s): os erros que os bloquearam viram avisos "
            "'Aprovado em vistoria'. Só funciona para itens que carregam a linha pronta (skipped/blocked do "
            "lote); itens informativos (duplicatas, contagem do rito) não têm o que publicar.",
        )
        self.vistoria_publish_button.pack(side=tk.LEFT)
        tip(
            ttk.Button(vistoria_actions, text="Descartar item", command=self._reject_selected_vistoria),
            "Fecha o(s) item(ns) selecionado(s) sem publicar (ex.: era mesmo um precedente citado, não um "
            "julgamento). Reversível: o histórico fica no arquivo da fila.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(vistoria_actions, text="Abrir vídeo", command=self._open_selected_vistoria_video),
            "Abre no navegador o vídeo da sessão de origem do item, já no trecho do julgamento quando houver "
            "timestamp registrado.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(vistoria_actions, text="Abrir artifacts", command=self._open_selected_vistoria_artifact),
            "Abre no Explorer a pasta de artifacts do vídeo que gerou o item (extrações e transcrições usadas "
            "como evidência).",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(vistoria_actions, text="Recarregar", command=self._reload_vistoria),
            "Relê a fila de vistoria do disco — use após rodar um lote ou uma auditoria em paralelo.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Label(vistoria_actions, text="Situação:"),
            "Filtra a tabela por tipo de pendência.",
        ).pack(side=tk.LEFT, padx=(24, 6))
        self.vistoria_filter_var = tk.StringVar(value="Todas")
        vistoria_filter = tip(
            ttk.Combobox(
                vistoria_actions, textvariable=self.vistoria_filter_var, state="readonly", width=18,
                values=("Todas", "skipped", "blocked", "duplicata_numero", "faltante_dje", "contagem_rito"),
            ),
            "Tipos de pendência:\n"
            "• skipped — o fluxo descartou o item (ex.: possível precedente citado, densidade baixa);\n"
            "• blocked — barrado por dados insuficientes/incoerentes (sem resultado, tema vazio, vista sem ministro);\n"
            "• duplicata_numero — mesmo julgamento aparece com dois números divergentes na base;\n"
            "• faltante_dje — consta do CSV oficial do DJE e é mencionado no vídeo, mas não tinha página;\n"
            "• contagem_rito — a transcrição indica mais julgamentos apregoados do que linhas na base.",
        )
        vistoria_filter.pack(side=tk.LEFT)
        vistoria_filter.bind("<<ComboboxSelected>>", lambda _e: self._reload_vistoria())
        self.vistoria_only_ts_var = tk.BooleanVar(value=False)
        tip(
            ttk.Checkbutton(
                vistoria_actions, text="Só com ⏱", variable=self.vistoria_only_ts_var,
                command=self._reload_vistoria,
            ),
            "Mostra apenas os itens cujo link do vídeo já tem o marcador de tempo (t=) apontando o momento do "
            "apregoamento — os casos mais fáceis de validar visualmente. A lista sempre ordena esses primeiro.",
        ).pack(side=tk.LEFT, padx=(12, 0))
        self.vistoria_summary_var = tk.StringVar(value="")
        tip(
            ttk.Label(vistoria_actions, textvariable=self.vistoria_summary_var, style="Muted.TLabel"),
            "Total de pendências e distribuição por situação (sem considerar o filtro).",
        ).pack(side=tk.RIGHT)

        vistoria_table = ttk.Frame(vistoria_tab)
        vistoria_table.grid(row=2, column=0, sticky="nsew")
        vistoria_table.columnconfigure(0, weight=1)
        vistoria_table.rowconfigure(0, weight=1)
        vistoria_columns = ("data", "ts", "numero", "tema", "disp", "origem", "video")
        self.vistoria_tree = tip(
            ttk.Treeview(vistoria_table, columns=vistoria_columns, show="headings", height=12),
            "Um item por julgamento candidato, ordenado com os que têm marcador de tempo (⏱) primeiro. Cores: "
            "âmbar = skipped (descartado pelo fluxo), vermelho = blocked (dados insuficientes), azul = "
            "informativo (duplicata/contagem). ⏱ = momento do apregoamento no vídeo (facilita a validação "
            "visual); Processo = nº CNJ quando conhecido; Fonte = quem detectou (batch, auditoria, dje, rito). "
            "Clique numa linha para ver os motivos completos no painel abaixo.",
        )
        self.vistoria_tree.grid(row=0, column=0, sticky="nsew")
        self.vistoria_tree.heading("data", text="Sessão")
        self.vistoria_tree.heading("ts", text="⏱")
        self.vistoria_tree.heading("numero", text="Processo")
        self.vistoria_tree.heading("tema", text="Tema")
        self.vistoria_tree.heading("disp", text="Situação")
        self.vistoria_tree.heading("origem", text="Fonte")
        self.vistoria_tree.heading("video", text="Vídeo")
        self.vistoria_tree.column("data", width=95, stretch=False)
        self.vistoria_tree.column("ts", width=75, stretch=False, anchor=tk.CENTER)
        self.vistoria_tree.column("numero", width=205, stretch=False)
        self.vistoria_tree.column("tema", width=420, stretch=True)
        self.vistoria_tree.column("disp", width=125, stretch=False)
        self.vistoria_tree.column("origem", width=80, stretch=False)
        self.vistoria_tree.column("video", width=110, stretch=False)
        vistoria_scroll = ttk.Scrollbar(vistoria_table, orient=tk.VERTICAL, command=self.vistoria_tree.yview)
        vistoria_scroll.grid(row=0, column=1, sticky="ns")
        self.vistoria_tree.configure(yscrollcommand=vistoria_scroll.set)
        self.vistoria_tree.bind("<<TreeviewSelect>>", self._show_vistoria_details)
        self.vistoria_tree.tag_configure("skipped", foreground="#7a4a00")
        self.vistoria_tree.tag_configure("blocked", foreground="#8a1f1f")
        self.vistoria_tree.tag_configure("info", foreground="#1f3a5f")

        details_frame = ttk.LabelFrame(vistoria_tab, text="Detalhes do item selecionado", padding=8)
        details_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        self.vistoria_details = tip(
            tk.Text(
                details_frame, wrap=tk.WORD, height=6, font=("Segoe UI", 9), relief=tk.FLAT,
                background="#fbf8f2", state=tk.DISABLED,
            ),
            "Detalhes do item selecionado na tabela: processo, sessão, motivos completos do descarte/bloqueio, "
            "ementa oficial do DJE (quando houver) e o caminho da pasta de artifacts com as evidências.",
        )
        self.vistoria_details.grid(row=0, column=0, sticky="nsew")
        details_scroll = ttk.Scrollbar(details_frame, orient=tk.VERTICAL, command=self.vistoria_details.yview)
        details_scroll.grid(row=0, column=1, sticky="ns")
        self.vistoria_details.configure(yscrollcommand=details_scroll.set)

        self.vistoria_items: dict[str, dict[str, Any]] = {}
        self._reload_vistoria()

    def _add_link(self) -> None:
        text = self.link_var.get()
        if self._add_links_from_text(text):
            self.link_var.set("")

    def _paste_links(self) -> None:
        try:
            text = self.root.clipboard_get()
        except Exception:
            messagebox.showwarning("Area de transferencia", "Nao foi possivel ler a area de transferencia.")
            return
        self._add_links_from_text(text)

    def _add_links_from_text(self, text: str) -> bool:
        added = False
        errors: list[str] = []
        existing_ids = {video.video_id for video in self.videos}
        for raw_url in split_candidate_urls(text):
            if len(self.videos) >= MAX_LINKS:
                errors.append(f"Limite de {MAX_LINKS} links atingido.")
                break
            try:
                video = normalize_video_input(len(self.videos) + 1, raw_url)
            except ValueError as exc:
                errors.append(str(exc))
                continue
            if video.video_id in existing_ids:
                errors.append(f"Video repetido ignorado: {video.video_id}")
                continue
            self.videos.append(video)
            existing_ids.add(video.video_id)
            added = True
            self.resume_root = None
            threading.Thread(target=self._probe_link_metadata, args=(video,), daemon=True).start()
        self._refresh_tree()
        if errors:
            messagebox.showwarning("Links", "\n".join(errors[:8]))
        return added

    def _refresh_tree(self) -> None:
        selected_id = self.tree.focus()
        self.tree.delete(*self.tree.get_children())
        refreshed: list[VideoInput] = []
        for index, video in enumerate(self.videos, start=1):
            refreshed_video = VideoInput(position=index, url=video.url, video_id=video.video_id)
            refreshed.append(refreshed_video)
            sessao, cc_badge = self.link_meta.get(video.video_id, ("(verificando...)", "?"))
            self.tree.insert(
                "",
                tk.END,
                iid=video.video_id,
                values=(index, video.video_id, sessao, cc_badge, "Pendente", "", video.url),
            )
        self.videos = refreshed
        self.count_var.set(f"{len(self.videos)}/{MAX_LINKS} links")
        if selected_id and self.tree.exists(selected_id):
            self.tree.selection_set(selected_id)
            self.tree.focus(selected_id)

    def _remove_selected(self) -> None:
        if self._is_running():
            messagebox.showwarning("Lote em execucao", "Nao altere a lista durante o processamento.")
            return
        selected = set(self.tree.selection())
        if not selected:
            return
        self.videos = [video for video in self.videos if video.video_id not in selected]
        self._refresh_tree()

    def _clear_links(self) -> None:
        if self._is_running():
            messagebox.showwarning("Lote em execucao", "Nao altere a lista durante o processamento.")
            return
        self.videos = []
        self.resume_root = None
        self._refresh_tree()
        self.output_text.delete("1.0", tk.END)

    def _load_resume_root(self) -> None:
        if self._is_running():
            messagebox.showwarning("Lote em execucao", "Nao altere a lista durante o processamento.")
            return
        picked = filedialog.askdirectory(
            title="Selecione a pasta do lote em artifacts",
            initialdir=str(BATCH_ARTIFACT_ROOT),
        )
        if not picked:
            return
        root = Path(picked)
        loaded: list[VideoInput] = []
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            match = re.match(r"^(\d+)_([A-Za-z0-9_-]+)$", child.name)
            if not match:
                continue
            position = int(match.group(1))
            video_id = match.group(2)
            loaded.append(
                VideoInput(
                    position=position,
                    video_id=video_id,
                    url=f"https://www.youtube.com/watch?v={video_id}",
                )
            )
        if not loaded:
            messagebox.showerror("Retomar artifacts", "Nenhum subdiretorio NN_videoid encontrado nessa pasta.")
            return
        self.resume_root = root
        self.batch_artifact_dir = str(root)
        self.videos = sorted(loaded, key=lambda item: item.position)[:MAX_LINKS]
        self._refresh_tree()
        self._append_output(f"Artifacts selecionados para retomada: {root}\n")

    def _options(self) -> BatchOptions:
        return BatchOptions(
            model=self.model_var.get().strip() or DEFAULT_GEMINI_MODEL,
            news_model=self.news_model_var.get().strip() or DEFAULT_NEWS_GEMINI_MODEL,
            with_news=bool(self.with_news_var.get()),
            publish=bool(self.publish_var.get()),
            continue_on_error=bool(self.continue_on_error_var.get()),
            post_publish_steps=("materia", "suspenso", "classe_nomes", "sanear") if self.post_publish_var.get() else (),
            recolor_labels=bool(self.recolor_labels_var.get()),
            watch_dje=bool(self.watch_dje_var.get()),
        )

    def _start_batch(self) -> None:
        if self._is_running():
            messagebox.showwarning("Lote em execucao", "Ja existe um lote em execucao.")
            return
        if not self.videos:
            messagebox.showerror("Links obrigatorios", "Adicione pelo menos um link do YouTube.")
            return
        if len(self.videos) > MAX_LINKS:
            messagebox.showerror("Links", f"O limite e de {MAX_LINKS} links.")
            return

        self.stop_event.clear()
        self.total_videos = len(self.videos)
        self.completed_videos = 0
        self.current_video_id = ""
        self.current_video_index = 0
        self.stage_labels.clear()
        self.stage_started_at.clear()
        self.progress_var.set(0.0)
        self.progress_text_var.set(f"Preparando lote com {self.total_videos} video(s)")
        self.start_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.output_text.delete("1.0", tk.END)
        for video in self.videos:
            self._update_tree_status(video.video_id, "Pendente", "")

        options = self._options()
        videos = list(self.videos)
        resume_root = self.resume_root
        self.worker = threading.Thread(
            target=self._run_worker,
            args=(videos, options, resume_root),
            daemon=True,
        )
        self.worker.start()

    def _run_worker(self, videos: list[VideoInput], options: BatchOptions, resume_root: Path | None) -> None:
        handler = QueueLogHandler(self.output_queue)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        root_logger = logging.getLogger()
        old_level = root_logger.level
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)
        try:
            summary = process_video_batch(
                videos,
                options,
                self.output_queue,
                self.stop_event,
                resume_root=resume_root,
            )
            self.output_queue.put(("batch_done", summary))
        except Exception as exc:
            self.output_queue.put(("fatal_error", str(exc), traceback.format_exc()))
        finally:
            root_logger.removeHandler(handler)
            root_logger.setLevel(old_level)
            self.output_queue.put(("worker_finished", None))

    def _request_stop(self) -> None:
        self.stop_event.set()
        self._append_output("\nParada solicitada. O video atual sera concluido antes de encerrar.\n")
        self.stop_button.configure(state=tk.DISABLED)

    def _open_artifacts(self) -> None:
        path = Path(self.batch_artifact_dir) if self.batch_artifact_dir else BATCH_ARTIFACT_ROOT
        open_path(path)

    def _is_running(self) -> bool:
        return bool(self.worker and self.worker.is_alive())

    def _drain_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                event = item[0]
                if event == "log":
                    self._append_output(str(item[1]))
                elif event == "status":
                    _, video_id, status, result = item
                    self._update_tree_status(str(video_id), str(status), str(result))
                elif event == "video_started":
                    _, video_id, index, total = item
                    self.current_video_id = str(video_id)
                    self.current_video_index = int(index)
                    self.total_videos = int(total)
                    self._refresh_live_progress(reschedule=False)
                elif event == "video_finished":
                    _, video_id, _result = item
                    self.completed_videos = min(self.completed_videos + 1, max(self.total_videos, 1))
                    self.stage_labels.pop(str(video_id), None)
                    self.stage_started_at.pop(str(video_id), None)
                    self.current_video_id = ""
                    self._refresh_live_progress(reschedule=False)
                elif event == "link_meta":
                    _, video_id, sessao, cc_badge = item
                    self.link_meta[str(video_id)] = (str(sessao), str(cc_badge))
                    if self.tree.exists(str(video_id)):
                        self.tree.set(str(video_id), "sessao", str(sessao))
                        self.tree.set(str(video_id), "cc", str(cc_badge))
                elif event == "vistoria_done":
                    self.vistoria_publish_button.configure(state=tk.NORMAL)
                    self._reload_vistoria()
                elif event == "batch_artifact_dir":
                    self.batch_artifact_dir = str(item[1])
                elif event == "batch_done":
                    summary = item[1]
                    message = (
                        "\nResumo: "
                        f"{summary.get('total_done', 0)} concluidos, "
                        f"{summary.get('total_error', 0)} com erro. "
                        f"Artifacts: {summary.get('artifact_dir', '')}\n"
                    )
                    self._append_output(message)
                elif event == "fatal_error":
                    _, error, detail = item
                    self._append_output(f"\nERRO FATAL: {error}\n{detail}\n")
                    messagebox.showerror("Erro no lote", str(error))
                elif event == "worker_finished":
                    self.start_button.configure(state=tk.NORMAL)
                    self.stop_button.configure(state=tk.DISABLED)
                    if not self._is_running():
                        self.current_video_id = ""
                        self._refresh_live_progress(reschedule=False)
        except queue.Empty:
            pass
        self.root.after(200, self._drain_output_queue)

    def _update_tree_status(self, video_id: str, status: str, result: str) -> None:
        if not self.tree.exists(video_id):
            return
        if status in TERMINAL_STATUSES or status == "Pendente":
            self.stage_labels.pop(video_id, None)
            self.stage_started_at.pop(video_id, None)
        else:
            previous = self.stage_labels.get(video_id)
            self.stage_labels[video_id] = status
            if previous != status:
                self.stage_started_at[video_id] = time.monotonic()

        self.tree.set(video_id, "status", self._display_status(video_id, status))
        if result:
            self.tree.set(video_id, "result", result)
        elif status not in TERMINAL_STATUSES and status != "Pendente":
            self.tree.set(video_id, "result", "Em execucao")
        self.tree.see(video_id)
        self._refresh_live_progress(reschedule=False)

    def _display_status(self, video_id: str, status: str) -> str:
        started = self.stage_started_at.get(video_id)
        if started is None or status in TERMINAL_STATUSES or status == "Pendente":
            return status
        return f"{status} ({format_elapsed(time.monotonic() - started)})"

    def _refresh_live_progress(self, *, reschedule: bool = True) -> None:
        if self.total_videos <= 0:
            self.progress_var.set(0.0)
            self.progress_text_var.set("Pronto")
            if reschedule:
                self.root.after(1000, self._refresh_live_progress)
            return

        active_id = self.current_video_id
        active_stage = self.stage_labels.get(active_id, "") if active_id else ""
        stage_fraction = STAGE_PROGRESS.get(active_stage, 0.0)
        percent = ((self.completed_videos + stage_fraction) / self.total_videos) * 100
        self.progress_var.set(max(0.0, min(100.0, percent)))

        if active_id and active_stage:
            elapsed = format_elapsed(time.monotonic() - self.stage_started_at.get(active_id, time.monotonic()))
            text = (
                f"Video {self.current_video_index}/{self.total_videos}: "
                f"{active_id} - {active_stage} ha {elapsed}"
            )
            self._update_active_row_clock(active_id)
        elif self.completed_videos >= self.total_videos and not self._is_running():
            text = f"Lote finalizado: {self.completed_videos}/{self.total_videos} video(s)"
            self.progress_var.set(100.0)
        elif self._is_running():
            text = f"Lote em execucao: {self.completed_videos}/{self.total_videos} video(s) concluidos"
        else:
            text = f"Pronto: {self.completed_videos}/{self.total_videos} video(s)"
        self.progress_text_var.set(text)
        if reschedule:
            self.root.after(1000, self._refresh_live_progress)

    def _update_active_row_clock(self, video_id: str) -> None:
        if not self.tree.exists(video_id):
            return
        status = self.stage_labels.get(video_id, "")
        if not status:
            return
        self.tree.set(video_id, "status", self._display_status(video_id, status))
        if self.tree.set(video_id, "result") in {"", "Em execucao"}:
            self.tree.set(video_id, "result", "Em execucao")

    def _append_output(self, text: str) -> None:
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    # ----- validação visual dos links (thread daemon; UI só via output_queue) -----

    def _probe_link_metadata(self, video: VideoInput) -> None:
        try:
            title = fetch_video_title_oembed(video.url)
        except Exception:
            title = ""
        sessao = summarize_link_meta(title)
        cc_badge = "?"
        try:
            api_cls = require_youtube_transcript_api()
            transcripts = api_cls().list(video.video_id)
            cc_badge = "CC" if any(True for _ in transcripts) else "—"
        except Exception:
            cc_badge = "—"
        self.output_queue.put(("link_meta", video.video_id, sessao, cc_badge))

    # ----- fila de vistoria -----

    def _reload_vistoria(self) -> None:
        try:
            items = vistoria_queue.load_items("pending")
        except Exception as exc:
            self._append_output(f"Falha ao carregar a fila de vistoria: {exc}\n")
            return
        total = len(items)
        counts = Counter(item.get("disposition", "?") for item in items)
        with_ts_total = sum(1 for item in items if item_timestamp_seconds(item) is not None)
        selected_filter = self.vistoria_filter_var.get() if hasattr(self, "vistoria_filter_var") else "Todas"
        if selected_filter != "Todas":
            items = [item for item in items if item.get("disposition") == selected_filter]
        if getattr(self, "vistoria_only_ts_var", None) and self.vistoria_only_ts_var.get():
            items = [item for item in items if item_timestamp_seconds(item) is not None]

        self.vistoria_items = {str(item["id"]): item for item in items}
        self.vistoria_tree.delete(*self.vistoria_tree.get_children())
        # Itens com marcador de tempo primeiro (validação visual mais fácil), depois por data.
        def sort_key(item):
            timestamp = item_timestamp_seconds(item)
            return (0 if timestamp is not None else 1, item.get("data_sessao") or "", item.get("id", ""))

        for item in sorted(items, key=sort_key):
            row = item.get("row") or {}
            dje = (item.get("extra") or {}).get("dje") or {}
            numero = row.get("numero_processo") or dje.get("numeroUnico") or ""
            tema = row.get("tema") or dje.get("ementa") or "; ".join(item.get("reasons") or [])
            disposition = item.get("disposition", "")
            tag = disposition if disposition in ("skipped", "blocked") else "info"
            timestamp = item_timestamp_seconds(item)
            self.vistoria_tree.insert(
                "",
                tk.END,
                iid=str(item["id"]),
                tags=(tag,),
                values=(
                    item.get("data_sessao", ""),
                    format_elapsed(timestamp) if timestamp is not None else "—",
                    numero,
                    str(tema)[:120],
                    disposition,
                    item.get("source", ""),
                    item.get("video_id", ""),
                ),
            )
        resumo = "  |  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        self.vistoria_summary_var.set(
            f"{total} pendente(s)  (⏱ {with_ts_total})   {resumo}" if total else "Fila vazia — nada aguardando revisão."
        )
        self.notebook.tab(self.vistoria_tab_index, text=f"  Fila de vistoria ({total})  ")
        self.vistoria_hint_var.set(f"⚠ {total} pendência(s) aguardam sua decisão — clique aqui" if total else "")

    def _show_vistoria_details(self, _event=None) -> None:
        selected = self._selected_vistoria_items()
        self.vistoria_details.configure(state=tk.NORMAL)
        self.vistoria_details.delete("1.0", tk.END)
        if selected:
            item = selected[0]
            row = item.get("row") or {}
            dje = (item.get("extra") or {}).get("dje") or {}
            lines = [
                f"Processo: {row.get('numero_processo') or dje.get('numeroUnico') or '(sem número)'}    "
                f"Sessão: {item.get('data_sessao') or '?'}    Situação: {item.get('disposition')}    "
                f"Fonte: {item.get('source')}",
            ]
            if row.get("tema"):
                lines.append(f"Tema: {row['tema']}")
            if dje.get("ementa"):
                lines.append(f"Ementa (DJE): {dje['ementa']}")
            lines.append("")
            lines.append("Motivos:")
            for reason in item.get("reasons") or ["(sem motivo registrado)"]:
                lines.append(f"  • {reason}")
            if item.get("artifact_dir"):
                lines.append("")
                lines.append(f"Artifacts: {item['artifact_dir']}")
            self.vistoria_details.insert("1.0", "\n".join(lines))
        self.vistoria_details.configure(state=tk.DISABLED)

    def _open_selected_vistoria_artifact(self) -> None:
        for item in self._selected_vistoria_items():
            artifact_dir = item.get("artifact_dir") or ""
            if artifact_dir and Path(artifact_dir).exists():
                open_path(Path(artifact_dir))
                return
        messagebox.showinfo("Fila de vistoria", "O item selecionado não tem pasta de artifacts registrada.")

    def _selected_vistoria_items(self) -> list[dict[str, Any]]:
        return [self.vistoria_items[iid] for iid in self.vistoria_tree.selection() if iid in self.vistoria_items]

    def _approve_selected_vistoria(self) -> None:
        selected = self._selected_vistoria_items()
        publishable = [item for item in selected if item.get("row")]
        if not selected:
            messagebox.showinfo("Fila de vistoria", "Selecione ao menos um item.")
            return
        if not publishable:
            messagebox.showinfo(
                "Fila de vistoria",
                "Os itens selecionados são informativos (sem linha para publicar). Use 'Descartar item' para fechá-los.",
            )
            return
        if not messagebox.askyesno(
            "Aprovar e publicar",
            f"Publicar {len(publishable)} item(ns) aprovados no Notion?",
        ):
            return
        self.vistoria_publish_button.configure(state=tk.DISABLED)
        threading.Thread(target=self._publish_vistoria_worker, args=(publishable,), daemon=True).start()

    def _publish_vistoria_worker(self, items: list[dict[str, Any]]) -> None:
        try:
            runtime = build_runtime_context()
            client = NotionSessoesClient(
                runtime["notion_api_key"],
                data_source_id=runtime["notion_data_source_id"],
                logger=LOGGER,
            )
            schema = client.fetch_schema()
            results = vistoria_queue.publish_approved_items(items, client, schema, apply=True)
            published_ids = [
                str(result.get("id"))
                for result in results
                if result.get("status") in {"created", "updated"}
            ]
            if published_ids:
                vistoria_queue.update_status(published_ids, "published")
            lines = [
                f"  {result.get('numero_processo', result.get('id', ''))}: {result.get('status')}"
                for result in results
            ]
            self.output_queue.put(("log", "Vistoria publicada:\n" + "\n".join(lines) + "\n"))
        except Exception as exc:
            self.output_queue.put(("log", f"ERRO ao publicar itens da vistoria: {exc}\n"))
        finally:
            self.output_queue.put(("vistoria_done",))

    def _reject_selected_vistoria(self) -> None:
        selected = self._selected_vistoria_items()
        if not selected:
            messagebox.showinfo("Fila de vistoria", "Selecione ao menos um item.")
            return
        if not messagebox.askyesno("Descartar", f"Descartar {len(selected)} item(ns) da fila?"):
            return
        vistoria_queue.update_status([str(item["id"]) for item in selected], "rejected")
        self._reload_vistoria()

    def _open_selected_vistoria_video(self) -> None:
        for item in self._selected_vistoria_items():
            url = item_video_link(item)
            if not url:
                video_id = item.get("video_id") or ""
                url = f"https://www.youtube.com/watch?v={video_id}" if video_id else ""
            if not url:
                continue
            if item_timestamp_seconds(item) is None:
                discovered = self._discover_vistoria_timestamp(item)
                if discovered is not None:
                    separator = "&" if "?" in url else "?"
                    url = f"{url}{separator}t={discovered}"
            webbrowser.open(url)
            break

    def _discover_vistoria_timestamp(self, item: dict[str, Any]) -> int | None:
        """Procura o momento do apregoamento nos artifacts do vídeo (nº do processo
        nos 02_judgment_NN.json) e grava a descoberta na fila para as próximas vezes."""
        artifact_dir = Path(item.get("artifact_dir") or "")
        row = item.get("row") or {}
        dje = (item.get("extra") or {}).get("dje") or {}
        numero = str(row.get("numero_processo") or dje.get("numeroUnico") or "")
        digits = re.sub(r"\D", "", numero)
        if not artifact_dir.exists() or len(digits) < 9:
            return None
        try:
            from import_dje_faltantes import find_mention_timestamp

            timestamp = find_mention_timestamp(artifact_dir, digits)
        except Exception:
            return None
        if timestamp is None:
            return None
        base_url = item_video_link(item) or f"https://www.youtube.com/watch?v={item.get('video_id', '')}"
        separator = "&" if "?" in base_url else "?"
        enriched_url = f"{base_url}{separator}t={timestamp}"
        try:
            vistoria_queue.update_status([str(item["id"])], item.get("status", "pending"), extra={"youtube_url": enriched_url})
            item["youtube_url"] = enriched_url
            self._reload_vistoria()
        except Exception:
            pass
        self._append_output(f"Timestamp localizado nos artifacts: t={timestamp}s para {numero or item.get('video_id')}\n")
        return timestamp


def main() -> None:
    root = tk.Tk()
    BatchGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
