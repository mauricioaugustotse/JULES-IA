from __future__ import annotations

import json
import logging
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
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
    extract_youtube_video_id,
    normalize_youtube_link,
    publish_preview_rows,
    validate_preview_row,
)


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
) -> dict[str, Any]:
    progress("analisando video")
    extractor = GeminiSessionExtractor(
        api_key=gemini_api_key,
        model=options.model,
        artifact_store=artifact_store,
        logger=LOGGER,
    )
    analysis = extractor.analyze_session(video.url)
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
    artifact_store.write_json("06_batch_video_summary.json", summary)
    return summary


def process_video_batch(
    videos: list[VideoInput],
    options: BatchOptions,
    output_queue: "queue.Queue[tuple[str, Any]]",
    stop_event: threading.Event,
    resume_root: Path | None = None,
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
            summary = process_single_video(
                video,
                artifact_store=artifact_store,
                notion_client=notion_client,
                notion_schema=notion_schema,
                gemini_api_key=gemini_key,
                options=options,
                progress=_progress,
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

    summary_payload = {
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
        self.root.geometry("1120x760")
        self.root.minsize(960, 680)

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

        self.link_var = tk.StringVar()
        self.model_var = tk.StringVar(value=DEFAULT_GEMINI_MODEL)
        self.news_model_var = tk.StringVar(value=DEFAULT_NEWS_GEMINI_MODEL)
        self.with_news_var = tk.BooleanVar(value=True)
        self.publish_var = tk.BooleanVar(value=True)
        self.continue_on_error_var = tk.BooleanVar(value=True)
        self.count_var = tk.StringVar(value=f"0/{MAX_LINKS} links")
        self.target_var = tk.StringVar(value=f"Notion: {DEFAULT_NOTION_DATABASE_URL}")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_text_var = tk.StringVar(value="Pronto")

        self._build_ui()
        self.root.after(200, self._drain_output_queue)
        self.root.after(1000, self._refresh_live_progress)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)
        main.rowconfigure(6, weight=2)

        header = ttk.Frame(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(header, text="TSE YouTube > Notion", font=("Segoe UI", 15, "bold")).pack(anchor=tk.W)
        ttk.Label(header, textvariable=self.target_var).pack(anchor=tk.W, pady=(2, 0))

        input_frame = ttk.LabelFrame(main, text="Links do YouTube", padding=8)
        input_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        input_frame.columnconfigure(0, weight=1)
        ttk.Entry(input_frame, textvariable=self.link_var).grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Button(input_frame, text="Adicionar link", command=self._add_link).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(input_frame, text="Colar da area", command=self._paste_links).grid(row=0, column=2)
        ttk.Label(input_frame, textvariable=self.count_var).grid(row=1, column=0, sticky=tk.W, pady=(6, 0))

        options = ttk.LabelFrame(main, text="Fluxo", padding=8)
        options.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)
        ttk.Label(options, text="Modelo Gemini").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(options, textvariable=self.model_var).grid(row=0, column=1, sticky="ew", padx=(0, 14))
        ttk.Label(options, text="Modelo noticias").grid(row=0, column=2, sticky=tk.W, padx=(0, 8))
        ttk.Entry(options, textvariable=self.news_model_var).grid(row=0, column=3, sticky="ew")
        ttk.Checkbutton(options, text="Buscar noticias antes de publicar", variable=self.with_news_var).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(6, 0),
        )
        ttk.Checkbutton(options, text="Publicar direto no Notion", variable=self.publish_var).grid(
            row=1,
            column=2,
            sticky=tk.W,
            pady=(6, 0),
        )
        ttk.Checkbutton(options, text="Continuar se um link falhar", variable=self.continue_on_error_var).grid(
            row=1,
            column=3,
            sticky=tk.W,
            pady=(6, 0),
        )

        actions = ttk.Frame(main)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.start_button = ttk.Button(actions, text="Processar lote", command=self._start_batch)
        self.start_button.pack(side=tk.LEFT)
        self.stop_button = ttk.Button(actions, text="Parar apos video atual", command=self._request_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Remover selecionado", command=self._remove_selected).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Limpar lista", command=self._clear_links).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Abrir artifacts", command=self._open_artifacts).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(actions, text="Retomar artifacts", command=self._load_resume_root).pack(side=tk.LEFT, padx=(8, 0))

        progress_frame = ttk.Frame(main)
        progress_frame.grid(row=4, column=0, sticky="ew", pady=(0, 8))
        progress_frame.columnconfigure(0, weight=1)
        ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=52).grid(row=0, column=1, sticky=tk.E)

        columns = ("pos", "video_id", "status", "result", "url")
        self.tree = ttk.Treeview(main, columns=columns, show="headings", height=9)
        self.tree.grid(row=5, column=0, sticky="nsew")
        self.tree.heading("pos", text="#")
        self.tree.heading("video_id", text="Video ID")
        self.tree.heading("status", text="Status")
        self.tree.heading("result", text="Resultado")
        self.tree.heading("url", text="URL")
        self.tree.column("pos", width=48, stretch=False, anchor=tk.CENTER)
        self.tree.column("video_id", width=130, stretch=False)
        self.tree.column("status", width=150, stretch=False)
        self.tree.column("result", width=360, stretch=True)
        self.tree.column("url", width=420, stretch=True)

        log_frame = ttk.LabelFrame(main, text="Saida", padding=8)
        log_frame.grid(row=6, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.output_text = tk.Text(log_frame, wrap=tk.WORD)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scroll.set)

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
            self.tree.insert(
                "",
                tk.END,
                iid=video.video_id,
                values=(index, video.video_id, "Pendente", "", video.url),
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

        values = list(self.tree.item(video_id, "values"))
        if len(values) < 5:
            return
        values[2] = self._display_status(video_id, status)
        if result:
            values[3] = result
        elif status not in TERMINAL_STATUSES and status != "Pendente":
            values[3] = "Em execucao"
        self.tree.item(video_id, values=values)
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
        values = list(self.tree.item(video_id, "values"))
        if len(values) < 5:
            return
        values[2] = self._display_status(video_id, status)
        if values[3] in {"", "Em execucao"}:
            values[3] = "Em execucao"
        self.tree.item(video_id, values=values)

    def _append_output(self, text: str) -> None:
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)


def main() -> None:
    root = tk.Tk()
    BatchGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
