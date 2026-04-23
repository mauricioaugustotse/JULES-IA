from __future__ import annotations

import queue
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from pipeline_pre_news import PLAYLIST_PLACEHOLDER, discover_playlist_url_for_year


DEFAULT_PLAYLIST = PLAYLIST_PLACEHOLDER
DEFAULT_SUPER_MODEL = "gpt-5.4-mini"
DEFAULT_SUPER_FOCUS = "quality-core"
DEFAULT_SUPER_MIN_CONFIDENCE = "medium"


class PipelineLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pipeline TSE até quality-core")
        self.root.geometry("980x720")

        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self.mode_var = tk.StringVar(value="from-scratch")
        self.playlist_var = tk.StringVar(value=DEFAULT_PLAYLIST)
        self.year_var = tk.StringVar(value="2022")
        self.limit_var = tk.StringVar(value="0")
        self.initial_workers_var = tk.StringVar(value="3")
        self.max_workers_var = tk.StringVar(value="3")
        self.auto_scale_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=False)
        self.super_model_var = tk.StringVar(value=DEFAULT_SUPER_MODEL)
        self.super_focus_var = tk.StringVar(value=DEFAULT_SUPER_FOCUS)
        self.super_min_confidence_var = tk.StringVar(value=DEFAULT_SUPER_MIN_CONFIDENCE)
        self.repair_focus_var = tk.StringVar(value="all")

        self._build_ui()
        self._autofill_playlist_from_year()
        self._refresh_command_preview()
        self.root.after(250, self._drain_output_queue)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        mode_frame = ttk.LabelFrame(main, text="Modo")
        mode_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(
            mode_frame,
            text="1) Rodar do zero",
            value="from-scratch",
            variable=self.mode_var,
            command=self._refresh_command_preview,
        ).pack(anchor=tk.W, padx=10, pady=4)
        ttk.Radiobutton(
            mode_frame,
            text="2) Continuar após a primeira passada Gemini",
            value="post-gemini",
            variable=self.mode_var,
            command=self._refresh_command_preview,
        ).pack(anchor=tk.W, padx=10, pady=4)

        form = ttk.Frame(main)
        form.pack(fill=tk.X, pady=(0, 10))
        form.columnconfigure(1, weight=1)

        self._add_labeled_entry(form, 0, "Playlist URL", self.playlist_var, width=90)
        self._add_labeled_entry(form, 1, "Ano", self.year_var, width=12)
        self._add_labeled_entry(form, 2, "Limite (0 = todos)", self.limit_var, width=12)
        self._add_labeled_entry(form, 3, "Workers iniciais", self.initial_workers_var, width=12)
        self._add_labeled_entry(form, 4, "Workers máximos", self.max_workers_var, width=12)
        self._add_labeled_entry(form, 5, "Super model", self.super_model_var, width=20)
        self._add_labeled_entry(form, 6, "Super focus", self.super_focus_var, width=20)
        self._add_labeled_entry(form, 7, "Min confidence", self.super_min_confidence_var, width=20)
        self._add_labeled_entry(form, 8, "Repair focus", self.repair_focus_var, width=20)

        options = ttk.Frame(main)
        options.pack(fill=tk.X, pady=(0, 10))
        ttk.Checkbutton(
            options,
            text="Auto scale",
            variable=self.auto_scale_var,
            command=self._refresh_command_preview,
        ).pack(side=tk.LEFT, padx=(0, 12))
        ttk.Checkbutton(
            options,
            text="Resume",
            variable=self.resume_var,
            command=self._refresh_command_preview,
        ).pack(side=tk.LEFT)

        preview_frame = ttk.LabelFrame(main, text="Comando gerado")
        preview_frame.pack(fill=tk.X, pady=(0, 10))
        self.command_preview = tk.Text(preview_frame, height=6, wrap=tk.WORD)
        self.command_preview.pack(fill=tk.X, padx=8, pady=8)
        self.command_preview.configure(state=tk.DISABLED)

        actions = ttk.Frame(main)
        actions.pack(fill=tk.X, pady=(0, 10))
        self.start_button = ttk.Button(actions, text="Iniciar pipeline", command=self._start_pipeline)
        self.start_button.pack(side=tk.LEFT)
        ttk.Button(actions, text="Copiar comando", command=self._copy_command).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Abrir pasta de artifacts", command=self._open_artifacts_folder).pack(side=tk.LEFT)

        log_frame = ttk.LabelFrame(main, text="Saída")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.output_text = tk.Text(log_frame, wrap=tk.WORD)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.output_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        for var in [
            self.playlist_var,
            self.year_var,
            self.limit_var,
            self.initial_workers_var,
            self.max_workers_var,
            self.super_model_var,
            self.super_focus_var,
            self.super_min_confidence_var,
            self.repair_focus_var,
        ]:
            var.trace_add("write", lambda *_: self._refresh_command_preview())
        self.year_var.trace_add("write", lambda *_: self._autofill_playlist_from_year())

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        width: int,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=tk.W, padx=(0, 12), pady=4)
        ttk.Entry(parent, textvariable=variable, width=width).grid(row=row, column=1, sticky=tk.EW, pady=4)

    def _build_command(self) -> list[str]:
        year = self.year_var.get().strip()
        playlist = self.playlist_var.get().strip()
        limit = self.limit_var.get().strip() or "0"
        initial_workers = self.initial_workers_var.get().strip() or "3"
        max_workers = self.max_workers_var.get().strip() or "3"
        command = [
            sys.executable,
            "pipeline_pre_news.py",
            "--playlist-url",
            playlist,
            "--year",
            year,
            "--super-model",
            self.super_model_var.get().strip() or DEFAULT_SUPER_MODEL,
            "--super-focus",
            self.super_focus_var.get().strip() or DEFAULT_SUPER_FOCUS,
            "--super-min-confidence",
            self.super_min_confidence_var.get().strip() or DEFAULT_SUPER_MIN_CONFIDENCE,
            "--repair-focus",
            self.repair_focus_var.get().strip() or "all",
        ]
        if self.mode_var.get() == "post-gemini":
            command.append("--skip-initial-backfill")
        else:
            command.extend(
                [
                    "--limit",
                    limit,
                    "--initial-workers",
                    initial_workers,
                    "--max-workers",
                    max_workers,
                ]
            )
            if self.auto_scale_var.get():
                command.append("--auto-scale")
            if self.resume_var.get():
                command.append("--resume")
        return command

    def _autofill_playlist_from_year(self) -> None:
        current = self.playlist_var.get().strip()
        if current and current != DEFAULT_PLAYLIST:
            return
        year_text = self.year_var.get().strip()
        if not year_text.isdigit():
            return
        try:
            playlist = discover_playlist_url_for_year(int(year_text))
        except Exception:
            return
        self.playlist_var.set(playlist)

    def _refresh_command_preview(self) -> None:
        command = shlex.join(self._build_command())
        self.command_preview.configure(state=tk.NORMAL)
        self.command_preview.delete("1.0", tk.END)
        self.command_preview.insert(tk.END, command)
        self.command_preview.configure(state=tk.DISABLED)

    def _copy_command(self) -> None:
        command = shlex.join(self._build_command())
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        self._append_output("Comando copiado para a área de transferência.\n")

    def _open_artifacts_folder(self) -> None:
        path = Path.cwd() / "artifacts" / "tse_youtube_notion" / "pipeline_pre_news"
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer.exe", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def _start_pipeline(self) -> None:
        if self.process and self.process.poll() is None:
            messagebox.showwarning("Pipeline em execução", "Já existe um pipeline em execução nesta janela.")
            return

        playlist = self.playlist_var.get().strip()
        year = self.year_var.get().strip()
        if not playlist or playlist == DEFAULT_PLAYLIST:
            messagebox.showerror("Campo obrigatório", "Informe a playlist.")
            return
        if not year.isdigit():
            messagebox.showerror("Campo obrigatório", "Informe um ano válido.")
            return

        command = self._build_command()
        self.output_text.delete("1.0", tk.END)
        self._append_output(f"Iniciando: {shlex.join(command)}\n\n")
        self.start_button.configure(state=tk.DISABLED)
        self.process = subprocess.Popen(
            command,
            cwd=Path.cwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        threading.Thread(target=self._read_process_output, daemon=True).start()

    def _read_process_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line)
        self.process.wait()
        self.output_queue.put(f"\nProcesso finalizado com código {self.process.returncode}.\n")
        self.output_queue.put("__PIPELINE_FINISHED__")

    def _drain_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item == "__PIPELINE_FINISHED__":
                    self.start_button.configure(state=tk.NORMAL)
                    continue
                self._append_output(item)
        except queue.Empty:
            pass
        self.root.after(250, self._drain_output_queue)

    def _append_output(self, text: str) -> None:
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)


def main() -> None:
    root = tk.Tk()
    app = PipelineLauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
