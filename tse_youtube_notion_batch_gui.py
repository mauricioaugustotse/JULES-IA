from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
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
from tse_normalization import (infer_session_date_from_video_title,
                               normalize_numero_processo_display)

import vistoria_queue


LOGGER = logging.getLogger("tse_youtube_notion_batch_gui")

# Faixa "Acervo do TSE": a MESMA leitura que a GUI DJE_relatorios_semanais_gui faz, para
# que as duas telas digam exatamente a mesma coisa sobre o acervo consolidado.
# `tse_acervo` e stdlib-pura, entao importa sem problema aqui no .venv-win (que nao tem
# as demais dependencias de ProjetoConversor).
PROJETO_CONVERSOR = Path(r"C:\Users\mauri\ProjetoConversor")
if str(PROJETO_CONVERSOR) not in sys.path:
    sys.path.append(str(PROJETO_CONVERSOR))
try:
    import tse_acervo as tse_acervo_mod
except Exception:  # noqa: BLE001 — a faixa e informativa; sem ela a GUI segue
    tse_acervo_mod = None
# Ledger das etapas do DJe: diz o que JA foi feito pela outra GUI, para esta nao refazer nem
# deixar buraco. Tambem stdlib-pura, pelo mesmo motivo acima.
try:
    import dje_etapas as dje_etapas_mod
except Exception:  # noqa: BLE001
    dje_etapas_mod = None

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
    # Desligado por default: o monitor independente da pasta DJe (Tarefa Agendada
    # WatchDJe_Notion, via CRIAR_TAREFA_WATCH_DJE.ps1) cuida disso continuamente.
    # A caixa segue na GUI para uma rodada --once manual, se desejado.
    watch_dje: bool = False              # processa CSVs ja em DJE (--once)
    dje_apply: bool = True               # grava no Notion (vs dry-run)
    dje_dir: str = r"C:\Users\mauri\OneDrive\Documentos\12 - Consultoria Legislativa\DJe"
    # Baixa do portal do TSE as decisoes publicadas desde a ultima coleta, ANTES de
    # processar os videos: dispensa o export manual de `jurisprudencia.csv`. O coletor
    # deposita o CSV novo em `dje_dir`, de onde o watcher (ou o --once desta GUI) o pega.
    atualizar_tse: bool = True
    tse_coletor: str = r"C:\Users\mauri\ProjetoConversor\tse_coletor.py"
    tse_max_idade_horas: float = 12.0
    # Espera curta pelo captcha: no passo 0 de um lote longo voce esta esperando o lote
    # terminar, nao olhando a janela do Edge. E teto de duracao, para que um captcha
    # persistente nao arraste a rodada por horas antes do primeiro video.
    tse_espera_captcha: float = 120.0
    tse_teto_minutos: float = 45.0
    # Relations no Notion (mesmo processo dentro do DJe + DJe <-> sessoes), rodadas
    # ao final da pos-publicacao. Incremental e idempotente: le a base, compara e
    # so grava o que falta. Vive em ProjetoConversor, como o tse_coletor acima.
    # DESLIGADO por padrao desde 26/08/2026. A promessa de "2 s quando nada mudou" nao se
    # cumpre NESTA GUI, e nao por causa de sujeira: dje_etapas.relations_precisa_rodar tem dois
    # portoes NESTA ORDEM (dje_etapas.py:391) -- primeiro o TETO DE IDADE, que dispara sozinho
    # quando o ultimo sucesso passa de relations_max_idade_horas (20 h) e nem consulta o
    # --se-sujo; so depois (dje_etapas.py:394) a comparacao de sujeira. Em 26/08/2026 o portao
    # que abriu foi o teto ("ultimo sucesso ha 137 h"). Numa GUI usada a cada sessao do TSE o
    # teto de 20 h estoura praticamente sempre, entao a etapa cobra o preco cheio (15,8 min so
    # na 'interna' naquele dia, mais a 'cross' que o usuario interrompeu).
    # E, mesmo sem o teto, o segundo portao tambem abriria: quem suja e a INGESTAO do DJe
    # (ingerir_dje, logo acima nesta mesma pos-publicacao), nao a publicacao das sessoes --
    # geracao_paginas() so enxerga o lote pelo batch_summary/vistoria, escritos DEPOIS.
    # Quem religa as bases por padrao e a GUI irma "DJE Relatorios Semanais" (checkbox
    # "Atualizar relations ao final", que segue marcado la). Marque no Avancado quando quiser.
    # Este default tambem vale para run_batch_videos.py (CLI); reprocess_videos.py e
    # republish_missing_days.py ja passavam False explicitamente.
    atualizar_relations: bool = False
    relations_script: str = r"C:\Users\mauri\ProjetoConversor\relations_manutencao.py"
    # Etapas do freio: com --se-sujo, "ligado por padrao" nao significa 50 min por lote.
    # `temas` NAO entra aqui. Ela liga a base temas <-> base DJe e nao enxerga a base de
    # sessoes, entao publicar video nao lhe da trabalho nenhum. Pior: o custo dela e quase todo
    # fixo -- medido em 19/08/2026, 200 min para 61 gravacoes, dos quais ~1h so perguntando ao
    # Notion "ja gravei este teor?" em 13.898 paginas que ja estavam prontas. E, como o lock de
    # relations_manutencao e unico, ela sequestraria por horas as duas etapas baratas.
    # Onde `temas` roda: no atalho "Atualizar base Temas (TSE)", que e o gatilho de verdade
    # (a base temas ganhando julgados novos), e numa passada semanal de madrugada.
    relations_etapas: str = "interna,cross"
    relations_max_idade_horas: float = 20.0
    # Popular a base DJe do Notion com os CSVs pendentes, ao final do lote. Ate 19/08/2026 so a
    # GUI "DJE Relatorios Semanais" fazia isto, e um delta coletado por aqui ficava no limbo.
    # Roda DEPOIS de publicar (o SJUR chama OpenAI e e caro -- adiaria o primeiro video) e ANTES
    # das relations (o cross liga DJe <-> sessoes: sem as paginas do DJe nao ha o que ligar).
    ingerir_dje: bool = True
    ingerir_script: str = r"C:\Users\mauri\ProjetoConversor\dje_ingerir.py"
    # create-only: a automacao nunca sobrescreve curadoria manual. A GUI semanal, onde voce esta
    # na frente decidindo, mantem upsert.
    ingerir_modo: str = "create-only"
    # Confronto dos CSVs com a base de sessoes: "pendente" (alcanca as paginas novas),
    # "forcar" (tudo), "nao".
    enriquecer_sessoes: str = "pendente"


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


STRONG_EVIDENCE_RE = re.compile(
    r"prova local forte"                      # gate reconheceu o julgamento no vídeo
    r"|partes com opç\w+ novas? no Notion:"   # partes NOMEADAS: a extração viu a
    r"|advogados com opç\w+ novas? no Notion:"  # qualificação de um julgamento real,
    r"|partes \(indicados da lista tríplice\)",  # não uma citação de precedente
    re.IGNORECASE,
)


def item_has_strong_evidence(item: dict[str, Any]) -> bool:
    """Fortes candidatos à aprovação, destacados em grupo próprio (⭐): o gate
    registrou prova local forte OU o motivo cita nomes próprios de partes/
    advogados — sinal de julgamento concreto, não de precedente citado."""
    return any(STRONG_EVIDENCE_RE.search(str(reason)) for reason in item.get("reasons") or [])


def item_video_link(item: dict[str, Any]) -> str:
    row = item.get("row") or {}
    return str(row.get("youtube_link") or item.get("youtube_url") or "")


def vistoria_item_numero_display(item: dict[str, Any]) -> str:
    """Numero do processo de um item da fila, LEGIVEL para decidir.

    Ordem: campo estruturado (row/extra.dje/hint); nao havendo, pesca o numero das
    `reasons` (os itens `faltante_dje` trazem o CNJ so no texto do motivo — na tela
    apareciam como "(sem número)", visto em 20/08/2026). Sempre formata: 20 digitos
    crus viram NNNNNNN-DD.AAAA.J.TR.OOOO.
    """
    row = item.get("row") or {}
    dje = (item.get("extra") or {}).get("dje") or {}
    bruto = str(row.get("numero_processo") or dje.get("numeroUnico")
                or item.get("numero_hint") or "")
    if not bruto.strip():
        texto = " ".join(str(r) for r in item.get("reasons") or [])
        m = re.search(r"\b\d{7}-\d{2}\.\d{4}\.\d\.\d{2}\.\d{4}\b|\b\d{20}\b|\b\d{7}-\d{2}\b", texto)
        bruto = m.group(0) if m else ""
    if not bruto.strip():
        return ""
    digits = re.sub(r"\D", "", bruto)
    if len(digits) == 20:
        return (f"{digits[0:7]}-{digits[7:9]}.{digits[9:13]}."
                f"{digits[13]}.{digits[14:16]}.{digits[16:20]}")
    return normalize_numero_processo_display(bruto) or bruto


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
    widgets aninhados não empilham balões) e fica aberto ENQUANTO o ponteiro
    estiver sobre o widget: as dicas do painel "Avançado" têm vários parágrafos,
    e o antigo prazo fixo de 8s fechava o balão no meio da leitura.

    A proteção contra balão órfão que aquele prazo dava virou o vigia _watch(),
    que cobre os casos em que o <Leave> pode nunca chegar: widget destruído, aba
    trocada, janela minimizada ou aplicação mandada para segundo plano.
    """

    _active: "Tooltip | None" = None
    WATCH_MS = 250

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 500, wraplength: int = 440) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.wraplength = wraplength
        self._tip: tk.Toplevel | None = None
        self._after_id: str | None = None
        self._watch_id: str | None = None
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
        self._watch_id = self.widget.after(self.WATCH_MS, self._watch)

    def _pointer_inside(self) -> bool:
        """O ponteiro segue sobre o widget, com a janela à vista e em primeiro plano?"""
        w = self.widget
        try:
            if not w.winfo_exists() or not w.winfo_viewable():
                return False
            if w.focus_displayof() is None:  # app foi para segundo plano (Alt+Tab)
                return False
            x0, y0 = w.winfo_rootx(), w.winfo_rooty()
            px, py = w.winfo_pointerx(), w.winfo_pointery()
            return x0 <= px < x0 + w.winfo_width() and y0 <= py < y0 + w.winfo_height()
        except Exception:
            return False

    def _watch(self) -> None:
        """Mantém o balão enquanto o mouse estiver em cima; fecha assim que ele sair."""
        self._watch_id = None
        if not self._tip:
            return
        if not self._pointer_inside():
            self._hide()
            return
        self._watch_id = self.widget.after(self.WATCH_MS, self._watch)

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._watch_id:
            try:
                self.widget.after_cancel(self._watch_id)
            except Exception:
                pass
            self._watch_id = None
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
    stop_event: threading.Event | None = None,
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
    # GATE DE PARADA — o pedido de parada tem de valer para o video EM CURSO, nao so
    # para os seguintes: e no video em curso que o usuario acaba de ver o defeito no
    # log (degeneracao do modelo, numero fabricado) e quer impedir que aquilo entre
    # na base. A analise ja foi paga e fica inteira nos artifacts, para conferencia e
    # reprocessamento; o que nao acontece e a ESCRITA no Notion.
    parou_antes_de_publicar = bool(
        options.publish and stop_event is not None and stop_event.is_set()
    )
    if parou_antes_de_publicar:
        progress("parada solicitada — NADA foi publicado no Notion")
        LOGGER.warning(
            "[%s] Parada solicitada: %s linha(s) extraidas ficam nos artifacts e NADA foi "
            "escrito no Notion. Para publicar depois, reprocessar o video.",
            video.video_id,
            len(rows),
        )
        artifact_store.write_json(
            "05_publish_skipped_by_stop.json",
            {
                "motivo": "parada solicitada na GUI antes da publicacao",
                "video_id": video.video_id,
                "url": video.url,
                "rows_extracted": len(rows),
            },
        )
    elif options.publish:
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
        published=bool(options.publish) and not parou_antes_de_publicar,
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
    summary["publish_skipped_by_stop"] = parou_antes_de_publicar
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


def _gf_python_com_playwright() -> str:
    """O .venv-win desta GUI nao tem playwright; o Python 3.13 global tem. O coletor
    tambem sabe se reexecutar sozinho, mas apontar direto evita o salto."""
    cand = r"C:\Users\mauri\AppData\Local\Programs\Python\Python313\python.exe"
    return cand if Path(cand).exists() else sys.executable


def _gf_run_streaming(cmd: list[str], prefixo: str, cwd: Path,
                      output_queue: "queue.Queue[tuple[str, Any]]",
                      timeout: int = 7200) -> Any:
    """Roda um subprocesso transmitindo o stdout AO VIVO, linha a linha.

    Unico jeito de rodar comando com saida ao vivo nesta GUI — antes _gf_run_tse_update
    e _gf_run_relations repetiam este mesmo bloco. O streaming e essencial no coletor:
    a coleta pode pausar esperando captcha, e o aviso tem de aparecer na hora, nao ao
    final. (Os outros dois helpers _gf_* NAO foram unificados aqui de proposito: eles
    capturam e FILTRAM a saida, e o de labels ainda orquestra o Edge oculto.)
    """
    import subprocess
    try:
        # CREATE_NO_WINDOW: como esta GUI roda sob pythonw, o python.exe filho ganhava um
        # console proprio — uma janela preta que, fechada pelo usuario, mandava CTRL_CLOSE
        # a toda a arvore (0xC000013A). Foi o que matou o relations em 20/08/2026, deixando
        # lock orfao e a etapa 'cross' sem rodar. Sem janela, nao ha o que fechar.
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                             encoding="utf-8", errors="replace",
                             creationflags=flags,
                             env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        assert p.stdout is not None
        for ln in p.stdout:
            if ln.strip():
                output_queue.put(("log", f"[{prefixo}] {ln.rstrip()}\n"))
        p.wait(timeout=timeout)
        return p.returncode
    except Exception as exc:  # noqa: BLE001
        output_queue.put(("log", f"[{prefixo}] erro: {exc}\n"))
        return str(exc)


def _gf_run_tse_update(coletor: str, max_idade_horas: float,
                       output_queue: "queue.Queue[tuple[str, Any]]",
                       espera_captcha: float = 120.0,
                       teto_minutos: float = 45.0) -> Any:
    """Passo 0: baixa do portal do TSE o que foi publicado desde a ultima coleta.

    SEM --janela-dias de proposito: a varredura longa de 120 dias, que pega decisoes
    julgadas ha meses e publicadas agora, so acontece quando a janela nao e passada
    explicitamente. Falha aqui nao aborta o lote.
    """
    if not Path(coletor).exists():
        output_queue.put(("log", f"[tse] coletor nao encontrado: {coletor}\n"))
        return "ausente"
    cmd = [_gf_python_com_playwright(), "-X", "utf8", coletor, "atualizar",
           "--max-idade-horas", str(max_idade_horas),
           "--espera-captcha", str(espera_captcha),
           "--teto-minutos", str(teto_minutos),
           "--log-file", str(PROJETO_CONVERSOR / "Artefatos" / "reports" / "tse_coletor.log")]
    codigo = _gf_run_streaming(cmd, "tse", Path(coletor).parent, output_queue)
    if codigo == 5:
        output_queue.put(("log", "[tse] a coleta nao trouxe nenhum CSV — veja o motivo "
                                 "na faixa 'Acervo do TSE' (captcha, teto ou falha do portal)\n"))
    return codigo


def _gf_run_dje_enrich(dje_dir: str, apply: bool, modo: str,
                       output_queue: "queue.Queue[tuple[str, Any]]") -> Any:
    """Confronta os CSVs do DJe com a base de SESSOES (watch_jurisprudencia_csv --once).

    `modo`:
      "pendente"  --alcancar-novas: reprocessa os CSVs cujo confronto ficou para tras das
                  paginas ja criadas. E o padrao, e existe por causa de 19/08/2026: o watcher
                  consumiu o delta as 09:57 e as paginas do lote nasceram as 11:23, tarde demais.
      "forcar"    --force: reprocessa tudo o que esta na pasta, aplicado ou nao.
      "simples"   comportamento antigo (so o que nunca passou).

    Roda com sys.executable (o .venv-win) porque o watcher e deste repo -- ao contrario dos
    outros _gf_*, que chamam scripts do ProjetoConversor. Streaming ao vivo: esta etapa leva de
    minutos a dezenas de minutos, e antes aparecia como janela congelada (era o que escondia os
    24 min que o fill_composicao gastava revertendo paginas).
    """
    cmd = [sys.executable, str(_gf_dir() / "watch_jurisprudencia_csv.py"),
           "--watch-dir", dje_dir, "--once"]
    if apply:
        cmd.append("--apply")
    if modo == "pendente":
        cmd.append("--alcancar-novas")
    elif modo == "forcar":
        cmd.append("--force")
    codigo = _gf_run_streaming(cmd, "dje-sessoes", _gf_dir(), output_queue, timeout=14400)
    if codigo == 4:
        # Nao e sucesso: outro watcher segurava o lock e o confronto NAO rodou. Como o watcher
        # continuo nunca alcanca paginas nascidas depois, isso deixa trabalho por fazer.
        output_queue.put(("log", "[dje-sessoes] ATENCAO: o confronto nao rodou (ha um watcher "
                                 "ativo). As paginas publicadas neste lote podem ficar sem "
                                 "partes/advogados ate a proxima passada.\n"))
    return codigo


def _gf_run_dje_ingest(script: str, modo: str,
                       output_queue: "queue.Queue[tuple[str, Any]]") -> Any:
    """Popula a base DJe do Notion com os CSVs pendentes (dje_ingerir.py, ProjetoConversor).

    A etapa que faltava nesta GUI: ate 19/08/2026 o botao de coleta daqui depositava o delta na
    pasta e parava, e so a GUI "DJE Relatorios Semanais" sabia ingerir. Quem grava o estado
    continua sendo o manifesto daquela GUI -- ingerir por aqui faz o CSV aparecer como processado
    la, sem estado paralelo.

    Codigos: 0 ok · 1 falha · 2 outro processo ingerindo · 3 nada a fazer (os dois ultimos sao
    informativos, nao erro).
    """
    if not Path(script).exists():
        output_queue.put(("log", f"[dje-base] script nao encontrado: {script}\n"))
        return "script ausente"
    cmd = [_gf_python_com_playwright(), "-X", "utf8", script, "--modo", modo]
    codigo = _gf_run_streaming(cmd, "dje-base", Path(script).parent, output_queue, timeout=14400)
    if codigo == 3:
        output_queue.put(("log", "[dje-base] base DJe ja estava em dia; nada a ingerir.\n"))
    elif codigo == 2:
        output_queue.put(("log", "[dje-base] outro processo esta ingerindo agora; nada feito.\n"))
    return codigo


def _gf_run_relations(script: str, output_queue: "queue.Queue[tuple[str, Any]]",
                      etapas: str = "interna,cross",
                      max_idade_horas: float = 20.0) -> Any:
    """Relations no Notion via relations_manutencao.py (ProjetoConversor).

    Mesmo padrao de streaming do _gf_run_tse_update: roda com o Python 3.13
    global (o .venv-win desta GUI nao tem as deps de ProjetoConversor) e mostra
    a saida ao vivo. Incremental: rodadas repetidas so gravam o delta.

    Com --se-sujo, roda so quando nasceram paginas ou entraram decisoes desde o ultimo sucesso.
    E o que permite deixar relations LIGADO por padrao: o primeiro lote do dia paga os ~50 min,
    o segundo custa 2 segundos.
    """
    if not Path(script).exists():
        output_queue.put(("log", f"[relations] script nao encontrado: {script}\n"))
        return "script ausente"
    cmd = [_gf_python_com_playwright(), "-X", "utf8", script,
           "--etapas", etapas, "--se-sujo", "--max-idade-horas", str(max_idade_horas)]
    return _gf_run_streaming(cmd, "relations", Path(script).parent, output_queue,
                             timeout=14400)


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

    # Passo 0 — acervo do TSE em dia ANTES dos videos, para que o confronto com os CSVs
    # do DJe (pos-publicacao, --once) e o watcher ja encontrem o material novo. Fica
    # fora de qualquer condicao de publicacao: atualizar o acervo nao depende disso.
    if options.atualizar_tse and not stop_event.is_set():
        output_queue.put(("status", "__post__", "Atualizando o acervo do TSE...", ""))
        LOGGER.info("Atualizando o acervo do TSE a partir do portal.")
        _gf_run_tse_update(options.tse_coletor, options.tse_max_idade_horas, output_queue,
                           espera_captcha=options.tse_espera_captcha,
                           teto_minutos=options.tse_teto_minutos)
        output_queue.put(("acervo_mudou", None))

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
                stop_event=stop_event,
            )
            if summary.get("publish_skipped_by_stop"):
                # Nao e "done": nada foi para o Notion. Marcar como done faria o
                # bloco de pos-publicacao considerar que houve publicacao.
                summaries.append({"status": "stopped", **summary})
                final_status = (
                    f"Interrompido antes de publicar: {summary['rows_extracted']} linha(s) "
                    "ficaram nos artifacts, NADA foi escrito no Notion"
                )
                output_queue.put(("status", video.video_id, "Interrompido", final_status))
                output_queue.put(("video_finished", video.video_id, "stopped"))
                LOGGER.warning("[%s] %s", video.video_id, final_status)
            else:
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

    # ===== POS-PUBLICACAO GOING-FORWARD =====
    # A ORDEM AQUI E A LICAO DE 19/08/2026:
    #   publicar -> confrontar sessoes -> popular base DJe -> relations
    # Confrontar ANTES de publicar (que era o efeito de deixar isso com o watcher continuo)
    # enriquece paginas que ainda nao existem: as 6 daquele dia sairam com partes vazias.
    #
    # Os tratamentos de dados dependem do que foi publicado e ficam sob `publicou`. As tres
    # etapas seguintes tratam dos CSVs, nao dos videos: se o lote inteiro falhou, o delta do DJe
    # continua precisando entrar nas bases. `stop_event` barra tudo -- e o que o botao
    # "Parar antes de publicar" promete.
    post_publish: dict[str, Any] = {}
    publicou = any(item.get("status") == "done" for item in summaries)
    if options.publish and publicou and not stop_event.is_set() and options.post_publish_steps:
        dsid = runtime["notion_data_source_id"]
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

    # `options.publish` continua mandando: e a promessa de "nao escrever no Notion" que
    # run_batch_videos.py --no-publish faz na propria ajuda. Tirar estas etapas de baixo dele
    # faria um lote declaradamente somente-leitura gravar partes, criar paginas na base DJe e
    # rodar relations. O que muda em relacao ao original e so a queda do `publicou`: elas tratam
    # dos CSVs, nao dos videos, entao um lote que nao publicou nada ainda deve fechar as bases.
    if options.publish and not stop_event.is_set():
        if options.enriquecer_sessoes != "nao":
            output_queue.put(("status", "__post__",
                              "Confrontando os CSVs do DJe com a base de sessoes...", ""))
            post_publish["dje_sessoes"] = _gf_run_dje_enrich(
                options.dje_dir, options.dje_apply, options.enriquecer_sessoes, output_queue)
        elif options.watch_dje:
            # Modo legado (so CSVs nunca processados). Hoje so alcancavel por chamador externo
            # que passe enriquecer_sessoes="nao" junto de watch_dje=True; a GUI usa a caixa
            # "Pular o confronto", que produz "nao" sem ligar este ramo.
            output_queue.put(("status", "__post__", "Pos-publicacao: CSVs DJE (--once)...", ""))
            post_publish["dje"] = _gf_run_dje_enrich(
                options.dje_dir, options.dje_apply, "simples", output_queue)

        if options.ingerir_dje:
            output_queue.put(("status", "__post__", "Populando a base DJe do Notion...", ""))
            post_publish["dje_base"] = _gf_run_dje_ingest(
                options.ingerir_script, options.ingerir_modo, output_queue)
            output_queue.put(("etapas_mudou", None))

        if options.atualizar_relations:
            output_queue.put(("status", "__post__", "Pos-publicacao: relations no Notion...", ""))
            post_publish["relations"] = _gf_run_relations(
                options.relations_script, output_queue, options.relations_etapas,
                options.relations_max_idade_horas)
        output_queue.put(("status", "__post__", "Pos-publicacao concluida.", ""))
        output_queue.put(("etapas_mudou", None))

    # Falhas das etapas pos-publicacao NAO podem sumir no "0 com erro" do resumo: em
    # 20/08/2026 o relations morreu com 0xC000013A (janela de console fechada) e o lote
    # fechou "4 concluidos, 0 com erro" — o usuario so descobriu fuçando o JSON. Codigos
    # informativos por etapa: dje_sessoes 4 = watcher residente com o lock (adiado);
    # dje_base 2 = outra ingestao em curso, 3 = nada a fazer.
    _POST_OK: dict[str, set] = {
        "dje_sessoes": {0, 4}, "dje": {0, 4}, "dje_base": {0, 2, 3}, "relations": {0},
    }
    post_publish_falhas = sorted(
        etapa_nome for etapa_nome, codigo in post_publish.items()
        if etapa_nome in _POST_OK and not (
            isinstance(codigo, int) and codigo in _POST_OK[etapa_nome])
    )
    if "treatments_error" in post_publish:
        post_publish_falhas.insert(0, "treatments")
    if post_publish_falhas:
        for etapa_nome in post_publish_falhas:
            output_queue.put(("log",
                              f"ATENCAO: etapa pos-publicacao '{etapa_nome}' NAO concluiu "
                              f"(retorno {post_publish.get(etapa_nome)!r}). As bases podem "
                              f"ter ficado desatualizadas — veja o log acima.\n"))
        post_publish["falhas"] = post_publish_falhas

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
        # Falha pos-publicacao conta como erro do LOTE: o resumo "0 com erro" com o
        # relations morto foi o que escondeu o incidente de 20/08/2026.
        "total_error": sum(1 for item in summaries if item.get("status") == "error")
        + len(post_publish.get("falhas") or []),
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
        self.root.title("TSE YouTube → Notion — lote de vídeos")
        # cabe na tela (inclusive com escala do Windows) e abre encostada no topo,
        # senão o rodapé (progresso + aviso de vistoria) nasce atrás da barra de tarefas
        larg = min(1280, self.root.winfo_screenwidth() - 40)
        alt = min(840, self.root.winfo_screenheight() - 110)
        x = max((self.root.winfo_screenwidth() - larg) // 2, 0)
        self.root.geometry(f"{larg}x{alt}+{x}+8")
        self.root.minsize(1000, 640)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.output_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        # Threads de fundo que NAO sao o lote (coleta manual, por ora). Contador em vez de
        # referencia porque podem coexistir; ver _is_running.
        self._threads_avulsas = 0
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
        self.watch_dje_var = tk.BooleanVar(value=False)  # monitor independente (Tarefa WatchDJe_Notion) assume; marque p/ rodada --once manual
        # A coleta do TSE virou comportamento fixo (passo 0, com freio de 12 h): a
        # faixa "Acervo do TSE" no alto da janela e que mostra o estado. O que sobrou
        # de controle e o override, no painel Avancado.
        self.pular_tse_var = tk.BooleanVar(value=False)
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.acervo_l1_var = tk.StringVar(value="Acervo do TSE: consultando...")
        self.acervo_l2_var = tk.StringVar(value="")
        self.etapas_var = tk.StringVar(value="Etapas do DJe: consultando...")
        # 19/08/2026 ligou as duas por padrao apostando no freio --se-sujo do relations_manutencao.
        # A aposta valeu para a ingestao do DJe, nao para as relations: o freio tem um TETO DE
        # IDADE que roda ANTES da checagem de sujeira (dje_etapas.py:391) e dispara sozinho aos
        # 20 h -- numa GUI usada a cada sessao ele estoura sempre. Ver o comentario em
        # BatchOptions.atualizar_relations. Desde 26/08/2026 relations sai DESMARCADA aqui; quem a
        # mantem marcada por padrao e a GUI irma "DJE Relatorios Semanais" ("Atualizar relations
        # ao final"), semanal, onde o teto de 20 h e justamente o comportamento desejado.
        self.atualizar_relations_var = tk.BooleanVar(value=False)  # pos-publicacao: relations no Notion
        self.ingerir_dje_var = tk.BooleanVar(value=True)           # popular a base DJe do Notion
        # Confronto com a base de sessoes alcancando as paginas criadas neste lote. Marcar a caixa
        # do "Avancado" troca "pendente" por "forcar" (reprocessa tudo o que esta na pasta).
        self.forcar_sessoes_var = tk.BooleanVar(value=False)
        self.pular_sessoes_var = tk.BooleanVar(value=False)
        self.count_var = tk.StringVar(value=f"0/{MAX_LINKS} links")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.last_batch_dir, ultimo_txt = self._find_last_batch()
        self.idle_status_text = f"Pronto — {ultimo_txt}" if ultimo_txt else "Pronto"
        self.progress_text_var = tk.StringVar(value=self.idle_status_text)

        self._build_ui()
        self.root.after(200, self._drain_output_queue)
        self.root.after(1000, self._refresh_live_progress)

    @staticmethod
    def _find_last_batch() -> tuple[Path | None, str]:
        """(pasta, rótulo) do lote mais recente em batch_gui — memória entre sessões."""
        try:
            pastas = sorted(
                (p for p in BATCH_ARTIFACT_ROOT.iterdir()
                 if p.is_dir() and re.match(r"^\d{8}_\d{6}$", p.name)),
                key=lambda p: p.name, reverse=True)
        except OSError:
            return None, ""
        if not pastas:
            return None, ""
        ultima = pastas[0]
        quando = f"{ultima.name[6:8]}/{ultima.name[4:6]} {ultima.name[9:11]}:{ultima.name[11:13]}"
        try:
            resumo = json.loads((ultima / "batch_summary.json").read_text(encoding="utf-8"))
            done, err = resumo.get("total_done", "?"), resumo.get("total_error", 0)
            detalhe = f"{done} vídeo(s) ok" + (f", {err} com erro" if err else "")
        except Exception:
            detalhe = "interrompido"
        return ultima, f"último lote: {quando} ({detalhe})"

    def _on_close(self) -> None:
        if self._is_running() and not messagebox.askyesno(
                "Sair?",
                "Há um lote em execução — sair mesmo assim?\n"
                "O vídeo atual será interrompido e os tratamentos "
                "pós-publicação não rodarão."):
            return
        self.root.destroy()

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
        # níveis da faixa "Acervo do TSE" (os mesmos da GUI DJE Relatorios Semanais)
        style.configure("Ok.TLabel", foreground="#0a6b22", font=("Segoe UI", 10, "bold"))
        style.configure("Erro.TLabel", foreground="#b00020", font=("Segoe UI", 10, "bold"))

        main = ttk.Frame(self.root, padding=(12, 10))
        main.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(main)
        header.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(header, text="TSE YouTube → Notion — lote de vídeos",
                  font=("Segoe UI", 16, "bold")).pack(side=tk.LEFT)
        notion_link = ttk.Label(
            header, text="abrir base no Notion ↗", foreground="#0b5cad", cursor="hand2",
            font=("Segoe UI", 9, "underline"),
        )
        notion_link.pack(side=tk.RIGHT, pady=(8, 0))
        notion_link.bind("<Button-1>", lambda _e: webbrowser.open(DEFAULT_NOTION_DATABASE_URL))
        Tooltip(notion_link, "Abre no navegador a base de sessões do TSE no Notion, onde as linhas são publicadas.")
        ttk.Label(main, text="Fluxo: 1 links → 2 opções → 3 processar → 4 acompanhar a saída → "
                             "5 decidir as pendências na Fila de vistoria.",
                  style="Muted.TLabel").pack(fill=tk.X, pady=(0, 6))

        # ---------------------------------------------------- faixa: acervo do TSE
        # Mesma faixa, mesmos textos e mesma posição da GUI "DJE Relatorios Semanais":
        # fora do notebook, para continuar visível também na aba da Fila de vistoria.
        acervo_box = ttk.LabelFrame(main, text="Acervo do TSE", padding=8)
        acervo_box.pack(fill=tk.X, pady=(0, 8))
        acervo_box.columnconfigure(0, weight=1)
        self.acervo_l1 = ttk.Label(acervo_box, textvariable=self.acervo_l1_var,
                                   font=("Segoe UI", 10, "bold"))
        self.acervo_l1.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(acervo_box, textvariable=self.acervo_l2_var,
                  style="Muted.TLabel").grid(row=1, column=0, sticky=tk.W, pady=(2, 0))
        # Terceira linha: o que ainda falta fazer com os CSVs ja coletados. Sem ela, um delta
        # baixado por aqui podia ficar semanas sem entrar na base DJe sem ninguem notar -- foi
        # exatamente o que aconteceu em 19/08/2026.
        self.etapas_l = ttk.Label(acervo_box, textvariable=self.etapas_var, style="Muted.TLabel")
        self.etapas_l.grid(row=2, column=0, sticky=tk.W, pady=(4, 0))
        Tooltip(self.etapas_l,
                "Estado das duas bases do Notion para os CSVs que estão na pasta do DJe.\n\n"
                "É o MESMO estado que a GUI 'DJE Relatórios Semanais' enxerga: se você ingerir "
                "por lá, aqui aparece em dia, e vice-versa.\n\n"
                "O lote cuida disso sozinho ao final (veja o Avançado).")
        self.btn_coletar = ttk.Button(acervo_box, text="Coletar do TSE agora",
                                      command=self._coletar_tse_agora)
        self.btn_coletar.grid(row=0, column=1, rowspan=3, sticky=tk.E)
        Tooltip(self.btn_coletar,
                "Baixa do portal do TSE as decisões publicadas desde a última coleta e as "
                "acrescenta ao acervo consolidado.\n\n"
                "Use para ADIANTAR: clique aqui e vá colando os links dos vídeos enquanto o "
                "Edge trabalha. Quando o lote começar, esta etapa pula sozinha (freio de 12 h).\n\n"
                "UMA JANELA DO EDGE VAI ABRIR — é o coletor. Se o portal pedir captcha, "
                "resolva NELA.")
        self._atualizar_faixa_acervo()

        # a barra de status é empacotada ANTES do notebook, com side=bottom:
        # ganha prioridade de espaço e nunca é cortada em tela baixa (é nela
        # que vivem o progresso, o "último lote" e o aviso âmbar da vistoria)
        statusbar = ttk.Frame(main)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X, pady=(8, 0))

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        process_tab = ttk.Frame(self.notebook, padding=10)
        vistoria_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(process_tab, text="  Processar lote  ")
        self.notebook.add(vistoria_tab, text="  5 · Fila de vistoria  ")
        self.vistoria_tab_index = 1
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
        process_tab.rowconfigure(3, weight=3)   # execucao
        process_tab.rowconfigure(4, weight=1)   # log

        input_frame = ttk.LabelFrame(
            process_tab,
            text="1 · Links do YouTube  (título, data da sessão e legenda são verificados ao adicionar)",
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

        # Regra desta tela: só fica visível o que muda de rodada a rodada. Tudo o que
        # os próprios tooltips descreviam como "normalmente desnecessário" ou "só
        # altere se souber o que está fazendo" virou comportamento fixo, com override
        # no painel "Avançado" logo abaixo.
        options = ttk.LabelFrame(process_tab, text="2 · Opções desta rodada", padding=8)
        options.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        options.columnconfigure(2, weight=1)
        tip(
            ttk.Checkbutton(options, text="Publicar direto no Notion", variable=self.publish_var),
            "Marcado: as linhas extraídas são gravadas na base do Notion ao final. Desmarcado: o lote apenas gera "
            "os arquivos de prévia (artifacts) para conferência, sem publicar nada.",
        ).grid(row=0, column=0, sticky=tk.W)
        tip(
            ttk.Checkbutton(options, text="Buscar notícias antes de publicar", variable=self.with_news_var),
            "Antes de gravar no Notion, procura notícias oficiais (TSE/TREs) e da imprensa sobre cada julgamento "
            "e anexa os links aprovados à linha. Desmarque para um lote mais rápido e sem notícias.",
        ).grid(row=0, column=1, sticky=tk.W, padx=(20, 0))
        tip(
            ttk.Checkbutton(options, text="Avançado", variable=self.show_advanced_var),
            "Mostra as opções que quase nunca mudam: modelos de IA, tratamentos "
            "pós-publicação, recolorir etiquetas, relations e pular a coleta do TSE.",
        ).grid(row=0, column=3, sticky=tk.E)

        # ---------------------------------------------------------------- Avançado
        self.adv_frame = ttk.LabelFrame(process_tab, text="Avançado", padding=8)
        self.adv_frame.columnconfigure(1, weight=1)
        self.adv_frame.columnconfigure(3, weight=1)
        adv = self.adv_frame

        tip(
            ttk.Label(adv, text="Modelo Gemini"),
            "Modelo de IA que assiste ao vídeo e extrai os julgamentos (número, relator, resultado, análise). "
            "O padrão preenchido é o recomendado; só altere se souber o que está fazendo.",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        tip(
            ttk.Entry(adv, textvariable=self.model_var),
            "Nome do modelo Gemini usado na EXTRAÇÃO dos julgamentos do vídeo.",
        ).grid(row=0, column=1, sticky="ew", padx=(0, 14))
        tip(
            ttk.Label(adv, text="Modelo p/ notícias"),
            "Modelo de IA usado na etapa de busca de notícias relacionadas a cada julgamento.",
        ).grid(row=0, column=2, sticky=tk.W, padx=(0, 8))
        tip(
            ttk.Entry(adv, textvariable=self.news_model_var),
            "Nome do modelo Gemini usado na BUSCA/validação de notícias (TSE, TREs e imprensa).",
        ).grid(row=0, column=3, sticky="ew")

        ttk.Separator(adv, orient="horizontal").grid(row=1, column=0, columnspan=4,
                                                     sticky="ew", pady=8)
        ttk.Label(adv, text="Comportamento fixo — desmarque só se souber por quê:",
                  style="Muted.TLabel").grid(row=2, column=0, columnspan=4, sticky=tk.W)

        tip(
            ttk.Checkbutton(adv, text="Continuar se um link falhar", variable=self.continue_on_error_var),
            "Se um vídeo der erro (ex.: transmissão recém-encerrada ainda sem VOD), o lote registra a falha e "
            "segue para o próximo vídeo em vez de parar tudo.",
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(adv, text="Tratar dados após publicar (matéria semelhante, suspensos, classes, advogados)",
                            variable=self.post_publish_var),
            "Depois de publicar, roda os tratamentos automáticos de qualidade: vincular matéria semelhante, "
            "reconciliar julgamentos 'Suspenso' concluídos depois, normalizar classe processual e sanear dados.",
        ).grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=(14, 0), pady=(6, 0))
        tip(
            ttk.Checkbutton(adv, text="Pular a coleta do TSE nesta rodada",
                            variable=self.pular_tse_var),
            "Antes do primeiro vídeo, o lote baixa sozinho do portal do TSE as decisões publicadas "
            "desde a última coleta. Quem mostra o estado dessa coleta é o quadro 'Acervo do TSE', "
            "logo abaixo do título desta janela.\n\nMarque esta caixa para NÃO abrir o Edge nesta "
            "rodada — por exemplo quando você acabou de clicar em 'Coletar do TSE agora', o botão "
            "à direita daquele quadro, e o acervo já está em dia.",
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))

        ttk.Separator(adv, orient="horizontal").grid(row=5, column=0, columnspan=4,
                                                     sticky="ew", pady=8)
        ttk.Label(adv, text="Etapas de manutenção das bases (rodam depois de publicar):",
                  style="Muted.TLabel").grid(row=6, column=0, columnspan=4, sticky=tk.W)

        tip(
            ttk.Checkbutton(adv, text="Atualizar relations no Notion (pós-publicação)",
                            variable=self.atualizar_relations_var),
            "Ao final da pós-publicação, religa as páginas do mesmo processo no Notion — dentro da base "
            "DJe e entre DJe ↔ sessões (inclui as sessões publicadas por este lote).\n\nVEM DESMARCADA: "
            "existe um freio, mas ele tem um teto de idade que dispara sozinho quando a última "
            "passada passou de 20 h — numa janela usada a cada sessão do TSE isso acontece "
            "quase sempre, e aí a etapa cobra o preço cheio (de ~1 h a ~2 h; em 26/08/2026 "
            "foram 15,8 min só na primeira das duas metades).\nQuem religa as bases por "
            "padrão é a GUI 'DJE Relatórios Semanais' ('Atualizar relations ao final'), semanal, "
            "onde esse teto é justamente o que se quer. Marque aqui só se precisar dos vínculos hoje mesmo.\n\nA base de TEMAS não entra aqui: "
            "ela não enxerga as sessões, então publicar vídeo não lhe dá trabalho nenhum. Ela roda no "
            "atalho 'Atualizar base Temas (TSE)', que é o gatilho de verdade.",
        ).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(adv, text="Alimentar a base DJe do Notion (pós-publicação)",
                            variable=self.ingerir_dje_var),
            "Ao final do lote, joga na base DJe do Notion os CSVs que o coletor baixou e que ainda não "
            "entraram — a mesma coisa que o '▶ Gerar relatórios' da GUI DJE Relatórios Semanais faz.\n"
            "Detecta sozinho o que já foi feito lá: se você já tinha ingerido por aquela GUI, esta etapa "
            "sai em um segundo dizendo 'base em dia'.\nUsa modo create-only: nunca sobrescreve curadoria "
            "manual sua na base.",
        ).grid(row=7, column=2, columnspan=2, sticky=tk.W, padx=(14, 0), pady=(6, 0))
        tip(
            ttk.Checkbutton(adv, text="Reprocessar TODOS os CSVs contra a base de sessões",
                            variable=self.forcar_sessoes_var),
            "Normalmente desnecessário. Por padrão o lote já reconfronta os CSVs cujo processamento "
            "ficou para trás das páginas novas — que é o caso das páginas publicadas por este próprio "
            "lote.\nMarque para reprocessar tudo o que está na pasta do DJe, inclusive o que já estava "
            "em dia (mais lento).",
        ).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(6, 0))
        tip(
            ttk.Checkbutton(adv, text="Pular o confronto com a base de sessões",
                            variable=self.pular_sessoes_var),
            "Não roda o confronto dos CSVs do DJe com a base de sessões nesta rodada.\n\n"
            "Marque só quando o watcher automático (tarefa WatchDJe_Notion) estiver ativo E você "
            "não tiver publicado páginas novas — porque o watcher contínuo não alcança páginas "
            "criadas depois da passada dele.",
        ).grid(row=8, column=2, columnspan=2, sticky=tk.W, padx=(14, 0), pady=(6, 0))

        def _toggle_advanced(*_args: Any) -> None:
            if self.show_advanced_var.get():
                self.adv_frame.grid(row=2, column=0, sticky="ew", pady=(8, 0))
            else:
                self.adv_frame.grid_remove()

        self.show_advanced_var.trace_add("write", _toggle_advanced)

        exec_frame = ttk.LabelFrame(process_tab, text="3 · Execução", padding=8)
        exec_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
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
            ttk.Button(actions, text="Parar antes de publicar", command=self._request_stop, state=tk.DISABLED),
            "Pede a parada do lote e PROTEGE O NOTION: o vídeo em andamento termina a análise (que já foi "
            "paga) e grava tudo nos artifacts, mas NÃO publica; os vídeos seguintes não são iniciados e os "
            "tratamentos pós-publicação não rodam. Use ao ver defeito no log — para publicar depois, "
            "reprocesse o vídeo.",
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
            "publicação) — útil para auditoria e diagnóstico. Sem lote nesta sessão, abre a pasta do ÚLTIMO "
            "lote processado.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(actions, text="Retomar artifacts", command=self._load_resume_root),
            "Refaz um lote a partir da pasta de artifacts dele: a lista de vídeos é remontada pelos "
            "subdiretórios NN_videoid encontrados ali, e o novo lote grava NESSA MESMA pasta, por cima. "
            "ATENÇÃO: a análise de vídeo é REFEITA — o Gemini roda de novo e a extração antiga é "
            "sobrescrita, então vídeo que não chegou a criar pasta NÃO volta para a lista.",
        ).pack(side=tk.LEFT, padx=(8, 0))

        columns = ("pos", "video_id", "sessao", "cc", "status", "result", "url")
        self.tree = tip(
            ttk.Treeview(exec_frame, columns=columns, show="headings", height=7),
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
        self.tree.column("sessao", width=300, stretch=False)
        self.tree.column("cc", width=36, stretch=False, anchor=tk.CENTER)
        self.tree.column("status", width=140, stretch=False)
        self.tree.column("result", width=320, stretch=False)
        self.tree.column("url", width=320, stretch=False)
        tree_xscroll = ttk.Scrollbar(exec_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        tree_xscroll.grid(row=2, column=0, sticky="ew")
        self.tree.configure(xscrollcommand=tree_xscroll.set)
        self.tree.bind("<Button-3>", self._videos_context_menu)
        self.tree.bind("<Control-c>", lambda _e: self._copy_tree_selection(self.tree))
        self.tree.bind("<Control-a>", lambda _e: self._select_all_tree(self.tree))

        log_frame = ttk.LabelFrame(process_tab, text="4 · Saída (log)", padding=8)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.output_text = tip(
            tk.Text(log_frame, wrap=tk.WORD, height=6, font=("Consolas", 9), relief=tk.FLAT, background="#f7f7f7"),
            "Log do processamento: mensagens de cada etapa, avisos, erros e o resumo final do lote. "
            "Uma cópia com data/hora fica em artifacts\\batch_gui\\gui.log.",
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
            "julgamento). Para desfazer: escolha a visão '🗑 descartados' no filtro Situação, selecione o item "
            "e clique em Restaurar item.",
        ).pack(side=tk.LEFT, padx=(8, 0))
        tip(
            ttk.Button(vistoria_actions, text="Restaurar item", command=self._restore_selected_vistoria),
            "Devolve para a fila (pendente) um item descartado. Use com a visão '🗑 descartados' selecionada "
            "no filtro Situação para localizar o item.",
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
                vistoria_actions, textvariable=self.vistoria_filter_var, state="readonly", width=22,
                values=(
                    "Todas", "⭐ prova local forte", "skipped", "blocked",
                    "duplicata_numero", "faltante_dje", "contagem_rito", "🗑 descartados",
                ),
            ),
            "Tipos de pendência (e a visão 🗑 descartados, que lista os itens já fechados para "
            "conferência ou restauração):\n"
            "• ⭐ prova local forte — o próprio vídeo comprova o julgamento (número citado em vários trechos "
            "OU partes/advogados NOMEADOS no motivo), mas o item foi barrado por outra razão: FORTES candidatos "
            "à aprovação (fundo verde, topo da lista);\n"
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
        # stretch=False em todas: coluna elástica "rouba" de volta o espaço quando a
        # janela redesenha, desfazendo o redimensionamento manual do usuário.
        self.vistoria_tree.column("data", width=95, stretch=False)
        self.vistoria_tree.column("ts", width=75, stretch=False, anchor=tk.CENTER)
        self.vistoria_tree.column("numero", width=205, stretch=False)
        self.vistoria_tree.column("tema", width=520, stretch=False)
        self.vistoria_tree.column("disp", width=125, stretch=False)
        self.vistoria_tree.column("origem", width=80, stretch=False)
        self.vistoria_tree.column("video", width=110, stretch=False)
        vistoria_scroll = ttk.Scrollbar(vistoria_table, orient=tk.VERTICAL, command=self.vistoria_tree.yview)
        vistoria_scroll.grid(row=0, column=1, sticky="ns")
        vistoria_xscroll = ttk.Scrollbar(vistoria_table, orient=tk.HORIZONTAL, command=self.vistoria_tree.xview)
        vistoria_xscroll.grid(row=1, column=0, sticky="ew")
        self.vistoria_tree.configure(yscrollcommand=vistoria_scroll.set, xscrollcommand=vistoria_xscroll.set)
        self.vistoria_tree.bind("<<TreeviewSelect>>", self._show_vistoria_details)
        self.vistoria_tree.bind("<Button-3>", self._vistoria_context_menu)
        self.vistoria_tree.bind("<Control-c>", lambda _e: self._copy_tree_selection(self.vistoria_tree))
        self.vistoria_tree.bind("<Control-a>", lambda _e: self._select_all_tree(self.vistoria_tree))
        self.vistoria_tree.tag_configure("skipped", foreground="#7a4a00")
        self.vistoria_tree.tag_configure("blocked", foreground="#8a1f1f")
        self.vistoria_tree.tag_configure("info", foreground="#1f3a5f")
        self.vistoria_tree.tag_configure("strong", background="#e6f4e6", foreground="#1b5e20")

        details_frame = ttk.LabelFrame(vistoria_tab, text="Detalhes do item selecionado", padding=8)
        details_frame.grid(row=3, column=0, sticky="nsew", pady=(8, 0))
        details_frame.columnconfigure(0, weight=1)
        details_frame.rowconfigure(0, weight=1)
        self.vistoria_details = tip(
            tk.Text(
                details_frame, wrap=tk.WORD, height=6, font=("Segoe UI", 9), relief=tk.FLAT,
                background="#fbf8f2",
            ),
            "Detalhes do item selecionado na tabela: processo, sessão, motivos completos do descarte/bloqueio, "
            "ementa oficial do DJE (quando houver) e o caminho da pasta de artifacts. O texto é selecionável: "
            "arraste o mouse e copie com Ctrl+C.",
        )
        self.vistoria_details.grid(row=0, column=0, sticky="nsew")
        # Somente leitura SEM state=DISABLED: o usuário seleciona e copia livremente,
        # mas qualquer tecla de edição é bloqueada.
        self.vistoria_details.bind("<Key>", self._readonly_text_key)
        self.vistoria_details.bind("<Button-3>", self._details_context_menu)
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
        # o log NÃO é apagado aqui — só no início de um novo lote (o efeito
        # colateral de "Limpar lista" apagar a saída surpreendia o usuário)

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
            watch_dje=bool(self.watch_dje_var.get()),
            atualizar_tse=not bool(self.pular_tse_var.get()),
            atualizar_relations=bool(self.atualizar_relations_var.get()),
            ingerir_dje=bool(self.ingerir_dje_var.get()),
            enriquecer_sessoes=("nao" if self.pular_sessoes_var.get()
                                else "forcar" if self.forcar_sessoes_var.get() else "pendente"),
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
        self._append_output("\nParada solicitada — o vídeo em andamento termina a análise e grava os "
                            "artifacts, mas NÃO publica no Notion; os seguintes não começam e os "
                            "tratamentos pós-publicação NÃO rodarão.\n")
        self.progress_text_var.set("Parada solicitada — o vídeo atual não será publicado "
                                   "(sem tratamentos pós-publicação)")
        self.stop_button.configure(state=tk.DISABLED)

    def _open_artifacts(self) -> None:
        # prioridade: lote da sessão atual > último lote encontrado no boot > raiz
        if self.batch_artifact_dir:
            path = Path(self.batch_artifact_dir)
        elif self.last_batch_dir is not None:
            path = self.last_batch_dir
        else:
            path = BATCH_ARTIFACT_ROOT
        open_path(path)

    def _is_running(self) -> bool:
        """Há trabalho pesado em andamento? Conta o lote E as threads avulsas.

        Antes olhava só `self.worker`, então a thread da coleta manual escapava: dava para
        clicar "Processar lote" com o Edge do coletor aberto, e as duas rodadas se atropelavam.
        """
        return bool((self.worker and self.worker.is_alive()) or self._threads_avulsas > 0)

    # ------------------------------------------------------ faixa: acervo do TSE
    def _atualizar_faixa_acervo(self) -> None:
        """Repinta a faixa do acervo. Barato: só lê o estado.json do coletor.

        As duas linhas são independentes: a de etapas é repintada num `finally`, senão uma falha
        do `tse_acervo` (índice sqlite corrompido, disco cheio) deixaria a linha de etapas presa
        em "consultando..." — perdendo justo o alarme de "há um delta esquecido".
        """
        estilos = {"ok": "Ok.TLabel", "atencao": "Hint.TLabel", "erro": "Erro.TLabel"}
        try:
            if tse_acervo_mod is None:
                self.acervo_l1_var.set("Acervo do TSE: indisponível")
                self.acervo_l2_var.set(f"não consegui importar tse_acervo de {PROJETO_CONVERSOR}")
                self.acervo_l1.configure(style="Erro.TLabel")
                return
            try:
                p = tse_acervo_mod.resumo_painel()
            except Exception as exc:  # noqa: BLE001
                self.acervo_l1_var.set("Acervo do TSE: indisponível")
                self.acervo_l2_var.set(str(exc)[:120])
                self.acervo_l1.configure(style="Erro.TLabel")
                return
            self.acervo_l1_var.set(p["linha1"])
            self.acervo_l2_var.set(p["linha2"])
            self.acervo_l1.configure(style=estilos.get(p["nivel"], "Muted.TLabel"))
        finally:
            self._atualizar_faixa_etapas()

    def _atualizar_faixa_etapas(self) -> None:
        """Repinta a linha de etapas do DJe. Barato: só lê JSONs locais, nunca o Notion."""
        if dje_etapas_mod is None:
            self.etapas_var.set("Etapas do DJe: indisponível (dje_etapas não importou)")
            return
        try:
            p = dje_etapas_mod.resumo_painel()
        except Exception as exc:  # noqa: BLE001
            self.etapas_var.set(f"Etapas do DJe: indisponível ({str(exc)[:80]})")
            return
        self.etapas_var.set(p["linha1"])
        self.etapas_l.configure(style="Hint.TLabel" if p["nivel"] == "atencao" else "Muted.TLabel")

    def _coletar_tse_agora(self) -> None:
        """Coleta antecipada: adianta o acervo enquanto você prepara os links.

        Depois disto, o passo 0 do lote pula sozinho pelo freio de 12 h — sem coleta
        duplicada e sem espera na hora de processar os vídeos.
        """
        if self._is_running():
            messagebox.showwarning("Lote em execucao",
                                   "Ha um lote em execucao; aguarde para coletar.")
            return
        opts = self._options()
        self.btn_coletar.configure(state=tk.DISABLED)
        self._append_output("[tse] coleta pedida a mao — uma janela do Edge vai abrir\n")

        def worker() -> None:
            try:
                # espera longa pelo captcha: aqui VOCE esta na frente e pode resolver
                _gf_run_tse_update(opts.tse_coletor, opts.tse_max_idade_horas,
                                   self.output_queue, espera_captcha=1800.0,
                                   teto_minutos=opts.tse_teto_minutos)
            finally:
                self._threads_avulsas -= 1
                self.output_queue.put(("acervo_mudou", None))

        self._threads_avulsas += 1
        threading.Thread(target=worker, daemon=True).start()

    def _drain_output_queue(self) -> None:
        try:
            while True:
                item = self.output_queue.get_nowait()
                event = item[0]
                if event == "log":
                    self._append_output(str(item[1]))
                elif event == "acervo_mudou":
                    # a coleta terminou (passo 0 ou botão da faixa): repinta o estado
                    self._atualizar_faixa_acervo()
                    if not self._is_running():
                        self.btn_coletar.configure(state=tk.NORMAL)
                elif event == "etapas_mudou":
                    # ingestão/confronto terminou: só a terceira linha precisa repintar
                    self._atualizar_faixa_etapas()
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
                    # O resumo tem de carregar TUDO que precisa de atencao: em 20/08/2026
                    # um video saiu com 7 julgamentos a menos (verdict "verificar") e o
                    # relations morreu — e o resumo dizia "4 concluidos, 0 com erro".
                    videos_verificar = [
                        str(v.get("video_id", "?"))
                        for v in (summary.get("videos") or [])
                        if isinstance(v, dict)
                        and (v.get("rito_count_check") or {}).get("verdict") == "verificar"
                    ]
                    pp_falhas = (summary.get("post_publish") or {}).get("falhas") or []
                    message = (
                        "\nResumo: "
                        f"{summary.get('total_done', 0)} concluidos, "
                        f"{summary.get('total_error', 0)} com erro. "
                        f"Artifacts: {summary.get('artifact_dir', '')}\n"
                    )
                    if videos_verificar:
                        message += (
                            f"ATENCAO: {len(videos_verificar)} video(s) com contagem do rito "
                            f"divergente ({', '.join(videos_verificar)}) — pode haver "
                            f"julgamento sem linha publicada; confira a fila de vistoria.\n"
                        )
                    if pp_falhas:
                        message += (
                            f"ATENCAO: etapa(s) pos-publicacao que NAO concluiram: "
                            f"{', '.join(pp_falhas)} — veja o log acima.\n"
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
            self.progress_text_var.set(self.idle_status_text)
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
        strong_total = sum(1 for item in items if item_has_strong_evidence(item))
        selected_filter = self.vistoria_filter_var.get() if hasattr(self, "vistoria_filter_var") else "Todas"
        showing_rejected = selected_filter.startswith("🗑")
        if showing_rejected:
            try:
                items = vistoria_queue.load_items("rejected")
            except Exception as exc:
                self._append_output(f"Falha ao carregar descartados: {exc}\n")
                return
        elif selected_filter.startswith("⭐"):
            items = [item for item in items if item_has_strong_evidence(item)]
        elif selected_filter != "Todas":
            items = [item for item in items if item.get("disposition") == selected_filter]
        if getattr(self, "vistoria_only_ts_var", None) and self.vistoria_only_ts_var.get():
            items = [item for item in items if item_timestamp_seconds(item) is not None]

        self.vistoria_items = {str(item["id"]): item for item in items}
        self.vistoria_tree.delete(*self.vistoria_tree.get_children())
        # Prioridade: ⭐ prova local forte no topo, depois quem tem marcador de tempo;
        # dentro de cada grupo, sessão mais RECENTE primeiro (sorts estáveis).
        ordered = sorted(items, key=lambda x: (x.get("data_sessao") or "", x.get("id", "")), reverse=True)
        ordered = sorted(
            ordered,
            key=lambda x: (
                0 if item_has_strong_evidence(x) else 1,
                0 if item_timestamp_seconds(x) is not None else 1,
            ),
        )

        for item in ordered:
            row = item.get("row") or {}
            dje = (item.get("extra") or {}).get("dje") or {}
            numero = vistoria_item_numero_display(item)
            tema = (
                row.get("tema") or dje.get("ementa") or item.get("tema_hint")
                or "; ".join(item.get("reasons") or [])
            )
            disposition = item.get("disposition", "")
            strong = item_has_strong_evidence(item)
            tag = "strong" if strong else (disposition if disposition in ("skipped", "blocked") else "info")
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
                    ("⭐ " if strong else "") + str(tema)[:120],
                    disposition,
                    item.get("source", ""),
                    item.get("video_id", ""),
                ),
            )
        resumo = "  |  ".join(f"{k}: {v}" for k, v in sorted(counts.items()))
        if showing_rejected:
            self.vistoria_summary_var.set(
                f"exibindo {len(items)} descartado(s) — selecione e use 'Restaurar item'   |   pendentes: {total}"
            )
        else:
            self.vistoria_summary_var.set(
                f"{total} pendente(s)  (⭐ {strong_total} | ⏱ {with_ts_total})   {resumo}"
                if total
                else "Fila vazia — nada aguardando revisão."
            )
        self.notebook.tab(self.vistoria_tab_index, text=f"  5 · Fila de vistoria ({total})  ")
        self.vistoria_hint_var.set(f"⚠ {total} pendência(s) aguardam sua decisão — clique aqui" if total else "")

    def _readonly_text_key(self, event):
        """Permite navegação e cópia no Text, bloqueando qualquer edição."""
        ctrl = bool(event.state & 0x4)
        if ctrl and event.keysym.lower() in ("c", "a"):
            if event.keysym.lower() == "a":
                event.widget.tag_add(tk.SEL, "1.0", tk.END)
                return "break"
            return None
        if event.keysym in ("Up", "Down", "Left", "Right", "Prior", "Next", "Home", "End"):
            return None
        return "break"

    def _copy_clip(self, text: str) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _copy_tree_selection(self, tree: ttk.Treeview) -> None:
        lines = ["\t".join(str(v) for v in tree.item(iid, "values")) for iid in tree.selection()]
        if lines:
            self._copy_clip("\n".join(lines))

    def _select_all_tree(self, tree: ttk.Treeview) -> str:
        tree.selection_set(tree.get_children())
        return "break"

    def _copy_entire_tree(self, tree: ttk.Treeview) -> None:
        """Copia TODAS as linhas visíveis (filtro atual) com cabeçalho, separadas
        por tab — pronto para colar em planilha."""
        headers = [tree.heading(column, "text") for column in tree["columns"]]
        lines = ["\t".join(headers)]
        for iid in tree.get_children():
            lines.append("\t".join(str(v) for v in tree.item(iid, "values")))
        if len(lines) > 1:
            self._copy_clip("\n".join(lines))
            self._append_output(f"Lista copiada: {len(lines) - 1} linha(s) + cabeçalho.\n")

    def _cell_under_cursor(self, tree: ttk.Treeview, event) -> tuple[str, str]:
        row_id = tree.identify_row(event.y)
        cell = ""
        if row_id:
            tree.selection_set(row_id)
            tree.focus(row_id)
            column_id = tree.identify_column(event.x)
            if column_id.startswith("#"):
                index = int(column_id[1:]) - 1
                values = tree.item(row_id, "values")
                if 0 <= index < len(values):
                    cell = str(values[index])
        return row_id, cell

    def _vistoria_context_menu(self, event) -> None:
        row_id, cell = self._cell_under_cursor(self.vistoria_tree, event)
        if not row_id:
            return
        item = self.vistoria_items.get(row_id) or {}
        row = item.get("row") or {}
        dje = (item.get("extra") or {}).get("dje") or {}
        numero = str(row.get("numero_processo") or dje.get("numeroUnico") or item.get("numero_hint") or "")
        link = item_video_link(item)
        menu = tk.Menu(self.vistoria_tree, tearoff=0)
        if cell and cell != "—":
            menu.add_command(label=f'Copiar "{cell[:60]}"', command=lambda v=cell: self._copy_clip(v))
        if numero:
            menu.add_command(label=f"Copiar processo  {numero}", command=lambda v=numero: self._copy_clip(v))
        if link:
            menu.add_command(label="Copiar link do vídeo", command=lambda v=link: self._copy_clip(v))
        reasons = "\n".join(item.get("reasons") or [])
        if reasons:
            menu.add_command(label="Copiar motivos completos", command=lambda v=reasons: self._copy_clip(v))
        menu.add_separator()
        menu.add_command(
            label="Copiar linha inteira",
            command=lambda: self._copy_tree_selection(self.vistoria_tree),
        )
        menu.add_command(
            label="Selecionar tudo  (Ctrl+A)",
            command=lambda: self._select_all_tree(self.vistoria_tree),
        )
        menu.add_command(
            label="Copiar LISTA inteira (filtro atual, com cabeçalho)",
            command=lambda: self._copy_entire_tree(self.vistoria_tree),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _videos_context_menu(self, event) -> None:
        row_id, cell = self._cell_under_cursor(self.tree, event)
        if not row_id:
            return
        values = self.tree.item(row_id, "values")
        url = values[-1] if values else ""
        menu = tk.Menu(self.tree, tearoff=0)
        if cell:
            menu.add_command(label=f'Copiar "{cell[:60]}"', command=lambda v=cell: self._copy_clip(v))
        if url:
            menu.add_command(label="Copiar URL do vídeo", command=lambda v=url: self._copy_clip(v))
        menu.add_separator()
        menu.add_command(label="Copiar linha inteira", command=lambda: self._copy_tree_selection(self.tree))
        menu.add_command(label="Selecionar tudo  (Ctrl+A)", command=lambda: self._select_all_tree(self.tree))
        menu.add_command(
            label="Copiar LISTA inteira (com cabeçalho)",
            command=lambda: self._copy_entire_tree(self.tree),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _details_context_menu(self, event) -> None:
        text = self.vistoria_details
        menu = tk.Menu(text, tearoff=0)
        has_selection = bool(text.tag_ranges(tk.SEL))
        if has_selection:
            menu.add_command(
                label="Copiar seleção",
                command=lambda: self._copy_clip(text.get(tk.SEL_FIRST, tk.SEL_LAST)),
            )
        menu.add_command(
            label="Copiar tudo",
            command=lambda: self._copy_clip(text.get("1.0", tk.END).strip()),
        )
        menu.add_command(
            label="Selecionar tudo",
            command=lambda: text.tag_add(tk.SEL, "1.0", tk.END),
        )
        menu.tk_popup(event.x_root, event.y_root)

    def _show_vistoria_details(self, _event=None) -> None:
        selected = self._selected_vistoria_items()
        self.vistoria_details.delete("1.0", tk.END)
        if selected:
            item = selected[0]
            row = item.get("row") or {}
            dje = (item.get("extra") or {}).get("dje") or {}
            lines = [
                f"Processo: {vistoria_item_numero_display(item) or '(sem número)'}    "
                f"Sessão: {item.get('data_sessao') or '?'}    Situação: {item.get('disposition')}    "
                f"Fonte: {item.get('source')}",
            ]
            if row.get("tema") or item.get("tema_hint"):
                lines.append(f"Tema: {row.get('tema') or item.get('tema_hint')}")
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
        if not selected:
            messagebox.showinfo("Fila de vistoria", "Selecione ao menos um item.")
            return
        publishable = [
            item for item in selected
            if item.get("row")
            or (item.get("disposition") in ("skipped", "blocked") and item.get("artifact_dir"))
        ]
        if not publishable:
            messagebox.showinfo(
                "Fila de vistoria",
                "Os itens selecionados são apenas informativos (duplicata de número / contagem do rito / "
                "faltante DJE): não carregam julgamento publicável por aqui. Duplicatas e contagens se "
                "resolvem corrigindo a base; faltantes DJE são importados pelo import_dje_faltantes.py. "
                "Use 'Descartar item' para fechá-los quando tratados.",
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

    def _restore_selected_vistoria(self) -> None:
        selected = self._selected_vistoria_items()
        if not selected:
            messagebox.showinfo(
                "Restaurar item",
                "Selecione o item a restaurar. Dica: escolha a visão '🗑 descartados' no filtro Situação "
                "para listar os itens fechados.",
            )
            return
        already_pending = [item for item in selected if item.get("status") == "pending"]
        to_restore = [item for item in selected if item.get("status") != "pending"]
        if not to_restore:
            messagebox.showinfo("Restaurar item", "O(s) item(ns) selecionado(s) já estão pendentes na fila.")
            return
        vistoria_queue.update_status(
            [str(item["id"]) for item in to_restore], "pending", extra={"triagem": "restaurado pelo usuário"}
        )
        self._append_output(f"Restaurado(s) para a fila: {len(to_restore)} item(ns).\n")
        if already_pending:
            self._append_output(f"(ignorados {len(already_pending)} já pendentes)\n")
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
    # log persistente da GUI: tudo que passa pelo logging durante os lotes fica
    # em artifacts\batch_gui\gui.log (o widget de saída morre com a janela)
    try:
        BATCH_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(BATCH_ARTIFACT_ROOT / "gui.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
    except OSError:
        pass
    root = tk.Tk()
    BatchGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
