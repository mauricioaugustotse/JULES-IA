import tse_youtube_notion_batch_gui as gui


def test_summarize_link_meta_extracts_session_date():
    display = gui.summarize_link_meta(
        "Sessão Plenária Jurisdicional do dia 30 de junho de 2026"
    )
    assert "2026-06-30" in display
    assert "Sessão Plenária" in display


def test_summarize_link_meta_without_date_keeps_title():
    assert gui.summarize_link_meta("Vídeo institucional") == "Vídeo institucional"
    assert gui.summarize_link_meta("") == "(título indisponível)"


def test_build_rito_count_check_reads_report(tmp_path):
    store = gui.RunArtifacts(tmp_path)
    assert gui._build_rito_count_check(store, []) is None
    store.write_json(
        "01b_rito_refinement.json",
        {"transcript_available": True, "apregoamentos_individuais": 3},
    )
    check = gui._build_rito_count_check(store, [object()])
    assert check == {"apregoamentos": 3, "rows": 1, "delta": 2, "verdict": "verificar"}
    check_ok = gui._build_rito_count_check(store, [object()] * 4)
    assert check_ok["verdict"] == "ok"


# ---------------------------------------------------------------------------
# Gate de parada ANTES da publicacao.
#
# O botao "Parar antes de publicar" existe para o usuario que ve defeito no log
# do video EM CURSO (degeneracao do modelo, numero de processo fabricado) e quer
# impedir que aquilo entre no Notion. Ate 19/08/2026 o stop_event so era
# consultado ENTRE videos: o video em curso ia ate o fim e PUBLICAVA. Numa
# rodada real a parada foi acionada e o video seguiu publicando — nao contaminou
# a base por acaso, porque a conexao com o Notion caiu antes.
#
# A analise ja foi paga e continua sendo gravada nos artifacts; o que o gate
# impede e a ESCRITA no Notion.
# ---------------------------------------------------------------------------
import threading

import pytest


class _LinhaFalsa:
    def model_dump(self, mode=None):
        return {"row": True}


class _AnaliseFalsa:
    def model_dump(self, mode=None):
        return {"analise": True}


@pytest.fixture
def gui_sem_rede(monkeypatch):
    """Neutraliza tudo que sairia para a rede, menos a publicacao (que e o objeto do teste)."""
    monkeypatch.setattr(gui, "build_preview_rows", lambda analysis, **kw: [_LinhaFalsa()])
    for nome in (
        "enrich_preview_rows_with_youtube_chapters",
        "enrich_preview_rows_with_session_date_from_title",
        "enrich_preview_rows_with_cnj",
        "enrich_preview_rows_with_process_metadata",
        "enrich_preview_rows_with_theme_punchline",
        "enrich_preview_rows_with_news",
    ):
        monkeypatch.setattr(gui, nome, lambda rows, **kw: rows)
    monkeypatch.setattr(gui, "dedupe_preview_rows", lambda rows, url: rows)
    monkeypatch.setattr(gui, "validate_preview_row", lambda row, schema: row)
    monkeypatch.setattr(gui, "_build_rito_count_check", lambda store, rows: None)
    monkeypatch.setattr(
        gui.vistoria_queue, "collect_video_vistoria_items", lambda *a, **kw: []
    )
    return gui


def _rodar(tmp_path, monkeypatch, *, parar: bool):
    chamadas = []
    monkeypatch.setattr(
        gui,
        "publish_preview_rows",
        lambda rows, client, schema: chamadas.append(len(rows)) or [],
    )
    evento = threading.Event()
    if parar:
        evento.set()
    resumo = gui.process_single_video(
        gui.VideoInput(position=1, video_id="wZQ9xLxzs9E", url="https://www.youtube.com/watch?v=wZQ9xLxzs9E"),
        artifact_store=gui.RunArtifacts(tmp_path),
        notion_client=None,
        notion_schema=None,
        gemini_api_key="fake",
        options=gui.BatchOptions(
            model="m", news_model="m", continue_on_error=False, publish=True, with_news=False
        ),
        progress=lambda mensagem: None,
        analysis=_AnaliseFalsa(),
        stop_event=evento,
    )
    return resumo, chamadas


def test_parada_acionada_nao_escreve_no_notion(tmp_path, monkeypatch, gui_sem_rede):
    resumo, chamadas = _rodar(tmp_path, monkeypatch, parar=True)

    assert chamadas == [], "publicou no Notion mesmo com a parada acionada"
    assert resumo["publish_skipped_by_stop"] is True
    assert not (tmp_path / "05_publish_results.json").exists()
    # o motivo fica registrado, para quem for auditar a pasta depois
    assert (tmp_path / "05_publish_skipped_by_stop.json").exists()
    # e a analise, que ja foi paga, continua salva para reprocessar sem gastar de novo
    assert (tmp_path / "03_analysis.json").exists()


def test_sem_parada_publica_normalmente(tmp_path, monkeypatch, gui_sem_rede):
    resumo, chamadas = _rodar(tmp_path, monkeypatch, parar=False)

    assert chamadas == [1]
    assert resumo["publish_skipped_by_stop"] is False
    assert (tmp_path / "05_publish_results.json").exists()


def test_chamada_sem_stop_event_segue_publicando(tmp_path, monkeypatch, gui_sem_rede):
    # process_single_video e importado por run_batch_videos/reprocess_videos/
    # republish_missing_days: o parametro novo e opcional e nao muda o default.
    chamadas = []
    monkeypatch.setattr(
        gui,
        "publish_preview_rows",
        lambda rows, client, schema: chamadas.append(len(rows)) or [],
    )
    resumo = gui.process_single_video(
        gui.VideoInput(position=1, video_id="abc", url="https://www.youtube.com/watch?v=abc"),
        artifact_store=gui.RunArtifacts(tmp_path),
        notion_client=None,
        notion_schema=None,
        gemini_api_key="fake",
        options=gui.BatchOptions(
            model="m", news_model="m", continue_on_error=False, publish=True, with_news=False
        ),
        progress=lambda mensagem: None,
        analysis=_AnaliseFalsa(),
    )
    assert chamadas == [1]
    assert resumo["publish_skipped_by_stop"] is False
