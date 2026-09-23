import json
import queue
import threading
from types import SimpleNamespace
from dataclasses import replace

import pytest

import tse_youtube_notion_batch_gui as gui


@pytest.fixture
def offline_batch(monkeypatch):
    monkeypatch.setattr(gui, "build_runtime_context", lambda: {
        "gemini_api_key": "fake", "notion_api_key": "fake", "notion_data_source_id": "fake",
    })

    class Client:
        def __init__(self, **kwargs):
            pass

        def fetch_schema(self):
            return None

    monkeypatch.setattr(gui, "NotionSessoesClient", Client)
    monkeypatch.setattr(gui.vistoria_queue, "append_items", lambda *args: 0)
    return gui.BatchOptions(
        model="fake", news_model="fake", with_news=False, publish=True, continue_on_error=True,
        atualizar_tse=False, ingerir_dje=False, enriquecer_sessoes="nao", post_publish_steps=(),
    )


def run(tmp_path, options, stop=None):
    video = gui.VideoInput(1, "https://www.youtube.com/watch?v=gY6-5zPrFlo", "gY6-5zPrFlo")
    return gui.process_video_batch([video], options, queue.Queue(), stop or threading.Event(), resume_root=tmp_path)


def test_incomplete_video_never_counts_as_done(tmp_path, monkeypatch, offline_batch):
    monkeypatch.setattr(gui, "process_single_video", lambda *args, **kw: {
        "video_id": "gY6-5zPrFlo", "rows_extracted": 4, "created": 3, "updated": 0,
        "blocked": 1, "skipped": 0, "coverage": {"status": "pending", "issues": [
            {"code": "unpublished_row", "message": "Processo bloqueado"},
        ]},
    })
    summary = run(tmp_path, offline_batch)
    assert summary["total_pending"] == 1
    assert summary["total_done"] == 0
    state = json.loads((tmp_path / "monitor_status.json").read_text(encoding="utf-8"))
    assert state["status"] == "pending"
    assert state["videos"]["gY6-5zPrFlo"]["status"] == "pending"
    assert "Processo bloqueado" in (tmp_path / "monitor.html").read_text(encoding="utf-8")


def test_stop_before_first_video_leaves_unprocessed_count(tmp_path, monkeypatch, offline_batch):
    monkeypatch.setattr(gui, "process_single_video", lambda *args, **kw: pytest.fail("Processou apos parada"))
    stop = threading.Event()
    stop.set()
    summary = run(tmp_path, offline_batch, stop)
    assert summary["total_unprocessed"] == 1
    assert summary["total_done"] == 0
    assert json.loads((tmp_path / "monitor_status.json").read_text())["status"] == "pending"


def test_partial_publication_is_accounted_after_failure(tmp_path, monkeypatch, offline_batch):
    def fail_after_one(video, *, artifact_store, **kwargs):
        artifact_store.write_json("05_publish_journal.json", [
            {"row_index": 0, "status": "created", "page_id": "saved-page"},
            {"row_index": 1, "status": "error", "errors": ["connection lost"]},
        ])
        raise RuntimeError("connection lost")

    monkeypatch.setattr(gui, "process_single_video", fail_after_one)
    summary = run(tmp_path, offline_batch)
    assert summary["total_error"] == 1
    assert summary["videos"][0]["created"] == 1
    assert summary["videos"][0]["publish_results"][0]["page_id"] == "saved-page"
    assert json.loads((tmp_path / "monitor_status.json").read_text())["status"] == "pending"


def test_fatal_startup_error_is_persisted(tmp_path, monkeypatch, offline_batch):
    monkeypatch.setattr(gui, "build_runtime_context", lambda: {"gemini_api_key": "", "notion_api_key": ""})
    with pytest.raises(RuntimeError, match="GEMINI"):
        run(tmp_path, offline_batch)
    state = json.loads((tmp_path / "monitor_status.json").read_text())
    assert state["status"] == "error"
    assert "GEMINI" in state["stage"]


DAY = "2026-09-17"
CNJ = "0600198-85.2024.6.02.0000"


def official(disposition="Julgado", **process_fields):
    from tse_official_session import normalize_official_session
    return normalize_official_session([{
        "id": "test", "tribunal": "TSE", "virtual": False, "dataSessao": DAY,
        "processos": [{"numeroProcesso": CNJ, "situacaoProcesso": disposition, **process_fields}],
    }], DAY)


def preview(**overrides):
    return gui.PublishPreviewRow(**{"numero_processo": CNJ, "data_sessao": DAY,
                                   "resultado": "Indeferido", "classe_processo": "RvE",
                                   "relator": "Min. Antônio Carlos Ferreira", **overrides})


@pytest.mark.parametrize("disposition, fields, expected", [
    ("Retirado de julgamento", {}, "withdrawn"),
    ("Julgado", {"blocoJulgamento": "Lista 1"}, "list"),
])
def test_gate_excludes_only_confirmed_withdrawn_or_collective_list(tmp_path, disposition, fields, expected):
    store = gui.RunArtifacts(tmp_path)
    rows, excluded = gui._apply_official_gate(store, [preview()], official(disposition, **fields))
    assert rows == []
    assert excluded[0]["status"] == "official_exclusion"
    assert excluded[0]["classification"] == expected
    assert json.loads((tmp_path / "04g_official_excluded_rows.json").read_text(encoding="utf-8")) == excluded


def test_wrong_full_cnj_is_blocked_even_when_short_number_is_withdrawn(tmp_path):
    row = preview(numero_processo="0600198-85.2026.6.02.0000")
    rows, excluded = gui._apply_official_gate(gui.RunArtifacts(tmp_path), [row], official("Retirado de julgamento"))
    assert rows == [row]
    assert not excluded
    assert row.blocked


def test_contradictory_outcome_stays_blocked_after_core_revalidation(tmp_path):
    row = preview(resultado="Suspenso")
    rows, excluded = gui._apply_official_gate(gui.RunArtifacts(tmp_path), [row], official(
        proclamacaoDecisao="O Tribunal, por maioria, indeferiu o pedido."))
    assert not excluded
    assert any("[oficial:official_field_mismatch]" in e for e in row.errors)
    gui.validate_preview_row(row, None)
    assert any("[oficial:official_field_mismatch]" in e for e in row.errors)


def test_no_date_never_fetches_or_reuses_stale_official_inventory(tmp_path, monkeypatch):
    store = gui.RunArtifacts(tmp_path)
    store.write_json(gui.OFFICIAL_INVENTORY_FILENAME, official())
    monkeypatch.setattr(gui, "fetch_official_session", lambda *a: pytest.fail("Invented date"))
    actual = gui._fetch_official_for_rows(store, [preview(data_sessao="")])
    assert actual["status"] == "unavailable"
    assert store.read_json(gui.OFFICIAL_INVENTORY_FILENAME)["status"] == "unavailable"


def test_title_enriched_date_is_used_for_fresh_official_query(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(gui, "fetch_official_session", lambda day, store: calls.append(day) or official())
    assert gui._fetch_official_for_rows(gui.RunArtifacts(tmp_path), [preview()])["status"] == "available"
    assert calls == [DAY]


def test_chapter_absence_is_informational_only_with_available_official_evidence(tmp_path):
    analysis = SimpleNamespace(model_dump=lambda **kw: {})
    store = gui.RunArtifacts(tmp_path)
    report = gui._video_coverage_report(store, analysis, [preview()], {"status": "unavailable"},
                                        {"status": "complete"}, official=official())
    assert not report["issues"]
    assert report["information"][0]["kind"] == "chapter_inventory_unavailable"
    unknown = gui._video_coverage_report(store, analysis, [preview()], {"status": "unavailable"},
                                         {"status": "complete"}, official={"status": "unavailable"})
    assert {"official_inventory_unavailable", "chapter_inventory_unavailable"} <= {i["code"] for i in unknown["issues"]}


def test_final_readback_detects_later_script_overwriting_outcome(tmp_path, monkeypatch):
    store = gui.RunArtifacts(tmp_path)
    store.write_json("04h_publish_preview_rows.json", [preview().model_dump(mode="json")])
    store.write_json(gui.OFFICIAL_INVENTORY_FILENAME, official())
    monkeypatch.setattr(gui, "_queue_monitor_issues", lambda *a: None)
    class Client:
        def build_properties_payload(self, schema, row):
            return {"resultado": {"select": {"name": row.resultado}}}
        def _request(self, *args):
            return {"id": "page-a", "properties": {"resultado": {"select": {"name": "Suspenso"}}}}
    summary = {"status": "done", "video_id": "abc", "artifact_dir": str(tmp_path), "created": 1,
               "publish_results": [{"status": "created", "page_id": "page-a"}],
               "coverage": {"status": "verified", "issues": []}}
    result = gui._verify_after_post_publish([summary], Client(), None, queue.Queue())
    assert result[0]["failed"] == 1
    assert summary["status"] == "pending"
    assert summary["coverage"]["issues"][0]["code"] == "notion_final_unverified"
    assert "resultado" in store.read_json("05c_final_notion_verification.json")[0]["error"]


def test_failed_internal_postpublish_script_is_exposed(tmp_path, monkeypatch, offline_batch):
    import post_publish_orchestrator
    monkeypatch.setattr(gui, "process_single_video", lambda *a, **kw: {
        "video_id": "gY6-5zPrFlo", "rows_extracted": 1, "created": 1, "updated": 0,
        "blocked": 0, "skipped": 0, "coverage": {"status": "verified", "issues": []}})
    monkeypatch.setattr(gui, "_verify_after_post_publish", lambda *a: [])
    monkeypatch.setattr(post_publish_orchestrator, "run_post_publish_treatments", lambda **kw: {
        "results": {"sanear": {"status": "failed", "returncode": 1}, "classe": {"status": "ok"}}})
    summary = run(tmp_path, replace(offline_batch, post_publish_steps=("sanear",)))
    assert summary["post_publish"]["falhas"] == ["treatments:sanear"]
    assert summary["total_error"] == 1
    assert json.loads((tmp_path / "monitor_status.json").read_text())["status"] == "pending"


def test_last_batch_resolution_label_and_monitor_preserve_original_summary(tmp_path, monkeypatch):
    folder = tmp_path / "20260919_095400_895000"
    folder.mkdir()
    original = {"total_done": 1, "total_pending": 2}
    (folder / "batch_summary.json").write_text(json.dumps(original))
    (folder / "monitor_resolution.json").write_text(json.dumps({"status": "verified"}))
    (folder / "resolution_monitor.html").write_text("confirmed")
    monkeypatch.setattr(gui, "BATCH_ARTIFACT_ROOT", tmp_path)
    selected, label = gui.BatchGuiApp._find_last_batch()
    assert selected == folder
    assert "corrigido e conferido" in label
    assert json.loads((folder / "batch_summary.json").read_text()) == original
    opened = []
    monkeypatch.setattr(gui.webbrowser, "open", opened.append)
    gui.BatchGuiApp._open_monitor(SimpleNamespace(batch_artifact_dir=None, last_batch_dir=folder))
    assert opened == [(folder / "resolution_monitor.html").resolve().as_uri()]
