import queue
from types import SimpleNamespace
from unittest.mock import Mock

import tse_youtube_notion_batch_gui as gui


def test_batch_completion_refreshes_queue_before_warning(monkeypatch):
    app = gui.BatchGuiApp.__new__(gui.BatchGuiApp)
    app.root = SimpleNamespace(after=Mock())
    app.output_queue = queue.Queue()
    app.output_queue.put(("batch_done", {"total_pending": 2}))
    app._append_output = Mock()
    events = []
    app._reload_vistoria = lambda: events.append("refreshed")
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a: events.append("warning"))
    app._drain_output_queue()
    assert events == ["refreshed", "warning"]


def test_external_queue_write_refreshes_once_without_resetting_unchanged_ui(tmp_path, monkeypatch):
    path = tmp_path / "queue.jsonl"
    monkeypatch.setattr(gui.vistoria_queue, "QUEUE_FILE", path)
    app = gui.BatchGuiApp.__new__(gui.BatchGuiApp)
    app.root = SimpleNamespace(after=Mock())
    app._vistoria_loaded_stamp = None
    refreshes = []

    def reload():
        refreshes.append(path.read_text() if path.exists() else "")
        app._vistoria_loaded_stamp = app._vistoria_queue_stamp()

    app._reload_vistoria = reload
    app._poll_vistoria_queue()
    assert refreshes == []
    path.write_text('{"id":"new-case","status":"pending"}\n')
    app._poll_vistoria_queue()
    app._poll_vistoria_queue()
    assert len(refreshes) == 1
    path.unlink()
    app._poll_vistoria_queue()
    assert refreshes[-1] == ""
    assert app.root.after.call_count == 4
