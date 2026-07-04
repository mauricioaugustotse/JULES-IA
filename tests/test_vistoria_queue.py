import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_tse_youtube_notion_core import FakeNotionClient, make_schema

import vistoria_queue
from tse_youtube_notion_core import PublishPreviewRow


def _row(index: int, **overrides) -> PublishPreviewRow:
    base = dict(
        tema=f"Tema {index}",
        numero_processo=f"060000{index}-1{index}",
        source_start_seconds=index * 100,
        data_sessao="2026-06-23",
    )
    base.update(overrides)
    return PublishPreviewRow(**base)


def test_collect_video_vistoria_items_matches_publish_results_order():
    rows = [_row(1), _row(2), _row(3)]
    publish_results = [
        {"status": "created", "page_id": "p1", "errors": [], "warnings": []},
        {"status": "skipped", "errors": [], "warnings": ["Item descartado: precedente citado."]},
        {"status": "blocked", "errors": ["Data da sessão ausente."], "warnings": []},
        {"status": "votacao_reconciled", "errors": [], "warnings": []},
    ]
    items = vistoria_queue.collect_video_vistoria_items(
        rows,
        publish_results,
        video_id="vid123",
        youtube_url="https://youtu.be/vid123",
        artifact_dir="X",
    )
    assert [item["disposition"] for item in items] == ["skipped", "blocked"]
    assert items[0]["row"]["tema"] == "Tema 2"
    assert items[1]["row"]["tema"] == "Tema 3"
    assert items[0]["data_sessao"] == "2026-06-23"


def test_collect_video_vistoria_items_adds_rito_divergence():
    items = vistoria_queue.collect_video_vistoria_items(
        [_row(1)],
        [{"status": "created", "errors": [], "warnings": []}],
        video_id="vid123",
        youtube_url="u",
        rito_check={"apregoamentos": 3, "rows": 1, "delta": 2, "verdict": "verificar"},
    )
    assert len(items) == 1
    assert items[0]["disposition"] == "contagem_rito"
    assert items[0]["row"] is None


def test_append_load_update_roundtrip(tmp_path):
    queue_file = tmp_path / "queue.jsonl"
    item = vistoria_queue.make_vistoria_item(
        source="batch",
        video_id="vid",
        youtube_url="u",
        disposition="skipped",
        reasons=["motivo"],
        row={"numero_processo": "0600904-54", "source_start_seconds": 5},
    )
    assert vistoria_queue.append_items([item], queue_file) == 1
    assert vistoria_queue.append_items([item], queue_file) == 0  # dedupe por id
    pending = vistoria_queue.load_items("pending", queue_file)
    assert len(pending) == 1
    vistoria_queue.update_status([item["id"]], "rejected", queue_file=queue_file)
    assert vistoria_queue.load_items("pending", queue_file) == []
    rejected = vistoria_queue.load_items("rejected", queue_file)
    assert len(rejected) == 1
    assert rejected[0]["reasons"] == ["motivo"]
    # item fechado não reentra na fila
    assert vistoria_queue.append_items([item], queue_file) == 0


def test_publish_approved_items_clears_block_with_audit_warning():
    schema = make_schema()
    notion = FakeNotionClient()
    row = _row(
        1,
        classe_processo="PA",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        origem="Brasília/DF",
        errors=[
            "Busca Google indicou que o número consultado aparece como precedente citado, não como processo julgado."
        ],
    )
    item = vistoria_queue.make_vistoria_item(
        source="batch",
        video_id="vid",
        youtube_url="u",
        disposition="skipped",
        reasons=["precedente citado"],
        row=row.model_dump(mode="json"),
    )
    results = vistoria_queue.publish_approved_items([item], notion, schema, apply=True)
    assert results[0]["status"] == "created"
    assert results[0]["id"] == item["id"]
    assert len(notion.created) == 1
    published_row = notion.created[0]
    assert not published_row.errors
    assert any(w.startswith("Aprovado em vistoria") for w in published_row.warnings)


def test_publish_approved_items_dry_run_reports_disposition():
    schema = make_schema()
    notion = FakeNotionClient()
    row = _row(2, relator="Min. Cármen Lúcia", resultado="Desprovido", votacao="Unânime")
    item = vistoria_queue.make_vistoria_item(
        source="batch",
        video_id="vid",
        youtube_url="u",
        disposition="skipped",
        reasons=[],
        row=row.model_dump(mode="json"),
    )
    results = vistoria_queue.publish_approved_items([item], notion, schema, apply=False)
    assert results[0]["status"].startswith("dry-run:")
    assert not notion.created
