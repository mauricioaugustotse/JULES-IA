import json

import pytest

import tse_youtube_notion_core as core


def make_row(number=1, **overrides):
    values = {
        "tema": "Propaganda eleitoral antecipada em rede social",
        "punchline": "A divulgação não contém pedido explícito de voto.",
        "numero_processo": "0600814-85.2022.6.00.0000",
        "tipo_registro": f"Julgamento {number}",
        "youtube_link": f"https://www.youtube.com/watch?v=abc123&t={number * 600}",
        "data_sessao": "2026-06-23",
        "relator": "Min. Cármen Lúcia",
        "resultado": "Desprovido",
        "votacao": "Unânime",
    }
    values.update(overrides)
    return core.PublishPreviewRow(**values)


@pytest.fixture
def schema():
    return core.NotionDataSourceSchema(
        "fake-source", {"properties": {"tema": {"type": "title", "title": {}}}}
    )


class FakeNotion:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.writes = []

    def _write(self, action, row):
        self.writes.append((action, row))
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    def create_row(self, schema, row):
        return self._write("create", row)

    def update_row(self, schema, page_id, row):
        return self._write("update", row)


def test_partial_publication_journal_survives_next_write_failure(tmp_path, schema):
    journal_path = tmp_path / "publication.jsonl"
    original_error = RuntimeError("Notion connection failed")
    notion = FakeNotion([{"id": "page-first"}, original_error, {"id": "never-written"}])

    def persist(result):
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")

    with pytest.raises(RuntimeError) as caught:
        core.publish_preview_rows(
            [make_row(1), make_row(2), make_row(3)], notion, schema, result_callback=persist
        )

    assert caught.value is original_error
    journal = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()]
    assert [entry["status"] for entry in journal] == ["created", "error"]
    assert [entry["row_index"] for entry in journal] == [0, 1]
    assert journal[0]["page_id"] == "page-first"
    assert journal[1]["errors"] == ["Notion connection failed"]
    assert len(notion.writes) == 2


def test_journal_covers_nonwrites_and_preserves_original_indexes(schema):
    rows = [
        make_row(1, errors=["Número identificado como precedente citado."]),
        make_row(2, errors=["Identificação conflitante do processo."]),
        make_row(3, action="update", page_id="page-existing"),
    ]
    notion = FakeNotion([{"id": "page-existing"}])
    journal = []
    results = core.publish_preview_rows(rows, notion, schema, result_callback=journal.append)

    assert [entry["status"] for entry in journal] == ["skipped", "blocked", "updated"]
    assert [entry["row_index"] for entry in journal] == [0, 1, 2]
    assert len(notion.writes) == 1
    assert notion.writes[0][0] == "update"
    assert rows[2].tipo_registro == "Julgamento 2"
    assert results == [{key: value for key, value in event.items() if key != "row_index"} for event in journal]


def test_publish_without_callback_retains_result_shape(schema):
    row = make_row()
    notion = FakeNotion([{"id": "page-first", "url": "https://notion.so/page-first"}])
    assert core.publish_preview_rows([row], notion, schema) == [
        {
            "tema": row.tema,
            "numero_processo": row.numero_processo,
            "status": "created",
            "page_id": "page-first",
            "url": "https://notion.so/page-first",
            "errors": [],
            "warnings": row.warnings,
        }
    ]


@pytest.mark.parametrize("action", ["create", "update"])
@pytest.mark.parametrize("response", [{}, {"id": ""}, {"id": "   "}, None])
def test_response_without_page_id_is_journaled_as_failure(schema, action, response):
    row = make_row(action=action, page_id="page-existing" if action == "update" else "")
    notion = FakeNotion([response, {"id": "never-written"}])
    journal = []
    with pytest.raises(RuntimeError, match="sem id"):
        core.publish_preview_rows([row, make_row(2)], notion, schema, result_callback=journal.append)
    assert [entry["status"] for entry in journal] == ["error"]
    assert journal[0]["row_index"] == 0
    assert len(notion.writes) == 1


def test_journal_failure_stops_subsequent_notion_writes(schema):
    notion = FakeNotion([{"id": "page-first"}, {"id": "never-written"}])

    def persist(result):
        raise OSError("Journal disk full")

    with pytest.raises(OSError, match="disk full"):
        core.publish_preview_rows([make_row(1), make_row(2)], notion, schema, result_callback=persist)
    assert len(notion.writes) == 1


def test_journal_failure_does_not_mask_original_write_exception(schema):
    original_error = RuntimeError("Notion write failed")
    notion = FakeNotion([original_error])

    def persist(result):
        raise OSError("Journal disk full")

    with pytest.raises(RuntimeError) as caught:
        core.publish_preview_rows([make_row()], notion, schema, result_callback=persist)
    assert caught.value is original_error
