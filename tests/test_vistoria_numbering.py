import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vistoria_queue as queue
from tse_youtube_notion_core import NotionRowMatch, NotionSessoesClient
from test_tse_youtube_notion_core import make_schema
from test_tse_publish_journal import make_row


DAY = "2026-06-23"


class NumberedNotion(NotionSessoesClient):
    def __init__(self, number=3):
        self.pages = {"existing": {"id": "existing", "properties": {
            "data_sessao": {"date": {"start": DAY}},
            "tipo_registro": {"select": {"name": f"Julgamento {number}"}},
        }}}
        self.writes = []
        self.filters = []
        self.failure = None
        self.match = None

    def query_data_source(self, filter_payload=None):
        self.filters.append(filter_payload)
        return list(self.pages.values())

    def find_existing_row(self, *args):
        return self.match

    def _request(self, method, path, **kwargs):
        assert method == "GET"
        return copy.deepcopy(self.pages[path.rsplit("/", 1)[-1]])

    def create_row(self, schema, row):
        failure, self.failure = self.failure, None
        if failure == "before":
            raise RuntimeError("Failure before write")
        result = self.update_row(schema, f"page-{len(self.writes)}", row)
        if failure == "after":
            raise RuntimeError("Response lost after write")
        return result

    def update_row(self, schema, page_id, row):
        self.writes.append(row.model_copy(deep=True))
        self.pages[page_id] = {"id": page_id, "properties": self.build_properties_payload(schema, row)}
        return {"id": page_id}


def candidate(n):
    # Suspended avoids unrelated reconciliation of older outcomes.
    row = make_row(n, votacao="Suspenso", resultado="Suspenso por vista")
    return {"id": str(n), "row": row.model_dump(mode="json")}


def test_saved_blocked_candidates_take_next_numbers_from_notion():
    client = NumberedNotion()
    results = queue.publish_approved_items([candidate(1), candidate(6)], client, make_schema())
    assert [r["status"] for r in results] == ["created", "created"]
    assert [r["verification"]["status"] for r in results] == ["verified", "verified"]
    assert [r.tipo_registro for r in client.writes] == ["Julgamento 4", "Julgamento 5"]
    assert client.filters == [{"or": [{"property": "data_sessao", "date": {"equals": DAY}}]}]


def test_retry_updates_existing_page_without_changing_its_number():
    client = NumberedNotion(2)
    client.match = NotionRowMatch("existing", "https://notion.so/existing")
    result = queue.publish_approved_items([candidate(7)], client, make_schema())[0]
    assert result["status"] == "updated"
    assert result["verification"]["status"] == "verified"
    assert client.writes[0].tipo_registro == "Julgamento 2"
    assert client.filters == []


@pytest.mark.parametrize("failure, expected", [("before", "Julgamento 4"), ("after", "Julgamento 5")])
def test_failed_write_rechecks_remote_number_before_next_candidate(failure, expected):
    client = NumberedNotion()
    client.failure = failure
    result = queue.publish_approved_items([candidate(1), candidate(6)], client, make_schema())
    assert result[0]["status"] == "erro"
    assert result[1]["status"] == "created"
    assert client.writes[-1].tipo_registro == expected
    assert len(client.filters) == 2
