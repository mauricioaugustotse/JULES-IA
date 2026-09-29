from copy import deepcopy

from tse_publication_repair import repair_confirmed_notion_fields
from tse_workflow_monitor import verify_notion_rows
from tse_youtube_notion_core import PublishPreviewRow


class Notion:
    def __init__(self):
        self.page = {"id": "page-a", "properties": {
            "resultado": {"select": {"name": "Provido"}},
            "punchline": {"rich_text": [{"text": {"content": "Texto enriquecido"}}]},
        }}
        self.writes = []

    def build_properties_payload(self, schema, row):
        return {"resultado": {"select": {"name": row.resultado}},
                "punchline": {"rich_text": [{"text": {"content": row.punchline}}]}}

    def _request(self, method, path, **kwargs):
        if method == "PATCH":
            self.writes.append(deepcopy(kwargs["json"]))
            self.page["properties"].update(kwargs["json"]["properties"])
        return deepcopy(self.page)


def context():
    row = PublishPreviewRow(numero_processo="0600162-45.2026.6.03.0000", data_sessao="2026-09-24",
                            resultado="Suspenso por vista", punchline="Texto original")
    result = {"page_id": "page-a", "status": "created"}
    audit = {"session_date": row.data_sessao,
             "matches": [{"numero_processo": row.numero_processo,
                          "confirmed_fields": ["resultado", "punchline"]}]}
    return row, result, audit


def test_repair_is_journaled_before_write_selective_verified_and_idempotent():
    row, result, audit = context()
    client = Notion()
    checkpoints = []
    def persist(events):
        checkpoints.append((deepcopy(events), len(client.writes)))
    events = repair_confirmed_notion_fields([row], [result], client, None, audit, checkpoint=persist)
    assert checkpoints[0][0][0]["status"] == "planned"
    assert checkpoints[0][1] == 0
    assert events[0]["status"] == "applied"
    assert set(client.writes[0]["properties"]) == {"resultado"}
    assert verify_notion_rows([row], [result], client, None, fields={"resultado"})[0]["status"] == "verified"
    assert repair_confirmed_notion_fields([row], [result], client, None, audit) == []
    assert len(client.writes) == 1


def test_absent_proof_wrong_session_and_archived_page_never_written():
    row, result, audit = context()
    client = Notion()
    assert repair_confirmed_notion_fields([row], [result], client, None, {}) == []
    audit["session_date"] = "2026-09-25"
    assert repair_confirmed_notion_fields([row], [result], client, None, audit) == []
    audit["session_date"] = row.data_sessao
    client.page["archived"] = True
    assert repair_confirmed_notion_fields([row], [result], client, None, audit)[0]["status"] == "error"
    assert not client.writes
