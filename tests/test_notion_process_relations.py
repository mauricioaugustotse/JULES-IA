import json
from importlib import import_module
from types import SimpleNamespace

import pytest

from notion_process_relations import (
    ProcessRelationRecord,
    build_full_cnj_relation_targets,
    full_cnj_relation_key,
    index_relation_records,
    reconcile_process_relations,
)

IRAN = "0600508-68.2018.6.14.0000"
RIACHINHO = "0600508-68.2024.6.13.0329"
MANUAL = "0000421-25.2016.6.00.0000"
UNCONFIRMED = "0600508-68.2025.6.14.0000"


@pytest.mark.parametrize("number", [IRAN, "06005086820186140000", " 0600508–68.2018.6.14.0000 "])
def test_relation_key_keeps_the_complete_identity(number):
    assert full_cnj_relation_key(number) == "06005086820186140000"
    assert full_cnj_relation_key(number) != full_cnj_relation_key(RIACHINHO)


@pytest.mark.parametrize("number", ["0600508-68", UNCONFIRMED, "", f"{IRAN}; {RIACHINHO}"])
def test_uncertain_number_cannot_authorize_automatic_links(number):
    assert full_cnj_relation_key(number) == ""


def records():
    return [
        ProcessRelationRecord("iran-1", IRAN, "iran-video-1", "2018-12-11"),
        ProcessRelationRecord("iran-2", IRAN, "iran-video-2", "2019-03-12"),
        ProcessRelationRecord("riachinho", RIACHINHO, "riachinho-video", "2026-04-09"),
        ProcessRelationRecord("manual", MANUAL, "manual-video", "2021-04-29"),
        ProcessRelationRecord("unknown", UNCONFIRMED, "unknown-video", "2025-01-01"),
    ]


def test_same_core_different_year_court_and_origin_does_not_link():
    target = build_full_cnj_relation_targets(records())
    assert target["iran-1"] == ["iran-2"]
    assert target["iran-2"] == ["iran-1"]
    assert target["riachinho"] == []
    assert "unknown" not in target


def test_reconciliation_removes_only_confirmed_collisions_and_preserves_manual_links():
    indexed = index_relation_records(records())
    current = ["riachinho", "manual", "unknown", "outside-loaded-scope"]
    proposed = build_full_cnj_relation_targets(indexed.values())
    assert reconcile_process_relations(indexed["iran-1"], current, proposed["iran-1"], indexed) == [
        "manual", "unknown", "outside-loaded-scope", "iran-2"
    ]
    assert reconcile_process_relations(indexed["unknown"], current, ["iran-1"], indexed) == current


def test_missing_session_context_does_not_create_or_remove_a_relation():
    left = ProcessRelationRecord("left", IRAN)
    right = ProcessRelationRecord("right", IRAN, "some-video", "2018-12-11")
    indexed = index_relation_records([left, right])
    assert build_full_cnj_relation_targets(indexed.values()) == {"left": [], "right": []}
    assert reconcile_process_relations(left, ["right"], [], indexed) == ["right"]


def test_conflicting_copies_of_a_page_do_not_authorize_changes():
    first, later = records()[:2]
    conflicting = ProcessRelationRecord(first.page_id, RIACHINHO, first.video_id, first.data_sessao)
    indexed = index_relation_records([first, conflicting, ProcessRelationRecord(first.page_id, ""), first, later])
    assert full_cnj_relation_key(indexed[first.page_id].numero_processo) == ""
    assert reconcile_process_relations(indexed[first.page_id], ["manual"], [later.page_id], indexed) == ["manual"]


def test_round2_preserves_each_column_and_does_not_cross_link_the_core_collision():
    from audit_notion_sessoes_round2 import PageChangeSet, add_relation_changes, build_relation_targets
    from tse_youtube_notion_core import NotionDataSourceSchema

    changed = {}
    for item in records():
        current = ["riachinho", "manual", "unknown", "external"] if item.page_id == "iran-1" else []
        if item.page_id == "riachinho":
            current = ["iran-1", "unknown"]
        record = SimpleNamespace(
            page_id=item.page_id, video_id=item.video_id, timestamp_seconds=0, index=0,
            page_url=f"https://notion.test/{item.page_id}",
            row=SimpleNamespace(numero_processo=item.numero_processo, data_sessao=item.data_sessao, tipo_registro="Julgamento 1"),
            page={"properties": {
                "materia_semelhante": {"type": "relation", "relation": [{"id": value} for value in current]},
                "materia_semelhante1": {"type": "relation", "relation": [{"id": "other-manual"}]},
            }},
        )
        changed[item.page_id] = PageChangeSet(record=record)
    schema = NotionDataSourceSchema("ds", {"properties": {
        "materia_semelhante": {"type": "relation", "relation": {}},
        "materia_semelhante1": {"type": "relation", "relation": {}},
    }})
    targets = build_relation_targets([item.record for item in changed.values()])
    add_relation_changes(changed, schema, targets)
    assert changed["iran-1"].changes["materia_semelhante"].new == ["manual", "unknown", "external", "iran-2"]
    assert changed["iran-1"].changes["materia_semelhante1"].new == ["other-manual", "iran-2"]
    assert changed["riachinho"].changes["materia_semelhante"].new == ["unknown"]
    assert not changed["unknown"].changes


def test_super_auditor_preserves_unknown_and_out_of_scope_relations():
    from super_auditor import build_global_relation_targets

    grouped = {}
    for item in records():
        current = ["riachinho", "manual", "unknown", "external"] if item.page_id == "iran-1" else []
        if item.page_id == "riachinho":
            current = ["iran-1", "unknown"]
        if item.page_id == "unknown":
            current = ["external"]
        record = SimpleNamespace(page_id=item.page_id, video_id=item.video_id,
                                 row=SimpleNamespace(numero_processo=item.numero_processo, data_sessao=item.data_sessao, materia_semelhante=current))
        grouped.setdefault(int(item.data_sessao[:4]), {}).setdefault(item.video_id, []).append(record)
    targets = build_global_relation_targets(grouped)
    assert targets["iran-1"] == ["manual", "unknown", "external", "iran-2"]
    assert targets["iran-2"] == ["iran-1"]
    assert targets["riachinho"] == ["unknown"]
    assert targets["unknown"] == ["external"]


@pytest.mark.parametrize("module_name", ["materia_semelhante_update", "fill_materia_semelhante"])
def test_cli_payloads_separate_full_identities_without_clearing_unconfirmed_links(module_name, monkeypatch, tmp_path):
    module = import_module(module_name)
    pages = []
    for item in records():
        current = ["riachinho", "manual", "unknown", "external"] if item.page_id == "iran-1" else []
        if item.page_id == "riachinho":
            current = ["iran-1"]
        if item.page_id == "unknown":
            current = ["iran-1", "external"]
        pages.append({"id": item.page_id, "values": {
            "numero_processo": item.numero_processo,
            "youtube_link": f"https://www.youtube.com/watch?v={item.video_id}",
            "data_sessao": item.data_sessao,
        }, "properties": {"materia_semelhante": {"type": "relation", "relation": [{"id": x} for x in current]}}})

    class Client:
        def fetch_schema(self):
            return SimpleNamespace(properties={"materia_semelhante": {}})

        def query_data_source(self):
            return pages

        def _extract_property_text(self, page, schema, field):
            return page["values"].get(field, "")

        def _build_property_value(self, schema, field, related):
            return {"relation": [{"id": item} for item in related]}

    writes = {}

    def fake_request(client, method, path, **kwargs):
        assert method == "PATCH"
        writes[path.rsplit("/", 1)[-1]] = [item["id"] for item in kwargs["json"]["properties"]["materia_semelhante"]["relation"]]
        return {}

    monkeypatch.setattr(module, "NotionSessoesClient", lambda **kwargs: Client())
    monkeypatch.setattr(module, "get_secret", lambda *args: "fake-test-token")
    monkeypatch.setattr(module, "notion_request_with_retry", fake_request)
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    monkeypatch.setattr("sys.argv", [module_name, "--apply"])
    assert module.main() == 0
    assert writes == {
        "iran-1": ["manual", "unknown", "external", "iran-2"],
        "iran-2": ["iran-1"],
        "riachinho": [],
    }
    if module_name == "materia_semelhante_update":
        report = json.loads(next(tmp_path.glob("*/changes.json")).read_text(encoding="utf-8"))
        first = next(item for item in report if item["page_id"] == "iran-1")
        assert first["before"] == ["riachinho", "manual", "unknown", "external"]
        assert first["after"] == writes["iran-1"]
        assert first["removed"] == ["riachinho"]
