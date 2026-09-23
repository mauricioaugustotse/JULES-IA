import importlib
import json
import sys

import pytest


OPENING = [
    "Min. Nunes Marques", "Min. André Mendonça", "Min. Antônio Carlos Ferreira",
    "Min. Ricardo Villas Bôas Cueva", "Min. Floriano de Azevedo Marques", "Min. Estela Aranha",
]
PRESERVED_CARMEN = OPENING + ["Min. Cármen Lúcia"]
PRESERVED_DIAS = OPENING + ["Min. Dias Toffoli"]


def page(page_id, composition, relator="Min. Ricardo Villas Bôas Cueva"):
    return {
        "id": page_id,
        "youtube_link": "https://www.youtube.com/watch?v=xeSt3PpR0xg&t=2500",
        "numero_processo": "0600198-85.2024.6.02.0000",
        "data_sessao": "2026-09-17",
        "relator": relator,
        "composicao": ", ".join(composition),
    }


def setup_script(monkeypatch, tmp_path, name, pages, extra_args=()):
    module = importlib.import_module(name)
    writes = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def fetch_schema(self):
            return object()

        def query_data_source(self):
            return pages

        def _extract_property_text(self, page, schema, field):
            return page.get(field, "")

        def _build_property_value(self, schema, field, value):
            return {"multi_select": [{"name": name} for name in value]}

    monkeypatch.setattr(module, "NotionSessoesClient", FakeClient)
    monkeypatch.setattr(module, "get_secret", lambda *args: "fake-key")
    monkeypatch.setattr(module, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(module, "ARTIFACT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(module.time, "sleep", lambda *args: None)
    monkeypatch.setattr(
        module, "notion_request_with_retry",
        lambda client, method, path, **kwargs: writes.append((path, kwargs["json"]["properties"])),
    )
    if name == "fix_composicao_from_transcript":
        monkeypatch.setattr(module, "fetch_transcript", lambda video_id: "Abertura de sessão")
        monkeypatch.setattr(module, "extract_present", lambda text, canon: OPENING.copy())
    else:
        monkeypatch.setattr(module, "gemini_opening", lambda *args: (OPENING.copy(), 20))
    monkeypatch.setattr(sys, "argv", [name, "--apply", "--data-source-id", "fake-source", *extra_args])
    return module, writes


@pytest.mark.parametrize("name", ["fix_composicao_from_transcript", "fix_composicao_via_gemini_opening"])
def test_bad_sibling_row_does_not_overwrite_preserved_case_votes(monkeypatch, tmp_path, name):
    # One incomplete row selects the whole session; two other cases legitimately
    # retain historical votes absent from the opening's physical attendance.
    pages = [
        page("preserved-carmen", PRESERVED_CARMEN),
        page("preserved-dias", PRESERVED_DIAS),
        page("six-present", list(reversed(OPENING))),
        page("incomplete", ["Min. Nunes Marques"]),
    ]
    module, writes = setup_script(monkeypatch, tmp_path, name, pages)
    assert module.main() == 0
    assert [path for path, _ in writes] == ["/pages/incomplete"]
    assert {item["name"] for item in writes[0][1]["composicao"]["multi_select"]} == set(OPENING)
    reports = list((tmp_path / "reports").glob("*/detalhe.json"))
    detail = json.loads(reports[0].read_text(encoding="utf-8"))
    assert {item["page_id"] for item in detail[0]["conflitos_por_processo"]} == {
        "preserved-carmen", "preserved-dias"
    }


def test_explicit_all_sessions_review_still_preserves_case_specific_composition(monkeypatch, tmp_path):
    module, writes = setup_script(
        monkeypatch, tmp_path, "fix_composicao_from_transcript",
        [page("preserved-carmen", PRESERVED_CARMEN)], extra_args=["--all-sessions"],
    )
    assert module.main() == 0
    assert writes == []


@pytest.mark.parametrize("name", ["fix_composicao_from_transcript", "fix_composicao_via_gemini_opening"])
def test_complete_sibling_without_relator_is_not_changed_by_opening(monkeypatch, tmp_path, name):
    module, writes = setup_script(
        monkeypatch, tmp_path, name,
        [page("unknown-relator", PRESERVED_CARMEN, relator=""), page("incomplete", [])],
    )
    assert module.main() == 0
    assert [path for path, _ in writes] == ["/pages/incomplete"]


@pytest.mark.parametrize("name", ["fix_composicao_from_transcript", "fix_composicao_via_gemini_opening"])
def test_partial_patch_failure_has_nonzero_exit_and_keeps_report(monkeypatch, tmp_path, name):
    module, _ = setup_script(
        monkeypatch, tmp_path, name,
        [page("first", []), page("failed", []), page("last", [])],
    )
    attempted = []

    def patch(client, method, path, **kwargs):
        attempted.append(path)
        if path == "/pages/failed":
            raise RuntimeError("simulated Notion write failure")

    monkeypatch.setattr(module, "notion_request_with_retry", patch)
    assert module.main() != 0
    assert attempted == ["/pages/first", "/pages/failed", "/pages/last"]
    assert list((tmp_path / "reports").glob("*/detalhe.json"))
