"""Offline integration of official repairs, publication and coverage accounting."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

import tse_youtube_notion_batch_gui as gui
import tse_youtube_notion_core as core
from tse_official_session import normalize_official_session


DAY = "2026-09-24"
CNJ = "0600957-40.2026.6.07.0000"
VIDEO = "iu2DRvNv4wc"
PANEL = ["Nunes Marques", "André Mendonça", "Dias Toffoli", "Ricardo Villas Bôas Cueva",
         "Sebastião Reis Júnior", "Floriano de Azevedo Marques", "Estela Aranha"]


def candidate(**overrides):
    return core.PublishPreviewRow(**{
        "tema": "Registro de candidatura e condenações por improbidade", "numero_processo": "0600957-40",
        "numero_origem_video": "0600957-40", "classe_processo": "RO", "origem": "Brasília/DF",
        "data_sessao": DAY, "relator": "Min. Dias Toffoli", "resultado": "Desprovido", "votacao": "Por maioria",
        "partes": ["José Roberto Arruda"], "composicao": ["Min. " + name for name in PANEL],
        "source_start_seconds": 7290, "source_bundle_index": 1, "source_item_index": 1,
        "youtube_link": f"https://www.youtube.com/watch?v={VIDEO}&t=7290",
        "analise_do_conteudo_juridico": "Recurso contra indeferimento de registro de candidatura por condenações de improbidade.",
        "punchline": "O TSE manteve o indeferimento do registro de candidatura.", **overrides,
    })


def official():
    return normalize_official_session([{
        "id": "synthetic", "tribunal": "TSE", "virtual": False, "dataSessao": DAY,
        "processos": [{"numeroProcesso": CNJ, "situacaoProcesso": "Julgado", "siglaClasseJudicial": "RO-El",
                       "relator": "DIAS TOFFOLI", "origem": "BRASÍLIA - DF",
                       "proclamacaoDecisao": "O Tribunal, por maioria, negou provimento ao recurso.\n\nComposição: " + ", ".join(PANEL) + "."}],
    }], DAY)


class MemoryNotion:
    def __init__(self, existing=False):
        self.existing = existing
        self.lookups = []
        self.written = []
        self.pages = {}

    def find_existing_row(self, schema, youtube_link, number, day):
        self.lookups.append((number, day))
        return SimpleNamespace(page_id="existing-page") if self.existing and number == CNJ else None

    def build_properties_payload(self, schema, row):
        return {name: {"rich_text": [{"text": {"content": str(getattr(row, name))}}]}
                for name in ("numero_processo", "relator", "resultado", "votacao")}

    def _request(self, method, path):
        assert method == "GET"
        return self.pages[path.rsplit("/", 1)[-1]]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    store = gui.RunArtifacts(tmp_path / "video")
    monkeypatch.setattr(gui.vistoria_queue, "QUEUE_FILE", tmp_path / "queue.jsonl")
    monkeypatch.setattr(core, "refresh_cached_rito_report", lambda store: None)
    monkeypatch.setattr(core, "load_scan_coverage_report", lambda store: {"status": "complete"})
    monkeypatch.setattr(gui, "ensure_chapter_inventory", lambda *args: {"status": "unavailable"})
    monkeypatch.setattr(gui, "_fetch_official_for_rows", lambda *args: official())
    for name in ("enrich_preview_rows_with_youtube_chapters", "enrich_preview_rows_with_session_date_from_title",
                 "enrich_preview_rows_with_process_metadata", "enrich_preview_rows_with_theme_punchline",
                 "enrich_preview_rows_with_news"):
        monkeypatch.setattr(gui, name, lambda rows, **kwargs: rows)
    monkeypatch.setattr(gui, "dedupe_preview_rows", lambda rows, url: rows)
    monkeypatch.setattr(gui, "enrich_preview_rows_with_cnj", lambda *args, **kwargs: pytest.fail("DataJud received officially confirmed identity"))

    def publish(rows, client, schema, result_callback):
        results = []
        for index, row in enumerate(rows):
            assert not row.errors, row.errors
            client.written.append(row.model_copy(deep=True))
            page_id = row.page_id or f"created-{index}"
            client.pages[page_id] = {"id": page_id, "properties": client.build_properties_payload(schema, row)}
            result = {"status": "updated" if row.action == "update" else "created", "page_id": page_id,
                      "numero_processo": row.numero_processo, "row_index": index}
            results.append(result)
            result_callback(result)
        return results
    monkeypatch.setattr(gui, "publish_preview_rows", publish)

    def run(rows, client=None):
        client = client or MemoryNotion()
        original = [row.model_copy(deep=True) for row in rows]
        monkeypatch.setattr(gui, "build_preview_rows", lambda *args, **kwargs: deepcopy(original))
        payload = {
            "session": {"data_sessao": DAY, "composicao": PANEL,
                        "judgments": [{"title_hint": "0600957-40", "start_seconds": 7290,
                                       "mentioned_process_numbers": [row.numero_processo for row in original]}]},
            "bundles": [{"title_hint": "0600957-40", "start_seconds": 7290,
                         "items": [row.model_dump(mode="json") for row in original]}],
        }
        analysis = SimpleNamespace(model_dump=lambda **kwargs: deepcopy(payload))
        summary = gui.process_single_video(
            gui.VideoInput(1, f"https://www.youtube.com/watch?v={VIDEO}", VIDEO),
            artifact_store=store, notion_client=client, notion_schema=None, gemini_api_key="unused",
            options=gui.BatchOptions(model="unused", news_model="unused", with_news=False, publish=True, continue_on_error=False),
            progress=lambda message: None, analysis=analysis,
        )
        return summary, client, store
    return run


def test_completed_official_identity_bypasses_datajud(pipeline):
    summary, client, store = pipeline([candidate()])
    assert client.written[0].numero_processo == CNJ
    assert client.written[0].relator == "Min. Dias Toffoli"
    assert summary["created"] == 1
    assert summary["coverage"]["status"] == "verified"
    audit = store.read_json("04i_automatic_reconciliation.json")
    assert audit["phases"][0]["matches"][0]["method"] == "unique_short_cnj"


def test_second_reconciliation_restores_metadata_drift(pipeline, monkeypatch):
    def overwrite(rows, **kwargs):
        rows[0].relator = "Min. Floriano de Azevedo Marques"
        rows[0].resultado = "Provido"
        rows[0].votacao = "Unânime"
        rows[0].composicao = ["Min. Antônio Carlos Ferreira"]
        return rows
    monkeypatch.setattr(gui, "enrich_preview_rows_with_process_metadata", overwrite)
    summary, client, store = pipeline([candidate()])
    written = client.written[0]
    assert (written.relator, written.resultado, written.votacao) == ("Min. Dias Toffoli", "Desprovido", "Por maioria")
    assert "Min. Sebastião Reis Júnior" in written.composicao
    assert "Min. Antônio Carlos Ferreira" not in written.composicao
    final_audit = store.read_json("04i_automatic_reconciliation.json")["phases"][-1]
    assert final_audit["phase"] == "before_publish"
    assert {"relator", "resultado", "votacao", "composicao"} <= {change["field"] for change in final_audit["corrections"]}
    assert not summary["coverage"]["issues"]


def test_completed_number_rechecks_upsert_before_publication(pipeline):
    summary, client, _ = pipeline([candidate()], MemoryNotion(existing=True))
    assert (CNJ, DAY) in client.lookups
    assert client.written[0].page_id == "existing-page"
    assert client.written[0].action == "update"
    assert summary["updated"] == 1
    assert summary["created"] == 0


def test_identity_confirmed_only_after_enrichment_also_rechecks_upsert(pipeline, monkeypatch):
    def supply_missing_relator(rows, **kwargs):
        rows[0].relator = "Min. Dias Toffoli"
        return rows
    monkeypatch.setattr(gui, "enrich_preview_rows_with_cnj", supply_missing_relator)
    source = candidate(numero_processo="", numero_origem_video="0060095740", relator="")
    summary, client, _ = pipeline([source], MemoryNotion(existing=True))
    assert client.written[0].numero_processo == CNJ
    assert client.written[0].page_id == "existing-page"
    assert summary["updated"] == 1


def test_cited_conviction_is_audited_and_does_not_return_as_missing_coverage(pipeline):
    citation = candidate(numero_processo="0013595-14", numero_origem_video="0013595-14", source_item_index=2,
                         analise_do_conteudo_juridico="Recurso de registro de candidatura baseado em condenação por improbidade.")
    summary, client, store = pipeline([candidate(), citation])
    assert len(client.written) == 1
    audit = store.read_json("04i_automatic_reconciliation.json")["phases"][0]
    assert audit["exclusions"][0]["row"]["numero_processo"] == "0013595-14"
    assert audit["exclusions"][0]["parent_numero_processo"] == CNJ
    assert {i.get("code") for i in summary["coverage"]["information"]} >= {"cited_process_number"}
    assert not summary["coverage"]["issues"]
    assert not gui.vistoria_queue.load_items("pending")


def test_automatic_rerun_closes_previously_blocked_candidate_after_verification(pipeline):
    old = gui.vistoria_queue.make_vistoria_item(
        source="batch", video_id=VIDEO, youtube_url="https://youtube.com/watch?v=" + VIDEO,
        disposition="blocked", reasons=["Número incompleto"], row=candidate().model_dump(mode="json"),
    )
    gui.vistoria_queue.append_items([old])
    summary, client, store = pipeline([candidate()])
    assert summary["coverage"]["status"] == "verified"
    assert gui.vistoria_queue.load_items("pending") == []
    published = gui.vistoria_queue.load_items("published")[0]
    assert published["row"]["numero_processo"] == CNJ
    assert published["published_page_id"] == "created-0"
