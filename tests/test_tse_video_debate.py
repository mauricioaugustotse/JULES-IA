"""Regression coverage for the accepted video-only inclusion criterion."""
import pytest

import tse_youtube_notion_core as core
from tse_video_debate import VideoDebateEvidence, video_debate_issues


def video_row(**changes):
    row = core.PublishPreviewRow(
        tema="Paridade de gênero na lista tríplice do TRE-RN",
        classe_processo="Lista Tríplice",
        numero_processo="0601075-11.2026.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=8nYGbjIGIMc&t=1620",
        data_sessao="2026-10-06",
        relator="Min. Nunes Marques",
        resultado="Devolvida",
        analise_do_conteudo_juridico="O plenário discutiu a composição da lista e determinou sua devolução.",
        video_debate_evidence=VideoDebateEvidence(
            video_id="8nYGbjIGIMc", session_date="2026-10-06",
            start_seconds=1620, end_seconds=1760,
            numero_processo="0601075-11",
            title_hint="Lista tríplice para o TRE-RN",
            analysis_excerpt="O plenário discutiu especificamente a exigência de paridade de gênero na lista tríplice e a devolução da lista ao tribunal de origem para nova formação.",
            source_bundle_index=2, source_item_index=1,
        ),
    )
    for name, value in changes.items():
        setattr(row, name, value)
    return row


def test_certificate_complete_case_without_video_proof_cannot_create():
    row = video_row(video_debate_evidence=None)
    disposition, reasons = core.assess_row_publishability(row)
    assert disposition == "blocked"
    assert any("trecho identificável" in reason for reason in reasons)


def test_individually_debated_lista_triplice_is_eligible():
    assert not video_debate_issues(video_row())
    assert core.assess_row_publishability(video_row())[0] == "publish"


def test_canonical_short_number_keeps_own_video_association():
    row = video_row(numero_processo="0000421-83")
    row.video_debate_evidence.numero_processo = "AREsp 42183"
    assert not video_debate_issues(row)


@pytest.mark.parametrize("change", [
    {"youtube_link": "https://www.youtube.com/watch?v=8nYGbjIGIMc"},
    {"youtube_link": "https://www.youtube.com/watch?v=8nYGbjIGIMc&t=1900"},
    {"youtube_link": "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=1620"},
    {"data_sessao": "2026-10-08"},
    {"numero_processo": "0600001-25.2026.6.00.0000"},
])
def test_evidence_cannot_be_reused_for_another_case_session_or_segment(change):
    assert video_debate_issues(video_row(**change))


@pytest.mark.parametrize("excerpt", [
    "O presidente anunciou o julgamento em lista dos processos. A certidão registra os resultados e os nomes dos ministros participantes da sessão.",
    "O vídeo apresenta apenas a leitura da ata da sessão anterior e a composição do plenário, não havendo discussão do processo neste trecho.",
    "O Presidente convoca uma sessão extraordinária para o julgamento dos recursos de direito de resposta, estabelecendo a pauta da sessão futura.",
])
def test_collective_list_and_scheduling_cannot_be_disguised_by_enriched_row(excerpt):
    row = video_row()
    row.video_debate_evidence.analysis_excerpt = excerpt
    assert core.assess_row_publishability(row)[0] == "blocked"


def test_missing_interval_is_not_filled_from_certificate():
    row = video_row()
    row.video_debate_evidence.end_seconds = None
    assert any("início e fim" in issue for issue in video_debate_issues(row))


def test_evidence_survives_editor_and_model_roundtrips():
    row = video_row()
    edited = core.PublishPreviewRow.from_editor_record(row.to_editor_record())
    restored = core.PublishPreviewRow.model_validate(row.model_dump())
    assert edited.video_debate_evidence == restored.video_debate_evidence == row.video_debate_evidence
    assert not video_debate_issues(edited)


@pytest.mark.parametrize("matched", [True, False])
def test_official_reconciliation_preserves_typed_video_evidence(matched):
    from tse_session_reconciliation import reconcile_session_rows
    row = video_row()
    original = row.model_dump()
    process = {"numeroProcesso": row.numero_processo, "situacaoProcesso": "Julgado",
               "siglaClasseJudicial": "LT", "relator": "NUNES MARQUES", "origem": "NATAL - RN"}
    output, _ = reconcile_session_rows([row], {"status": "available", "session_date": row.data_sessao,
        "processes": [process] if matched else []})
    assert len(output) == 1
    assert isinstance(output[0].video_debate_evidence, VideoDebateEvidence)
    assert output[0].video_debate_evidence == row.video_debate_evidence
    assert not video_debate_issues(output[0])
    assert core.PublishPreviewRow.from_editor_record(output[0].to_editor_record()).video_debate_evidence == row.video_debate_evidence
    assert row.model_dump() == original


def test_direct_certificate_only_post_is_rejected_before_network(monkeypatch):
    client = core.NotionSessoesClient("fake-token", data_source_id="fake-ds")
    monkeypatch.setattr(client.session, "request", lambda *a, **k: pytest.fail("Unexpected network write"))
    with pytest.raises(RuntimeError, match="evidência do debate"):
        client._request("POST", "/pages", json={"parent": {"data_source_id": "fake-ds"}, "properties": {}})


def test_create_guard_runs_before_schema_side_effects(monkeypatch):
    client = core.NotionSessoesClient("fake-token")
    monkeypatch.setattr(client, "_prepare_row_schema", lambda *a: pytest.fail("Unexpected schema write"))
    with pytest.raises(RuntimeError, match="Nova inclusão bloqueada"):
        client.create_row(core.NotionDataSourceSchema("fake", {}), video_row(video_debate_evidence=None))


def test_raw_payload_cannot_change_the_proven_case(monkeypatch):
    client = core.NotionSessoesClient("fake-token")
    monkeypatch.setattr(client.session, "request", lambda *a, **k: pytest.fail("Unexpected network write"))
    row = video_row()
    with pytest.raises(RuntimeError, match="Payload diverge"):
        client._request("POST", "/pages", _video_debate_row=row, json={"properties": {
            "numero_processo": {"rich_text": [{"text": {"content": "0600001-25"}}]},
            "data_sessao": {"date": {"start": row.data_sessao}},
            "youtube_link": {"url": row.youtube_link},
        }})


def test_metadata_only_update_of_existing_page_remains_possible(monkeypatch):
    client = core.NotionSessoesClient("fake-token")
    schema = core.NotionDataSourceSchema("fake", {"properties": {"tema": {"type": "title"}}})
    monkeypatch.setattr(client, "_prepare_row_schema", lambda schema, row: schema)
    requests = []
    monkeypatch.setattr(client, "_request", lambda method, path, **kw: requests.append((method, path)) or {"id": "existing"})
    row = video_row(action="update", page_id="existing", video_debate_evidence=None)
    assert client.update_row(schema, row.page_id, row)["id"] == "existing"
    assert requests == [("PATCH", "/pages/existing")]


def test_blocked_new_row_does_not_reserve_judgment_number():
    first = video_row(tipo_registro="Julgamento 1", video_debate_evidence=None)
    second = video_row(tipo_registro="Julgamento 2")
    disposition = core.assess_row_publishability(first)[0]
    core._renumber_judgments_after_skips([(first, disposition), (second, "publish")])
    assert disposition == "blocked"
    assert second.tipo_registro == "Julgamento 1"


def test_preview_captures_original_evidence_before_official_enrichment():
    original = video_row()
    bundle = core.JudgmentBundleExtraction(start_seconds=1620, end_seconds=1760, title_hint="Lista Tríplice", items=[
        core.JudgmentItemExtraction(numero_processo="0601075-11", tema=original.tema,
            analise_do_conteudo_juridico=original.video_debate_evidence.analysis_excerpt,
            resultado_final="Devolvida", relator="Min. Nunes Marques")])
    rows = core.build_preview_rows(core.AnalysisResult(
        session=core.SessionExtraction(data_sessao="2026-10-06"), bundles=[bundle]), original.youtube_link)
    assert len(rows) == 1
    row = rows[0]
    assert row.video_debate_evidence.numero_processo == "0601075-11"
    row.numero_processo = original.numero_processo
    assert not video_debate_issues(row)


def test_list_header_is_rejected_even_after_rich_legal_enrichment():
    row = video_row()
    row.video_debate_evidence.title_hint = "LISTA 1 - MACF"
    assert any("lista" in issue for issue in video_debate_issues(row))


def test_manual_vistoria_approval_cannot_replace_video_evidence():
    import vistoria_queue
    from test_tse_youtube_notion_core import FakeNotionClient, make_schema
    notion = FakeNotionClient()
    item = {"id": "review", "row": video_row(video_debate_evidence=None).model_dump()}
    result = vistoria_queue.publish_approved_items([item], notion, make_schema())
    assert result[0]["status"] == "blocked"
    assert any("trecho identificável" in text for text in result[0]["errors"])
    assert not notion.created


def test_proven_video_creation_reaches_api_without_internal_proof_argument(monkeypatch):
    client = core.NotionSessoesClient("fake-token", data_source_id="fake-ds")
    schema = core.NotionDataSourceSchema("fake-ds", {"properties": {
        "tema": {"type": "title"}, "numero_processo": {"type": "rich_text"},
        "youtube_link": {"type": "url"}, "data_sessao": {"type": "date"},
    }})
    monkeypatch.setattr(client, "_prepare_row_schema", lambda schema, row: schema)
    seen = []
    class Response:
        status_code = 200
        content = b'{}'
        def json(self):
            return {"id": "new-proven-page"}
    def request(method, url, **kw):
        seen.append((method, url, kw))
        assert "_video_debate_row" not in kw
        assert kw['json']['properties']['youtube_link']['url'] == video_row().youtube_link
        return Response()
    monkeypatch.setattr(client.session, "request", request)
    assert client.create_row(schema, video_row())["id"] == "new-proven-page"
    assert len(seen) == 1 and seen[0][0] == "POST"
