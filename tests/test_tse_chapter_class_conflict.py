import tse_youtube_notion_core as core


def test_chapter_cannot_reclassify_confirmed_force_request_as_triple_list(monkeypatch):
    # Real contradictory source: the chapter labels PA 0600742-04 as LT.
    monkeypatch.setattr(core, "fetch_youtube_description", lambda _: "00:25:48 LT 060074204")
    row = core.PublishPreviewRow(
        numero_processo="0600742-04",
        classe_processo="PA",
        tema="Requisição de força federal para segurança das eleições",
        analise_do_conteudo_juridico=(
            "O TRE-RN encaminhou ao TSE requisição de força federal para garantir "
            "a votação em nove municípios."
        ),
        youtube_link="https://www.youtube.com/watch?v=gY6-5zPrFlo",
    )
    core.enrich_preview_rows_with_youtube_chapters([row], row.youtube_link)
    assert row.classe_processo == "PA"
    assert row.youtube_link.endswith("&t=1548")
    assert any("capítulo" in warning and "mantida PA" in warning for warning in row.warnings)
    assert not row.errors


def test_unsubstantiated_pa_remains_correctable_by_matching_chapter(monkeypatch):
    monkeypatch.setattr(core, "fetch_youtube_description", lambda _: "00:25:48 LT 060074204")
    row = core.PublishPreviewRow(
        numero_processo="0600742-04", classe_processo="PA",
        youtube_link="https://www.youtube.com/watch?v=gY6-5zPrFlo&t=1550",
    )
    core.enrich_preview_rows_with_youtube_chapters([row], row.youtube_link)
    assert row.classe_processo == "Lista Tríplice"
    assert row.youtube_link.endswith("&t=1550")
    assert not row.warnings


def test_triple_list_description_still_allows_pa_label_correction(monkeypatch):
    monkeypatch.setattr(core, "fetch_youtube_description", lambda _: "00:23:15 LT 060104136")
    row = core.PublishPreviewRow(
        numero_processo="0601041-36", classe_processo="PA",
        tema="Encaminhamento de lista tríplice ao Poder Executivo",
        youtube_link="https://www.youtube.com/watch?v=gY6-5zPrFlo",
    )
    core.enrich_preview_rows_with_youtube_chapters([row], row.youtube_link)
    assert row.classe_processo == "Lista Tríplice"
    assert not row.warnings
