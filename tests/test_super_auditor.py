from types import SimpleNamespace

from tse_backfill_2025_notion import ExistingPageRecord
from tse_youtube_notion_core import NotionDataSourceSchema, PublishPreviewRow

from super_auditor import (
    FieldSuggestion,
    SuperAuditSuggestion,
    _extract_json_payload,
    _normalize_suggested_value,
    _target_fields_for_record,
    apply_super_audit_suggestions,
    build_global_relation_targets,
    origin_can_be_upgraded_from_local_text,
    punchline_looks_weak_for_super_audit,
    should_apply_suggestion,
    tema_looks_weak_for_super_audit,
)
from tse_backfill_2025_notion import RepairArtifactContext


def make_schema(with_relation: bool = True) -> NotionDataSourceSchema:
    properties = {
        "tema": {"type": "title", "title": {}},
        "classe_processo": {
            "type": "select",
            "select": {"options": [{"name": "REspe"}, {"name": "CTA"}, {"name": "AREspe"}]},
        },
        "tipo_registro": {
            "type": "select",
            "select": {"options": [{"name": "Julgamento 1"}]},
        },
        "origem": {
            "type": "select",
            "select": {"options": [{"name": "TSE"}, {"name": "Brasília/DF"}, {"name": "Granjeiro/CE"}]},
        },
        "tribunal": {
            "type": "select",
            "select": {"options": [{"name": "TSE"}, {"name": "TRE-CE"}]},
        },
        "numero_processo": {"type": "rich_text", "rich_text": {}},
        "youtube_link": {"type": "url", "url": {}},
        "resultado": {
            "type": "select",
            "select": {"options": [{"name": "Aprovada"}, {"name": "Desprovido"}, {"name": "Suspenso por vista"}]},
        },
        "votacao": {
            "type": "select",
            "select": {"options": [{"name": "Unânime"}, {"name": "Por maioria"}, {"name": "Suspenso"}]},
        },
        "pedido_vista": {
            "type": "select",
            "select": {"options": [{"name": "Min. André Mendonça"}]},
        },
        "partes": {"type": "multi_select", "multi_select": {"options": [{"name": "Alice (Recorrente)"}]}},
        "advogados": {"type": "multi_select", "multi_select": {"options": [{"name": "Dr. João da Silva"}]}},
        "composicao": {
            "type": "multi_select",
            "multi_select": {"options": [{"name": "Min. Cármen Lúcia"}, {"name": "Min. André Mendonça"}]},
        },
        "punchline": {"type": "rich_text", "rich_text": {}},
        "analise_do_conteudo_juridico": {"type": "rich_text", "rich_text": {}},
        "fundamentacao_normativa": {"type": "rich_text", "rich_text": {}},
        "precedentes_citados": {"type": "rich_text", "rich_text": {}},
        "raciocinio_juridico": {"type": "rich_text", "rich_text": {}},
        "resoluções_citadas": {"type": "rich_text", "rich_text": {}},
        "data_sessao": {"type": "date", "date": {}},
        "eleicao": {
            "type": "select",
            "select": {"options": [{"name": "2020"}, {"name": "2022"}, {"name": "2024"}]},
        },
        "relator": {
            "type": "select",
            "select": {"options": [{"name": "Min. Cármen Lúcia"}]},
        },
    }
    if with_relation:
        properties["materia_semelhante"] = {"type": "relation", "relation": {}}
    return NotionDataSourceSchema("fake-ds", {"properties": properties})


def make_record(page_id: str, video_id: str, data_sessao: str, numero: str) -> ExistingPageRecord:
    return ExistingPageRecord(
        page_id=page_id,
        url=f"https://notion.so/{page_id}",
        video_id=video_id,
        row=PublishPreviewRow(
            tema="Tema jurídico",
            punchline="Resumo jurídico completo.",
            tipo_registro="Julgamento 1",
            origem="TSE",
            tribunal="TSE",
            classe_processo="CTA",
            numero_processo=numero,
            youtube_link=f"https://www.youtube.com/watch?v={video_id}&t=10",
            data_sessao=data_sessao,
        ),
    )


def test_extract_json_payload_accepts_fenced_json():
    payload = _extract_json_payload("```json\n{\"tema\": {\"action\": \"keep\"}}\n```")
    assert payload["tema"]["action"] == "keep"


def test_should_apply_suggestion_requires_evidence_for_hard_field():
    suggestion = FieldSuggestion(
        action="update",
        suggested_value="TSE",
        confidence="high",
        evidence_snippets=[],
        reason="",
    )
    assert should_apply_suggestion("origem", suggestion, "high") is False


def test_normalize_suggested_value_flattens_structured_multivalue_dict():
    value = {
        "agravantes": ["Pablo Sergio dos Santos", "Mauro Roberto Pinheiro"],
        "agravados": ["Coligação A Voz do Povo"],
    }

    normalized = _normalize_suggested_value("partes", value)

    assert normalized == [
        "Pablo Sergio dos Santos",
        "Mauro Roberto Pinheiro",
        "Coligação A Voz do Povo",
    ]


def test_build_global_relation_targets_only_links_distinct_sessions():
    grouped = {
        2025: {
            "video-a": [
                make_record("page-1", "video-a", "2025-03-01", "0600001-01.2025.6.00.0000"),
                make_record("page-2", "video-a", "2025-03-01", "0600001-01.2025.6.00.0000"),
            ],
            "video-b": [
                make_record("page-3", "video-b", "2025-03-08", "0600001-01.2025.6.00.0000"),
            ],
        }
    }

    relations = build_global_relation_targets(grouped)

    assert relations["page-1"] == ["page-3"]
    assert relations["page-2"] == ["page-3"]
    assert relations["page-3"] == ["page-1", "page-2"]


def test_apply_super_audit_suggestions_respects_schema_and_relations():
    schema = make_schema(with_relation=True)
    row = PublishPreviewRow(
        tema="Caso do município de Aracaju",
        punchline="O processo trata de uma consulta.",
        tipo_registro="Julgamento 1",
        origem="TSE",
        tribunal="TSE",
        classe_processo="CTA",
        numero_processo="0600003-05",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-05-07",
    )
    suggestion = SuperAuditSuggestion(
        tema=FieldSuggestion(
            action="update",
            suggested_value="Fraude à cota de gênero e modulação dos efeitos da cassação",
            confidence="high",
            evidence_snippets=["fraude à cota de gênero", "modulação dos efeitos"],
            reason="",
        ),
        punchline=FieldSuggestion(
            action="update",
            suggested_value="Fraude à cota de gênero com debate sobre modulação dos efeitos da cassação.",
            confidence="high",
            evidence_snippets=["debate sobre modulação dos efeitos"],
            reason="",
        ),
    )

    repaired, changed_fields, review_fields, skipped_fields = apply_super_audit_suggestions(
        row,
        suggestion,
        schema,
        min_confidence="high",
        deterministic_relations=["page-2", "page-3"],
    )

    assert repaired.tema.startswith("Fraude à cota de gênero")
    assert repaired.materia_semelhante == ["page-2", "page-3"]
    assert "tema" in changed_fields
    assert "punchline" in changed_fields
    assert "materia_semelhante" in changed_fields
    assert review_fields == []
    assert skipped_fields == []


def test_apply_super_audit_suggestions_skips_relation_when_schema_missing():
    schema = make_schema(with_relation=False)
    row = PublishPreviewRow(
        tema="Tema jurídico",
        punchline="Resumo jurídico.",
        tipo_registro="Julgamento 1",
        origem="TSE",
        tribunal="TSE",
        classe_processo="CTA",
        numero_processo="0600003-05",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-05-07",
    )

    repaired, changed_fields, _review_fields, _skipped_fields = apply_super_audit_suggestions(
        row,
        SuperAuditSuggestion(),
        schema,
        min_confidence="high",
        deterministic_relations=["page-2"],
    )

    assert repaired.materia_semelhante == []
    assert "materia_semelhante" not in changed_fields


def test_apply_super_audit_suggestions_rejects_invalid_composition_size():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema jurídico",
        punchline="Resumo jurídico.",
        tipo_registro="Julgamento 1",
        origem="TSE",
        tribunal="TSE",
        classe_processo="CTA",
        numero_processo="0600003-05",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-05-07",
        composicao=[
            "Min. A",
            "Min. B",
            "Min. C",
            "Min. D",
            "Min. E",
            "Min. F",
            "Min. G",
        ],
    )
    suggestion = SuperAuditSuggestion(
        composicao=FieldSuggestion(
            action="update",
            suggested_value=[
                "Min. A",
                "Min. B",
                "Min. C",
                "Min. D",
                "Min. E",
                "Min. F",
                "Min. G",
                "Min. H",
            ],
            confidence="high",
            evidence_snippets=["sessão"],
            reason="",
        )
    )

    repaired, changed_fields, review_fields, skipped_fields = apply_super_audit_suggestions(
        row,
        suggestion,
        schema,
        min_confidence="high",
        deterministic_relations=[],
    )

    assert repaired.composicao == row.composicao
    assert "composicao" not in changed_fields
    assert review_fields == []
    assert "composicao" in skipped_fields


def test_theme_and_punchline_weak_heuristics_flag_meta_text():
    row = PublishPreviewRow(
        tema="Caso do município de Aracaju, relator Ministro Raul Araújo",
        punchline="O processo trata de uma consulta formulada pelo PSDB sobre a possibilidade de utilização de recursos do Fundo Partidário para o pagamento.",
        tipo_registro="Julgamento 1",
        origem="TSE",
        tribunal="TSE",
        classe_processo="CTA",
        numero_processo="0600003-05",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-05-07",
    )
    assert tema_looks_weak_for_super_audit(row) is True
    assert punchline_looks_weak_for_super_audit(row) is True


def test_theme_weak_heuristics_flag_result_proclamation():
    row = PublishPreviewRow(
        tema="Negado provimento ao agravo regimental, mantendo a decisão que aprovou com ressalvas as contas de campanha",
        punchline="O TSE negou provimento ao agravo regimental e manteve a aprovação com ressalvas das contas de campanha, reconhecendo a extrapolação dos limites de gastos.",
        tipo_registro="Julgamento 1",
        origem="TRE/SE",
        tribunal="TRE-SE",
        classe_processo="AgRg-REspe",
        numero_processo="0600279-20",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2025-10-09",
    )
    assert tema_looks_weak_for_super_audit(row) is True


def test_target_fields_focus_quality_core_targets_theme_and_punchline_when_only_text_is_weak():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Caso do município de Aracaju, relator Ministro Raul Araújo",
            punchline="O processo trata de uma consulta formulada pelo PSDB.",
            partes=[],
            tipo_registro="Julgamento 1",
            origem="TSE",
            tribunal="TSE",
            classe_processo="CTA",
            numero_processo="0600003-05",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2024-05-07",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, set(), artifact_context, focus="quality-core")

    assert fields == ["tema", "punchline", "relator", "resultado", "votacao", "eleicao"]
    assert "tema_weak_local" in reasons
    assert "punchline_weak_local" in reasons
    assert "relator_missing" in reasons
    assert "resultado_missing" in reasons
    assert "votacao_improvable" in reasons
    assert "eleicao_missing" in reasons


def test_origin_can_be_upgraded_from_local_text_when_row_is_only_tre():
    row = PublishPreviewRow(
        tema="Tema jurídico",
        punchline="Fraude à cota de gênero em candidatura vinculada ao município de Granjeiro/CE, com análise da prova e da votação.",
        analise_do_conteudo_juridico="O caso examina fatos ocorridos em Granjeiro/CE e a repercussão na eleição municipal.",
        tipo_registro="Julgamento 1",
        origem="TRE/CE",
        tribunal="TRE-CE",
        classe_processo="REspe",
        numero_processo="0600003-05",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-05-07",
    )

    assert origin_can_be_upgraded_from_local_text(row) == "Granjeiro/CE"


def test_origin_can_be_upgraded_from_local_text_when_current_is_tre_extenso():
    row = PublishPreviewRow(
        tema="Contas de campanha eleitoral",
        punchline="Resumo jurídico completo.",
        analise_do_conteudo_juridico=(
            "O recurso discute a aprovação com ressalvas das contas de campanha de candidata "
            "ao cargo de vereadora no município de Campo do Brito/SE nas eleições de 2024."
        ),
        tipo_registro="Julgamento 1",
        origem="Tribunal Regional Eleitoral de Sergipe",
        tribunal="TRE-SE",
        classe_processo="AgRg-REspe",
        numero_processo="0600279-20",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2025-10-09",
    )

    assert origin_can_be_upgraded_from_local_text(row) == "Campo do Brito/SE"


def test_target_fields_focus_quality_core_can_pull_origem_upgrade():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Fraude à cota de gênero",
            punchline="Fraude à cota de gênero em candidatura vinculada ao município de Granjeiro/CE, com análise da prova e da votação.",
            analise_do_conteudo_juridico="O caso examina fatos ocorridos em Granjeiro/CE e a repercussão na eleição municipal.",
            tipo_registro="Julgamento 1",
            origem="TRE/CE",
            tribunal="TRE-CE",
            classe_processo="REspe",
            numero_processo="0600003-05",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2024-05-07",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, set(), artifact_context, focus="quality-core")

    assert "origem" in fields
    assert "origem_city_from_local_text" in reasons


def test_target_fields_focus_origem_uses_upgrade_signal_only():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Fraude à cota de gênero",
            punchline="Fraude à cota de gênero em candidatura vinculada ao município de Granjeiro/CE, com análise da prova e da votação.",
            analise_do_conteudo_juridico="O caso examina fatos ocorridos em Granjeiro/CE e a repercussão na eleição municipal.",
            tipo_registro="Julgamento 1",
            origem="TRE/CE",
            tribunal="TRE-CE",
            classe_processo="REspe",
            numero_processo="0600003-05",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2024-05-07",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, set(), artifact_context, focus="origem")

    assert fields == ["origem"]
    assert "origem_city_from_local_text" in reasons


def test_target_fields_focus_quality_core_targets_only_punchline_when_other_signals_are_clean():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Abuso de poder político por uso indevido da máquina pública em campanha eleitoral",
            punchline="O voto da relatora foi acompanhado pelos ministros André Ramos Tavares, Nunes Marques e Cristiano Zanin.",
            tipo_registro="Julgamento 1",
            origem="TSE",
            tribunal="TSE",
            classe_processo="AREspe",
            numero_processo="0600682-94.2024.6.00.0000",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2024-05-07",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, set(), artifact_context, focus="quality-core")

    assert fields == ["punchline", "relator", "resultado", "votacao", "eleicao"]
    assert "punchline_weak_local" in reasons


def test_target_fields_focus_quality_core_combines_origin_theme_and_multivalues():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Negado provimento ao agravo regimental, mantendo a decisão que aprovou com ressalvas as contas de campanha",
            punchline="O TSE negou provimento ao agravo regimental e manteve a aprovação com ressalvas das contas de campanha, reconhecendo a extrapolação dos limites de gastos.",
            analise_do_conteudo_juridico=(
                "O recurso discute a aprovação com ressalvas das contas de campanha de candidata "
                "ao cargo de vereadora no município de Campo do Brito/SE nas eleições de 2024."
            ),
            partes=[],
            advogados=[],
            tipo_registro="Julgamento 1",
            origem="Tribunal Regional Eleitoral de Sergipe",
            tribunal="TRE-SE",
            classe_processo="AgRg-REspe",
            numero_processo="0600279-20",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2025-10-09",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext
    from tse_normalization import canonicalize_numero_processo

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2025-10-09",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            canonicalize_numero_processo("0600279-20"): SimpleNamespace(
                partes=["Crisnádia Passos Cruz"],
                advogados=["Dr. João da Silva"],
                precedentes_citados="",
                resolucoes_citadas="",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, set(), artifact_context, focus="quality-core")

    assert fields == [
        "origem",
        "tema",
        "punchline",
        "partes",
        "advogados",
        "relator",
        "resultado",
        "votacao",
        "eleicao",
    ]
    assert "origem_city_from_local_text" in reasons
    assert "tema_weak_local" in reasons
    assert "punchline_weak_local" in reasons
    assert "partes_missing_with_artifact" in reasons
    assert "advogados_improvable" in reasons


def test_target_fields_focus_quality_core_prioritizes_blank_completion_fields():
    record = ExistingPageRecord(
        page_id="page-1",
        url="https://notion.so/page-1",
        video_id="video-a",
        row=PublishPreviewRow(
            tema="Abuso de poder político por uso promocional da máquina pública em campanha eleitoral",
            punchline="O julgamento examinou abuso de poder político mediante uso promocional da máquina pública em campanha eleitoral, com análise concreta da prova produzida.",
            partes=["Alice (Recorrente)"],
            advogados=["Dr. João da Silva"],
            composicao=["Min. Cármen Lúcia"],
            tipo_registro="Julgamento 1",
            origem="Granjeiro/CE",
            tribunal="",
            classe_processo="REspe",
            numero_processo="0600003-05",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
            data_sessao="2024-05-07",
            relator="",
            pedido_vista="",
            resultado="",
            votacao="",
            eleicao="",
        ),
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[
            "Min. Cármen Lúcia",
            "Min. André Mendonça",
            "Min. Nunes Marques",
        ],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(
        record,
        {"relator_empty", "resultado_empty", "votacao_empty", "composicao_incomplete"},
        artifact_context,
        focus="quality-core",
    )

    assert fields == [
        "relator",
        "tribunal",
        "resultado",
        "votacao",
        "eleicao",
        "composicao",
    ]
    assert "relator_missing" in reasons
    assert "tribunal_missing" in reasons
    assert "resultado_missing" in reasons
    assert "votacao_improvable" in reasons
    assert "eleicao_missing" in reasons
    assert "composicao_incomplete" in reasons


def test_target_fields_focus_relator_targets_only_missing_relator():
    row = PublishPreviewRow(
        tema="Tema jurídico",
        punchline="Resumo jurídico completo.",
        tipo_registro="Julgamento 1",
        origem="TSE",
        tribunal="TSE",
        classe_processo="CTA",
        numero_processo="0600001-00.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc&t=10",
        data_sessao="2024-05-07",
        relator="",
        resultado="Desprovido",
        votacao="Unânime",
        eleicao="2024",
    )
    record = ExistingPageRecord(
        page_id="page-relator",
        url="https://notion.so/page-relator",
        video_id="abc",
        row=row,
    )
    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-07",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(record, {"relator_empty"}, artifact_context, focus="relator")

    assert fields == ["relator"]
    assert reasons == ["relator_missing"]


def test_target_fields_focus_residual_core_targets_only_requested_residual_fields():
    row = PublishPreviewRow(
        tema="Caso do município de Aracaju, relator Ministro Raul Araújo",
        punchline="O processo trata de consulta formulada sobre uso de recursos públicos em campanha eleitoral.",
        tipo_registro="Julgamento 1",
        origem="Tribunal Regional Eleitoral de Sergipe",
        tribunal="TRE-SE",
        classe_processo="",
        numero_processo="0600279-20",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2025-10-09",
        relator="",
        resultado="",
        votacao="",
        analise_do_conteudo_juridico=(
            "O recurso discute a aprovação com ressalvas das contas de campanha de candidata "
            "ao cargo de vereadora no município de Campo do Brito/SE nas eleições de 2024."
        ),
    )
    record = ExistingPageRecord(
        page_id="page-residual",
        url="https://notion.so/page-residual",
        video_id="video-a",
        row=row,
    )
    from tse_backfill_2025_notion import RepairArtifactContext

    artifact_context = RepairArtifactContext(
        artifact_dir=None,
        session_date="2025-10-09",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    fields, reasons = _target_fields_for_record(
        record,
        {"origem_tre_extenso", "classe_empty", "relator_empty", "resultado_empty", "votacao_empty"},
        artifact_context,
        focus="residual-core",
    )

    assert fields == ["origem", "classe_processo", "relator", "resultado", "votacao"]
    assert "origem_city_from_local_text" in reasons
    assert "classe_improvable" in reasons
    assert "relator_missing" in reasons
    assert "resultado_missing" in reasons
    assert "votacao_improvable" in reasons
