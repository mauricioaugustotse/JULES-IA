import json
import logging

import tse_youtube_notion_core as core
from tse_youtube_notion_core import (
    AnalysisResult,
    enrich_preview_rows_with_process_metadata,
    GeminiSessionExtractor,
    GeminiNewsEnricher,
    GeminiProcessMetadataEnricher,
    InstitutionalRepairResult,
    JudgmentBundleExtraction,
    JudgmentItemExtraction,
    NewsEnrichmentResult,
    NotionDataSourceSchema,
    NotionSessoesClient,
    NotionRowMatch,
    PublishPreviewRow,
    RunArtifacts,
    SessionExtraction,
    SessionWindow,
    StartRefinementResult,
    TranscriptChunk,
    TranscriptSnippet,
    build_preview_rows,
    build_fallback_tema,
    build_fundamentacao_column_text,
    build_gemini_http_options,
    build_raciocinio_column_text,
    enrich_preview_rows_with_news,
    extract_retry_delay_seconds,
    filter_general_news_urls,
    infer_classe_from_row_text,
    infer_origin_from_row_text,
    infer_relator_from_row_text,
    infer_votacao_from_row_text,
    normalize_party_list,
    publish_preview_rows,
    should_disable_model,
    create_gemini_client,
    validate_preview_row,
)


def make_schema() -> NotionDataSourceSchema:
    raw_payload = {
        "properties": {
            "tema": {"type": "title", "title": {}},
            "classe_processo": {
                "type": "select",
                "select": {"options": [{"name": "PA"}, {"name": "AgRg-REspe"}, {"name": "Lista Tríplice"}]},
            },
            "tipo_registro": {
                "type": "select",
                "select": {"options": [{"name": "Julgamento 1"}, {"name": "Julgamento 2"}, {"name": "Julgamento 3"}]},
            },
            "eleicao": {
                "type": "select",
                "select": {"options": [{"name": "2020"}, {"name": "2022"}, {"name": "2024"}, {"name": "2026"}]},
            },
            "origem": {
                "type": "select",
                "select": {"options": [{"name": "Porto Alegre/RS"}, {"name": "Brasília/DF"}]},
            },
            "tribunal": {
                "type": "select",
                "select": {"options": [{"name": "TRE-RS"}, {"name": "TRE-DF"}, {"name": "TSE"}]},
            },
            "numero_processo": {"type": "rich_text", "rich_text": {}},
            "youtube_link": {"type": "url", "url": {}},
            "relator": {
                "type": "select",
                "select": {"options": [{"name": "Min. Cármen Lúcia"}, {"name": "Min. André Mendonça"}]},
            },
            "pedido_vista": {
                "type": "select",
                "select": {"options": [{"name": "Min. André Mendonça"}, {"name": "Min. Isabel Gallotti"}]},
            },
            "resultado": {
                "type": "select",
                "select": {"options": [{"name": "Aprovada"}, {"name": "Desprovido"}, {"name": "Suspenso por vista"}]},
            },
            "votacao": {
                "type": "select",
                "select": {"options": [{"name": "Unânime"}, {"name": "Por maioria"}, {"name": "Suspenso"}]},
            },
            "partes": {
                "type": "multi_select",
                "multi_select": {"options": [{"name": "Alice"}, {"name": "Bob"}]},
            },
            "advogados": {
                "type": "multi_select",
                "multi_select": {"options": [{"name": "Dr. João da Silva"}]},
            },
            "composicao": {
                "type": "multi_select",
                "multi_select": {
                    "options": [{"name": "Min. Cármen Lúcia"}, {"name": "Min. André Mendonça"}]
                },
            },
            "punchline": {"type": "rich_text", "rich_text": {}},
            "analise_do_conteudo_juridico": {"type": "rich_text", "rich_text": {}},
            "fundamentacao_normativa": {"type": "rich_text", "rich_text": {}},
            "precedentes_citados": {"type": "rich_text", "rich_text": {}},
            "raciocinio_juridico": {"type": "rich_text", "rich_text": {}},
            "resoluções_citadas": {"type": "rich_text", "rich_text": {}},
            "data_sessao": {"type": "date", "date": {}},
            "noticia_TSE": {"type": "url", "url": {}},
            "noticia_TRE": {"type": "url", "url": {}},
            "noticia_geral_1": {"type": "url", "url": {}},
            "noticia_geral_2": {"type": "url", "url": {}},
            "noticia_geral_3": {"type": "url", "url": {}},
        }
    }
    return NotionDataSourceSchema("fake-ds", raw_payload)


def make_analysis() -> AnalysisResult:
    return AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
            judgments=[
                SessionWindow(title_hint="Julgamento 1", start_seconds=931),
                SessionWindow(title_hint="Lista final", start_seconds=1900, should_ignore=True, ignore_reason="lista"),
            ],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=931,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="processo administrativo",
                        numero_processo="0600249-07.2024.6.13.0000",
                        origem="Porto Alegre - RS",
                        tre="",
                        partes=["Alice", "Ministério Público Eleitoral", "Bob"],
                        advogados=["João da Silva"],
                        composicao=["Ministra Cármen Lúcia", "Ministro André Mendonça"],
                        relator="Ministra Cármen Lúcia",
                        pedido_vista="",
                        tema="Tema do julgamento",
                        punchline="Resumo forte",
                        analise_do_conteudo_juridico="Análise",
                        fundamentacao_normativa="CF, Lei 9.504/97",
                        precedentes_citados="Precedente TSE",
                        raciocinio_juridico="Tese vencedora",
                        pontos_processuais_relevantes="Questão de ordem",
                        efeitos_e_providencias_praticas="Comunicar ao TRE",
                        resolucoes_citadas="Res.-TSE 23.000",
                        votacao="por unanimidade",
                        resultado_final="aprovada",
                    )
                ],
            ),
            JudgmentBundleExtraction(
                should_ignore=True,
                ignore_reason="lista",
                items=[],
            ),
        ],
    )


class FakeNotionClient:
    def __init__(self) -> None:
        self.created = []
        self.updated = []

    def find_existing_row(self, schema, youtube_link: str, numero_processo: str):
        if numero_processo == "0600249-07":
            return NotionRowMatch(page_id="page-123", url="https://notion.so/page-123")
        return None

    def create_row(self, schema, row: PublishPreviewRow):
        self.created.append(row)
        return {"id": "page-created", "url": "https://notion.so/page-created"}

    def update_row(self, schema, page_id: str, row: PublishPreviewRow):
        self.updated.append((page_id, row))
        return {"id": page_id, "url": f"https://notion.so/{page_id}"}


def test_build_preview_rows_ignores_list_block_and_marks_update():
    schema = make_schema()
    notion = FakeNotionClient()
    rows = build_preview_rows(make_analysis(), "https://youtu.be/abc123", schema, notion)
    assert len(rows) == 1
    row = rows[0]
    assert row.action == "update"
    assert row.page_id == "page-123"
    assert row.numero_processo == "0600249-07.2024.6.13.0000"
    assert row.tribunal == "TRE-RS"
    assert row.data_sessao == "2026-03-20"
    assert row.youtube_link == "https://www.youtube.com/watch?v=abc123&t=931"
    assert row.partes == ["Alice", "Bob"]
    assert row.advogados == ["Dr. João da Silva"]


def test_build_preview_rows_dedupes_same_process_and_keeps_earliest_timestamp():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=120,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="Agravo Regimental no Agravo em Recurso Especial Eleitoral",
                        numero_processo="060036879",
                        origem="Brasília/DF",
                        tre="TRE-DF",
                        partes=["Alice"],
                        advogados=[],
                        composicao=["Ministra Cármen Lúcia"],
                        relator="Ministro André Mendonça",
                        tema="Tema cedo",
                        punchline="Resumo cedo",
                        analise_do_conteudo_juridico="Análise cedo",
                        fundamentacao_normativa="Fundamentação cedo",
                        precedentes_citados="",
                        raciocinio_juridico="",
                        pontos_processuais_relevantes="",
                        efeitos_e_providencias_praticas="",
                        resolucoes_citadas="",
                        votacao="Unânime",
                        resultado_final="Agravo regimental desprovido",
                    )
                ],
            ),
            JudgmentBundleExtraction(
                start_seconds=180,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="Agravo Regimental no Agravo em Recurso Especial Eleitoral",
                        numero_processo="060036879",
                        origem="Brasília/DF",
                        tre="TRE-DF",
                        partes=["Bob"],
                        advogados=["João da Silva"],
                        composicao=["Ministra Cármen Lúcia", "Ministro André Mendonça"],
                        relator="Ministro André Mendonça",
                        tema="Tema tarde",
                        punchline="Resumo tarde",
                        analise_do_conteudo_juridico="Análise tarde",
                        fundamentacao_normativa="Fundamentação tarde",
                        precedentes_citados="Precedente",
                        raciocinio_juridico="Raciocínio",
                        pontos_processuais_relevantes="Ponto processual",
                        efeitos_e_providencias_praticas="Efeito prático",
                        resolucoes_citadas="Res.-TSE 23.000",
                        votacao="Unânime",
                        resultado_final="Agravo regimental desprovido",
                    )
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/abc123", schema, notion)
    assert len(rows) == 1
    row = rows[0]
    assert row.youtube_link == "https://www.youtube.com/watch?v=abc123&t=120"
    assert row.partes == ["Alice", "Bob"]
    assert row.advogados == ["Dr. João da Silva"]
    assert row.composicao == ["Min. Cármen Lúcia", "Min. André Mendonça"]
    assert row.tipo_registro == "Julgamento 1"


def test_build_preview_rows_prefers_session_composition_when_item_is_sparse():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=[
                "Ministra Cármen Lúcia",
                "Ministro André Mendonça",
                "Ministra Isabel Gallotti",
                "Ministro Kassio Nunes Marques",
                "Ministro Floriano de Azevedo Marques",
                "Ministro Alexandre de Moraes",
                "Ministro Ramos Tavares",
            ],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=120,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="PA",
                        numero_processo="0600001-01.2024.6.00.0000",
                        origem="Brasília/DF",
                        tre="TSE",
                        partes=["Alice"],
                        composicao=["Ministra Cármen Lúcia", "Ministro André Mendonça"],
                        relator="Ministro André Mendonça",
                        tema="Tema útil",
                        punchline="Resumo",
                        resultado_final="Aprovada",
                        votacao="Unânime",
                    )
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/abc123", schema, notion)
    assert rows[0].composicao == [
        "Min. Cármen Lúcia",
        "Min. André Mendonça",
        "Min. Isabel Gallotti",
        "Min. Nunes Marques",
        "Min. Floriano de Azevedo Marques",
        "Min. Alexandre de Moraes",
        "Min. Ramos Tavares",
    ]


def test_build_preview_rows_orders_joint_cases_by_item_position_not_process_number():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=500,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="PA",
                        numero_processo="0600999-99.2024.6.00.0000",
                        origem="Brasília/DF",
                        tre="TSE",
                        partes=["Alice"],
                        relator="Ministra Cármen Lúcia",
                        tema="Conduta vedada",
                        punchline="Tema 1",
                        resultado_final="Aprovada",
                        votacao="Unânime",
                    ),
                    JudgmentItemExtraction(
                        data_sessao="20/03/2026",
                        eleicao="2024",
                        classe_processo="PA",
                        numero_processo="0600001-01.2024.6.00.0000",
                        origem="Brasília/DF",
                        tre="TSE",
                        partes=["Bob"],
                        relator="Ministra Cármen Lúcia",
                        tema="Fraude à cota de gênero",
                        punchline="Tema 2",
                        resultado_final="Aprovada",
                        votacao="Unânime",
                    ),
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/abc123", schema, notion)
    assert [row.numero_processo for row in rows] == [
        "0600999-99.2024.6.00.0000",
        "0600001-01.2024.6.00.0000",
    ]
    assert [row.tipo_registro for row in rows] == ["Julgamento 1", "Julgamento 2"]


def test_build_raciocinio_column_text_keeps_reasoning_as_primary_section():
    value = build_raciocinio_column_text(
        "O relator aplicou a Súmula 24 porque a tese recursal exigia rediscutir a prova do uso do bem público.",
        "O agravo reiterou as razões do recurso especial.",
        "Ficou mantida a multa aplicada pelo TRE.",
    )
    assert value.startswith("O relator aplicou a Súmula 24")
    assert "Súmula 24" in value
    assert "Pontos Processuais Relevantes" not in value
    assert "Efeitos e Providências Práticas" not in value


def test_build_fundamentacao_column_text_keeps_devices_explicit():
    value = build_fundamentacao_column_text(
        "Art. 73, I, da Lei 9.504/1997; Súmula 24 do TSE."
    )
    assert value.startswith("Art. 73, I")
    assert "Art. 73, I" in value
    assert "Súmula 24" in value


def test_build_fallback_tema_uses_punchline_when_tema_is_empty():
    row = PublishPreviewRow(
        tema="",
        punchline="Propaganda eleitoral antecipada negativa.",
        classe_processo="AgRg-REspe",
        numero_processo="0600564-43.2024.6.26.0199",
    )
    assert build_fallback_tema(row) == "Propaganda eleitoral antecipada negativa"


def test_build_fallback_tema_never_falls_back_to_process_number():
    row = PublishPreviewRow(
        tema="Processo 0600564-43.2024.6.26.0199",
        punchline="",
        classe_processo="AgRg-REspe",
        numero_processo="0600564-43.2024.6.26.0199",
    )
    assert build_fallback_tema(row) == ""


def test_build_fallback_tema_treats_result_only_theme_as_generic():
    row = PublishPreviewRow(
        tema="Provido",
        punchline="",
        classe_processo="",
        numero_processo="0600067-69.2024.6.07.0001",
        resultado="Provido",
    )
    assert build_fallback_tema(row) == ""


def test_build_fallback_tema_treats_class_plus_result_theme_as_generic():
    row = PublishPreviewRow(
        tema="PC Aprovada",
        punchline="",
        classe_processo="",
        numero_processo="0601650-29.2020.6.00.0000",
        resultado="Aprovada",
    )
    assert build_fallback_tema(row) == ""


def test_build_fallback_tema_infers_from_analysis_text_when_theme_is_generic():
    row = PublishPreviewRow(
        tema="Provido",
        punchline="",
        classe_processo="",
        numero_processo="0600067-69.2024.6.07.0001",
        resultado="Provido",
        analise_do_conteudo_juridico=(
            "O recurso discute a manutenção de medidas cautelares de busca e apreensão, "
            "sequestro e bloqueio de bens e valores na Operação Fundo do Poço."
        ),
    )
    assert build_fallback_tema(row) == "Manutenção de medidas cautelares patrimoniais"


def test_build_fallback_tema_infers_fundo_partidario_consultoria_theme():
    row = PublishPreviewRow(
        tema="Processo 0600366-24.2022.6.00.0000",
        punchline="",
        numero_processo="0600366-24.2022.6.00.0000",
        analise_do_conteudo_juridico=(
            "A consulta versa sobre a possibilidade de utilização de recursos do Fundo Partidário "
            "para o pagamento de despesas com a contratação de serviços de consultoria jurídica e contábil."
        ),
    )
    assert build_fallback_tema(row) == "Uso do Fundo Partidário para custear consultoria jurídica e contábil"


def test_build_fallback_tema_infers_publicidade_institucional_theme():
    row = PublishPreviewRow(
        tema="Processo 0600557-55.2022.6.05.0000",
        punchline="",
        numero_processo="0600557-55.2022.6.05.0000",
        analise_do_conteudo_juridico=(
            "O recurso trata de suposta conduta vedada a agente público, consistente na realização "
            "de publicidade institucional em período vedado."
        ),
    )
    assert build_fallback_tema(row) == "Publicidade institucional em período vedado"


def test_build_fallback_tema_infers_programa_social_theme():
    row = PublishPreviewRow(
        tema="Processo 0600469-83.2022.6.19.0000",
        punchline="",
        numero_processo="0600469-83.2022.6.19.0000",
        analise_do_conteudo_juridico=(
            "O recurso trata de suposta prática de conduta vedada a agente público consistente na "
            "utilização de programas sociais com finalidade eleitoral durante o período de campanha."
        ),
    )
    assert build_fallback_tema(row) == "Uso promocional de programa social como conduta vedada"


def test_build_fallback_tema_infers_panfletagem_theme():
    row = PublishPreviewRow(
        tema="Processo 0600607-35.2022.6.00.0000",
        punchline="",
        numero_processo="0600607-35.2022.6.00.0000",
        analise_do_conteudo_juridico=(
            "O caso trata da desaprovação das contas de campanha. A controvérsia central girou em torno "
            "da regularidade dos gastos com panfletagem e da ausência de contratos individuais de trabalho."
        ),
    )
    assert build_fallback_tema(row) == "Comprovação de gastos com panfletagem em prestação de contas"


def test_build_fallback_tema_discards_truncated_inelegibilidade_theme():
    row = PublishPreviewRow(
        tema="Processo 0600273-14.2024.6.26.0000",
        punchline="",
        numero_processo="0600273-14.2024.6.26.0000",
        analise_do_conteudo_juridico="O recurso discute a inelegibilidade do art. 1º, I, g, da LC 64/1990.",
    )
    assert build_fallback_tema(row) == ""


def test_build_fallback_tema_replaces_overbroad_inelegibilidade_with_specific_theme():
    row = PublishPreviewRow(
        tema="Inelegibilidade",
        numero_processo="0600058-01.2024.6.14.0022",
        analise_do_conteudo_juridico=(
            "Trata-se de julgamento de contas com imputação de débito relativo a convênio "
            "para fortalecimento da agricultura familiar. O TRE reconheceu a causa de "
            "inelegibilidade, mas o objeto do convênio foi executado integralmente."
        ),
        raciocinio_juridico=(
            "O relator concluiu que a execução integral do objeto pactuado afasta a "
            "configuração de ato doloso de improbidade administrativa."
        ),
    )
    assert build_fallback_tema(row) == "Execução integral de convênio afasta inelegibilidade por rejeição de contas"


def test_infer_relator_from_row_text_reads_relatoria_phrase():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico="O processo estava sob relatoria do Ministro Ramos Tavares e foi levado a julgamento."
    )
    assert infer_relator_from_row_text(row) == "Min. Ramos Tavares"


def test_infer_votacao_from_row_text_detects_unanimity():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico="O Plenário aprovou, por unanimidade, a lista tríplice."
    )
    assert infer_votacao_from_row_text(row) == "Unânime"


def test_infer_classe_from_row_text_detects_consulta():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "O processo trata de uma consulta formulada ao Tribunal Superior Eleitoral "
            "sobre federação partidária."
        )
    )
    assert infer_classe_from_row_text(row) == "CTA"


def test_infer_origin_from_row_text_extracts_city_uf():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico="Discute-se a inelegibilidade de candidatos a prefeito e vice-prefeito de Paranhos/MS."
    )
    assert infer_origin_from_row_text(row) == "Paranhos/MS"


def test_validate_preview_row_keeps_safe_new_select_options():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0600001-01.2024.6.00.0000",
        data_sessao="2025-03-20",
        classe_processo="REspe",
        relator="Min. Alexandre de Moraes",
        votacao="Unânime",
    )

    validated = validate_preview_row(row, schema)

    assert validated.classe_processo == "REspe"
    assert validated.relator == "Min. Alexandre de Moraes"
    assert validated.votacao == "Unânime"
    assert any("opção nova no Notion" in message for message in validated.warnings)


def test_normalize_party_list_keeps_role_at_end_and_drops_lawyers():
    values = normalize_party_list(
        [
            "Recorrente: Alice",
            "Dr. João da Silva",
            "Agravante: Bob",
            "OAB/DF 12345",
            "MPE",
        ]
    )
    assert values == ["Alice (Recorrente)", "Bob (Agravante)"]


def test_normalize_party_list_parses_serialized_role_mapping():
    values = normalize_party_list(
        [
            "{'embargante': 'Cláudia Aparecida dos Santos'",
            "'embargados': ['Denilson Aparecido Martins'",
            "'Federação Brasil da Esperança de Santa Luzia']}",
        ]
    )
    assert values == [
        "Cláudia Aparecida dos Santos (Embargante)",
        "Denilson Aparecido Martins (Embargado)",
        "Federação Brasil da Esperança de Santa Luzia (Embargado)",
    ]


def test_normalize_party_list_normalizes_suffix_role_label():
    values = normalize_party_list(["Thiago Soares de Godoy (agravante)"])
    assert values == ["Thiago Soares de Godoy (Agravante)"]


def test_validate_preview_row_strips_legacy_section_headings():
    validated = validate_preview_row(
        PublishPreviewRow(
            tema="Tema jurídico",
            raciocinio_juridico="Raciocínio Jurídico Aplicado ao Caso Concreto\nFundamento aplicado.",
            fundamentacao_normativa="Fundamentação Normativa e Dispositivos Citados\nArt. 73, I, da Lei 9.504/1997.",
            data_sessao="20/03/2026",
        ),
        None,
    )
    assert validated.raciocinio_juridico == "Fundamento aplicado."
    assert validated.fundamentacao_normativa == "Art. 73, I, da Lei 9.504/1997."


def test_build_properties_payload_does_not_fallback_title_to_process_number():
    schema = make_schema()
    client = NotionSessoesClient(api_key="token", data_source_id="fake-ds")
    payload = client.build_properties_payload(
        schema,
        PublishPreviewRow(
            tema="",
            punchline="",
            numero_processo="0600564-43.2024.6.26.0199",
            action="update",
        ),
    )
    assert schema.title_property_name not in payload


def test_build_properties_payload_can_clear_title_on_update():
    schema = make_schema()
    client = NotionSessoesClient(api_key="token", data_source_id="fake-ds")
    payload = client.build_properties_payload(
        schema,
        PublishPreviewRow(
            tema="",
            punchline="",
            numero_processo="0600564-43.2024.6.26.0199",
            action="update",
            force_clear_title=True,
        ),
    )
    assert payload[schema.title_property_name] == {"title": []}


def test_build_properties_payload_can_clear_multi_select_on_update():
    schema = make_schema()
    client = NotionSessoesClient(api_key="token", data_source_id="fake-ds")
    payload = client.build_properties_payload(
        schema,
        PublishPreviewRow(
            partes=[],
            action="update",
            clear_properties=["partes"],
        ),
    )
    assert payload["partes"] == {"multi_select": []}


def test_validate_preview_row_downgrades_noncritical_invalid_selects_to_warning():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="Classe Inventada",
        tipo_registro="Julgamento 9",
        eleicao="2099",
        origem="Cidade/ZZ",
        tribunal="TRE-ZZ",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        relator="Min. Relator Desconhecido",
        resultado="Resultado Inventado",
        votacao="Qualquer",
        data_sessao="20/03/2026",
    )
    validated = validate_preview_row(row, schema)
    assert validated.blocked
    assert any("tribunal" in error for error in validated.errors)
    assert any("classe_processo" in warning for warning in validated.warnings)
    assert any("resultado" in warning for warning in validated.warnings)
    assert validated.classe_processo == ""
    assert validated.resultado == ""


def test_validate_preview_row_clears_stale_dynamic_errors_after_normalization():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="Não especificada",
        origem="Cidade/UF",
        tribunal="TRE-UF",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        relator="Min. Maria Isabel Gallotti",
        resultado="Homologado",
        votacao="Pedido de vista pelo Ministro Nunes Marques.",
        data_sessao="20/03/2026",
        warnings=["resultado fora das opções do Notion; valor omitido: Homologado"],
        errors=[
            "Valor inválido para resultado: Homologado",
            "Valor inválido para relator: Min. Maria Isabel Gallotti",
            "Valor inválido para eleicao: Não especificada",
            "Valor inválido para votacao: Pedido de vista pelo Ministro Nunes Marques.",
        ],
    )
    validated = validate_preview_row(row, schema)
    assert "Valor inválido para resultado: Homologado" not in validated.errors
    assert "Valor inválido para relator: Min. Maria Isabel Gallotti" not in validated.errors
    assert "Valor inválido para eleicao: Não especificada" not in validated.errors
    assert "Valor inválido para votacao: Pedido de vista pelo Ministro Nunes Marques." not in validated.errors
    assert validated.resultado == "Aprovada"
    assert validated.relator == "Min. Isabel Gallotti"
    assert validated.eleicao == ""
    assert validated.pedido_vista == ""
    assert not any("relator fora das opções do Notion" in warning for warning in validated.warnings)


def test_publish_preview_rows_skips_blocked_and_updates_existing():
    schema = make_schema()
    notion = FakeNotionClient()
    valid_row = build_preview_rows(make_analysis(), "https://youtu.be/abc123", schema, notion)[0]
    blocked_row = PublishPreviewRow(
        tema="Bloqueado",
        classe_processo="Classe Inventada",
        tipo_registro="Julgamento 2",
        eleicao="2024",
        origem="Brasília/DF",
        tribunal="TRE-ZZ",
        numero_processo="0000001-99",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=2000",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2026-03-20",
        errors=["Valor inválido para classe_processo"],
    )
    results = publish_preview_rows([valid_row, blocked_row], notion, schema)
    assert results[0]["status"] == "updated"
    assert results[1]["status"] == "blocked"
    assert notion.updated and not notion.created


def test_find_existing_row_matches_same_video_even_with_different_timestamp():
    schema = make_schema()
    client = NotionSessoesClient(api_key="fake-token", data_source_id="fake-ds")
    candidate_page = {
        "id": "page-123",
        "url": "https://notion.so/page-123",
        "properties": {
            "tema": {"title": [{"plain_text": "Tema"}]},
            "numero_processo": {"rich_text": [{"plain_text": "0600249-07"}]},
            "youtube_link": {"url": "https://www.youtube.com/watch?v=abc123&t=10"},
        },
    }

    def fake_query_data_source(filter_payload=None):
        return [candidate_page]

    client.query_data_source = fake_query_data_source
    match = client.find_existing_row(
        schema,
        youtube_link="https://www.youtube.com/watch?v=abc123&t=931",
        numero_processo="0600249-07",
    )
    assert match is not None
    assert match.page_id == "page-123"


def test_validate_preview_row_normalizes_news_urls(monkeypatch):
    class FakeResponse:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 200
            self.headers = {"Content-Type": "text/html; charset=utf-8"}
            self.content = b"<html><body>Noticia valida</body></html>"
            self.text = self.content.decode("utf-8")

    def fake_get(url, *args, **kwargs):
        return FakeResponse(url)

    monkeypatch.setattr(core.requests, "get", fake_get)
    core.fetch_candidate_page_snapshot.cache_clear()
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Porto Alegre/RS",
        tribunal="TRE-RS",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="20/03/2026",
        noticia_TSE="tse.jus.br/comunicacao/noticia",
        noticia_TRE="https://tre-rs.jus.br/noticia",
        noticias_gerais=[
            "g1.globo.com/noticia-1",
            "https://conjur.com.br/noticia-2",
        ],
    )
    validated = validate_preview_row(row, schema)
    assert validated.noticia_TSE == "https://tse.jus.br/comunicacao/noticia"
    assert validated.noticia_TRE == "https://tre-rs.jus.br/noticia"
    assert validated.noticias_gerais == [
        "https://g1.globo.com/noticia-1",
        "https://conjur.com.br/noticia-2",
    ]


def test_validate_preview_row_discards_unavailable_tse_news(monkeypatch):
    class FakeResponse:
        def __init__(self, url: str):
            self.url = url
            self.status_code = 404
            self.headers = {"Content-Type": "text/html; charset=utf-8"}
            self.content = (
                "<html><head><title>Página não encontrada</title></head>"
                "<body>Página não encontrada.</body></html>"
            ).encode("utf-8")
            self.text = self.content.decode("utf-8")

    def fake_get(url, *args, **kwargs):
        return FakeResponse(url)

    monkeypatch.setattr(core.requests, "get", fake_get)
    core.fetch_candidate_page_snapshot.cache_clear()
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Porto Alegre/RS",
        tribunal="TRE-RS",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="20/03/2026",
        noticia_TSE="https://www.tse.jus.br/comunicacao/noticias/2025/Dezembro/link-inexistente",
    )
    validated = validate_preview_row(row, make_schema())
    assert validated.noticia_TSE == ""
    assert "noticia_TSE descartada por indisponibilidade da página." in validated.warnings


def test_validate_preview_row_resolves_grounding_redirect_urls(monkeypatch):
    class FakeResponse:
        url = "https://www.tre-mt.jus.br/comunicacao/noticias/2026/Marco/biometria"

    def fake_get(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(core.requests, "get", fake_get)
    core.fetch_candidate_page_snapshot.cache_clear()
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Porto Alegre/RS",
        tribunal="TRE-RS",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="20/03/2026",
        noticias_gerais=[
            "https://vertexaisearch.cloud.google.com/grounding-api-redirect/abc",
        ],
    )
    validated = validate_preview_row(row, make_schema())
    assert validated.noticias_gerais == [
        "https://www.tre-mt.jus.br/comunicacao/noticias/2026/Marco/biometria"
    ]


def test_build_properties_payload_includes_news_urls():
    schema = make_schema()
    notion_client = NotionSessoesClient(api_key="fake-token", data_source_id="fake-ds")
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Porto Alegre/RS",
        tribunal="TRE-RS",
        numero_processo="0600249-07",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2026-03-20",
        noticia_TSE="https://tse.jus.br/noticia-tse",
        noticia_TRE="https://tre-rs.jus.br/noticia-tre",
        noticias_gerais=[
            "https://g1.globo.com/noticia-1",
            "https://conjur.com.br/noticia-2",
        ],
    )
    payload = notion_client.build_properties_payload(schema, row)
    assert payload["noticia_TSE"]["url"] == "https://tse.jus.br/noticia-tse"
    assert payload["noticia_TRE"]["url"] == "https://tre-rs.jus.br/noticia-tre"
    assert payload["noticia_geral_1"]["url"] == "https://g1.globo.com/noticia-1"
    assert payload["noticia_geral_2"]["url"] == "https://conjur.com.br/noticia-2"


def test_enrich_preview_rows_with_news_uses_optional_second_stage():
    class FakeNewsEnricher:
        def enrich_rows(self, rows):
            enriched = []
            for row in rows:
                candidate = row.model_copy(deep=True)
                candidate.noticia_TSE = "https://tse.jus.br/noticia"
                candidate.noticia_TRE = "https://tre-rs.jus.br/noticia"
                candidate.noticias_gerais = ["https://g1.globo.com/noticia"]
                enriched.append(candidate)
            return enriched

    rows = build_preview_rows(make_analysis(), "https://youtu.be/abc123", make_schema(), FakeNotionClient())
    enriched = enrich_preview_rows_with_news(
        rows,
        api_key="fake-key",
        enricher=FakeNewsEnricher(),
    )
    assert enriched[0].noticia_TSE == "https://tse.jus.br/noticia"
    assert enriched[0].noticia_TRE == "https://tre-rs.jus.br/noticia"
    assert enriched[0].noticias_gerais == ["https://g1.globo.com/noticia"]


def test_gemini_news_enricher_repairs_broken_tse_slug(monkeypatch):
    broken_url = (
        "https://www.tse.jus.br/comunicacao/noticias/2025/Dezembro/"
        "mantida-multa-a-candidata-em-marechal-deodoro-al-por-repasse-irregular-de-verba"
    )
    fixed_url = (
        "https://www.tse.jus.br/comunicacao/noticias/2025/Dezembro/"
        "mantida-multa-a-candidata-em-marechal-deodoro-al-por-repasse-irregular-de-verba-do-fundo-eleitoral-1"
    )
    row = PublishPreviewRow(
        tema="Desaprovação de contas",
        punchline="Repasse irregular de verba do fundo eleitoral.",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Marechal Deodoro/AL",
        tribunal="TRE-AL",
        numero_processo="0600249-07.2024.6.02.0001",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Desprovido",
        votacao="Unânime",
        data_sessao="2025-12-18",
    )

    class ArtifactStore:
        def exists(self, *args, **kwargs):
            return False

        def write_json(self, *args, **kwargs):
            return None

    enricher = object.__new__(GeminiNewsEnricher)
    enricher.logger = logging.getLogger(__name__)
    enricher.artifact_store = ArtifactStore()

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        if response_model is NewsEnrichmentResult:
            return NewsEnrichmentResult(noticia_TSE=[broken_url]), []
        if response_model is InstitutionalRepairResult:
            return InstitutionalRepairResult(urls=[fixed_url]), []
        raise AssertionError(response_model)

    def fake_filter_accessible(urls):
        if broken_url in urls and fixed_url not in urls:
            return [], [broken_url]
        if fixed_url in urls:
            return [fixed_url], []
        return list(urls), []

    monkeypatch.setattr(enricher, "_call_grounded_json", fake_call_grounded_json)
    monkeypatch.setattr(core, "filter_accessible_news_urls", fake_filter_accessible)

    enriched = enricher.enrich_rows([row])
    assert enriched[0].noticia_TSE == fixed_url


def test_process_metadata_enricher_reuses_cached_artifact(tmp_path):
    artifact_store = RunArtifacts(tmp_path)
    cached_row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Brasília/DF",
        tribunal="TSE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2025-03-20",
    )
    artifact_store.write_json(
        "04a_process_metadata_01.json",
        {
            "context": "ctx",
            "parsed": {},
            "applied": cached_row.model_dump(mode="json"),
        },
    )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = artifact_store

    enriched = enricher.enrich_rows([PublishPreviewRow()])

    assert enriched[0].numero_processo == cached_row.numero_processo
    assert enriched[0].origem == cached_row.origem


def test_process_metadata_enricher_skips_grounding_when_only_origem_is_missing(tmp_path):
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="",
        tribunal="TSE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2025-03-20",
    )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = RunArtifacts(tmp_path)
    enricher.ground_origem_with_search = False

    def should_not_run(*args, **kwargs):
        raise AssertionError("grounding não deveria ser chamado quando só falta origem")

    enricher._call_grounded_json = should_not_run

    enriched = enricher.enrich_rows([row])

    assert enriched[0].numero_processo == row.numero_processo
    assert enriched[0].origem == ""
    assert not (tmp_path / "04a_process_metadata_01.json").exists()


def test_process_metadata_enricher_still_grounds_when_cnj_is_incomplete(tmp_path):
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="",
        tribunal="TSE",
        numero_processo="0600001-01",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2025-03-20",
    )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = RunArtifacts(tmp_path)
    enricher.ground_origem_with_search = False

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        assert "0600001-01" in prompt
        return core.ProcessMetadataResult(
            full_numero_processo="0600001-01.2024.6.00.0000",
            origem="Brasília/DF",
            is_judged_process=True,
        )

    enricher._call_grounded_json = fake_call_grounded_json

    enriched = enricher.enrich_rows([row])

    assert enriched[0].numero_processo == "0600001-01.2024.6.00.0000"
    assert enriched[0].origem == "Brasília/DF"


def test_news_enricher_reuses_cached_artifact(tmp_path):
    artifact_store = RunArtifacts(tmp_path)
    artifact_store.write_json(
        "06_news_enrichment_01.json",
        {
            "applied": {
                "noticia_TSE": "https://www.tse.jus.br/noticia",
                "noticia_TRE": "",
                "noticias_gerais": ["https://g1.globo.com/noticia"],
            }
        },
    )
    enricher = object.__new__(GeminiNewsEnricher)
    enricher.artifact_store = artifact_store

    enriched = enricher.enrich_rows([PublishPreviewRow(tema="Tema")])

    assert enriched[0].noticia_TSE == "https://www.tse.jus.br/noticia"
    assert enriched[0].noticias_gerais == ["https://g1.globo.com/noticia"]


def test_filter_general_news_urls_discards_irrelevant_candidates(monkeypatch):
    row = PublishPreviewRow(
        tema="Conduta vedada e uso de bens públicos em campanha eleitoral",
        punchline="Uso de bens públicos em campanha eleitoral inacessíveis aos demais candidatos.",
        classe_processo="AgRg-REspe",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Potiretama/CE",
        tribunal="TRE-CE",
        numero_processo="0600368-79.2024.6.06.0086",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=3081",
        relator="Min. André Mendonça",
        resultado="Desprovido",
        votacao="Unânime",
        data_sessao="2026-02-02",
        partes=[
            "Luan Dantas Félix (Agravante)",
            "Solange Mary Holanda Campelo Balbino",
        ],
    )

    class FakeResponse:
        def __init__(self, text: str):
            self.text = text
            self.headers = {"Content-Type": "text/html; charset=utf-8"}

        def raise_for_status(self):
            return None

    def fake_get(url, *args, **kwargs):
        if "blogdoedisonsilva" in url:
            return FakeResponse(
                "<html><body>TSE mantém decisão do TRE cearense condenando o prefeito de Potiretama."
                " Luan Dantas Félix e Solange Campelo foram multados por conduta vedada"
                " com uso de bens públicos em campanha eleitoral.</body></html>"
            )
        return FakeResponse(
            "<html><body>PGR diz que Ministério Público vai atuar para garantir que a vontade do eleitor"
            " seja traduzida nas urnas.</body></html>"
        )

    monkeypatch.setattr(core.requests, "get", fake_get)
    filtered = filter_general_news_urls(
        [
            "https://www.mpf.mp.br/pgr/noticias-pgr/pgr-diz-que-ministerio-publico-vai-atuar-para-garantir-que-vontade-do-eleitor-seja-traduzida-na-urnas",
            "https://blogdoedisonsilva.com.br/tse-mantem-decisao-do-tre-cearense-condenando-o-prefeito-de-potiretama/",
        ],
        row,
    )
    assert filtered == [
        "https://blogdoedisonsilva.com.br/tse-mantem-decisao-do-tre-cearense-condenando-o-prefeito-de-potiretama/"
    ]


def test_enrich_preview_rows_with_process_metadata_updates_full_number_and_blocks_precedent():
    class FakeMetadataEnricher:
        def enrich_rows(self, rows):
            first = rows[0].model_copy(deep=True)
            first.numero_processo = "0600368-79.2024.6.06.0086"
            first.origem = "Potiretama/CE"

            second = rows[1].model_copy(deep=True)
            second.add_error("Busca Google indicou que o número consultado aparece como precedente citado, não como processo julgado.")
            return [first, second]

    rows = [
        PublishPreviewRow(
            tema="Tema 1",
            classe_processo="AgRg-REspe",
            tipo_registro="Julgamento 1",
            eleicao="2024",
            origem="",
            tribunal="TRE-CE",
            numero_processo="0600368-79",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=120",
            relator="Min. André Mendonça",
            resultado="Desprovido",
            votacao="Unânime",
            data_sessao="2026-02-02",
        ),
        PublishPreviewRow(
            tema="Tema 2",
            classe_processo="AgRg-REspe",
            tipo_registro="Julgamento 2",
            eleicao="2024",
            origem="",
            tribunal="TRE-CE",
            numero_processo="0600448-31",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=3150",
            relator="Min. André Mendonça",
            resultado="Desprovido",
            votacao="Unânime",
            data_sessao="2026-02-02",
        ),
    ]
    enriched = enrich_preview_rows_with_process_metadata(
        rows,
        api_key="fake-key",
        enricher=FakeMetadataEnricher(),
        notion_schema=make_schema(),
    )
    assert enriched[0].numero_processo == "0600368-79.2024.6.06.0086"
    assert enriched[0].origem == "Potiretama/CE"
    assert enriched[1].blocked is True


def test_process_metadata_enricher_keeps_row_when_grounding_fails(monkeypatch, tmp_path):
    class DummyGenAI:
        class Client:
            def __init__(self, api_key: str):
                self.api_key = api_key

    class DummyTypes:
        pass

    monkeypatch.setattr(core, "require_google_genai", lambda: (DummyGenAI, DummyTypes))
    monkeypatch.setattr(
        GeminiProcessMetadataEnricher,
        "_call_grounded_json",
        lambda self, **kwargs: (_ for _ in ()).throw(RuntimeError("empty grounded response")),
    )

    enricher = GeminiProcessMetadataEnricher(
        api_key="fake-key",
        artifact_store=RunArtifacts(tmp_path),
        logger=logging.getLogger("test"),
        client=DummyGenAI.Client("fake-key"),
    )
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="AgRg-REspe",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="",
        tribunal="TRE-CE",
        numero_processo="0600368-79",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=120",
        relator="Min. André Mendonça",
        resultado="Desprovido",
        votacao="Unânime",
        data_sessao="2026-02-02",
    )

    enriched = enricher.enrich_rows([row])
    assert len(enriched) == 1
    assert enriched[0].numero_processo == "0600368-79"
    assert any("Metadados processuais não enriquecidos" in warning for warning in enriched[0].warnings)
    assert (tmp_path / "04a_process_metadata_01.json").exists()


def test_create_gemini_client_passes_http_timeout():
    captured = {}

    class DummyHttpOptions:
        def __init__(self, timeout: int | None = None):
            self.timeout = timeout

    class DummyClient:
        def __init__(self, api_key: str, http_options=None):
            captured["api_key"] = api_key
            captured["http_options"] = http_options

    class DummyGenAI:
        Client = DummyClient

    class DummyTypes:
        HttpOptions = DummyHttpOptions

    options = build_gemini_http_options(DummyTypes, timeout_seconds=123)
    assert options.timeout == 123

    create_gemini_client(DummyGenAI, DummyTypes, "fake-key", timeout_seconds=123)
    assert captured["api_key"] == "fake-key"
    assert captured["http_options"].timeout == 123


def test_extract_judgment_bundle_uses_refined_start_seconds():
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_refine_start")

    captured = {}

    def fake_refine_bundle_start_seconds(**kwargs):
        return 915

    def fake_call_gemini(**kwargs):
        captured["start_seconds"] = kwargs["start_seconds"]
        captured["end_seconds"] = kwargs["end_seconds"]
        return JudgmentBundleExtraction(items=[])

    extractor._refine_bundle_start_seconds = fake_refine_bundle_start_seconds
    extractor._call_gemini = fake_call_gemini

    session = SessionExtraction(
        data_sessao="20/03/2026",
        composicao=["Min. Cármen Lúcia"],
        judgments=[],
    )
    window = SessionWindow(
        title_hint="Julgamento 1",
        start_seconds=931,
        end_seconds=1000,
        mentioned_process_numbers=["0600249-07"],
    )
    bundle = extractor._extract_judgment_bundle(
        youtube_url="https://youtu.be/abc123",
        session=session,
        window=window,
        index=1,
    )

    assert captured["start_seconds"] == 915
    assert captured["end_seconds"] == 1000
    assert bundle.start_seconds == 915


def test_analyze_session_generates_placeholder_when_bundle_extraction_fails(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_bundle_placeholder")
    extractor.artifact_store = RunArtifacts(tmp_path)

    session = SessionExtraction(
        data_sessao="19/12/2025",
        composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
        judgments=[
            SessionWindow(
                title_hint="AgR no REspe 0600433-71",
                start_seconds=1420,
                end_seconds=1650,
                mentioned_process_numbers=["0600433-71"],
            )
        ],
    )
    extractor.artifact_store.write_json("01_session_windows.json", session.model_dump(mode="json"))

    def fail_extract(*args, **kwargs):
        raise RuntimeError("The read operation timed out")

    extractor._extract_judgment_bundle = fail_extract

    analysis = extractor.analyze_session("https://www.youtube.com/watch?v=abc123")
    assert len(analysis.bundles) == 1
    bundle = analysis.bundles[0]
    assert bundle.should_ignore is False
    assert bundle.items[0].numero_processo == "0600433-71"
    assert bundle.items[0].tema == "AgR no REspe 0600433-71"
    assert (tmp_path / "02_judgment_01.error.json").exists()
    assert (tmp_path / "02_judgment_01.json").exists()


def test_extract_session_windows_skips_failed_chunk_and_keeps_successful_ones(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_global_chunk_placeholder")
    extractor.artifact_store = RunArtifacts(tmp_path)

    calls = {"count": 0}

    def fake_call_gemini(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("The read operation timed out")
        return SessionExtraction(
            data_sessao="19/12/2025",
            composicao=["Min. Cármen Lúcia"],
            judgments=[
                SessionWindow(
                    title_hint="AgR no REspe 0600433-71",
                    start_seconds=1420,
                    end_seconds=1650,
                    mentioned_process_numbers=["0600433-71"],
                )
            ],
        )

    extractor._call_gemini = fake_call_gemini
    extractor._merge_session_chunks = GeminiSessionExtractor._merge_session_chunks.__get__(extractor, GeminiSessionExtractor)

    original_fetch = core.fetch_youtube_duration_seconds
    original_chunker = core.chunk_video_windows
    try:
        core.fetch_youtube_duration_seconds = lambda youtube_url: 600
        core.chunk_video_windows = lambda duration_seconds, window_seconds=None, overlap_seconds=None: [(0, 300), (270, 600)]
        merged = extractor._extract_session_windows("https://www.youtube.com/watch?v=abc123")
    finally:
        core.fetch_youtube_duration_seconds = original_fetch
        core.chunk_video_windows = original_chunker

    assert merged.data_sessao == "2025-12-19"
    assert len(merged.judgments) == 1
    assert (tmp_path / "raw_global_response_chunk_01.error.json").exists()
    assert (tmp_path / "raw_global_response_chunk_02.json").exists()


def test_extract_session_windows_fails_fast_after_consecutive_initial_errors(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_global_chunk_failfast")
    extractor.artifact_store = RunArtifacts(tmp_path)

    def always_fail(**kwargs):
        raise RuntimeError("The read operation timed out")

    extractor._call_gemini = always_fail

    original_fetch = core.fetch_youtube_duration_seconds
    original_chunker = core.chunk_video_windows
    original_threshold = core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS
    try:
        core.fetch_youtube_duration_seconds = lambda youtube_url: 900
        core.chunk_video_windows = (
            lambda duration_seconds, window_seconds=None, overlap_seconds=None: [(0, 300), (270, 600), (540, 840), (810, 1110)]
        )
        core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = 3
        try:
            extractor._extract_session_windows("https://www.youtube.com/watch?v=abc123")
        except RuntimeError as exc:
            assert "Abortando varredura global" in str(exc)
        else:
            raise AssertionError("Era esperado fail-fast da varredura global.")
    finally:
        core.fetch_youtube_duration_seconds = original_fetch
        core.chunk_video_windows = original_chunker
        core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = original_threshold

    assert (tmp_path / "raw_global_response_chunk_01.error.json").exists()
    assert (tmp_path / "raw_global_response_chunk_02.error.json").exists()
    assert (tmp_path / "raw_global_response_chunk_03.error.json").exists()
    assert not (tmp_path / "raw_global_response_chunk_04.error.json").exists()
    assert (tmp_path / "raw_global_fallback_response_chunk_01.error.json").exists()
    assert (tmp_path / "raw_global_fallback_response_chunk_02.error.json").exists()
    assert (tmp_path / "raw_global_fallback_response_chunk_03.error.json").exists()
    assert not (tmp_path / "raw_global_fallback_response_chunk_04.error.json").exists()


def test_extract_session_windows_uses_fallback_plan_after_primary_fail_fast(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_global_chunk_fallback_plan")
    extractor.artifact_store = RunArtifacts(tmp_path)
    extractor._merge_session_chunks = GeminiSessionExtractor._merge_session_chunks.__get__(extractor, GeminiSessionExtractor)

    calls = {"primary": 0, "fallback": 0}

    def fake_call_gemini(**kwargs):
        start_seconds = kwargs["start_seconds"]
        if start_seconds < 600:
            calls["primary"] += 1
            raise RuntimeError("The read operation timed out")
        calls["fallback"] += 1
        return SessionExtraction(
            data_sessao="19/12/2025",
            composicao=["Min. Cármen Lúcia"],
            judgments=[
                SessionWindow(
                    title_hint="AgR no REspe 0600433-71",
                    start_seconds=1420,
                    end_seconds=1650,
                    mentioned_process_numbers=["0600433-71"],
                )
            ],
        )

    extractor._call_gemini = fake_call_gemini

    original_fetch = core.fetch_youtube_duration_seconds
    original_chunker = core.chunk_video_windows
    original_threshold = core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS
    original_primary_window = core.GLOBAL_SCAN_WINDOW_SECONDS
    original_fallback_window = core.GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS
    try:
        core.fetch_youtube_duration_seconds = lambda youtube_url: 900
        def fake_chunker(duration_seconds, window_seconds=None, overlap_seconds=None):
            if window_seconds == core.GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS:
                return [(600, 720), (705, 825)]
            return [(0, 300), (270, 570), (540, 840), (810, 1110)]
        core.chunk_video_windows = fake_chunker
        core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = 3
        core.GLOBAL_SCAN_WINDOW_SECONDS = 300
        core.GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS = 120
        merged = extractor._extract_session_windows("https://www.youtube.com/watch?v=abc123")
    finally:
        core.fetch_youtube_duration_seconds = original_fetch
        core.chunk_video_windows = original_chunker
        core.GLOBAL_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = original_threshold
        core.GLOBAL_SCAN_WINDOW_SECONDS = original_primary_window
        core.GLOBAL_SCAN_FALLBACK_WINDOW_SECONDS = original_fallback_window

    assert merged.data_sessao == "2025-12-19"
    assert len(merged.judgments) == 1
    assert calls["primary"] == 3
    assert calls["fallback"] >= 1
    assert (tmp_path / "raw_global_response_chunk_01.error.json").exists()
    assert (tmp_path / "raw_global_fallback_response_chunk_01.json").exists()


def test_extract_session_windows_uses_transcript_after_video_plans_fail(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_global_transcript_fallback")
    extractor.artifact_store = RunArtifacts(tmp_path)

    def always_fail(*args, **kwargs):
        raise RuntimeError("The read operation timed out")

    extractor._extract_session_windows_for_plan = always_fail
    extractor._extract_session_windows_from_transcript = lambda youtube_url: [
        SessionExtraction(
            data_sessao="19/12/2025",
            composicao=["Min. Cármen Lúcia"],
            judgments=[
                SessionWindow(
                    title_hint="AgR no REspe 0600433-71",
                    start_seconds=1420,
                    end_seconds=1650,
                    mentioned_process_numbers=["0600433-71"],
                )
            ],
        )
    ]
    extractor._merge_session_chunks = GeminiSessionExtractor._merge_session_chunks.__get__(extractor, GeminiSessionExtractor)

    original_fetch = core.fetch_youtube_duration_seconds
    original_chunker = core.chunk_video_windows
    try:
        core.fetch_youtube_duration_seconds = lambda youtube_url: 900
        core.chunk_video_windows = (
            lambda duration_seconds, window_seconds=None, overlap_seconds=None: [(0, 300), (270, 570), (540, 840)]
        )
        merged = extractor._extract_session_windows("https://www.youtube.com/watch?v=abc123")
    finally:
        core.fetch_youtube_duration_seconds = original_fetch
        core.chunk_video_windows = original_chunker

    assert merged.data_sessao == "2025-12-19"
    assert len(merged.judgments) == 1


def test_extract_session_windows_from_transcript_builds_chunks(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_transcript_chunk_scan")
    extractor.artifact_store = RunArtifacts(tmp_path)
    extractor._transcript_snippets_cache = None
    extractor._get_transcript_snippets = lambda youtube_url: [
        TranscriptSnippet(text="Abertura da sessão.", start_seconds=0, end_seconds=5),
        TranscriptSnippet(text="Chamo para julgamento o AgR no REspe 0600433-71.", start_seconds=120, end_seconds=128),
        TranscriptSnippet(text="O relator profere voto.", start_seconds=129, end_seconds=145),
    ]

    captured = []

    def fake_call_gemini_text(**kwargs):
        captured.append(kwargs["prompt"])
        return SessionExtraction(
            data_sessao="19/12/2025",
            composicao=["Min. Cármen Lúcia"],
            judgments=[
                SessionWindow(
                    title_hint="AgR no REspe 0600433-71",
                    start_seconds=120,
                    end_seconds=145,
                    mentioned_process_numbers=["0600433-71"],
                )
            ],
        )

    extractor._call_gemini_text = fake_call_gemini_text

    chunks = extractor._extract_session_windows_from_transcript("https://www.youtube.com/watch?v=abc123")

    assert len(chunks) == 1
    assert "Transcrição com timestamps absolutos" in captured[0]
    assert (tmp_path / "raw_transcript_chunk_01.txt").exists()
    assert (tmp_path / "raw_transcript_response_chunk_01.json").exists()


def test_extract_session_windows_from_transcript_fails_fast_after_initial_errors(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_transcript_failfast")
    extractor.artifact_store = RunArtifacts(tmp_path)
    extractor._get_transcript_snippets = lambda youtube_url: [
        TranscriptSnippet(text="Trecho 1", start_seconds=0, end_seconds=5),
        TranscriptSnippet(text="Trecho 2", start_seconds=10, end_seconds=15),
        TranscriptSnippet(text="Trecho 3", start_seconds=20, end_seconds=25),
    ]

    def fail_text_call(**kwargs):
        raise RuntimeError("The read operation timed out")

    extractor._call_gemini_text = fail_text_call

    original_threshold = core.TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS
    original_chunk_builder = core.build_transcript_chunks
    try:
        core.TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = 2
        core.build_transcript_chunks = lambda snippets, **kwargs: [
            TranscriptChunk(start_seconds=0, end_seconds=5, text="[0s-5s] Trecho 1", snippet_count=1),
            TranscriptChunk(start_seconds=10, end_seconds=15, text="[10s-15s] Trecho 2", snippet_count=1),
            TranscriptChunk(start_seconds=20, end_seconds=25, text="[20s-25s] Trecho 3", snippet_count=1),
        ]
        try:
            extractor._extract_session_windows_from_transcript("https://www.youtube.com/watch?v=abc123")
        except RuntimeError as exc:
            assert "Abortando varredura por transcrição" in str(exc)
        else:
            raise AssertionError("Era esperado fail-fast da transcrição.")
    finally:
        core.TRANSCRIPT_SCAN_FAIL_FAST_CONSECUTIVE_ERRORS = original_threshold
        core.build_transcript_chunks = original_chunk_builder

    assert (tmp_path / "raw_transcript_response_chunk_01.error.json").exists()
    assert (tmp_path / "raw_transcript_response_chunk_02.error.json").exists()


def test_extract_judgment_bundle_falls_back_to_transcript_on_video_failure(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_detail_transcript_fallback")
    extractor.artifact_store = RunArtifacts(tmp_path)

    def fake_refine_bundle_start_seconds(**kwargs):
        return 915

    def fail_video_call(**kwargs):
        raise RuntimeError("The read operation timed out")

    def fake_text_call(**kwargs):
        return JudgmentBundleExtraction(
            items=[
                JudgmentItemExtraction(
                    numero_processo="0600433-71.2024.6.00.0000",
                    tema="Tema pela transcrição",
                )
            ]
        )

    extractor._refine_bundle_start_seconds = fake_refine_bundle_start_seconds
    extractor._call_gemini = fail_video_call
    extractor._call_gemini_text = fake_text_call
    extractor._build_transcript_detail_chunk = lambda youtube_url, start_seconds, end_seconds: TranscriptChunk(
        start_seconds=900,
        end_seconds=1000,
        text="[915s-930s] Chamo para julgamento o processo 0600433-71.\n[931s-980s] O relator vota.",
        snippet_count=2,
    )

    session = SessionExtraction(
        data_sessao="20/03/2026",
        composicao=["Min. Cármen Lúcia"],
        judgments=[],
    )
    window = SessionWindow(
        title_hint="Julgamento 1",
        start_seconds=931,
        end_seconds=1000,
        mentioned_process_numbers=["0600433-71"],
    )
    bundle = extractor._extract_judgment_bundle(
        youtube_url="https://youtu.be/abc123",
        session=session,
        window=window,
        index=1,
    )

    assert bundle.start_seconds == 915
    assert bundle.items[0].numero_processo == "0600433-71.2024.6.00.0000"
    assert (tmp_path / "raw_detail_transcript_01.input.txt").exists()


def test_coerce_session_extraction_parses_string_composicao():
    payload = {
        "data_da_sessao": "18/12/2025",
        "composicao_da_sessao": "Min. Cármen Lúcia, Min. André Mendonça",
        "julgamentos": [],
    }

    result = core._coerce_gemini_response_model(SessionExtraction, json.dumps(payload, ensure_ascii=False))

    assert result.composicao == ["Min. Cármen Lúcia", "Min. André Mendonça"]


def test_coerce_session_extraction_accepts_list_of_chunk_payloads():
    payload = [
        {
            "data_da_sessao": None,
            "composicao_colegiado": None,
            "julgamentos": [],
            "should_ignore": True,
            "motivo": "trecho institucional",
        },
        {
            "data_da_sessao": "18/12/2025",
            "composicao_da_sessao": "Min. Cármen Lúcia, Min. André Mendonça",
            "julgamentos": [
                {
                    "titulo": "AgR no REspe 0600433-71",
                    "timestamp_inicial": 120,
                    "timestamp_final": 240,
                    "processo": "0600433-71",
                }
            ],
        },
    ]

    result = core._coerce_gemini_response_model(SessionExtraction, json.dumps(payload, ensure_ascii=False))

    assert result.data_sessao == "18/12/2025"
    assert result.composicao == ["Min. Cármen Lúcia", "Min. André Mendonça"]
    assert len(result.judgments) == 1
    assert result.judgments[0].mentioned_process_numbers == ["0600433-71"]


def test_coerce_judgment_bundle_parses_string_multi_value_fields():
    payload = {
        "items": [
            {
                "numero_processo": "0600433-71.2024.6.00.0000",
                "partes": "",
                "advogados": "Adv. A; Adv. B",
                "composicao": "Min. Cármen Lúcia, Min. André Mendonça",
                "tema": "Tema teste",
            }
        ]
    }

    result = core._coerce_gemini_response_model(
        JudgmentBundleExtraction,
        json.dumps(payload, ensure_ascii=False),
    )

    assert result.items[0].partes == []
    assert result.items[0].advogados == ["Adv. A", "Adv. B"]
    assert result.items[0].composicao == ["Min. Cármen Lúcia", "Min. André Mendonça"]


def test_coerce_process_metadata_result_accepts_list_payload():
    payload = [
        {
            "processo": "0600180-72.2024.6.00.0000",
            "origem": "Cidade/UF",
            "julgado_na_sessao": True,
        }
    ]

    result = core._coerce_gemini_response_model(
        core.ProcessMetadataResult,
        json.dumps(payload, ensure_ascii=False),
    )

    assert result.full_numero_processo == "0600180-72.2024.6.00.0000"
    assert result.origem == "Cidade/UF"
    assert result.is_judged_process is True


def test_refinement_anchor_uses_previous_administrative_window():
    session = SessionExtraction(
        data_sessao="20/03/2026",
        composicao=[],
        judgments=[
            SessionWindow(
                title_hint="Abertura da sessão e leitura da ata",
                start_seconds=2970,
                end_seconds=3120,
                should_ignore=True,
                ignore_reason="Procedimentos administrativos de abertura de sessão",
            ),
            SessionWindow(
                title_hint="AgR no AREspe 060036879 / POTIRETAMA - CE",
                start_seconds=3120,
                end_seconds=3270,
                mentioned_process_numbers=["060036879"],
            ),
        ],
    )

    anchor = GeminiSessionExtractor._refinement_anchor_start_seconds(session, session.judgments[1])
    assert anchor == 2970


def test_should_refine_bundle_start_skips_precise_numbered_window_without_admin():
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.enable_start_refinement = True
    extractor.conditional_start_refinement = True

    session = SessionExtraction(
        data_sessao="20/03/2026",
        composicao=[],
        judgments=[],
    )
    window = SessionWindow(
        title_hint="AgR no AREspe 060036879 / POTIRETAMA - CE",
        start_seconds=3120,
        end_seconds=3270,
        mentioned_process_numbers=["060036879"],
    )

    assert extractor._should_refine_bundle_start(session=session, window=window, previous_admin_window=None) is False


def test_should_refine_bundle_start_keeps_transition_after_admin():
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.enable_start_refinement = True
    extractor.conditional_start_refinement = True

    previous_window = SessionWindow(
        title_hint="Abertura da sessão",
        start_seconds=2970,
        end_seconds=3120,
        should_ignore=True,
        ignore_reason="Procedimentos administrativos",
    )
    session = SessionExtraction(
        data_sessao="20/03/2026",
        composicao=[],
        judgments=[previous_window],
    )
    window = SessionWindow(
        title_hint="AgR no AREspe 060036879 / POTIRETAMA - CE",
        start_seconds=3120,
        end_seconds=3270,
        mentioned_process_numbers=["060036879"],
    )

    assert extractor._should_refine_bundle_start(
        session=session,
        window=window,
        previous_admin_window=previous_window,
    ) is True


def test_merge_session_chunks_ignores_ceremonial_windows_and_coalesces_duplicate_processes():
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)

    merged = extractor._merge_session_chunks(
        [
            SessionExtraction(
                data_sessao="null",
                composicao=["Ministra Cármen Lúcia"],
                judgments=[
                    SessionWindow(
                        title_hint="Sessão de Abertura do Ano Judiciário Eleitoral",
                        start_seconds=1086,
                        end_seconds=1110,
                        should_ignore=False,
                    ),
                    SessionWindow(
                        title_hint="AgR no AREspe 060036879",
                        start_seconds=3082,
                        end_seconds=3270,
                        mentioned_process_numbers=["060036879"],
                    ),
                    SessionWindow(
                        title_hint="AgR no AREspe 0600368-79.2024.6.06.0000",
                        start_seconds=3240,
                        end_seconds=3467,
                        mentioned_process_numbers=["060036879"],
                    ),
                    SessionWindow(
                        title_hint="AgR no AREspe 0600448-31.2024.6.06.0000",
                        start_seconds=3467,
                        end_seconds=3540,
                        mentioned_process_numbers=["060044831"],
                    ),
                    SessionWindow(
                        title_hint="AgR no AREspe - 060036879 / POTIRETAMA - CE",
                        start_seconds=3510,
                        end_seconds=3640,
                        mentioned_process_numbers=["060036879"],
                    ),
                ],
            ),
            SessionExtraction(
                data_sessao="02 de fevereiro de 2026",
                composicao=["Ministro André Mendonça"],
                judgments=[],
            ),
        ]
    )

    assert merged.data_sessao == "2026-02-02"
    assert merged.composicao == ["Ministra Cármen Lúcia", "Ministro André Mendonça"]
    assert merged.judgments[0].should_ignore is True
    assert merged.judgments[1].mentioned_process_numbers == ["0600368-79"]
    assert merged.judgments[1].start_seconds == 3082
    assert merged.judgments[1].end_seconds == 3467
    assert merged.judgments[2].mentioned_process_numbers == ["0600448-31"]
    assert len([item for item in merged.judgments if item.mentioned_process_numbers]) == 2


def test_call_gemini_disables_unavailable_model_after_first_fallback(tmp_path):
    extractor = GeminiSessionExtractor.__new__(GeminiSessionExtractor)
    extractor.logger = logging.getLogger("test_gemini_fallback")
    extractor.artifact_store = RunArtifacts(tmp_path)
    extractor.api_key = "fake-key"
    extractor.model = "gemini-3.1-pro-preview"
    extractor.model_candidates = ["gemini-3.1-pro-preview", "gemini-2.5-flash"]
    extractor.disabled_models = set()

    calls = []

    original_rest_call = core.call_gemini_generate_content_rest

    def fake_rest_call(**kwargs):
        calls.append(kwargs["model_name"])
        if kwargs["model_name"] == "gemini-3.1-pro-preview":
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, limit: 0, model: gemini-3.1-pro")
        parsed = SessionExtraction(data_sessao="02/02/2026", composicao=[], judgments=[])
        return parsed, parsed.model_dump_json(), {}

    core.call_gemini_generate_content_rest = fake_rest_call

    try:
        first = extractor._call_gemini(
            youtube_url="https://youtu.be/abc123",
            prompt="Teste",
            response_model=SessionExtraction,
            system_prompt="Sistema",
            artifact_name="first.txt",
        )
        second = extractor._call_gemini(
            youtube_url="https://youtu.be/abc123",
            prompt="Teste 2",
            response_model=SessionExtraction,
            system_prompt="Sistema",
            artifact_name="second.txt",
        )
    finally:
        core.call_gemini_generate_content_rest = original_rest_call

    assert first.data_sessao == "02/02/2026"
    assert second.data_sessao == "02/02/2026"
    assert extractor.disabled_models == {"gemini-3.1-pro-preview"}
    assert calls == [
        "gemini-3.1-pro-preview",
        "gemini-2.5-flash",
        "gemini-2.5-flash",
    ]


def test_should_disable_model_only_for_zero_quota_or_unsupported():
    zero_quota = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, limit: 0, model: gemini-3.1-pro")
    transient_quota = RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded, limit: 250000, model: gemini-2.5-flash")
    unsupported = RuntimeError("Model is not supported for generateContent")

    assert should_disable_model(zero_quota) is True
    assert should_disable_model(transient_quota) is False
    assert should_disable_model(unsupported) is True


def test_extract_retry_delay_seconds_parses_response_text():
    exc = RuntimeError("Please retry in 46.167278487s. {'retryDelay': '46s'}")
    assert extract_retry_delay_seconds(exc) == 46.167278487
