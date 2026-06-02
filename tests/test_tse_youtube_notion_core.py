import json
import logging

import tse_youtube_notion_core as core
from tse_youtube_notion_core import (
    AnalysisResult,
    enrich_preview_rows_with_process_metadata,
    GeminiSessionExtractor,
    GeminiNewsEnricher,
    GeminiProcessMetadataEnricher,
    GeminiThemePunchlineEnricher,
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
    ThemePunchlineRepairBatchResult,
    build_preview_rows,
    compute_suspenso_star_updates,
    build_fallback_tema,
    build_editorial_punchline_fallback,
    build_fundamentacao_column_text,
    build_gemini_http_options,
    build_raciocinio_column_text,
    enrich_preview_rows_with_news,
    extract_retry_delay_seconds,
    filter_general_news_urls,
    infer_classe_from_row_text,
    infer_full_numero_processo_from_row_text,
    infer_resultado_from_row_text,
    infer_origin_from_row_text,
    infer_punchline_from_row_text,
    infer_relator_from_row_text,
    infer_votacao_from_row_text,
    normalize_party_list,
    punchline_looks_generic,
    theme_punchline_pair_needs_rewrite,
    theme_punchline_pair_too_similar,
    publish_preview_rows,
    should_replace_classe_processo,
    should_disable_model,
    tema_looks_generic,
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
                "select": {
                    "options": [
                        {"name": "Aprovada"},
                        {"name": "Desprovido"},
                        {"name": "Suspenso por vista"},
                        {"name": "Suspenso mas julgado depois"},
                    ]
                },
            },
            "votacao": {
                "type": "select",
                "select": {"options": [{"name": "Unânime"}, {"name": "Por maioria"}, {"name": "Suspenso"}, {"name": "Suspenso*"}]},
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
                "Ministro Antônio Carlos Ferreira",
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
        "Min. Antônio Carlos Ferreira",
        "Min. Ramos Tavares",
    ]


def test_build_preview_rows_extracts_relator_and_pedido_vista_from_composicao_markers():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="03/02/2022",
            composicao=["Min. Alexandre de Moraes", "Min. Sérgio Banhos"],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=4860,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="08/09/2020",
                        eleicao="2020",
                        classe_processo="AgRg-REspe",
                        numero_processo="0000697-22.2016.6.13.0000",
                        origem="Belo Horizonte/MG",
                        tre="TRE-MG",
                        partes=["Alice"],
                        advogados=[],
                        composicao=[
                            "Ministro Sérgio Banhos (Relator)",
                            "Ministro Alexandre de Moraes (Voto-vista)",
                        ],
                        relator="",
                        pedido_vista="",
                        tema="Tema útil",
                        punchline="Resumo forte",
                        analise_do_conteudo_juridico="Análise",
                        fundamentacao_normativa="Fundamentação",
                        precedentes_citados="",
                        raciocinio_juridico="",
                        pontos_processuais_relevantes="",
                        efeitos_e_providencias_praticas="",
                        resolucoes_citadas="",
                        votacao="Por maioria",
                        resultado_final="Desprovido",
                    )
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/NALJtQaMUSs", schema, notion)

    assert len(rows) == 1
    assert rows[0].relator == "Min. Sérgio Banhos"
    assert rows[0].pedido_vista == "Min. Alexandre de Moraes"
    assert rows[0].data_sessao == "2022-02-03"


def test_build_preview_rows_ignores_invalid_session_composition_with_eight_names():
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
                "Ministro Excedente",
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
    assert rows[0].composicao == ["Min. Cármen Lúcia", "Min. André Mendonça"]


def test_choose_preferred_composition_rejects_regimentally_impossible_seven_names():
    contaminated = [
        "Min. Rosa Weber",
        "Min. Luís Roberto Barroso",
        "Min. Jorge Mussi",
        "Min. Og Fernandes",
        "Min. Tarcísio Vieira de Carvalho Neto",
        "Min. Sérgio Banhos",
        "Min. Admar Gonzaga",
    ]
    regimental = [
        "Min. Rosa Weber",
        "Min. Luís Roberto Barroso",
        "Min. Edson Fachin",
        "Min. Jorge Mussi",
        "Min. Og Fernandes",
        "Min. Admar Gonzaga",
        "Min. Tarcísio Vieira de Carvalho Neto",
    ]

    assert core.choose_preferred_composition(contaminated, regimental) == regimental


def test_build_preview_rows_keeps_seven_names_with_one_unrostered_minister():
    # Regressao guard (Fix #1): uma composicao de sessao com 7 nomes, sendo 1
    # ministro fora do roster (ex.: recem-empossado / grafia sem alias), NAO pode
    # zerar a coluna composicao. Antes do fix, o gate regimental ("unknown_institution")
    # descartava todo o fallback e a coluna saia VAZIA mesmo com os 7 nomes corretos.
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=[
                "Ministra Cármen Lúcia",
                "Ministro André Mendonça",
                "Ministro Nunes Marques",
                "Ministra Isabel Gallotti",
                "Ministro Antônio Carlos Ferreira",
                "Ministro Floriano de Azevedo Marques",
                "Ministro Joaquim Pereira Lima",
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
                        composicao=[],
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
    assert len(rows[0].composicao) == 7
    assert "Min. Joaquim Pereira Lima" in rows[0].composicao


def test_build_preview_rows_warns_when_published_composition_is_regimentally_off():
    # Fix #1: composicoes de tamanho plausivel mas com divergencia regimental real
    # (ex.: 2 nomes nao classificados) continuam sendo publicadas (nao mais vazias),
    # porem com um aviso explicito na linha para revisao humana.
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=[
                "Ministra Cármen Lúcia",
                "Ministro André Mendonça",
                "Ministro Nunes Marques",
                "Ministra Isabel Gallotti",
                "Ministro Antônio Carlos Ferreira",
                "Ministro Joaquim Pereira Lima",
                "Ministro Tadeu Soares Quintino",
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
                        composicao=[],
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
    assert len(rows[0].composicao) == 7
    assert any("divergencia regimental" in warning for warning in rows[0].warnings)


def test_infer_origin_keeps_compound_municipality_name():
    # A origem composta "Santo Antônio do Tauá/PA" não pode ser truncada para
    # "Antônio do Tauá/PA" (perdendo "Santo") pela regex de topônimos.
    row = PublishPreviewRow(
        numero_processo="0601309-60",
        origem="Tribunal Regional Eleitoral do Pará",
        analise_do_conteudo_juridico=(
            "Agravo contra decisão do TRE/PA que manteve a desaprovação das contas de "
            "candidato a vereador em Santo Antônio do Tauá/PA nas eleições de 2024."
        ),
    )
    assert core.infer_origin_from_row_text(row) == "Santo Antônio do Tauá/PA"


def test_validate_preview_row_prefers_municipality_over_tre_capital():
    # Quando a origem veio como referência ao tribunal (normalizada para a capital
    # "Belém/PA"), deve ceder ao município citado no texto do caso.
    row = PublishPreviewRow(
        numero_processo="0601309-60",
        origem="Tribunal Regional Eleitoral do Pará",
        tema="Desaprovação de contas de campanha",
        analise_do_conteudo_juridico=(
            "Agravo em recurso especial contra decisão do TRE/PA sobre contas de "
            "candidato a vereador em Santo Antônio do Tauá/PA nas eleições de 2024."
        ),
    )
    out = validate_preview_row(row, None)
    assert out.origem == "Santo Antônio do Tauá/PA"
    assert out.tribunal == "TRE-PA"


def test_validate_preview_row_keeps_specific_municipality_from_model():
    # Controle: se o modelo já trouxe um município específico, não sobrescreve com
    # outro nome que apareça no texto.
    row = PublishPreviewRow(
        numero_processo="0601",
        origem="Belém/PA",
        tema="Tema",
        analise_do_conteudo_juridico="Processo de Belém/PA.",
    )
    assert validate_preview_row(row, None).origem == "Belém/PA"


def test_compute_suspenso_star_updates_flips_prior_suspension_when_later_definitive():
    records = [
        {"page_id": "p1", "numero_processo": "0600100-10", "votacao": "Suspenso", "data_sessao": "2025-03-01"},
        {"page_id": "p2", "numero_processo": "0600100-10", "votacao": "Unânime", "data_sessao": "2025-05-01"},
    ]
    updates = compute_suspenso_star_updates(records)
    assert [u["page_id"] for u in updates] == ["p1"]


def test_compute_suspenso_star_updates_ignores_process_without_definitive_vote():
    records = [
        {"page_id": "p1", "numero_processo": "0600100-10", "votacao": "Suspenso", "data_sessao": "2025-03-01"},
        {"page_id": "p2", "numero_processo": "0600100-10", "votacao": "Suspenso", "data_sessao": "2025-05-01"},
    ]
    assert compute_suspenso_star_updates(records) == []


def test_compute_suspenso_star_updates_does_not_flip_new_suspension_after_definitive():
    # Suspensão POSTERIOR ao julgamento definitivo é uma nova suspensão: não rebaixa.
    records = [
        {"page_id": "p1", "numero_processo": "0600100-10", "votacao": "Unânime", "data_sessao": "2025-03-01"},
        {"page_id": "p2", "numero_processo": "0600100-10", "votacao": "Suspenso", "data_sessao": "2025-09-01"},
    ]
    assert compute_suspenso_star_updates(records) == []


def test_parse_youtube_chapter_timestamps_extracts_numero_and_seconds():
    desc = "\n".join(
        [
            "Sessão Plenária - 12 de Fevereiro 2026",
            "00:00:00 Início da transmissão",
            "00:26:22 Abertura da sessão",
            "00:47:42 RO - 060290922",
            "01:20:17 AgR no AREspe - 060006171",
            "01:32:17 Julgamento em lista",
            "01:33:54 Rp - 060018305 / Rp - 06009479",
        ]
    )
    result = core.parse_youtube_chapter_timestamps(desc)
    assert result["0602909-22"] == 47 * 60 + 42
    assert result["0600061-71"] == 1 * 3600 + 20 * 60 + 17
    # linha com dois processos: ambos recebem o mesmo timestamp
    assert result["0600183-05"] == 1 * 3600 + 33 * 60 + 54
    # linhas administrativas/de lista não viram processo
    assert all("lista" not in key for key in result)


def test_enrich_preview_rows_with_youtube_chapters(monkeypatch):
    description = "\n".join(
        [
            "00:47:42 RO - 060290922",
            "01:20:17 AgR no AREspe - 060006171",
        ]
    )
    monkeypatch.setattr(core, "fetch_youtube_description", lambda *args, **kwargs: description)
    rows = [
        PublishPreviewRow(numero_processo="0602909-22", classe_processo="", youtube_link="https://www.youtube.com/watch?v=VID"),
        PublishPreviewRow(numero_processo="0600061-71", classe_processo="PA", youtube_link="https://www.youtube.com/watch?v=VID&t=10"),
    ]
    out = core.enrich_preview_rows_with_youtube_chapters(rows, youtube_url="https://www.youtube.com/watch?v=VID")
    # classe vazia preenchida e link sem timestamp recebe &t=2862 (47:42)
    assert out[0].classe_processo == "RO"
    assert out[0].youtube_link.endswith("&t=2862")
    # 'PA' corrigido para a classe do capítulo; link que já tinha timestamp é preservado
    assert out[1].classe_processo == "AgRg-AREspe"
    assert out[1].youtube_link.endswith("&t=10")


def test_classe_is_specificity_downgrade_guards_internal_recourse():
    assert core.classe_is_specificity_downgrade("ED-AgRg-AREspe", "AgRg-AREspe") is True
    assert core.classe_is_specificity_downgrade("AgRg-REspe", "REspe") is True
    assert core.classe_is_specificity_downgrade("REspe", "AgRg-REspe") is False
    assert core.classe_is_specificity_downgrade("AIJE", "RO") is False


def test_compute_suspenso_star_updates_separates_distinct_processes():
    records = [
        {"page_id": "a1", "numero_processo": "0600100-10", "votacao": "Suspenso", "data_sessao": "2025-03-01"},
        {"page_id": "a2", "numero_processo": "0600100-10", "votacao": "Por maioria", "data_sessao": "2025-05-01"},
        {"page_id": "b1", "numero_processo": "0600200-20", "votacao": "Suspenso", "data_sessao": "2025-03-01"},
    ]
    assert [u["page_id"] for u in compute_suspenso_star_updates(records)] == ["a1"]


def test_build_preview_rows_prefers_session_date_over_item_date():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="20/03/2026",
            composicao=["Ministra Cármen Lúcia"],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=120,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="16/05/2023",
                        eleicao="2024",
                        classe_processo="PA",
                        numero_processo="0600001-01.2024.6.00.0000",
                        origem="Brasília/DF",
                        tre="TSE",
                        relator="Ministro André Mendonça",
                        tema="Tema útil",
                        resultado_final="Aprovada",
                        votacao="Unânime",
                    )
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/abc123", schema, notion)
    assert len(rows) == 1
    assert rows[0].data_sessao == "2026-03-20"


def test_build_preview_rows_drops_timestamp_when_item_date_conflicts_with_authoritative_session_date():
    schema = make_schema()
    notion = FakeNotionClient()
    analysis = AnalysisResult(
        session=SessionExtraction(
            data_sessao="11/02/2021",
            composicao=["Ministro Luís Roberto Barroso"],
            judgments=[],
        ),
        bundles=[
            JudgmentBundleExtraction(
                start_seconds=114,
                items=[
                    JudgmentItemExtraction(
                        data_sessao="23/06/2021",
                        numero_processo="0600378-65.2020.6.00.0000",
                        tema="Tema útil",
                        relator="Ministro Luís Roberto Barroso",
                        resultado_final="Aprovado o voto do relator.",
                        votacao="Unânime.",
                    )
                ],
            ),
        ],
    )

    rows = build_preview_rows(analysis, "https://youtu.be/s9Ts40TfDas", schema, notion)

    assert len(rows) == 1
    assert rows[0].data_sessao == "2021-02-11"
    assert rows[0].youtube_link == "https://www.youtube.com/watch?v=s9Ts40TfDas"


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


def test_build_fallback_tema_discards_decision_sentence_and_infers_legal_issue():
    row = PublishPreviewRow(
        tema="O Tribunal, por unanimidade, negou provimento ao agravo regimental",
        punchline="O Tribunal, por unanimidade, negou provimento ao agravo regimental.",
        classe_processo="AgRg-REspe",
        numero_processo="0600136-55",
        resultado="Desprovido",
        votacao="Unânime",
        analise_do_conteudo_juridico=(
            "O cerne da discussão jurídica reside na contagem do prazo de inelegibilidade "
            "decorrente de condenação criminal quando há parcelamento da multa."
        ),
        raciocinio_juridico=(
            "O relator reafirmou que o parcelamento da multa não posterga o prazo de inelegibilidade."
        ),
        fundamentacao_normativa="Súmula 61 do TSE.",
    )
    assert build_fallback_tema(row) == "Prazo de inelegibilidade e parcelamento da pena de multa"


def test_build_fallback_tema_infers_cota_feminina_fefc_theme_from_context():
    row = PublishPreviewRow(
        tema="O Tribunal negou provimento ao agravo regimental, mantendo a decisão que desaprovou as contas de campanha da candidata",
        punchline="O Tribunal negou provimento ao agravo regimental.",
        classe_processo="AgRg-REspe",
        numero_processo="0600433-71",
        resultado="Desprovido",
        votacao="Unânime",
        analise_do_conteudo_juridico=(
            "A irregularidade apontada foi o repasse de verba do Fundo Especial de Financiamento "
            "de Campanha, destinado à cota feminina, para candidato do gênero masculino, sem a "
            "demonstração de benefício comum à campanha da candidata."
        ),
        raciocinio_juridico="Aplicou-se a Súmula 30 do TSE sobre o desvirtuamento da cota feminina.",
    )
    assert build_fallback_tema(row) == "Desvio de recursos da cota feminina do FEFC para candidatura masculina"


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


def test_build_fallback_tema_infers_consulta_conduta_vedada_theme():
    row = PublishPreviewRow(
        tema=(
            "Além disso, destacou que a análise sobre a configuração de conduta vedada exige "
            "a verificação de fatos e provas, o que é incompatível com o rito da consulta eleitoral"
        ),
        classe_processo="CTA",
        numero_processo="0600090-81",
        analise_do_conteudo_juridico=(
            "Consulta formulada questionando a possibilidade de exame de conduta vedada em tese. "
            "O relator votou pelo não conhecimento da consulta por ausência de abstração e por "
            "tratar de casos concretos, além de não caber ao TSE analisar a configuração de "
            "condutas vedadas por meio de consulta."
        ),
        raciocinio_juridico=(
            "A análise sobre a configuração de conduta vedada exige a verificação de fatos e provas, "
            "o que é incompatível com o rito da consulta eleitoral."
        ),
    )
    assert build_fallback_tema(row) == "Cabimento de consulta eleitoral para análise abstrata de conduta vedada"


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


def test_build_fallback_tema_infers_specific_party_accounts_theme():
    row = PublishPreviewRow(
        tema="Prestação de contas partidárias",
        classe_processo="PC",
        numero_processo="0600231-08.2019.6.00.0000",
        analise_do_conteudo_juridico=(
            "O processo trata da prestação de contas do exercício de 2015, relatada pelo Ministro "
            "Tarcísio Vieira de Carvalho, que foi aprovada com ressalvas."
        ),
        punchline=(
            "As contas partidárias do exercício de 2015 foram aprovadas com ressalvas, "
            "com determinação de devolução ao erário do valor de R$ 1.253.409,31."
        ),
    )
    assert build_fallback_tema(row) == "Prestação de contas partidárias do exercício de 2015 com devolução ao erário"


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


def test_tema_looks_generic_flags_decision_sentence_theme():
    row = PublishPreviewRow(
        classe_processo="AgRg-REspe",
        numero_processo="0600136-55",
        resultado="Desprovido",
        votacao="Unânime",
    )
    assert tema_looks_generic(
        "O Tribunal, por unanimidade, negou provimento ao agravo regimental",
        row,
    )


def test_tema_looks_generic_flags_o_processo_trata_de_formula():
    row = PublishPreviewRow(
        classe_processo="PC",
        numero_processo="0600444-44.0000.6.00.0000",
    )
    assert tema_looks_generic(
        "O processo trata de prestação de contas",
        row,
    )


def test_tema_looks_generic_flags_relational_case_stub():
    row = PublishPreviewRow(
        classe_processo="REspe",
        numero_processo="0600003-05",
        relator="Min. Ramos Tavares",
    )
    assert tema_looks_generic(
        "Caso do município de Aracaju, relator Ministro Raul Araújo",
        row,
    )


def test_tema_looks_generic_flags_reporting_sentence_theme():
    row = PublishPreviewRow(
        classe_processo="CTA",
        numero_processo="0600090-81",
    )
    assert tema_looks_generic(
        "Além disso, destacou que a análise sobre a configuração de conduta vedada exige a verificação de fatos e provas, o que é incompatível com o rito da consulta eleitoral",
        row,
    )


def test_tema_looks_generic_flags_overbroad_party_accounts_theme():
    row = PublishPreviewRow(
        classe_processo="PC",
        numero_processo="0600231-08.2019.6.00.0000",
    )
    assert tema_looks_generic("Prestação de contas partidárias", row)


def test_build_fallback_tema_prefers_more_specific_cota_genero_theme():
    row = PublishPreviewRow(
        tema="Fraude à cota de gênero",
        classe_processo="REspe",
        numero_processo="0600003-05",
        origem="Granjeiro/CE",
        resultado="Suspenso por vista",
        analise_do_conteudo_juridico=(
            "O recurso especial eleitoral discute a configuração de fraude à cota de gênero "
            "nas eleições de 2020 para vereador em Granjeiro/CE. A Ministra Cármen Lúcia "
            "pediu vista para examinar a modulação dos efeitos da cassação e o impacto da "
            "decisão sobre a representação feminina."
        ),
        raciocinio_juridico=(
            "O voto propõe reconhecer a fraude e cassar o mandato, mas o colegiado debate "
            "como modular os efeitos para não reduzir a representação feminina."
        ),
    )
    assert build_fallback_tema(row) == "Fraude à cota de gênero e modulação dos efeitos da cassação"


def test_build_fallback_tema_infers_aime_cota_genero_theme():
    row = PublishPreviewRow(
        tema="Fraude à cota de gênero",
        classe_processo="REspe",
        numero_processo="0600003-05",
        analise_do_conteudo_juridico=(
            "O processo trata de recurso especial eleitoral contra acórdão do TRE/CE que julgou "
            "improcedente ação de impugnação de mandato eletivo (AIME) por suposta fraude à cota "
            "de gênero nas eleições de 2020 para vereador em Granjeiro/CE."
        ),
    )
    assert build_fallback_tema(row) == "Fraude à cota de gênero em ação de impugnação de mandato eletivo"


def test_punchline_looks_generic_keeps_pedido_de_vista_sentence():
    row = PublishPreviewRow(
        classe_processo="PC",
        numero_processo="0600444-44.0000.6.00.0000",
    )
    assert not punchline_looks_generic("Julgamento adiado por pedido de vista.", row)


def test_punchline_looks_generic_flags_o_processo_trata_de_prestacao_de_contas():
    row = PublishPreviewRow(
        classe_processo="PC",
        numero_processo="0600444-44.0000.6.00.0000",
    )
    assert punchline_looks_generic("O processo trata de prestação de contas.", row)


def test_punchline_looks_generic_flags_long_recurso_intro():
    row = PublishPreviewRow(
        classe_processo="REspe",
        numero_processo="0600003-05",
    )
    assert punchline_looks_generic(
        "O processo trata de recurso especial eleitoral contra acórdão do TRE/CE que julgou improcedente ação de impugnação de mandato eletivo por suposta fraude à cota de gênero.",
        row,
    )


def test_punchline_looks_generic_flags_relator_stub():
    row = PublishPreviewRow(
        classe_processo="REspe",
        numero_processo="0600003-05",
    )
    assert punchline_looks_generic(
        "Caso do município de Aracaju, relator Ministro Raul Araújo com julgamento suspenso por pedido de vista.",
        row,
    )


def test_punchline_looks_generic_flags_meta_raciocinio_sentence():
    row = PublishPreviewRow(
        classe_processo="AIJE",
        numero_processo="0600003-05",
    )
    assert punchline_looks_generic(
        "O raciocínio jurídico baseou-se na premissa de que a fraude à cota de gênero compromete a lisura do pleito.",
        row,
    )


def test_punchline_looks_generic_flags_relator_vote_stub():
    row = PublishPreviewRow(
        classe_processo="REspe",
        numero_processo="0600863-49.2022.6.05.0000",
    )
    assert punchline_looks_generic(
        "O relator, Ministro Raul Araújo, votou pelo desprovimento do recurso.",
        row,
    )


def test_punchline_looks_generic_flags_vote_followed_by_other_ministers():
    row = PublishPreviewRow(
        classe_processo="AREspe",
        numero_processo="0600682-94.2024.6.00.0000",
    )
    assert punchline_looks_generic(
        "O voto da relatora foi acompanhado pelos ministros André Ramos Tavares, Nunes Marques e Cristiano Zanin.",
        row,
    )


def test_punchline_looks_generic_flags_result_only_sentence():
    row = PublishPreviewRow(
        classe_processo="AgRg-REspe",
        numero_processo="0600279-20",
    )
    assert punchline_looks_generic(
        "Negado provimento ao agravo regimental, mantendo-se a decisão que aprovou com ressalvas as contas de campanha.",
        row,
    )


def test_punchline_looks_generic_flags_citation_only_sentence():
    row = PublishPreviewRow(
        classe_processo="PC",
        numero_processo="0600231-08.2019.6.00.0000",
    )
    assert punchline_looks_generic(
        "Artigo 73, inciso VI, alínea b, da Lei 9.504/1997; Lei 9.996/96.",
        row,
    )


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


def test_infer_classe_from_row_text_detects_aije():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "Trata-se de Ação de Investigação Judicial Eleitoral (AIJE) proposta para apurar "
            "abuso de poder e desinformação sobre o sistema eletrônico de votação."
        )
    )
    assert infer_classe_from_row_text(row) == "AIJE"


def test_infer_classe_from_row_text_detects_prestacao_de_contas():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "O processo versa sobre prestação de contas partidárias com exame da regularidade "
            "de despesas custeadas pelo Fundo Partidário."
        )
    )
    assert infer_classe_from_row_text(row) == "PC"


def test_infer_classe_from_row_text_detects_arespe():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "Agravo em recurso especial eleitoral sobre fraude à cota de gênero e cassação dos diplomas."
        )
    )
    assert infer_classe_from_row_text(row) == "AREspe"


def test_infer_classe_from_row_text_detects_plural_agravos_em_recurso_especial():
    row = PublishPreviewRow(
        resultado="Parcial provimento aos agravos em recurso especial para afastar a multa imposta na origem.",
        analise_do_conteudo_juridico="Abuso de poder econômico em Tucuruí/PA.",
    )
    assert infer_classe_from_row_text(row) == "AREspe"


def test_infer_origin_from_row_text_extracts_city_uf():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico="Discute-se a inelegibilidade de candidatos a prefeito e vice-prefeito de Paranhos/MS."
    )
    assert infer_origin_from_row_text(row) == "Paranhos/MS"


def test_infer_origin_from_row_text_falls_back_to_tre_sigla_and_tse():
    row_tre = PublishPreviewRow(
        analise_do_conteudo_juridico="Acórdão do Tribunal Regional Eleitoral de Sergipe (TRE-SE)."
    )
    row_tse = PublishPreviewRow(
        analise_do_conteudo_juridico="Questão administrativa interna do Tribunal Superior Eleitoral."
    )
    assert infer_origin_from_row_text(row_tre) == "Aracaju/SE"
    assert infer_origin_from_row_text(row_tse) == "Brasília/DF"


def test_infer_origin_from_row_text_ignores_plural_tre_listing_noise():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "Consulta enviada aos Tribunais Regionais Eleitorais do Pará, Paraná/RJ, "
            "sem indicação segura da cidade de origem do processo."
        )
    )
    assert infer_origin_from_row_text(row) == ""


def test_infer_origin_from_row_text_ignores_state_listing_ending_with_fake_city_uf():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "Foram citados Distrito Federal, Rio Grande do Sul, Rio Grande do Norte, Acre, Amapá/RO, "
            "sem indicação segura da cidade de origem do processo."
        )
    )
    assert infer_origin_from_row_text(row) == ""


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


def test_validate_preview_row_accepts_safe_dynamic_aije_and_resultado():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Integridade do sistema eletrônico de votação",
        numero_processo="0600263-54.2022.6.00.0000",
        data_sessao="2024-05-07",
        classe_processo="AIJE",
        resultado="Improcedente",
        votacao="Unânime",
        relator="Min. Alexandre de Moraes",
    )

    validated = validate_preview_row(row, schema)

    assert validated.classe_processo == "AIJE"
    assert validated.resultado == "Improcedente"


def test_validate_preview_row_sets_tse_as_subsidiary_origin_for_cta():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Uso do Fundo Partidário para custear consultoria jurídica e contábil",
        numero_processo="0600814-85.2022.6.00.0000",
        data_sessao="2025-02-26",
        classe_processo="CTA",
        origem="",
        tribunal="",
        resultado="Aprovada",
        votacao="Unânime",
    )

    validated = validate_preview_row(row, schema)

    assert validated.origem == "Brasília/DF"
    assert validated.tribunal == "TSE"


def test_normalize_party_list_strips_process_role_and_drops_lawyers():
    values = normalize_party_list(
        [
            "Recorrente: Alice",
            "Dr. João da Silva",
            "Agravante: Bob",
            "OAB/DF 12345",
            "MPE",
        ]
    )
    assert values == ["Alice", "Bob"]


def test_normalize_party_list_parses_serialized_role_mapping():
    values = normalize_party_list(
        [
            "{'embargante': 'Cláudia Aparecida dos Santos'",
            "'embargados': ['Denilson Aparecido Martins'",
            "'Federação Brasil da Esperança de Santa Luzia']}",
        ]
    )
    assert values == [
        "Cláudia Aparecida dos Santos",
        "Denilson Aparecido Martins",
        "Federação Brasil da Esperança de Santa Luzia",
    ]


def test_normalize_party_list_strips_suffix_role_label():
    values = normalize_party_list(["Thiago Soares de Godoy (agravante)"])
    assert values == ["Thiago Soares de Godoy"]


def test_normalize_party_list_preserves_entity_acronym_but_strips_process_role():
    values = normalize_party_list(
        [
            'Coligação "Bora Continuar Avançando" (Recorrente)',
            "Partido Socialista Brasileiro (PSB) (Agravado)",
            "Impetrado: Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ)",
        ]
    )
    assert values == [
        "Coligação Bora Continuar Avançando",
        "Partido Socialista Brasileiro (PSB)",
        "Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ)",
    ]


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


def test_collect_missing_multiselect_properties_only_tracks_partes_and_advogados():
    schema = make_schema()
    client = NotionSessoesClient(api_key="token", data_source_id="fake-ds")
    row = PublishPreviewRow(
        partes=["Alice", "Carol"],
        advogados=["Dr. João da Silva", "Dr. Maria"],
        composicao=["Min. Ministro Novo"],
    )
    assert client._collect_missing_multiselect_properties(schema, row) == ["partes", "advogados"]


def test_collect_missing_multiselect_options_collects_new_labels_once():
    schema = make_schema()
    client = NotionSessoesClient(api_key="token", data_source_id="fake-ds")
    row = PublishPreviewRow(
        partes=["Alice", "Carol", "Carol"],
        advogados=["Dr. João da Silva", "Dr. Maria"],
        composicao=["Min. Ministro Novo"],
    )
    assert client._collect_missing_multiselect_options(schema, row) == {
        "partes": ["Carol"],
        "advogados": ["Dr. Maria"],
    }


def test_create_row_preseeds_new_partes_and_advogados_in_default_schema():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.created_payload = None
            self.preseeded_options = None

        def _request(self, method, path, **kwargs):
            if method == "POST" and path == "/pages":
                self.created_payload = kwargs.get("json")
                return {"id": "page-created", "url": "https://notion.so/page-created"}
            raise AssertionError(f"unexpected request: {method} {path}")

        def ensure_multiselect_options_default(self, missing_options):
            self.preseeded_options = dict(missing_options)
            return {"updated": True}

    schema = make_schema()
    client = RecordingClient()
    row = PublishPreviewRow(
        tema="Tema útil",
        partes=["Alice", "Carol"],
        advogados=["Dr. João da Silva", "Dr. Maria"],
        data_sessao="2026-03-20",
    )
    response = client.create_row(schema, row)
    assert response["id"] == "page-created"
    assert client.created_payload["properties"]["partes"]["multi_select"] == [{"name": "Alice"}, {"name": "Carol"}]
    assert client.created_payload["properties"]["advogados"]["multi_select"] == [
        {"name": "Dr. João da Silva"},
        {"name": "Dr. Maria"},
    ]
    assert client.preseeded_options == {
        "partes": ["Carol"],
        "advogados": ["Dr. Maria"],
    }


def test_create_row_preseeds_missing_select_options_for_relator_and_pedido_vista():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.created_payload = None
            self.preseeded_select_options = None
            self.preseeded_multiselect_options = None

        def _request(self, method, path, **kwargs):
            if method == "POST" and path == "/pages":
                self.created_payload = kwargs.get("json")
                return {"id": "page-created", "url": "https://notion.so/page-created"}
            raise AssertionError(f"unexpected request: {method} {path}")

        def ensure_select_options_default(self, missing_options):
            self.preseeded_select_options = dict(missing_options)
            return {"updated": True}

        def ensure_multiselect_options_default(self, missing_options):
            self.preseeded_multiselect_options = dict(missing_options)
            return {"updated": True}

    schema = make_schema()
    client = RecordingClient()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0000697-22.2016.6.13.0000",
        relator="Ministro Sérgio Banhos",
        pedido_vista="Alexandre de Moraes",
        data_sessao="03/02/2022",
    )

    response = client.create_row(schema, row)

    assert response["id"] == "page-created"
    assert client.preseeded_select_options == {
        "relator": ["Ministro Sérgio Banhos"],
        "pedido_vista": ["Alexandre de Moraes"],
    }
    assert client.preseeded_multiselect_options is None


def test_ensure_multiselect_options_default_ignores_remote_existing_option_with_legacy_color():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.patch_calls = []
            self.get_count = 0

        def _request(self, method, path, **kwargs):
            if method == "GET" and path == "/data_sources/fake-ds":
                self.get_count += 1
                options = [{"name": "Alice", "color": "default"}]
                if self.get_count >= 2:
                    options.append({"name": "Carol", "color": "blue"})
                return {
                    "properties": {
                        "partes": {
                            "type": "multi_select",
                            "multi_select": {"options": options},
                        }
                    }
                }
            if method == "PATCH" and path == "/data_sources/fake-ds":
                self.patch_calls.append(kwargs.get("json"))
                raise RuntimeError(
                    'Notion API error 400: {"object":"error","status":400,"code":"validation_error",'
                    '"message":"Cannot update color of select with name: Carol."}'
                )
            raise AssertionError(f"unexpected request: {method} {path}")

    client = RecordingClient()

    result = client.ensure_multiselect_options_default({"partes": ["Carol"]})

    assert result == {"updated": False, "properties": []}
    assert len(client.patch_calls) == 1
    assert client.get_count == 2


def test_ensure_multiselect_options_default_preserves_existing_options_when_adding_new_one():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.options = [
                {"name": "Alice", "color": "default"},
                {"name": "Bob", "color": "default"},
            ]
            self.patch_calls = []

        def _request(self, method, path, **kwargs):
            if method == "GET" and path == "/data_sources/fake-ds":
                return {
                    "properties": {
                        "partes": {
                            "type": "multi_select",
                            "multi_select": {"options": list(self.options)},
                        }
                    }
                }
            if method == "PATCH" and path == "/data_sources/fake-ds":
                payload_options = kwargs.get("json", {}).get("properties", {}).get("partes", {}).get("multi_select", {}).get("options", [])
                self.patch_calls.append(payload_options)
                self.options = list(payload_options)
                return {"ok": True}
            raise AssertionError(f"unexpected request: {method} {path}")

    client = RecordingClient()

    result = client.ensure_multiselect_options_default({"partes": ["Carol"]})

    assert result == {"updated": True, "properties": [{"property": "partes", "created_options": 1}]}
    assert client.options == [
        {"name": "Alice", "color": "default"},
        {"name": "Bob", "color": "default"},
        {"name": "Carol", "color": "default"},
    ]
    assert client.patch_calls[-1] == client.options


def test_ensure_select_options_default_preserves_existing_options_when_adding_new_one():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.options = [
                {"name": "Min. Cármen Lúcia", "color": "default"},
                {"name": "Min. André Mendonça", "color": "default"},
            ]
            self.patch_calls = []

        def _request(self, method, path, **kwargs):
            if method == "GET" and path == "/data_sources/fake-ds":
                return {
                    "properties": {
                        "relator": {
                            "type": "select",
                            "select": {"options": list(self.options)},
                        }
                    }
                }
            if method == "PATCH" and path == "/data_sources/fake-ds":
                payload_options = kwargs.get("json", {}).get("properties", {}).get("relator", {}).get("select", {}).get("options", [])
                self.patch_calls.append(payload_options)
                self.options = list(payload_options)
                return {"ok": True}
            raise AssertionError(f"unexpected request: {method} {path}")

    client = RecordingClient()

    result = client.ensure_select_options_default({"relator": ["Min. Sérgio Banhos"]})

    assert result == {"updated": True, "properties": [{"property": "relator", "created_options": 1}]}
    assert client.options == [
        {"name": "Min. Cármen Lúcia", "color": "default"},
        {"name": "Min. André Mendonça", "color": "default"},
        {"name": "Min. Sérgio Banhos", "color": "default"},
    ]
    assert client.patch_calls[-1] == client.options


def test_ensure_multiselect_options_default_skips_schema_patch_when_property_is_already_above_limit():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.patch_calls = []

        def _request(self, method, path, **kwargs):
            if method == "GET" and path == "/data_sources/fake-ds":
                options = [{"name": f"Opt {index}", "color": "default"} for index in range(101)]
                return {
                    "properties": {
                        "partes": {
                            "type": "multi_select",
                            "multi_select": {"options": options},
                        }
                    }
                }
            if method == "PATCH" and path == "/data_sources/fake-ds":
                self.patch_calls.append(kwargs.get("json"))
                return {"ok": True}
            raise AssertionError(f"unexpected request: {method} {path}")

    client = RecordingClient()

    result = client.ensure_multiselect_options_default({"partes": ["Carol"]})

    assert result == {
        "updated": True,
        "properties": [
            {
                "property": "partes",
                "created_options": 0,
                "skipped_schema_update_due_to_limit": True,
            }
        ],
    }
    assert client.patch_calls == []


def test_update_row_skips_schema_normalization_when_no_new_multiselect_option_exists():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.updated_payload = None
            self.preseeded_options = None

        def _request(self, method, path, **kwargs):
            if method == "PATCH" and path == "/pages/page-123":
                self.updated_payload = kwargs.get("json")
                return {"id": "page-123", "url": "https://notion.so/page-123"}
            raise AssertionError(f"unexpected request: {method} {path}")

        def ensure_multiselect_options_default(self, missing_options):
            self.preseeded_options = dict(missing_options)
            return {"updated": True}

    schema = make_schema()
    client = RecordingClient()
    row = PublishPreviewRow(
        tema="Tema útil",
        partes=["Alice"],
        advogados=["Dr. João da Silva"],
        data_sessao="2026-03-20",
    )
    response = client.update_row(schema, "page-123", row)
    assert response["id"] == "page-123"
    assert client.updated_payload["properties"]["partes"]["multi_select"] == [{"name": "Alice"}]
    assert client.updated_payload["properties"]["advogados"]["multi_select"] == [{"name": "Dr. João da Silva"}]
    assert client.preseeded_options is None


def test_create_row_skips_schema_normalization_when_post_write_is_disabled():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(
                api_key="token",
                data_source_id="fake-ds",
                normalize_multiselect_colors_post_write=False,
            )
            self.created_payload = None

        def _request(self, method, path, **kwargs):
            if method == "POST" and path == "/pages":
                self.created_payload = kwargs.get("json")
                return {"id": "page-created", "url": "https://notion.so/page-created"}
            raise AssertionError(f"unexpected request: {method} {path}")

        def ensure_multiselect_options_default(self, missing_options):
            raise AssertionError("schema normalization should stay disabled in this mode")

    schema = make_schema()
    client = RecordingClient()
    row = PublishPreviewRow(
        tema="Tema útil",
        partes=["Alice", "Carol"],
        advogados=["Dr. João da Silva", "Dr. Maria"],
        data_sessao="2026-03-20",
    )
    response = client.create_row(schema, row)
    assert response["id"] == "page-created"
    assert client.created_payload["properties"]["partes"]["multi_select"] == [{"name": "Alice"}, {"name": "Carol"}]
    assert client.created_payload["properties"]["advogados"]["multi_select"] == [
        {"name": "Dr. João da Silva"},
        {"name": "Dr. Maria"},
    ]


def test_rebuild_multiselect_property_with_default_colors_swaps_property_schema():
    class RecordingClient(NotionSessoesClient):
        def __init__(self):
            super().__init__(api_key="token", data_source_id="fake-ds")
            self.properties = {
                "partes": {
                    "id": "prop-partes",
                    "type": "multi_select",
                    "multi_select": {
                        "options": [
                            {"id": "opt-a", "name": "Alice", "color": "red"},
                            {"id": "opt-b", "name": "Bob", "color": "blue"},
                        ]
                    },
                }
            }
            self.pages = [
                {"id": "page-1", "properties": {"partes": {"multi_select": [{"name": "Alice"}]}}},
                {"id": "page-2", "properties": {"partes": {"multi_select": [{"name": "Bob"}]}}},
            ]

        def fetch_schema(self):
            schema = make_schema()
            schema.properties["partes"].options = [
                option.get("name", "")
                for option in self.properties.get("partes", {}).get("multi_select", {}).get("options", [])
                if option.get("name", "")
            ]
            return schema

        def _request(self, method, path, **kwargs):
            if method == "GET" and path == "/data_sources/fake-ds":
                return {"properties": self.properties}
            if method == "POST" and path == "/data_sources/fake-ds/query":
                return {"results": self.pages, "has_more": False, "next_cursor": None}
            if method == "PATCH" and path == "/data_sources/fake-ds":
                properties = kwargs.get("json", {}).get("properties", {})
                for key, value in properties.items():
                    if value is None:
                        self.properties.pop(key, None)
                        continue
                    if "name" in value and key in {prop.get("id") for prop in self.properties.values()}:
                        for current_name, prop in list(self.properties.items()):
                            if prop.get("id") == key:
                                new_name = value["name"]
                                self.properties[new_name] = self.properties.pop(current_name)
                                break
                        continue
                    target = self.properties.setdefault(key, {"id": f"id-{key}", "type": "multi_select", "multi_select": {"options": []}})
                    options = value.get("multi_select", {}).get("options")
                    if options is not None:
                        existing = {opt["name"]: opt for opt in target["multi_select"]["options"]}
                        for option in options:
                            existing[option["name"]] = {
                                "id": existing.get(option["name"], {}).get("id", f"id-{key}-{option['name']}"),
                                "name": option["name"],
                                "color": option.get("color", "default"),
                            }
                        target["multi_select"]["options"] = list(existing.values())
                return {"properties": self.properties}
            if method == "PATCH" and path.startswith("/pages/"):
                page_id = path.split("/")[-1]
                properties = kwargs.get("json", {}).get("properties", {})
                for page in self.pages:
                    if page["id"] == page_id:
                        page["properties"].update(properties)
                        return {"id": page_id}
                raise AssertionError(f"unknown page: {page_id}")
            raise AssertionError(f"unexpected request: {method} {path}")

    client = RecordingClient()
    summary = client.rebuild_multiselect_property_with_default_colors("partes")
    assert summary["updated"] is True
    assert "partes" in client.properties
    assert "partes__legacy_color" not in client.properties
    assert "partes__default_tmp" not in client.properties
    assert [opt["color"] for opt in client.properties["partes"]["multi_select"]["options"]] == ["default", "default"]
    assert client.pages[0]["properties"]["partes"]["multi_select"] == [{"name": "Alice"}]
    assert client.pages[1]["properties"]["partes"]["multi_select"] == [{"name": "Bob"}]


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


def test_validate_preview_row_keeps_dynamic_tipo_registro_beyond_schema_options():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        tipo_registro="Julgamento 10",
        numero_processo="0600249-07",
        youtube_link="https://youtu.be/abc123?t=10",
        data_sessao="20/03/2026",
    )
    validated = validate_preview_row(row, schema)
    assert validated.tipo_registro == "Julgamento 10"
    assert any("tipo_registro com opção nova no Notion: Julgamento 10" in warning for warning in validated.warnings)
    assert not any("tipo_registro fora das opções do Notion" in warning for warning in validated.warnings)


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


def test_validate_preview_row_keeps_canonical_new_relator_and_pedido_vista_options():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0000697-22.2016.6.13.0000",
        relator="Ministro Sérgio Banhos",
        pedido_vista="Alexandre de Moraes",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="03/02/2022",
    )

    validated = validate_preview_row(row, schema)

    assert validated.relator == "Min. Sérgio Banhos"
    assert validated.pedido_vista == "Min. Alexandre de Moraes"
    assert any("relator com opção nova no Notion: Min. Sérgio Banhos" in warning for warning in validated.warnings)
    assert any(
        "pedido_vista com opção nova no Notion: Min. Alexandre de Moraes" in warning
        for warning in validated.warnings
    )
    assert not any("relator fora das opções do Notion" in warning for warning in validated.warnings)
    assert not any("pedido_vista fora das opções do Notion" in warning for warning in validated.warnings)


def test_validate_preview_row_collapses_multi_name_relator_to_single_canonical_option():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="2354620-17",
        relator="Luís Felipe Salomão (original), Raul Araújo (sucessor)",
        pedido_vista="Ministro Raul Araújo",
        data_sessao="31/08/2023",
    )

    validated = validate_preview_row(row, schema)

    assert validated.relator == "Min. Raul Araújo"
    assert validated.pedido_vista == "Min. Raul Araújo"
    assert not any("Luís Felipe Salomão, Raul Araújo" in warning for warning in validated.warnings)


def test_validate_preview_row_overrides_result_when_judgment_is_suspended_by_vista():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0613678-87",
        classe_processo="AgRg-REspe",
        resultado="Provido",
        votacao="Suspenso",
        pedido_vista="Ministro Nunes Marques",
        data_sessao="20/03/2026",
    )
    validated = validate_preview_row(row, schema)
    assert validated.votacao == "Suspenso"
    assert validated.resultado == "Suspenso por vista"


def test_validate_preview_row_preserves_suspenso_julgado_depois_marker():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0613678-87",
        classe_processo="AgRg-REspe",
        resultado="Suspenso mas julgado depois",
        votacao="Suspenso",
        data_sessao="20/03/2026",
    )
    validated = validate_preview_row(row, schema)
    assert validated.votacao == "Suspenso*"
    assert validated.resultado == "Suspenso mas julgado depois"


def test_validate_preview_row_replaces_generic_pa_with_arespe_when_text_supports_it():
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0600071-96.2025.6.00.0000",
        classe_processo="PA",
        analise_do_conteudo_juridico="Agravo em recurso especial eleitoral envolvendo fraude à cota de gênero.",
        data_sessao="20/03/2026",
    )

    validated = validate_preview_row(row, None)

    assert validated.classe_processo == "AREspe"


def test_validate_preview_row_removes_textual_numero_and_splits_person_partes():
    row = PublishPreviewRow(
        tema="Abuso de poder político",
        numero_processo="Recursos Ordinários de Luiz Augusto Barcelos Lara e Divaldo Vieira Lara",
        partes=["Luiz Augusto Barcelos Lara e Divaldo Vieira Lara"],
        data_sessao="03/03/2022",
    )

    validated = validate_preview_row(row, None)

    assert validated.numero_processo == ""
    assert validated.partes == ["Luiz Augusto Barcelos Lara", "Divaldo Vieira Lara"]
    assert any(
        "Número do processo textual inválido removido" in warning
        for warning in validated.warnings
    )


def test_validate_preview_row_keeps_final_result_when_view_was_historical_but_vote_is_final():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0600020-93",
        classe_processo="AgRg-REspe",
        resultado="Desprovido",
        votacao="Por maioria",
        pedido_vista="Ministro André Ramos Tavares",
        data_sessao="20/03/2026",
    )
    validated = validate_preview_row(row, schema)
    assert validated.votacao == "Por maioria"
    assert validated.resultado == "Desprovido"


def test_infer_resultado_from_row_text_maps_consulta_respondida_to_aprovada():
    row = PublishPreviewRow(
        tema="Utilização de Fundo Partidário para defesa de filiados",
        punchline="",
        analise_do_conteudo_juridico="Consulta formulada pelo partido sobre o uso do Fundo Partidário.",
        raciocinio_juridico="Consulta respondida nos termos do voto do relator.",
    )
    assert infer_resultado_from_row_text(row) == "Aprovada"


def test_infer_resultado_from_row_text_maps_improcedente():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "O relator votou por julgar improcedente a ação de investigação judicial eleitoral, "
            "afastando as alegações de fraude nas urnas eletrônicas."
        ),
    )
    assert infer_resultado_from_row_text(row) == "Improcedente"


def test_infer_resultado_from_row_text_maps_procedente_em_parte():
    row = PublishPreviewRow(
        analise_do_conteudo_juridico=(
            "O voto conclui pela procedência parcial da representação, com imposição de multa "
            "e manutenção de ordem de remoção do conteúdo irregular."
        ),
    )
    assert infer_resultado_from_row_text(row) == "Procedente em parte"


def test_validate_preview_row_infers_missing_resultado_from_text():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Utilização de Fundo Partidário para defesa de filiados",
        numero_processo="0600814-85.2022.6.00.0000",
        resultado="",
        votacao="Unânime",
        relator="Ministro Alexandre de Moraes",
        analise_do_conteudo_juridico="Consulta formulada pelo partido sobre o uso do Fundo Partidário.",
        raciocinio_juridico="Consulta respondida nos termos do voto do relator.",
        data_sessao="21/05/2024",
    )
    validated = validate_preview_row(row, schema)
    assert validated.resultado == "Aprovada"


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
            self.content = (
                b"<html><body>Noticia valida sobre Porto Alegre/RS no processo 0600249-07 "
                b"julgado pelo TSE.</body></html>"
            )
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

    def fake_filter_relevant(urls, _row):
        if broken_url in urls and fixed_url not in urls:
            return [], [broken_url], []
        if fixed_url in urls:
            return [fixed_url], [], []
        return list(urls), [], []

    monkeypatch.setattr(enricher, "_call_grounded_json", fake_call_grounded_json)
    monkeypatch.setattr(core, "filter_relevant_institutional_news_urls", fake_filter_relevant)

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


def test_process_metadata_enricher_keeps_item_when_grounding_marks_precedent_but_local_video_proves_overlay_judgment(tmp_path):
    row = PublishPreviewRow(
        tema="ED na PC - 060122740",
        classe_processo="PC",
        tipo_registro="Julgamento 5",
        eleicao="2018",
        origem="Brasília/DF",
        tribunal="TSE",
        numero_processo="0601227-40",
        youtube_link="https://www.youtube.com/watch?v=BLnwPIWKZv4&t=4963",
        resultado="Rejeitados",
        votacao="Unânime",
        data_sessao="2023-04-18",
        source_start_seconds=4963,
        source_bundle_index=9,
        source_item_index=1,
    )
    artifact_store = RunArtifacts(tmp_path)
    artifact_store.write_text(
        "raw_global_response_chunk_19.txt",
        json.dumps(
            [
                {
                    "data_sessao": "18 de abril de 2023",
                    "julgamentos": [
                        {
                            "processo": "ED na PC - 060122740",
                            "timestamp_inicial": 4963,
                            "timestamp_final": 5106,
                            "should_ignore": False,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = artifact_store

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        return core.ProcessMetadataResult(
            full_numero_processo="0601227-40.2018.6.00.0000",
            origem="Brasília/DF",
            is_judged_process=False,
        )

    enricher._call_grounded_json = fake_call_grounded_json

    enriched = enricher.enrich_rows([row])

    assert enriched[0].blocked is False
    assert enriched[0].numero_processo == "0601227-40.2018.6.00.0000"
    assert any("prova local forte do julgamento" in warning for warning in enriched[0].warnings)


def test_process_metadata_enricher_still_blocks_single_chunk_false_positive_when_grounding_marks_precedent(tmp_path):
    row = PublishPreviewRow(
        tema="0600378-65",
        classe_processo="CTA",
        tipo_registro="Julgamento 1",
        tribunal="TSE",
        numero_processo="0600378-65",
        youtube_link="https://www.youtube.com/watch?v=s9Ts40TfDas&t=114",
        data_sessao="2021-02-11",
        source_start_seconds=114,
        source_bundle_index=1,
        source_item_index=1,
    )
    artifact_store = RunArtifacts(tmp_path)
    artifact_store.write_text(
        "raw_global_response_chunk_01.txt",
        json.dumps(
            [
                {
                    "data_sessao": "23 de junho de 2021",
                    "julgamentos": [
                        {
                            "processo": "0600378-65.2020.6.00.0000",
                            "timestamp_inicial": 114,
                            "timestamp_final": 299,
                            "should_ignore": False,
                        }
                    ],
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
    )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = artifact_store

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        return core.ProcessMetadataResult(
            full_numero_processo="0600378-65.2020.6.00.0000",
            origem="",
            is_judged_process=False,
        )

    enricher._call_grounded_json = fake_call_grounded_json

    enriched = enricher.enrich_rows([row])

    assert enriched[0].blocked is True
    assert any("precedente citado" in error for error in enriched[0].errors)


def test_row_has_strong_local_judgment_evidence_accepts_processos_plural(tmp_path):
    row = PublishPreviewRow(
        tema="Agravo regimental em recurso especial eleitoral",
        classe_processo="AgRg-REspe",
        numero_processo="0602136-21.2022.6.17.0000",
    )
    artifact_store = RunArtifacts(tmp_path)
    for chunk_name, start_seconds in [("10", 2679), ("11", 2700)]:
        artifact_store.write_text(
            f"raw_global_response_chunk_{chunk_name}.txt",
            json.dumps(
                [
                    {
                        "julgamentos": [
                            {
                                "processos": ["060213621", "060210598"],
                                "timestamp_inicial": start_seconds,
                                "should_ignore": False,
                            }
                        ]
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
        )

    assert core._row_has_strong_local_judgment_evidence(row, artifact_store) is True


def test_process_metadata_enricher_keeps_judged_process_when_local_chunks_use_processos_plural(tmp_path):
    row = PublishPreviewRow(
        tema="Agravo regimental em recurso especial eleitoral",
        classe_processo="AgRg-REspe",
        numero_processo="0602136-21",
        youtube_link="https://www.youtube.com/watch?v=r_TMEJe3iIg&t=2679",
        data_sessao="2023-08-17",
    )
    artifact_store = RunArtifacts(tmp_path)
    for chunk_name, start_seconds in [("10", 2679), ("11", 2700)]:
        artifact_store.write_text(
            f"raw_global_response_chunk_{chunk_name}.txt",
            json.dumps(
                [
                    {
                        "julgamentos": [
                            {
                                "processos": ["060213621", "060210598"],
                                "timestamp_inicial": start_seconds,
                                "should_ignore": False,
                            }
                        ]
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
        )
    enricher = object.__new__(GeminiProcessMetadataEnricher)
    enricher.artifact_store = artifact_store

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        return core.ProcessMetadataResult(
            full_numero_processo="0602136-21.2022.6.17.0000",
            origem="Vitória/ES",
            is_judged_process=False,
        )

    enricher._call_grounded_json = fake_call_grounded_json

    enriched = enricher.enrich_rows([row])

    assert enriched[0].blocked is False
    assert any("prova local forte do julgamento" in warning for warning in enriched[0].warnings)
    assert not any("precedente citado" in error for error in enriched[0].errors)


def test_news_enricher_reuses_cached_artifact(tmp_path):
    artifact_store = RunArtifacts(tmp_path)
    row = PublishPreviewRow(tema="Tema")
    artifact_store.write_json(
        "06_news_enrichment_01.json",
        {
            "context": core.build_news_enrichment_context(row),
            "applied": {
                "noticia_TSE": "https://www.tse.jus.br/noticia",
                "noticia_TRE": "",
                "noticias_gerais": ["https://g1.globo.com/noticia"],
            }
        },
    )
    enricher = object.__new__(GeminiNewsEnricher)
    enricher.artifact_store = artifact_store

    enriched = enricher.enrich_rows([row])

    assert enriched[0].noticia_TSE == "https://www.tse.jus.br/noticia"
    assert enriched[0].noticias_gerais == ["https://g1.globo.com/noticia"]


def test_news_response_accepts_plain_url_list():
    parsed = core._coerce_gemini_response_model(
        NewsEnrichmentResult,
        json.dumps(["https://www.poder360.com.br/noticia"]),
    )

    assert parsed.noticia_geral == ["https://www.poder360.com.br/noticia"]


def test_optional_enrichment_response_accepts_empty_text():
    assert core._coerce_gemini_response_model(NewsEnrichmentResult, "").noticia_geral == []
    assert core._coerce_gemini_response_model(InstitutionalRepairResult, "").urls == []
    assert core._coerce_gemini_response_model(ThemePunchlineRepairBatchResult, "").items == []


def test_institutional_repair_response_accepts_plain_url_list():
    parsed = core._coerce_gemini_response_model(
        InstitutionalRepairResult,
        json.dumps(["https://www.tre-mt.jus.br/noticia"]),
    )

    assert parsed.urls == ["https://www.tre-mt.jus.br/noticia"]


def test_theme_punchline_batch_response_accepts_plain_item_list():
    parsed = core._coerce_gemini_response_model(
        ThemePunchlineRepairBatchResult,
        json.dumps(
            [
                {
                    "key": "row_001",
                    "tema": "Conduta vedada",
                    "punchline": "O tribunal manteve a sanção por uso indevido de bens públicos.",
                    "confidence": "medium",
                }
            ],
            ensure_ascii=False,
        ),
    )

    assert len(parsed.items) == 1
    assert parsed.items[0].key == "row_001"
    assert parsed.items[0].tema == "Conduta vedada"


def test_news_enricher_keeps_row_when_grounding_fails(tmp_path, monkeypatch):
    row = PublishPreviewRow(
        tema="Conduta vedada por uso de bens públicos",
        numero_processo="0600249-07",
    )
    enricher = object.__new__(GeminiNewsEnricher)
    enricher.artifact_store = RunArtifacts(tmp_path)
    enricher.model = "gemini-3.1-flash-lite"

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        raise RuntimeError("empty grounded response")

    monkeypatch.setattr(enricher, "_call_grounded_json", fake_call_grounded_json)

    enriched = enricher.enrich_rows([row])

    assert len(enriched) == 1
    assert enriched[0].tema == row.tema
    assert any("Enriquecimento de notícias falhou" in warning for warning in enriched[0].warnings)
    assert (tmp_path / "06_news_enrichment_01.json").exists()


def test_news_enricher_ignores_stale_cached_artifact(tmp_path, monkeypatch):
    artifact_store = RunArtifacts(tmp_path)
    artifact_store.write_json(
        "06_news_enrichment_01.json",
        {
            "context": "tema: outro contexto",
            "applied": {
                "noticia_TSE": "https://www.tse.jus.br/noticia-antiga",
            },
        },
    )
    row = PublishPreviewRow(tema="Tema atual", numero_processo="0600249-07")
    enricher = object.__new__(GeminiNewsEnricher)
    enricher.artifact_store = artifact_store
    enricher.model = "gemini-2.5-flash-lite"

    def fake_call_grounded_json(*, prompt, response_model, artifact_name):
        return NewsEnrichmentResult(noticia_TSE=["https://www.tse.jus.br/noticia-nova"]), []

    monkeypatch.setattr(enricher, "_call_grounded_json", fake_call_grounded_json)
    monkeypatch.setattr(
        core,
        "filter_relevant_institutional_news_urls",
        lambda urls, _row: (list(urls), [], []),
    )
    monkeypatch.setattr(core, "filter_general_news_urls", lambda urls, _row: list(urls))

    enriched = enricher.enrich_rows([row])

    assert enriched[0].noticia_TSE == "https://www.tse.jus.br/noticia-nova"


def test_news_enricher_reuses_existing_valid_links_without_grounding(tmp_path, monkeypatch):
    row = PublishPreviewRow(
        tema="Conduta vedada por uso de bens públicos",
        numero_processo="0600249-07",
        noticia_TSE="https://www.tse.jus.br/noticia",
    )
    enricher = object.__new__(GeminiNewsEnricher)
    enricher.artifact_store = RunArtifacts(tmp_path)

    def should_not_call(*args, **kwargs):
        raise AssertionError("grounding não deveria ser chamado quando já há notícia válida")

    monkeypatch.setattr(enricher, "_call_grounded_json", should_not_call)
    monkeypatch.setattr(
        core,
        "filter_relevant_institutional_news_urls",
        lambda urls, _row: (list(urls), [], []),
    )
    monkeypatch.setattr(core, "filter_general_news_urls", lambda urls, _row: list(urls))

    enriched = enricher.enrich_rows([row])

    assert enriched[0].noticia_TSE == "https://www.tse.jus.br/noticia"
    assert (tmp_path / "06_news_enrichment_01.json").exists()


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


def test_filter_relevant_institutional_news_urls_discards_unrelated_tse_page(monkeypatch):
    row = PublishPreviewRow(
        tema="Conduta vedada por uso de bens públicos",
        origem="Potiretama/CE",
        tribunal="TRE-CE",
        numero_processo="0600368-79.2024.6.06.0086",
        partes=["Luan Dantas Félix", "Solange Mary Holanda Campelo Balbino"],
    )

    def fake_snapshot(url):
        if url.endswith("/relevante"):
            return (
                url,
                200,
                "text/html",
                "<html><body>TSE mantém decisão sobre Luan Dantas Félix em Potiretama/CE no processo 0600368-79.</body></html>",
            )
        return (
            url,
            200,
            "text/html",
            "<html><body>TSE divulga calendário de atendimento biométrico nacional.</body></html>",
        )

    monkeypatch.setattr(core, "fetch_candidate_page_snapshot", fake_snapshot)

    accepted, unavailable, irrelevant = core.filter_relevant_institutional_news_urls(
        [
            "https://www.tse.jus.br/relevante",
            "https://www.tse.jus.br/irrelevante",
        ],
        row,
    )

    assert accepted == ["https://www.tse.jus.br/relevante"]
    assert unavailable == []
    assert irrelevant == ["https://www.tse.jus.br/irrelevante"]


def test_filter_relevant_institutional_news_urls_discards_generic_homepage(monkeypatch):
    row = PublishPreviewRow(
        tema="Transporte especial de eleitores com deficiência",
        origem="Brasília/DF",
        tribunal="TSE",
        numero_processo="0000276-65",
    )

    def fake_snapshot(url):
        return (
            url,
            200,
            "text/html",
            "<html><body>TSE noticia processo 0000276-65 em Brasília/DF.</body></html>",
        )

    monkeypatch.setattr(core, "fetch_candidate_page_snapshot", fake_snapshot)

    accepted, unavailable, irrelevant = core.filter_relevant_institutional_news_urls(
        [
            "https://www.tse.jus.br/",
            "https://www.tse.jus.br/comunicacao/noticias",
            "https://www.tse.jus.br/comunicacao/noticias/2026/Maio/noticia-especifica",
        ],
        row,
    )

    assert accepted == ["https://www.tse.jus.br/comunicacao/noticias/2026/Maio/noticia-especifica"]
    assert unavailable == []
    assert irrelevant == ["https://www.tse.jus.br/", "https://www.tse.jus.br/comunicacao/noticias"]


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


def test_validate_preview_row_promotes_full_cnj_and_fills_origin_from_tribunal():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        origem="Amapá",
        tribunal="TRE-AP",
        numero_processo="0600249-07",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        raciocinio_juridico="O processo 0600249-07.2024.6.02.0001 trata de publicidade institucional.",
        analise_do_conteudo_juridico="Publicidade institucional em período vedado.",
        data_sessao="20/03/2026",
    )

    validated = validate_preview_row(row, schema)

    assert validated.numero_processo == "0600249-07.2024.6.02.0001"
    assert validated.origem == "Macapá/AP"


def test_validate_preview_row_promotes_special_adi_number_from_text():
    schema = make_schema()
    row = PublishPreviewRow(
        tema="Distribuição de sobras eleitorais",
        numero_processo="7228",
        analise_do_conteudo_juridico="O julgamento trata das ADI 7228, 7263 e 7325.",
    )

    validated = validate_preview_row(row, schema)

    assert validated.numero_processo == "ADI 7228"
    assert validated.classe_processo == ""


def test_infer_punchline_from_row_text_uses_contextual_sentence_not_decision_formula():
    row = PublishPreviewRow(
        tema="",
        classe_processo="REspe",
        resultado="Desprovido",
        votacao="Unânime",
        analise_do_conteudo_juridico=(
            "Publicidade institucional em período vedado antes das eleições municipais. "
            "O Tribunal, por unanimidade, negou provimento ao recurso."
        ),
        raciocinio_juridico="A propaganda institucional teve caráter promocional.",
        fundamentacao_normativa="Art. 73, VI, b, da Lei 9.504/1997.",
    )

    assert infer_punchline_from_row_text(row) == "Publicidade institucional em período vedado antes das eleições municipais."


def test_infer_punchline_from_row_text_rebuilds_consulta_sentence_without_truncation():
    row = PublishPreviewRow(
        tema="Uso do Fundo Partidário para custear consultoria jurídica e contábil",
        classe_processo="CTA",
        resultado="Aprovada",
        votacao="Unânime",
        analise_do_conteudo_juridico=(
            "Consulta formulada pelo PSDB sobre a possibilidade de utilização de recursos do Fundo Partidário "
            "para o pagamento de serviços de consultoria jurídica e contábil relacionados à defesa de filiados."
        ),
    )

    assert infer_punchline_from_row_text(row) == (
        "Consulta sobre uso do Fundo Partidário para custear consultoria jurídica e contábil em defesa de filiados."
    )


def test_infer_punchline_from_row_text_rebuilds_consulta_without_truncated_pagamento():
    row = PublishPreviewRow(
        tema="Uso do Fundo Partidário para custear consultoria jurídica e contábil",
        classe_processo="CTA",
        resultado="Aprovada",
        analise_do_conteudo_juridico=(
            "O processo trata de uma consulta formulada pelo PSDB sobre a possibilidade de utilização "
            "de recursos do Fundo Partidário para o pagamento de despesas com a contratação de serviços "
            "de consultoria jurídica e contábil."
        ),
    )

    assert infer_punchline_from_row_text(row) == (
        "Consulta sobre uso do Fundo Partidário para custear consultoria jurídica e contábil."
    )


def test_infer_punchline_from_row_text_rejects_generic_aije_intro_and_uses_theme():
    row = PublishPreviewRow(
        tema="Integridade do sistema eletrônico de votação nas eleições de 2022",
        classe_processo="AIJE",
        votacao="Unânime",
        analise_do_conteudo_juridico=(
            "Trata-se de Ação de Investigação Judicial Eleitoral (AIJE) proposta pelo Partido Liberal "
            "contra Jair Messias Bolsonaro, questionando a integridade do sistema eletrônico de votação."
        ),
    )

    assert infer_punchline_from_row_text(row) == (
        "Julgamento sobre integridade do sistema eletrônico de votação nas eleições de 2022."
    )


def test_infer_punchline_from_row_text_discards_long_truncated_sentence_after_crop():
    row = PublishPreviewRow(
        tema="Propaganda eleitoral irregular",
        classe_processo="Rp",
        resultado="Procedente em parte",
        raciocinio_juridico=(
            "O relator fundamenta seu voto na interpretação restritiva do art. 57-D da Lei das Eleições, "
            "argumentando que o impulsionamento de conteúdo deve observar estritamente as finalidades "
            "permitidas pela norma, sob pena de desequilíbrio do pleito e violação do art. 57-D."
        ),
    )
    assert infer_punchline_from_row_text(row) == "Julgamento sobre propaganda eleitoral irregular."


def test_infer_punchline_from_row_text_rebuilds_cota_genero_vista_sentence():
    row = PublishPreviewRow(
        tema="Fraude à cota de gênero e modulação dos efeitos da cassação",
        classe_processo="REspe",
        numero_processo="0600003-05",
        origem="Granjeiro/CE",
        resultado="Suspenso por vista",
        votacao="Suspenso",
        analise_do_conteudo_juridico=(
            "O recurso especial eleitoral discute fraude à cota de gênero nas eleições de 2020 "
            "em Granjeiro/CE. A Ministra Cármen Lúcia pediu vista para examinar a modulação dos "
            "efeitos da cassação e evitar redução da representação feminina."
        ),
        raciocinio_juridico=(
            "O voto do relator reconhece a fraude, mas o colegiado debate a preservação da "
            "representação feminina e a modulação dos efeitos."
        ),
    )
    assert infer_punchline_from_row_text(row) == (
        "Julgamento sobre fraude à cota de gênero em Granjeiro/CE foi suspenso por vista após debate sobre modulação dos efeitos da cassação e preservação da representação feminina."
    )


def test_theme_punchline_pair_needs_rewrite_flags_short_repetitive_pair():
    row = PublishPreviewRow(
        tema="Fraude à cota de gênero",
        punchline="Fraude à cota de gênero.",
        analise_do_conteudo_juridico="A discussão envolveu candidatura feminina fictícia e consequência na chapa proporcional.",
    )

    assert theme_punchline_pair_too_similar(row.tema, row.punchline)
    assert theme_punchline_pair_needs_rewrite(row)


def test_theme_punchline_enricher_applies_complementary_repair_item():
    row = PublishPreviewRow(
        tema="Julgamento",
        punchline="Recurso provido.",
        resultado="Provido",
        analise_do_conteudo_juridico=(
            "A controvérsia envolveu a utilização promocional de programa social por agente público durante "
            "a campanha municipal, com debate sobre desequilíbrio eleitoral e alcance da sanção."
        ),
    )
    repair_item = core.ThemePunchlineRepairItem(
        key="row_001",
        tema="Uso promocional de programa social em campanha municipal",
        punchline=(
            "A disputa examinou se a divulgação eleitoral de benefícios sociais pela gestão municipal comprometeu "
            "a igualdade da disputa, e o TSE reconheceu o impacto concreto da conduta no desfecho do pleito."
        ),
    )
    enricher = object.__new__(GeminiThemePunchlineEnricher)

    repaired = enricher._apply_repair_item(row, repair_item)

    assert repaired.tema == "Uso promocional de programa social em campanha municipal"
    assert repaired.punchline.startswith("A disputa examinou")
    assert not theme_punchline_pair_needs_rewrite(repaired)


def test_build_editorial_punchline_fallback_adds_context_and_result():
    row = PublishPreviewRow(
        tema="Publicidade institucional em período vedado",
        resultado="Desprovido",
        analise_do_conteudo_juridico=(
            "Prefeito manteve publicidade institucional em canais oficiais durante o período vedado das eleições "
            "municipais, e a controvérsia examinou se a exposição beneficiou sua candidatura à reeleição."
        ),
    )

    punchline = build_editorial_punchline_fallback(row, row.tema)

    assert "publicidade institucional" in punchline.lower()
    assert "desprovido" in punchline.lower()
    assert len(punchline) >= 90


def test_should_replace_classe_processo_rejects_speculative_adi_over_consulta():
    row = PublishPreviewRow(
        numero_processo="0601984-92.2022.6.00.0000",
        classe_processo="CTA",
        analise_do_conteudo_juridico="O voto menciona julgamentos de ADI relacionados ao tema.",
    )

    assert should_replace_classe_processo("CTA", "ADI", row) is False


def test_should_replace_classe_processo_rejects_adi_even_when_numero_mentions_it():
    row = PublishPreviewRow(
        numero_processo="ADI 7228",
        classe_processo="CTA",
        analise_do_conteudo_juridico="O voto discute a ADI 7228.",
    )

    assert should_replace_classe_processo("CTA", "ADI", row) is False


def test_dedupe_preview_rows_preserves_overlay_class_same_process_same_video():
    rows = [
        PublishPreviewRow(
            tema="Prestação de contas",
            numero_processo="262-19",
            classe_processo="PC",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=1620",
        ),
        PublishPreviewRow(
            tema="Embargos de declaração na prestação de contas",
            numero_processo="262-19",
            classe_processo="ED-PC",
            youtube_link="https://www.youtube.com/watch?v=abc123&t=1890",
        ),
    ]

    deduped = core.dedupe_preview_rows(rows, "https://www.youtube.com/watch?v=abc123")

    assert len(deduped) == 2
    assert {row.classe_processo for row in deduped} == {"PC", "ED-PC"}


def test_dedupe_preview_rows_prefers_richer_full_cnj_when_short_row_is_semantically_weaker():
    rows = [
        PublishPreviewRow(
            tema="Aprovação de resolução",
            classe_processo="PA",
            numero_processo="0600127-72",
            youtube_link="https://www.youtube.com/watch?v=FkjHl4xgbfQ&t=1029",
            relator="Min. Mauro Campbell Marques",
            origem="Lago do Junco/MA",
            analise_do_conteudo_juridico=(
                "O Tribunal aprovou a resolução nos termos do voto do relator."
            ),
            raciocinio_juridico=(
                "O colegiado deliberou pela aprovação unânime do texto proposto."
            ),
        ),
        PublishPreviewRow(
            tema="Adiamento de julgamento por falha técnica",
            classe_processo="REspe",
            numero_processo="0600127-72.2020.6.10.0000",
            youtube_link="https://www.youtube.com/watch?v=FkjHl4xgbfQ&t=1223",
            origem="Lago do Junco/MA",
            partes=["Recorrente: Francisca Josenita Soares de Arruda Moraes"],
            advogados=["Carlos Eduardo Barros"],
            analise_do_conteudo_juridico=(
                "O julgamento do recurso especial foi interrompido por falha técnica "
                "na sustentação oral da defesa."
            ),
            raciocinio_juridico=(
                "O relator submeteu ao colegiado o adiamento do julgamento após a "
                "instabilidade na conexão do advogado."
            ),
        ),
    ]

    deduped = core.dedupe_preview_rows(rows, "https://www.youtube.com/watch?v=FkjHl4xgbfQ")

    assert len(deduped) == 1
    row = deduped[0]
    assert row.numero_processo == "0600127-72.2020.6.10.0000"
    assert row.classe_processo == "REspe"
    assert row.tema == "Adiamento de julgamento por falha técnica"
    assert row.youtube_link == "https://www.youtube.com/watch?v=FkjHl4xgbfQ&t=1223"
    assert row.analise_do_conteudo_juridico.startswith("O julgamento do recurso especial")
    assert row.relator == ""


def test_punchline_looks_generic_flags_generic_aije_intro():
    row = PublishPreviewRow(
        tema="Integridade do sistema eletrônico de votação nas eleições de 2022",
        classe_processo="AIJE",
        numero_processo="0600814-85.2022.6.00.0000",
    )
    assert punchline_looks_generic(
        "O processo trata de uma Ação de Investigação Judicial Eleitoral (AIJE) proposta contra Jair Bolsonaro.",
        row,
    )


def test_infer_full_numero_processo_from_row_text_matches_existing_short_number():
    row = PublishPreviewRow(
        numero_processo="0600249-07",
        analise_do_conteudo_juridico="No processo 0600249-07.2024.6.02.0001 houve discussão sobre publicidade institucional.",
    )

    assert infer_full_numero_processo_from_row_text(row) == "0600249-07.2024.6.02.0001"
