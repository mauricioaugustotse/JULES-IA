from __future__ import annotations

import copy

from audit_notion_sessoes import PageRecord
from audit_notion_sessoes_round2 import (
    PageChangeSet,
    add_punchline_change,
    add_relation_changes,
    add_suspension_resolution_changes,
    build_relation_targets,
    normalized_partes_values,
    strict_origem_value,
)
from tse_youtube_notion_core import NotionDataSourceSchema, PublishPreviewRow


def _record(
    page_id: str,
    *,
    numero_processo: str = "0600001-00.2024.6.09.0000",
    data_sessao: str = "2024-01-01",
    video_id: str = "video-1",
    timestamp_seconds: int | None = 10,
    resultado: str = "Desprovido",
    votacao: str = "Unânime",
) -> PageRecord:
    return PageRecord(
        page={
            "id": page_id,
            "url": f"https://notion.test/{page_id}",
            "properties": {
                "materia_semelhante": {"type": "relation", "relation": []},
                "materia_semelhante1": {"type": "relation", "relation": []},
            },
        },
        row=PublishPreviewRow(
            tema="Propaganda eleitoral antecipada negativa",
            punchline="Punchline antiga.",
            numero_processo=numero_processo,
            data_sessao=data_sessao,
            youtube_link=f"https://www.youtube.com/watch?v={video_id}&t={timestamp_seconds or 0}",
            resultado=resultado,
            votacao=votacao,
        ),
        index=0,
        page_id=page_id,
        page_url=f"https://notion.test/{page_id}",
        video_id=video_id,
        timestamp_seconds=timestamp_seconds,
    )


def _relation_schema() -> NotionDataSourceSchema:
    return NotionDataSourceSchema(
        "ds",
        {
            "properties": {
                "tema": {"type": "title", "title": {}},
                "materia_semelhante": {"type": "relation", "relation": {}},
                "materia_semelhante1": {"type": "relation", "relation": {}},
            }
        },
    )


def test_strict_origem_replaces_organs_and_state_only_values_with_capitals() -> None:
    assert strict_origem_value(PublishPreviewRow(origem="TRE/GO"))[0] == "Goiânia/GO"
    assert strict_origem_value(PublishPreviewRow(origem="TSE"))[0] == "Brasília/DF"
    assert strict_origem_value(PublishPreviewRow(origem="SP"))[0] == "São Paulo/SP"
    assert strict_origem_value(PublishPreviewRow(origem="SE"))[0] == "Aracaju/SE"
    assert strict_origem_value(PublishPreviewRow(origem="Tribunal Regional Eleitoral de Sergipe"))[0] == "Aracaju/SE"
    assert strict_origem_value(PublishPreviewRow(origem="Goiânia"))[0] == "Goiânia/GO"
    assert strict_origem_value(PublishPreviewRow(origem="Tribunal Regional Federal"))[0] == "Brasília/DF"


def test_strict_origem_preserves_city_from_zona_eleitoral_and_adds_missing_uf_from_tribunal() -> None:
    row = PublishPreviewRow(origem="92ª Zona Eleitoral de Araruama/RJ")
    assert strict_origem_value(row)[0] == "Araruama/RJ"

    row = PublishPreviewRow(origem="Curitiba", tribunal="TRE-PR")
    assert strict_origem_value(row)[0] == "Curitiba/PR"


def test_partes_split_encavaladas_por_x() -> None:
    partes = normalized_partes_values(
        ["Ronaldo Barroso Tabosa dos Reis (recorrente) x Jeferson Anjos da Silva e outro (Recorrido)"]
    )

    assert partes == [
        "Ronaldo Barroso Tabosa dos Reis (Recorrente)",
        "Jeferson Anjos da Silva e outro (Recorrido)",
    ]


def test_suspension_rows_are_marked_when_same_process_is_later_decided() -> None:
    suspended = _record(
        "page-1",
        data_sessao="2024-01-01",
        resultado="Suspenso por vista",
        votacao="Suspenso",
    )
    decided = _record("page-2", data_sessao="2024-01-08", resultado="Desprovido", votacao="Unânime")
    change_sets = {
        suspended.page_id: PageChangeSet(record=copy.deepcopy(suspended)),
        decided.page_id: PageChangeSet(record=copy.deepcopy(decided)),
    }

    add_suspension_resolution_changes(change_sets)

    assert change_sets["page-1"].changes["votacao"].new == "Suspenso*"
    assert change_sets["page-1"].changes["resultado"].new == "Suspenso mas julgado depois"
    assert not change_sets["page-2"].changes


def test_relation_targets_link_distinct_sessions_of_same_process() -> None:
    first = _record("page-1", data_sessao="2024-01-01", video_id="video-1")
    same_session = _record("page-2", data_sessao="2024-01-01", video_id="video-1")
    later = _record("page-3", data_sessao="2024-01-08", video_id="video-2")

    targets = build_relation_targets([first, same_session, later])

    assert targets["page-1"] == ["page-3"]
    assert targets["page-2"] == ["page-3"]
    assert targets["page-3"] == ["page-1", "page-2"]


def test_relation_changes_update_both_relation_columns_when_present() -> None:
    first = _record("page-1", data_sessao="2024-01-01", video_id="video-1")
    later = _record("page-2", data_sessao="2024-01-08", video_id="video-2")
    change_sets = {
        first.page_id: PageChangeSet(record=copy.deepcopy(first)),
        later.page_id: PageChangeSet(record=copy.deepcopy(later)),
    }

    add_relation_changes(change_sets, _relation_schema(), {"page-1": ["page-2"]})

    assert change_sets["page-1"].changes["materia_semelhante"].new == ["page-2"]
    assert change_sets["page-1"].changes["materia_semelhante1"].new == ["page-2"]


def test_punchline_exclusion_preserves_previous_round_pages() -> None:
    record = _record("page-1")
    change_set = PageChangeSet(record=copy.deepcopy(record))

    add_punchline_change(change_set, {"page-1"})

    assert not change_set.changes
