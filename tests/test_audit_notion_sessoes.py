from __future__ import annotations

import copy

from audit_notion_sessoes import (
    PageChangeSet,
    PageRecord,
    add_tipo_registro_changes,
    better_theme,
    candidate_looks_bad_punchline,
    canonical_ministro,
)
from tse_youtube_notion_core import PublishPreviewRow


def _record(page_id: str, youtube_link: str, tipo_registro: str, index: int) -> PageRecord:
    return PageRecord(
        page={"id": page_id, "url": f"https://notion.test/{page_id}", "properties": {}},
        row=PublishPreviewRow(
            tema="Propaganda eleitoral antecipada",
            tipo_registro=tipo_registro,
            youtube_link=youtube_link,
            data_sessao="2024-01-01",
            numero_processo=f"060000{index}-00",
        ),
        index=index,
        page_id=page_id,
        page_url=f"https://notion.test/{page_id}",
        video_id="video1",
        timestamp_seconds=None if "&t=" not in youtube_link else int(youtube_link.rsplit("&t=", 1)[1]),
    )


def test_tipo_registro_resequences_by_timestamp_and_places_missing_timestamp_last() -> None:
    records = [
        _record("a", "https://www.youtube.com/watch?v=video1&t=200", "Julgamento 9", 0),
        _record("b", "https://www.youtube.com/watch?v=video1", "Julgamento 1", 1),
        _record("c", "https://www.youtube.com/watch?v=video1&t=100", "Julgamento 3", 2),
    ]
    change_sets = {record.page_id: PageChangeSet(record=copy.deepcopy(record)) for record in records}

    warnings = add_tipo_registro_changes(change_sets, [item.record for item in change_sets.values()])

    assert change_sets["c"].changes["tipo_registro"].new == "Julgamento 1"
    assert change_sets["a"].changes["tipo_registro"].new == "Julgamento 2"
    assert change_sets["b"].changes["tipo_registro"].new == "Julgamento 3"
    assert warnings and warnings[0]["page_id"] == "b"


def test_canonical_ministro_rejects_sentence_capture_and_repairs_double_min() -> None:
    assert canonical_ministro("Min. Min Cármen Lúcia") == "Min. Cármen Lúcia"
    assert canonical_ministro("Min. apresentou o voto sobre o encaminhamento") == ""


def test_better_theme_does_not_replace_citation_with_generic_vista_theme() -> None:
    row = PublishPreviewRow(
        tema="ADI 4583, ADI 5398",
        punchline="Julgamento adiado por pedido de vista.",
        analise_do_conteudo_juridico="O julgamento foi adiado por pedido de vista, sem exposição segura do núcleo temático.",
    )

    assert better_theme(row) == ""


def test_candidate_looks_bad_punchline_rejects_bare_citation() -> None:
    row = PublishPreviewRow(tema="Propaganda eleitoral antecipada")

    assert candidate_looks_bad_punchline("Súmula 24 do TSE.", row)
    assert candidate_looks_bad_punchline("2º da Resolução TSE nº 23.458/2017.", row)
