"""Metadata artifacts must follow process identity when chapters reorder rows."""
import pytest

import tse_youtube_notion_core as core

URL = "https://www.youtube.com/watch?v=pNK-L5X6fXw"
A = "0601179-62.2026.6.05.0000"
B = "0601155-34.2026.6.05.0000"


def row(number, **updates):
    values = dict(
        numero_processo=number, classe_processo="REspe", data_sessao="2026-09-10",
        youtube_link=URL, tema=f"Controvérsia de {number}",
    )
    values.update(updates)
    return core.PublishPreviewRow(**values)


def test_swapped_rows_do_not_reuse_metadata_by_index_and_preserve_rejected_cache(tmp_path):
    store = core.RunArtifacts(tmp_path)
    cached = [row(A, origem="Origem antiga A/BA"), row(B, origem="Origem antiga B/BA")]
    for index, item in enumerate(cached, 1):
        store.write_json(f"04a_process_metadata_{index:02d}.json", {"applied": item.model_dump()})
        store.write_text(f"04a_process_metadata_{index:02d}.txt", f"old raw case {item.numero_processo}")
    obj = core.GeminiProcessMetadataEnricher.__new__(core.GeminiProcessMetadataEnricher)
    obj.artifact_store = store
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs["prompt"])
        number = B if "numero_processo: 0601155-34" in kwargs["prompt"] else A
        return core.ProcessMetadataResult(full_numero_processo=number, origem="Salvador/BA", is_judged_process=True)

    obj._call_grounded_json = resolve
    result = obj.enrich_rows([row("0601155-34"), row("0601179-62")])
    assert [item.numero_processo for item in result] == [B, A]
    assert [item.origem for item in result] == ["Salvador/BA", "Salvador/BA"]
    assert len(calls) == 2
    for index, original in enumerate(cached, 1):
        backup = f"04a_process_metadata_{index:02d}.identity_mismatch.json"
        assert store.read_json(backup)["applied"]["numero_processo"] == original.numero_processo
        assert (tmp_path / backup.replace(".json", ".txt")).read_text() == f"old raw case {original.numero_processo}"
        assert store.read_json(f"04a_process_metadata_{index:02d}.json")["applied"]["numero_processo"] != original.numero_processo


def test_completed_rows_ignore_swapped_cache_without_extra_grounding(tmp_path):
    store = core.RunArtifacts(tmp_path)
    store.write_json("04a_process_metadata_01.json", {"applied": row(A).model_dump()})
    obj = core.GeminiProcessMetadataEnricher.__new__(core.GeminiProcessMetadataEnricher)
    obj.artifact_store = store
    obj._call_grounded_json = lambda **kw: pytest.fail("Complete CNJ follows ordinary no-call path")
    result = obj.enrich_rows([row(B)])[0]
    assert result.numero_processo == B
    assert store.exists("04a_process_metadata_01.identity_mismatch.json")
    # Retrying without replacing the cache preserves one copy of the same evidence.
    obj.enrich_rows([row(B)])
    assert len(list(tmp_path.glob("*.identity_mismatch*.json"))) == 1


@pytest.mark.parametrize("changes", [
    {"data_sessao": "2026-09-17"},
    {"youtube_link": "https://www.youtube.com/watch?v=xeSt3PpR0xg"},
    {"numero_processo": "0601179-62.2024.6.05.0000"},
    {"numero_processo": ""},
    {"data_sessao": ""},
    {"youtube_link": ""},
])
def test_metadata_cache_requires_complete_matching_identity(changes):
    current = row(A)
    other = current.model_copy(update=changes)
    assert not core._metadata_cache_identity_matches(current, other, [current])


def test_unique_short_number_can_reuse_full_cnj_for_same_video_and_date():
    current = row("0601179-62")
    cached = row(A, youtube_link=URL + "&t=1578", data_sessao="10 de setembro de 2026")
    assert core._metadata_cache_identity_matches(current, cached, [current])


def test_ambiguous_short_number_cannot_select_between_full_cnjs():
    current = row("0601179-62")
    duplicate_core = row("0601179-62.2024.6.05.0000")
    assert not core._metadata_cache_identity_matches(current, row(A), [current, duplicate_core])
