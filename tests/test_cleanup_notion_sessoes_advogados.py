from __future__ import annotations

from cleanup_notion_sessoes_advogados import (
    build_canonical_map,
    expand_advogado_value,
    sanitize_advogados_values,
)


def _expanded(value: str) -> list[str]:
    return [item.display for item in expand_advogado_value(value)]


def test_expand_advogado_value_removes_process_context_suffix() -> None:
    assert _expanded("Dr. José Eduardo Cardoso (pelo embargante)") == ["Dr. José Eduardo Cardoso"]
    assert _expanded("Dra. Maria Cláudia Bucchianeri (pela recorrida).") == ["Dra. Maria Cláudia Bucchianeri"]


def test_expand_advogado_value_splits_two_titled_lawyers_in_same_tag() -> None:
    assert _expanded("Dr. Luiz Fernando Pereira e Dr. Nicolau Dino (pela coligação)") == [
        "Dr. Luiz Fernando Pereira",
        "Dr. Nicolau Dino",
    ]


def test_expand_advogado_value_cleans_professor_title_and_all_caps() -> None:
    assert _expanded("Dr. Prof. JOSÉ EDUARDO CARDOZO") == ["Dr. José Eduardo Cardozo"]


def test_expand_advogado_value_fixes_common_feminine_title() -> None:
    assert _expanded("Dr. Karina Kufa") == ["Dra. Karina Kufa"]


def test_sanitize_advogados_values_prefers_full_canonical_name_for_safe_alias() -> None:
    values_by_page = [
        ["Dra. Maria Claudia Bucchianeri"],
        ["Dra. Maria Cláudia Bucchianeri Pinheiro (pela recorrida)"],
    ]
    canonical_by_key, root_by_key, _groups = build_canonical_map(values_by_page)

    repaired, reasons, _transformations = sanitize_advogados_values(
        ["Dra. Maria Claudia Bucchianeri (pela recorrente)"],
        canonical_by_key,
        root_by_key,
    )

    assert repaired == ["Dra. Maria Cláudia Bucchianeri Pinheiro"]
    assert "duplicidade falsa reduzida ao nome completo" in reasons


def test_sanitize_advogados_values_drops_empty_or_mpe_noise() -> None:
    canonical_by_key, root_by_key, _groups = build_canonical_map([["MPE"], ["Dr. João da Silva"]])

    repaired, reasons, _transformations = sanitize_advogados_values(
        ["MPE", "sem informação", "Dr. João da Silva (pelo recorrente)"],
        canonical_by_key,
        root_by_key,
    )

    assert repaired == ["Dr. João da Silva"]
    assert "etiqueta vazia ou espuria removida" in reasons
