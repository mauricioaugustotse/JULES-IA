from tse_normalization import (
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    normalize_advogados_list,
    normalize_classe_processo,
    normalize_eleicao_value,
    normalize_ministro_name,
    normalize_numero_processo_display,
    normalize_origem_value,
    normalize_partes_list,
    normalize_pedido_vista_value,
    normalize_resultado_final,
    normalize_session_date_to_iso,
    remove_mpe_from_partes,
)


def test_canonicalize_numero_processo_extracts_short_number():
    assert (
        canonicalize_numero_processo("0600249-07.2024.6.13.0000")
        == "0600249-07"
    )
    assert canonicalize_numero_processo("060036879") == "0600368-79"


def test_normalize_classe_processo_canonicalizes_known_alias():
    assert normalize_classe_processo("agravo regimental no recurso especial eleitoral") == "AgRg-REspe"


def test_normalize_resultado_final_normalizes_suspenso_por_vista():
    assert normalize_resultado_final("julgamento suspenso por pedido de vista") == "Suspenso por vista"


def test_normalize_ministro_and_pedido_vista_names():
    assert normalize_ministro_name("Ministra Cármen Lúcia") == "Min. Cármen Lúcia"
    assert normalize_pedido_vista_value("Relator Ministro André Mendonça") == "Min. André Mendonça"


def test_normalize_origem_and_date():
    assert normalize_origem_value("Porto Alegre - RS") == "Porto Alegre/RS"
    assert normalize_origem_value("São Gonçalo do Amarante, RN") == "São Gonçalo do Amarante/RN"
    assert normalize_origem_value("São Gonçalo do Amarante, Rio Grande do Norte") == "São Gonçalo do Amarante/RN"
    assert normalize_origem_value("Tribunal Regional Eleitoral do Ceará") == "TRE/CE"
    assert normalize_origem_value("Tribunal Regional Eleitoral de Sergipe/SE") == "TRE/SE"
    assert normalize_origem_value("TRE-SP") == "TRE/SP"
    assert normalize_eleicao_value("Eleições 2024") == "2024"
    assert normalize_session_date_to_iso("20/03/2026") == "2026-03-20"


def test_normalize_advogados_and_remove_mpe():
    assert normalize_advogados_list("João da Silva e Maria Souza") == "Dr. João da Silva, Dra. Maria Souza"
    assert remove_mpe_from_partes("Alice, Ministério Público Eleitoral, Bob") == "Alice, Bob"


def test_build_timestamped_youtube_link_canonicalizes_watch_url():
    assert (
        build_timestamped_youtube_link("https://youtu.be/abc123?t=5", 931)
        == "https://www.youtube.com/watch?v=abc123&t=931"
    )


def test_normalize_numero_processo_display_strips_class_prefix_and_formats_short_number():
    assert normalize_numero_processo_display("REsp 60350714") == "603507-14"


def test_normalize_partes_list_parses_serialized_mapping_payload():
    value = """{'embargante': 'Cláudia Aparecida dos Santos', 'embargados': ['Denilson Aparecido Martins', 'Federação Brasil da Esperança de Santa Luzia']}"""
    assert normalize_partes_list(value) == (
        "Cláudia Aparecida dos Santos (Embargante), "
        "Denilson Aparecido Martins (Embargado), "
        "Federação Brasil da Esperança de Santa Luzia (Embargado)"
    )


def test_normalize_partes_list_parses_loose_label_value_payload():
    value = "'agravado': 'Jânio Natal Andrade Borges'}"
    assert normalize_partes_list(value) == "Jânio Natal Andrade Borges (Agravado)"


def test_normalize_partes_list_moves_impetrado_role_to_suffix_without_degrading_acronym():
    value = "Impetrado: Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ)"
    assert normalize_partes_list(value) == "Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ) (Impetrado)"


def test_normalize_partes_list_does_not_reapply_first_role_to_following_items():
    value = [
        "Impetrante: Diego Fernandes da Silva",
        "Impetrado: Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ)",
        "Damiana Sidneia Oliveira e outros (Interessado)",
    ]
    assert normalize_partes_list(value) == (
        "Diego Fernandes da Silva (Impetrante), "
        "Tribunal Regional Eleitoral do Rio de Janeiro (TRE-RJ) (Impetrado), "
        "Damiana Sidneia Oliveira e outros (Interessado)"
    )


def test_normalize_partes_list_expands_plural_loose_mapping_to_all_names():
    value = [
        "'agravados': 'Caio Faria Donatelli",
        "César Lima de Nascimento",
        "Maurícia Marciel Pessanha'}",
    ]
    assert normalize_partes_list(value) == (
        "Caio Faria Donatelli (Agravado), "
        "César Lima de Nascimento (Agravado), "
        "Maurícia Marciel Pessanha (Agravado)"
    )


def test_normalize_partes_list_drops_placeholder_names():
    value = ["Não especificado (Recorrente)", "Recorrido: Não informado"]
    assert normalize_partes_list(value) == ""
