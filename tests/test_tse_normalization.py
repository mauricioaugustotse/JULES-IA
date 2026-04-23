from tse_normalization import (
    build_video_only_youtube_link,
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
    assert canonicalize_numero_processo("06007196") == "0600071-96"
    assert canonicalize_numero_processo("060061316874") == "0613168-74"
    assert canonicalize_numero_processo("ED na PC nº 26219") == "262-19"


def test_normalize_classe_processo_canonicalizes_known_alias():
    assert normalize_classe_processo("agravo regimental no recurso especial eleitoral") == "AgRg-REspe"


def test_normalize_resultado_final_normalizes_suspenso_por_vista():
    assert normalize_resultado_final("julgamento suspenso por pedido de vista") == "Suspenso por vista"
    assert normalize_resultado_final("Consulta respondida nos termos do voto do relator.") == "Aprovada"


def test_normalize_ministro_and_pedido_vista_names():
    assert normalize_ministro_name("Ministra Cármen Lúcia") == "Min. Cármen Lúcia"
    assert normalize_pedido_vista_value("Relator Ministro André Mendonça") == "Min. André Mendonça"
    assert (
        normalize_ministro_name("Luís Felipe Salomão (original), Raul Araújo (sucessor)")
        == "Min. Raul Araújo"
    )
    assert normalize_ministro_name("Min. Luís Felipe Salomão, Raul Araújo") == "Min. Luís Felipe Salomão"


def test_normalize_origem_and_date():
    assert normalize_origem_value("Porto Alegre - RS") == "Porto Alegre/RS"
    assert normalize_origem_value("São Gonçalo do Amarante, RN") == "São Gonçalo do Amarante/RN"
    assert normalize_origem_value("São Gonçalo do Amarante, Rio Grande do Norte") == "São Gonçalo do Amarante/RN"
    assert normalize_origem_value("Tribunal Regional Eleitoral do Ceará") == "TRE/CE"
    assert normalize_origem_value("Tribunal Regional Eleitoral de Sergipe/SE") == "TRE/SE"
    assert normalize_origem_value("Tribunal Superior Eleitoral") == "TSE"
    assert normalize_origem_value("Tribunal de Justiça de São Paulo/SP") == "São Paulo/SP"
    assert normalize_origem_value("Decisões do TRE/PR") == "TRE/PR"
    assert normalize_origem_value("Jurisprudência do TRE/PR") == "TRE/PR"
    assert normalize_origem_value("Municipal de Cascavel/CE") == "Cascavel/CE"
    assert normalize_origem_value("TSE/CE") == "TRE/CE"
    assert normalize_origem_value("TRE-SP") == "TRE/SP"
    assert normalize_origem_value("Titular do TRE/MS") == "TRE/MS"
    assert normalize_origem_value("Eleitoral de Macapá/AP") == "Macapá/AP"
    assert normalize_origem_value("92ª Zona Eleitoral de Araruama/RJ") == "Araruama/RJ"
    assert normalize_origem_value("Tribunais Regionais Eleitorais do Pará, Paraná/RJ") == ""
    assert normalize_origem_value("Tribunais Regionais Eleitorais de Ceará, Sergipe/MA") == ""
    assert normalize_origem_value("Distrito Federal, Rio Grande do Sul, Rio Grande do Norte, Acre, Amapá/RO") == ""
    assert normalize_origem_value("Amapá") == ""
    assert normalize_eleicao_value("Eleições 2024") == "2024"
    assert normalize_session_date_to_iso("20/03/2026") == "2026-03-20"


def test_normalize_advogados_and_remove_mpe():
    assert normalize_advogados_list("João da Silva e Maria Souza") == "Dr. João da Silva, Dra. Maria Souza"
    assert remove_mpe_from_partes("Alice, Ministério Público Eleitoral, Bob") == "Alice, Bob"


def test_normalize_advogados_list_parses_serialized_mapping_payload():
    value = """{'pelo_agravado': 'Dr. Alessandro Silverio, Dr. Vitor Augusto Esprada Rossetin'}"""
    assert normalize_advogados_list(value) == (
        "Dr. Alessandro Silverio (pelo agravado), "
        "Dr. Vitor Augusto Esprada Rossetin (pelo agravado)"
    )


def test_normalize_advogados_list_parses_loose_fragments_with_titles_before_keys():
    value = [
        "Dr. {'pelo_agravado': 'Dr. Alessandro Silverio",
        "Dr. Vitor Augusto Esprada Rossetin'}",
    ]
    assert normalize_advogados_list(value) == (
        "Dr. Alessandro Silverio (pelo agravado), "
        "Dr. Vitor Augusto Esprada Rossetin (pelo agravado)"
    )


def test_normalize_advogados_list_drops_empty_structured_values():
    value = [
        "Dr. {'advogado_do_agravante': ''",
        "Dr. 'advogado_do_agravado': 'João Guilherme Gualberto Torres'}",
    ]
    assert normalize_advogados_list(value) == "Dr. João Guilherme Gualberto Torres (advogado do agravado)"


def test_build_timestamped_youtube_link_canonicalizes_watch_url():
    assert (
        build_timestamped_youtube_link("https://youtu.be/abc123?t=5", 931)
        == "https://www.youtube.com/watch?v=abc123&t=931"
    )


def test_build_video_only_youtube_link_strips_timestamp():
    assert (
        build_video_only_youtube_link("https://youtu.be/abc123?t=5")
        == "https://www.youtube.com/watch?v=abc123"
    )


def test_normalize_numero_processo_display_strips_class_prefix_and_formats_short_number():
    assert normalize_numero_processo_display("REsp 60350714") == "603507-14"
    assert normalize_numero_processo_display("ADI 7228") == "ADI 7228"
    assert normalize_numero_processo_display("ADO 38") == "ADO 38"
    assert normalize_numero_processo_display("PC 26219") == "262-19"
    assert normalize_numero_processo_display("ED na PC nº 26219") == "262-19"
    assert normalize_numero_processo_display("060071-96") == "0600071-96"
    assert normalize_numero_processo_display("06007196") == "0600071-96"
    assert normalize_numero_processo_display("060061316874") == "0613168-74"


def test_normalize_numero_processo_display_rejects_textual_descriptors_without_digits():
    assert (
        normalize_numero_processo_display(
            "Recursos Ordinários de Luiz Augusto Barcelos Lara e Divaldo Vieira Lara"
        )
        == ""
    )
    assert (
        normalize_numero_processo_display(
            "Recurso Especial do Ministério Público Eleitoral e da Coligação Independência e Luta para Mudar o Rio Grande"
        )
        == ""
    )


def test_normalize_classe_processo_recognizes_adi_ado_and_force_federal():
    assert normalize_classe_processo("Ação Direta de Inconstitucionalidade (ADI)") == "ADI"
    assert normalize_classe_processo("Ação Direta de Inconstitucionalidade por Omissão") == "ADO"
    assert normalize_classe_processo("Homologação de requisições de Força Federal") == "PA"


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


def test_normalize_partes_list_splits_conjoined_person_names():
    assert normalize_partes_list("Luiz Augusto Barcelos Lara e Divaldo Vieira Lara") == (
        "Luiz Augusto Barcelos Lara, Divaldo Vieira Lara"
    )


def test_normalize_partes_list_does_not_split_organization_name_on_e():
    assert normalize_partes_list(
        "Coligação Independência e Luta para Mudar o Rio Grande"
    ) == "Coligação Independência e Luta para Mudar o Rio Grande"


def test_normalize_partes_list_splits_conjoined_names_with_shared_role_suffix():
    value = "José Auricchio Júnior (Prefeito eleito) e Carlos Humberto Seraphim (Vice-prefeito eleito) (Recorrido)"
    assert normalize_partes_list(value) == (
        "José Auricchio Júnior (Prefeito eleito) (Recorrido), "
        "Carlos Humberto Seraphim (Vice-prefeito eleito) (Recorrido)"
    )


def test_normalize_advogados_list_propagates_shared_suffix_after_conjunction_split():
    value = "Dr. João da Silva e Dra. Maria Souza (pelo recorrente)"
    assert normalize_advogados_list(value) == (
        "Dr. João da Silva (pelo recorrente), "
        "Dra. Maria Souza (pelo recorrente)"
    )
