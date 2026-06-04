from tse_normalization import (
    build_video_only_youtube_link,
    build_timestamped_youtube_link,
    canonicalize_numero_processo,
    composicao_regimental_issue,
    extract_youtube_video_id,
    is_regimentally_valid_composicao,
    normalize_advogados_list,
    normalize_classe_processo,
    normalize_eleicao_value,
    normalize_ministro_name,
    normalize_numero_processo_display,
    normalize_origem_value,
    normalize_partes_list,
    normalize_party_entry,
    normalize_pedido_vista_value,
    standardize_tribunal_party_name,
    normalize_resultado_final,
    normalize_session_date_to_iso,
    normalize_votacao,
    normalize_youtube_link,
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


def test_normalize_ministro_name_dedup_aliases_for_same_person():
    assert normalize_ministro_name("Min. Stella Aranha") == "Min. Estela Aranha"
    assert normalize_ministro_name("Min. André Luiz Mendonça") == "Min. André Mendonça"
    assert normalize_ministro_name("Min. Antônio Carlos") == "Min. Antônio Carlos Ferreira"


def test_standardize_tribunal_party_name_maps_state_to_uf():
    assert standardize_tribunal_party_name("Tribunal Regional Eleitoral de Sergipe (TRE-SE)") == "TRE/SE"
    assert standardize_tribunal_party_name("Tribunal Regional Eleitoral da Paraíba") == "TRE/PB"
    assert standardize_tribunal_party_name("Tribunal Regional Eleitoral do Pará") == "TRE/PA"
    assert standardize_tribunal_party_name("Tribunal Regional Eleitoral do Rio Grande do Norte") == "TRE/RN"
    assert standardize_tribunal_party_name("Tribunal Superior Eleitoral") == "TSE"
    # Corregedoria/Presidente não são o tribunal-parte e permanecem como estão.
    assert standardize_tribunal_party_name("Corregedoria Regional Eleitoral de São Paulo (CRE-SP)") == ""


def test_normalize_party_entry_drops_role_noise_and_standardizes_tribunal():
    assert normalize_party_entry("Candidato ao cargo de Deputado Estadual de Roraima nas eleições 2018") == ""
    assert normalize_party_entry("Deputado Federal e Vereador") == ""
    assert normalize_party_entry("Prefeito e Vice-Prefeito de Baixo Guandu/ES") == ""
    assert normalize_party_entry("Não mencionado") == ""
    assert normalize_party_entry("N/A") == ""
    assert normalize_party_entry("Tribunal Regional Eleitoral de Sergipe (TRE-SE)") == "TRE/SE"
    # entidades/pessoas legítimas são preservadas
    assert normalize_party_entry("Município de Governador Edison Lobão") == "Município de Governador Edison Lobão"
    assert normalize_party_entry("Prefeito João Silva") == "Prefeito João Silva"
    # autoridade coatora referenciando tribunal permanece (parte legítima)
    assert normalize_party_entry("Presidente do TRE-RR (autoridade coatora)") == "Presidente do TRE-RR (autoridade coatora)"


def test_normalize_classe_processo_preserves_agravo_regimental_official_abbrev():
    # 'AgR-' é a sigla oficial do TSE para agravo regimental e deve canonizar para a
    # forma 'AgRg-' usada na base; antes era reduzida indevidamente à classe-base.
    assert normalize_classe_processo("AgR-AREspe") == "AgRg-AREspe"
    assert normalize_classe_processo("AgR-REspe") == "AgRg-REspe"
    assert normalize_classe_processo("AgR-RO") == "AgRg-RO"
    # 'AgR-HC' é uma classe canônica própria da base e deve ser preservada como está.
    assert normalize_classe_processo("AgR-HC") == "AgR-HC"


def test_normalize_resultado_final_normalizes_suspenso_por_vista():
    assert normalize_resultado_final("julgamento suspenso por pedido de vista") == "Suspenso por vista"
    assert normalize_resultado_final("suspenso mas julgado depois") == "Suspenso mas julgado depois"
    assert normalize_votacao("Suspenso*") == "Suspenso*"
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
    assert normalize_origem_value("Tribunal Regional Eleitoral do Ceará") == "Fortaleza/CE"
    assert normalize_origem_value("Tribunal Regional Eleitoral de Sergipe/SE") == "Aracaju/SE"
    assert normalize_origem_value("Tribunal Superior Eleitoral") == "Brasília/DF"
    assert normalize_origem_value("Tribunal de Justiça de São Paulo/SP") == "São Paulo/SP"
    assert normalize_origem_value("Decisões do TRE/PR") == "Curitiba/PR"
    assert normalize_origem_value("Jurisprudência do TRE/PR") == "Curitiba/PR"
    assert normalize_origem_value("Municipal de Cascavel/CE") == "Cascavel/CE"
    assert normalize_origem_value("TSE/CE") == "Fortaleza/CE"
    assert normalize_origem_value("TRE-SP") == "São Paulo/SP"
    assert normalize_origem_value("Titular do TRE/MS") == "Campo Grande/MS"
    assert normalize_origem_value("Eleitoral de Macapá/AP") == "Macapá/AP"
    assert normalize_origem_value("92ª Zona Eleitoral de Araruama/RJ") == "Araruama/RJ"
    assert normalize_origem_value("Tribunais Regionais Eleitorais do Pará, Paraná/RJ") == ""
    assert normalize_origem_value("Tribunais Regionais Eleitorais de Ceará, Sergipe/MA") == ""
    assert normalize_origem_value("Distrito Federal, Rio Grande do Sul, Rio Grande do Norte, Acre, Amapá/RO") == ""
    assert normalize_origem_value("Amapá") == "Macapá/AP"
    assert normalize_origem_value("SP") == "São Paulo/SP"
    assert normalize_origem_value("Goiânia") == "Goiânia/GO"
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


def test_extract_youtube_video_id_reconcilia_formatos():
    # Concilia os dois formatos: /live/ (sessoes antigas do TSE) e watch?v= (recentes),
    # alem de youtu.be, /shorts/ e /embed/. Nenhum formato pode ser inviabilizado.
    vid = "KJbTq9hzG_s"
    casos = [
        f"https://www.youtube.com/watch?v={vid}",
        f"https://www.youtube.com/watch?v={vid}&t=120",
        f"https://youtu.be/{vid}",
        f"https://youtu.be/{vid}?t=5",
        f"https://www.youtube.com/live/{vid}?si=8o8rO-Aby7rvS2YY",
        f"https://www.youtube.com/live/{vid}",
        f"https://www.youtube.com/shorts/{vid}",
        f"https://www.youtube.com/embed/{vid}",
    ]
    for url in casos:
        assert extract_youtube_video_id(url) == vid, url


def test_normalize_youtube_link_reconcilia_live_e_watch():
    vid = "KJbTq9hzG_s"
    # /live/?si=... e watch?v= convergem para a mesma forma canonica
    assert (
        normalize_youtube_link(f"https://www.youtube.com/live/{vid}?si=abc")
        == f"https://www.youtube.com/watch?v={vid}"
    )
    assert (
        normalize_youtube_link(f"https://www.youtube.com/watch?v={vid}")
        == f"https://www.youtube.com/watch?v={vid}"
    )
    # preserva o timestamp quando presente
    assert (
        normalize_youtube_link(f"https://youtu.be/{vid}?t=931")
        == f"https://www.youtube.com/watch?v={vid}&t=931"
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


_COMPOSICAO_PLENA_3_2_2 = [
    "Min. Cármen Lúcia",
    "Min. André Mendonça",
    "Min. Nunes Marques",
    "Min. Isabel Gallotti",
    "Min. Antônio Carlos Ferreira",
    "Min. Floriano de Azevedo Marques",
    "Min. Ramos Tavares",
]


def test_composicao_regimental_issue_accepts_full_3_2_2_bench():
    assert composicao_regimental_issue(_COMPOSICAO_PLENA_3_2_2) == ""
    assert is_regimentally_valid_composicao(_COMPOSICAO_PLENA_3_2_2) is True


def test_composicao_regimental_issue_tolerates_single_unrostered_minister():
    # Fix #2a: 7 nomes com 1 ministro fora do roster (recem-empossado/substituto)
    # continua sendo uma bancada aproveitavel: nao deve disparar issue.
    bench = _COMPOSICAO_PLENA_3_2_2[:-1] + ["Min. Joaquim Pereira Lima"]
    assert composicao_regimental_issue(bench) == ""
    # Mas nao e uma plenaria regimental "perfeita" (3,2,2,0) para fins de auditoria.
    assert is_regimentally_valid_composicao(bench) is False


def test_composicao_regimental_issue_flags_two_or_more_unrostered():
    bench = _COMPOSICAO_PLENA_3_2_2[:-2] + [
        "Min. Joaquim Pereira Lima",
        "Min. Tadeu Soares Quintino",
    ]
    assert composicao_regimental_issue(bench) == "unknown_institution"


def test_composicao_regimental_issue_flags_size_bounds():
    assert composicao_regimental_issue(_COMPOSICAO_PLENA_3_2_2 + ["Min. Excedente Extra"]) == "gt7"
    assert composicao_regimental_issue(_COMPOSICAO_PLENA_3_2_2[:5]) == "lt6"


def test_composicao_regimental_issue_flags_category_excess():
    contaminated = [
        "Min. Rosa Weber",
        "Min. Luís Roberto Barroso",
        "Min. Jorge Mussi",
        "Min. Og Fernandes",
        "Min. Tarcísio Vieira de Carvalho Neto",
        "Min. Sérgio Banhos",
        "Min. Admar Gonzaga",
    ]
    assert composicao_regimental_issue(contaminated) == "category_excess"


def test_composicao_regimental_issue_accepts_valid_six_member_bench():
    assert composicao_regimental_issue(_COMPOSICAO_PLENA_3_2_2[:6]) == ""
    assert is_regimentally_valid_composicao(_COMPOSICAO_PLENA_3_2_2[:6]) is False
