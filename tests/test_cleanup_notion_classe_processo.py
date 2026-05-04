from cleanup_notion_classe_processo import (
    explicit_current_cleanup_proposal,
    infer_classe_processo_for_cleanup,
    should_apply_proposal,
)
from tse_youtube_notion_core import PublishPreviewRow


def test_classe_cleanup_infere_rpp_sem_transformar_mencao_a_adi_em_classe() -> None:
    row = PublishPreviewRow(
        numero_processo="0000843-68",
        analise_do_conteudo_juridico=(
            "O Partido Novo apresentou pedido de registro de alteracoes estatutarias. "
            "A relatora mencionou a ADI 5875 apenas como precedente do STF."
        ),
    )

    proposal = infer_classe_processo_for_cleanup(row)

    assert proposal is not None
    assert proposal.value == "RPP"


def test_classe_cleanup_infere_cta_em_consulta_com_mencao_a_adi() -> None:
    row = PublishPreviewRow(
        numero_processo="0600415-27",
        analise_do_conteudo_juridico=(
            "O processo trata de uma consulta formulada por partido nacional. "
            "O relator destacou que o tema tambem e objeto da ADI 6374 no STF."
        ),
    )

    proposal = infer_classe_processo_for_cleanup(row)

    assert proposal is not None
    assert proposal.value == "CTA"


def test_classe_cleanup_infere_pa_para_instrucao_normativa() -> None:
    row = PublishPreviewRow(
        numero_processo="0600748-13.2019.6.00.0000",
        analise_do_conteudo_juridico=(
            "O processo trata de instrucao para regulamentacao permanente dos procedimentos "
            "relativos a escolha e ao registro de candidaturas."
        ),
    )

    proposal = infer_classe_processo_for_cleanup(row)

    assert proposal is not None
    assert proposal.value == "PA"


def test_classe_cleanup_prioriza_tutela_cautelar_sobre_mencao_incidental_a_ms() -> None:
    row = PublishPreviewRow(
        numero_processo="0613339-31",
        analise_do_conteudo_juridico=(
            "O requerente interpos tutela cautelar antecedente com pedido de efeito suspensivo. "
            "O voto mencionou mandado de seguranca apenas em precedente correlato."
        ),
    )

    proposal = infer_classe_processo_for_cleanup(row)

    assert proposal is not None
    assert proposal.value == "TutCautAnt"


def test_classe_cleanup_prioriza_rcand_em_registro_presidencial_com_drap_citado() -> None:
    row = PublishPreviewRow(
        numero_processo="0600852-39.2018.6.00.0000",
        analise_do_conteudo_juridico=(
            "O julgamento trata do pedido de registro de candidatura aos cargos de Presidente "
            "e Vice-Presidente da Republica. O relator tambem deferiu o DRAP."
        ),
    )

    proposal = infer_classe_processo_for_cleanup(row)

    assert proposal is not None
    assert proposal.value == "RCand"


def test_classe_cleanup_nao_infere_adi_apenas_por_acao_do_stf() -> None:
    row = PublishPreviewRow(
        numero_processo="ADI 7228",
        analise_do_conteudo_juridico=(
            "O julgamento trata de Acoes Diretas de Inconstitucionalidade ADI 7228, "
            "7263 e 7325, cuja materia de fundo sera julgada pelo STF."
        ),
    )

    assert infer_classe_processo_for_cleanup(row) is None


def test_classe_cleanup_limpa_adi_quando_nao_ha_classe_segura() -> None:
    should_apply, new_value, reason, confidence, _evidence = should_apply_proposal("ADI", None)

    assert should_apply is True
    assert new_value == ""
    assert "ADI/ADO" in reason
    assert confidence == "review"


def test_classe_cleanup_remove_pc_quando_contas_e_mencao_incidental_em_registro() -> None:
    row = PublishPreviewRow(
        tema="Deferimento de registro de candidatura por filiacao partidaria",
        analise_do_conteudo_juridico="O recurso discute registro de candidatura e comprovacao de filiacao partidaria.",
        raciocinio_juridico="A alegacao de ausencia de prestacao de contas foi considerada inovacao recursal.",
    )

    proposal = explicit_current_cleanup_proposal(row, "PC")
    should_apply, new_value, reason, confidence, _evidence = should_apply_proposal("PC", proposal)

    assert should_apply is True
    assert new_value == ""
    assert "PC removida" in reason
    assert confidence == "review"
