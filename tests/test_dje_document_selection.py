"""Regression: morning relief and evening withdrawal share CNJ and date."""
import pytest
from import_dje_faltantes import resultado_votacao_from_dispositivo
from manutencao_sessoes.dje_document_selection import choose_document


def choose(docs, result='Deferido', cls='Rec-DR', window=False):
    return choose_document(docs, '2022-10-28', result, cls,
                           resultado_votacao_from_dispositivo, allow_window=window)


def doc(text, **extra):
    return {'id': 'source', 'data': '2022-10-28', 'decisao': text, 'ementa': '', **extra}


def test_same_day_evening_homologation_cannot_fill_morning_relief():
    evening = doc('O Tribunal, por unanimidade, homologou o pedido de desistência de todos os pedidos de direito de resposta relacionados à televisão, nos termos acordados pelas partes.')
    assert choose([evening])[0] is None


def test_selects_morning_source_even_when_evening_source_is_first():
    evening = doc('O Tribunal, por unanimidade, homologou o pedido de desistência.', id='evening')
    morning = doc('O Tribunal, por maioria, deferiu o pedido de efeito suspensivo ao recurso, nos termos do voto do Ministro Ricardo Lewandowski.', id='morning')
    selected, reason = choose([evening, morning])
    assert selected['id'] == 'morning'
    assert reason == 'data_exata_resultado_e_fase_conferem'


def test_same_day_unknown_outcome_is_not_an_automatic_match():
    assert choose([doc('O Tribunal deliberou conforme o voto do relator.')])[0] is None


def test_same_day_embargos_cannot_fill_principal_resource():
    source = doc('O Tribunal, por unanimidade, negou provimento aos embargos de declaração.', classe='ED-REspe')
    assert choose([source], result='Desprovido', cls='REspe')[0] is None
    assert choose([source], result='Desprovido', cls='ED-REspe')[0] is source


def test_two_different_same_day_decisions_with_same_outcome_require_review():
    first = doc('O Tribunal deferiu o pedido de efeito suspensivo.', id='first')
    second = doc('O Tribunal deferiu a tutela incidental em novo pedido.', id='second')
    assert choose([first, second]) == (None, 'acordaos_distintos_exigem_revisao_de_fase')


def test_identical_source_rows_can_share_one_verified_document():
    first = doc('O Tribunal deferiu o pedido de efeito suspensivo.', id='first')
    second = {**first, 'id': 'second'}
    assert choose([first, second])[0] is first


def test_window_keeps_required_outcome_and_phase_checks():
    source = doc('O Tribunal negou provimento ao recurso.', data='2022-10-29', classe='REspe')
    assert choose([source], result='Desprovido', cls='REspe')[0] is None
    assert choose([source], result='Desprovido', cls='REspe', window=True)[0] is source
    assert choose([source], result='Provido', cls='REspe', window=True)[0] is None


@pytest.mark.parametrize('date', ['', 'invalid', '2022-99-99'])
def test_invalid_source_dates_do_not_match(date):
    assert choose([doc('O Tribunal deferiu o pedido.', data=date)])[0] is None
