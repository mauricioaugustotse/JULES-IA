"""Real historical API placeholders cannot replace an own signed decision."""
from copy import deepcopy
import pytest
from tse_official_session import normalize_official_session, compare_official_rows, _classification
from tse_official_recovery import recover_ignored_windows

DAY='2020-08-27'
NUMBER='0000002-64.2000.6.00.0000'


def entry(proclamation='Proclamação teste 1.'):
    return {'numeroProcesso':NUMBER,'situacaoProcesso':'Julgado',
            'siglaClasseJudicial':'PA','relator':'LUÍS ROBERTO BARROSO',
            'proclamacaoDecisao':proclamation,'blocoJulgamento':None}


def normalize(processes):
    return normalize_official_session([{'id':'TSE-1260','tribunal':'TSE',
        'virtual':'Não','dataSessao':DAY+'T10:00:00.000','processos':processes}],DAY)


@pytest.mark.parametrize('placeholder',['Proclamação teste 1.','PROCLAMAÇÃO TESTE 2','Proclamação teste','Proclamação teste1.'])
def test_explicit_placeholder_is_unknown_and_cannot_certify_coverage(placeholder):
    process=entry(placeholder)
    result=normalize([process])
    assert result['status']=='available'  # keep identity and provenance for own-source review
    assert result['processes'][0]['numeroProcesso']==NUMBER
    assert result['processes'][0]['classification']=='unknown'
    assert result['counts']['unknown']==1 and result['expected_count']==0
    issues=compare_official_rows(result,[])
    assert any(i['code']=='official_placeholder_decision' and i['coverage_status']=='unknown' for i in issues)
    assert not any(i['code']=='official_missing_row' for i in issues)


def test_cached_judged_classification_is_rechecked_before_scan_recovery():
    process=entry();process['classification']='judged'
    inventory={'status':'available','session_date':DAY,'processes':[process]}
    session={'data_sessao':DAY,'judgments':[{'mentioned_process_numbers':[NUMBER],
        'start_seconds':510,'end_seconds':570,'should_ignore':True,'ignore_reason':'Lista.'}]}
    before=deepcopy(session)
    recovered,report=recover_ignored_windows(session,inventory,confirmed_date=DAY)
    assert recovered==before and session==before and not report['corrections']
    issues=compare_official_rows(inventory,[])
    assert any(i['code']=='official_placeholder_decision' for i in issues)


@pytest.mark.parametrize('decision',[
    'O Tribunal, por unanimidade, aprovou a proposta, nos termos do voto do relator.',
    'O Tribunal aprovou a realização do teste.',
    'O relatório cita a expressão Proclamação teste 1., mas o Tribunal negou provimento.',
    None,
])
def test_legitimate_dispositions_and_missing_proclamation_keep_existing_classification(decision):
    assert _classification(entry(decision))=='judged'


def test_one_placeholder_does_not_discard_other_real_processes():
    own=entry('O Tribunal, por unanimidade, aprovou a proposta.');own['numeroProcesso']='0601330-76.2020.6.00.0000'
    result=normalize([entry(),own])
    assert len(result['processes'])==2 and result['expected_count']==1
    assert result['counts']['unknown']==1 and result['counts']['judged']==1
