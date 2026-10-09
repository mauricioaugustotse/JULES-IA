"""Regressions from the 6 and 8 October sessions, with refusal boundaries."""
import pytest
from copy import deepcopy

from tse_official_session import _class, _official_decision
from tse_session_reconciliation import reconcile_session_rows
from tse_official_recovery import recover_ignored_windows


@pytest.mark.parametrize('label,expected', [
    ('ED no(a) AgR no(a) PC-PP', 'pc'), ('ED-AgR-PC', 'pc'),
    ('Ref-Rp', 'rp'), ('Inst', 'inst'), ('AREspE', 'arespe'),
])
def test_official_classes_keep_base_identity(label, expected):
    assert _class(label) == expected


def test_refusal_to_endorse_denial_preserves_grant_of_urgent_relief():
    p = {'proclamacaoDecisao': 'O Tribunal, por unanimidade, não referendou a decisão não concessiva da liminar e deferiu o pedido de tutela de urgência, para determinar a remoção das postagens.'}
    assert _official_decision(p) == ('deferido', 'unanime')
    p['proclamacaoDecisao'] = 'O Tribunal, por unanimidade, não referendou a decisão liminar.'
    assert _official_decision(p)[0] != 'referendada'


def test_preliminary_unanimity_does_not_replace_merits_majority():
    proclamation = 'O Tribunal, por unanimidade, rejeitou a preliminar e não homologou o pedido de desistência do recurso ordinário eleitoral, e, no mérito, por maioria, deu-lhe parcial provimento para afastar a causa de inelegibilidade, mantendo, contudo, o indeferimento do registro.'
    p = {'numeroProcesso': '0602359-26.2026.6.19.0000', 'situacaoProcesso': 'Julgado',
         'siglaClasseJudicial': 'RO-El', 'proclamacaoDecisao': proclamation}
    assert _official_decision(p) == ('parcialmente provido', 'por maioria')
    row = {'numero_processo': p['numeroProcesso'], 'data_sessao': '2026-10-08',
           'resultado': 'Desprovido', 'votacao': 'Unânime', 'errors': [], 'warnings': []}
    out, audit = reconcile_session_rows([row], {'status': 'available', 'session_date': '2026-10-08', 'processes': [p]})
    assert out[0]['resultado'] == 'Provido em parte'
    assert out[0]['votacao'] == 'Por maioria'
    assert {'resultado', 'votacao'} <= set(audit['matches'][0]['confirmed_fields'])


def test_distinct_appeals_remain_ambiguous():
    p = {'proclamacaoDecisao': 'O Tribunal, por unanimidade, deu parcial provimento ao recurso de A; e, no mérito, por maioria, deu-lhe parcial provimento ao recurso de B.'}
    assert _official_decision(p) == ('', '')


def test_invalid_cnj_cannot_cross_an_unrecognized_class_in_joint_judgment():
    row={'numero_processo':'0601915-02.2018.6.13.0000','data_sessao':'2019-09-10',
         'classe_processo':'REspe','origem':'Belo Horizonte/MG','relator':'Min. Tarcísio Vieira de Carvalho Neto'}
    process={'numeroProcesso':'0601915-02.2018.6.00.0000','siglaClasseJudicial':'AC',
             'origem':'BELO HORIZONTE - MG','relator':'TARCISIO VIEIRA DE CARVALHO NETO',
             'situacaoProcesso':'Julgado'}
    out,audit=reconcile_session_rows([row],{'status':'available','session_date':'2019-09-10','processes':[process]})
    assert out[0]['numero_processo']==row['numero_processo']
    assert not audit['matches']


@pytest.mark.parametrize('status,reason,recover', [
    ('Não julgado', 'Pedido de Vista', True), ('Julgado', None, True),
    ('Não julgado', 'Adiado', False), ('Retirado de julgamento', None, False),
])
def test_official_individual_judgment_overrides_scan_list_label(status, reason, recover):
    scan = {'data_sessao': '2026-10-06', 'judgments': [{
        'title_hint': 'Lista 1, 0600389-92', 'mentioned_process_numbers': ['0600389-92'],
        'start_seconds': 9316, 'end_seconds': 9388, 'should_ignore': True,
        'ignore_reason': 'Julgamento em lista'}]}
    inv = {'status': 'available', 'session_date': '2026-10-08', 'processes': [{
        'numeroProcesso': '0600389-92.2021.6.00.0000', 'situacaoProcesso': status,
        'motivoRetiradaPauta': reason}]}
    out, audit = recover_ignored_windows(scan, inv, confirmed_date='2026-10-08')
    assert out['judgments'][0]['should_ignore'] is not recover
    assert bool(audit['corrections']) is recover
    assert scan['judgments'][0]['should_ignore'] is True
    assert out['data_sessao'] == '2026-10-08'
    assert audit['date_correction']['before'] == '2026-10-06'
    assert audit['date_correction']['after'] == '2026-10-08'
    inv['processes'][0]['blocoJulgamento'] = 'Lista 1'
    inv['processes'][0]['situacaoProcesso'] = 'Julgado'
    out, _ = recover_ignored_windows(scan, inv, confirmed_date='2026-10-08')
    assert out['judgments'][0]['should_ignore'] is True
    out, _ = recover_ignored_windows(scan, inv, confirmed_date='2026-10-06')
    assert out == scan


def duplicate_context():
    prose = ('A formação da lista tríplice para preenchimento da vaga titular da classe advogado no Tribunal Regional Eleitoral '
             'considerou candidatos inscritos escolhidos unanimidade edital publicado participação feminina observância paridade gênero '
             'representatividade requisitos legais encaminhamento Poder Executivo escolha nomeação magistratura exercício biênio término '
             'candidatura habilitada existência composição complementar tramitação processo jurista mulheres homens equilíbrio indicações.')
    number = '0601075-11.2026.6.00.0000'
    row = {'numero_processo': number, 'data_sessao':'2026-10-06', 'classe_processo':'Lista Tríplice',
           'relator':'Min. Dias Toffoli', 'origem':'Natal/RN', 'source_item_index':1,
           'source_bundle_index':2, 'source_start_seconds':1620, 'analise_do_conteudo_juridico':prose}
    bad = {**row, 'numero_processo':'0600107-51.2026.6.20.0000', 'source_bundle_index':1,'source_start_seconds':1613}
    process = {'numeroProcesso':number, 'siglaClasseJudicial':'LT','relator':'DIAS TOFFOLI',
               'origem':'NATAL - RN','situacaoProcesso':'Julgado'}
    evidence = {'session': {'data_sessao':'2026-10-06','judgments':[
        {'start_seconds':1613},{'start_seconds':1620}]},
        'bundles':[{'start_seconds':1613,'end_seconds':1650}, {'start_seconds':1620,'end_seconds':1820}]}
    return [bad,row], {'status':'available','session_date':'2026-10-06','processes':[process]},evidence


def test_corrupt_overlapping_duplicate_does_not_create_a_second_judgment():
    rows, inv, evidence = duplicate_context()
    before = deepcopy(rows)
    output, audit = reconcile_session_rows(rows,inv,evidence=evidence)
    assert len(output)==1
    assert rows==before
    assert audit['exclusions'][0]['code']=='duplicate_extraction'
    assert audit['exclusions'][0]['row']==before[0]


@pytest.mark.parametrize('mutation', ['valid_number','separate_excerpt','different_origin','different_narrative','two_official_cases'])
def test_ambiguous_nearby_cases_are_not_suppressed(mutation):
    rows, inv, evidence = duplicate_context()
    if mutation=='valid_number': rows[0]['numero_processo']='0600502-70.2026.6.00.0000'
    if mutation=='separate_excerpt':
        rows[0]['source_start_seconds']=1500
        evidence['bundles'][0].update(start_seconds=1500,end_seconds=1600)
        evidence['session']['judgments'][0]['start_seconds']=1500
    if mutation=='different_origin': rows[0]['origem']='Salvador/BA'
    if mutation=='different_narrative': rows[0]['analise_do_conteudo_juridico']='Outra vaga e outros candidatos.'
    if mutation=='two_official_cases': inv['processes'].append({**inv['processes'][0],'numeroProcesso':'0600502-70.2026.6.00.0000'})
    output,audit=reconcile_session_rows(rows,inv,evidence=evidence)
    assert len(output)==2 and not audit['exclusions']


def test_registration_prose_cannot_invert_the_winning_party():
    text='O Tribunal, por unanimidade, negou provimento ao recurso do MPE, a fim de manter o deferimento do registro de candidatura.'
    process={'numeroProcesso':'0600317-21.2026.6.04.0000','situacaoProcesso':'Julgado','proclamacaoDecisao':text}
    row={'numero_processo':process['numeroProcesso'],'data_sessao':'2026-10-06',
         'resultado':'Desprovido','punchline':'O TSE manteve o indeferimento do registro de candidatura.',
         'analise_do_conteudo_juridico':'O candidato havia impugnado o indeferimento do registro em outra oportunidade.'}
    out,audit=reconcile_session_rows([row],{'status':'available','session_date':'2026-10-06','processes':[process]})
    assert out[0]['punchline']==text
    assert out[0]['analise_do_conteudo_juridico']==row['analise_do_conteudo_juridico']
    assert 'punchline' in audit['matches'][0]['confirmed_fields']


@pytest.mark.parametrize('prose', [
    'Inicialmente o relator negou provimento; após reajuste, houve parcial provimento.',
    'O TRE manteve o indeferimento do registro; o TSE deu parcial provimento ao recurso.',
    'Houve parcial provimento do recurso, com manutenção da inelegibilidade por outro fundamento.',
])
def test_partial_disposition_preserves_qualified_procedural_history(prose):
    process = {'numeroProcesso':'0602359-26.2026.6.19.0000','situacaoProcesso':'Julgado',
               'proclamacaoDecisao':'O Tribunal, no mérito, por maioria, deu-lhe parcial provimento para afastar a causa de inelegibilidade.'}
    row = {'numero_processo':process['numeroProcesso'],'data_sessao':'2026-10-08',
           'raciocinio_juridico':prose}
    out,_ = reconcile_session_rows([row],{'status':'available','session_date':'2026-10-08','processes':[process]})
    assert out[0]['raciocinio_juridico']==prose


@pytest.mark.parametrize('mutation,repaired', [
    ('none',True), ('valid_foreign_number',False), ('no_original_origin',False),
    ('other_original_relator',False), ('other_outcome',False), ('two_candidates',False),
])
def test_lt_devolution_requires_unique_original_context(mutation,repaired):
    source={'numero_processo':'0600599-88','data_sessao':'2026-10-08',
            'classe_processo':'Lista Tríplice','origem':'Salvador/BA',
            'relator':'Min. Ricardo Villas Bôas Cueva','resultado':'Devolvida',
            'source_start_seconds':1350,'source_bundle_index':1,'source_item_index':1}
    item={'origem':'Salvador/BA','relator':source['relator'],
          'resultado_final':'Devolvida','analise_do_conteudo_juridico':'Formação de lista tríplice para o TRE-BA.'}
    p={'numeroProcesso':'0601026-67.2026.6.00.0000','origem':'SALVADOR - BA',
       'relator':'RICARDO VILLAS BÔAS CUEVA','siglaClasseJudicial':'LT','situacaoProcesso':'Julgado',
       'proclamacaoDecisao':'O Tribunal, por unanimidade, determinou a devolução da lista tríplice.'}
    inv={'status':'available','session_date':'2026-10-08','processes':[p]}
    evidence={'session':{'data_sessao':'2026-10-08','judgments':[{'start_seconds':1350}]},
              'bundles':[{'start_seconds':1350,'items':[item]}]}
    if mutation=='valid_foreign_number': source['numero_processo']='0600502-70.2026.6.00.0000'
    if mutation=='no_original_origin': item['origem']=''
    if mutation=='other_original_relator': item['relator']='Min. Dias Toffoli'
    if mutation=='other_outcome': item['resultado_final']='Aprovada'
    if mutation=='two_candidates': inv['processes'].append({**p,'numeroProcesso':'0601027-52.2026.6.00.0000'})
    output,audit=reconcile_session_rows([source],inv,evidence=evidence)
    assert bool(audit['matches']) is repaired
    assert output[0]['numero_processo']==(p['numeroProcesso'] if repaired else source['numero_processo'])
