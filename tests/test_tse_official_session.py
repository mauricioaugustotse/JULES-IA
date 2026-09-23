"""Offline regression cases for independent official-session reconciliation."""
from copy import deepcopy

import pytest

from tse_official_session import (
    INVENTORY_FILENAME,
    compare_official_rows,
    fetch_official_session,
    normalize_official_session,
)


DAY = "2026-09-17"
CNJ = "0600198-85.2024.6.02.0000"


def process(**kw):
    p = {
        "numeroProcesso": CNJ, "situacaoProcesso": "Julgado",
        "blocoJulgamento": None, "agrupadorOrgaoJulgador": False,
        "siglaClasseJudicial": "RvE", "classeJudicial": "REVISÃO DE ELEITORADO",
        "origem": "MINADOR DO NEGRÃO - AL", "relator": "ANTONIO CARLOS FERREIRA",
        "proclamacaoDecisao": "O Tribunal, por maioria, indeferiu o pedido, nos termos do voto do Relator.",
        "votantes": None,
    }
    p.update(kw)
    return p


def session(processes=None, **kw):
    s = {"id": "TSE-1", "tribunal": "TSE", "virtual": "Não", "dataSessao": DAY+"T10:00:00.000", "processos": [process()] if processes is None else processes}
    s.update(kw)
    return s


def row(**kw):
    r = {"numero_processo": CNJ, "data_sessao": DAY, "classe_processo": "RvE", "origem": "Minador do Negrão/AL", "relator": "Min. Antônio Carlos Ferreira", "resultado": "Indeferido", "votacao": "Por maioria"}
    r.update(kw)
    return r


def inventory(processes=None):
    return normalize_official_session([session(processes)], DAY)


def codes(issues):
    return [i["code"] for i in issues]


class Store:
    def __init__(self):
        self.files = {INVENTORY_FILENAME: {"status": "available", "session_date": "2020-01-01"}}

    def write_json(self, name, value):
        self.files[name] = deepcopy(value)


class HTTP:
    def __init__(self, payload=None, error=None):
        self.payload, self.error, self.calls = payload, error, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_fetch_is_fresh_date_bound_and_snapshotted():
    store, http = Store(), HTTP([session()])
    out = fetch_official_session(DAY, store, http)
    assert out["status"] == "available"
    assert out["session_date"] == DAY
    assert out["expected_count"] == 1
    assert http.calls[0][1]["params"] == {"sgTribunal": "TSE", "dataSessao": DAY}
    assert store.files[INVENTORY_FILENAME] == out


def test_failed_refresh_does_not_reuse_cache_or_leak_error_details():
    store = Store()
    out = fetch_official_session(DAY, store, HTTP(error=RuntimeError("secret-token")))
    assert out["status"] == "unavailable"
    assert out["expected_count"] is None
    assert "secret" not in out["error"]
    result = compare_official_rows(out, [])
    assert result[0]["code"] == "official_inventory_unavailable"
    assert result[0]["coverage_status"] == "unknown"


@pytest.mark.parametrize("change", [{"tribunal":"TRE-AL"}, {"virtual":"Sim"}, {"virtual":None}, {"dataSessao":"2026-09-15T10:00:00"}])
def test_no_wrong_tribunal_virtual_unknown_mode_or_wrong_date(change):
    out = normalize_official_session([session(**change)], DAY)
    assert out["status"] == "unavailable"
    assert out["expected_count"] is None


@pytest.mark.parametrize("raw", [None, {}, [session(processos=None)], [session(processos=[None])]])
def test_malformed_inventory_stays_unknown(raw):
    out = normalize_official_session(raw, DAY)
    assert out["status"] == "unavailable"
    assert compare_official_rows(out, [row()])[0]["coverage_status"] == "unknown"


def test_empty_confirmed_session_is_distinct_from_missing_session():
    out = normalize_official_session([session([])], DAY)
    assert out["status"] == "available"
    assert out["expected_count"] == 0
    assert normalize_official_session([], DAY)["status"] == "unavailable"


def test_individual_lista_triplice_not_collective_list_and_unknown_not_excluded():
    processes = [process(siglaClasseJudicial="LT", classeJudicial="LISTA TRÍPLICE"), process(numeroProcesso="0600104-40.2024.6.02.0000", blocoJulgamento="LISTA 1 - MFAM", agrupadorOrgaoJulgador=True), process(numeroProcesso="0600077-57.2024.6.02.0000", situacaoProcesso="Retirado de julgamento"), process(numeroProcesso="0600224-83.2024.6.02.0000", situacaoProcesso="Em julgamento")]
    out = inventory(processes)
    assert out["expected_count"] == 1
    assert out["counts"] == {"judged":1,"list":1,"withdrawn":1,"unknown":1}
    assert len(out["excluded"]) == 2
    issues = compare_official_rows(out, [])
    assert codes(issues).count("official_exclusion") == 2
    assert codes(issues).count("official_unknown_status") == 1
    assert codes(issues).count("official_missing_judgment") == 1


def test_unknown_collection_status_not_silently_excluded():
    out = inventory([process(blocoJulgamento="LISTA", situacaoProcesso="Pautado")])
    assert out["counts"]["unknown"] == 1
    assert not out["excluded"]
    assert "official_unknown_status" in codes(compare_official_rows(out, []))


def test_correct_row_has_no_issue_and_inputs_not_mutated():
    inv, rows = inventory(), [row()]
    backup = deepcopy((inv, rows))
    assert compare_official_rows(inv, rows) == []
    assert (inv, rows) == backup


@pytest.mark.parametrize("prefix", ["Ref-", "AgR-", "AgRg-", "ED-", "ED-AgRg-"])
def test_known_procedural_prefix_keeps_original_class(prefix):
    inv = inventory([process(siglaClasseJudicial="TutCautAnt")])
    assert not compare_official_rows(inv, [row(classe_processo=prefix + "TutCautAnt")])
    assert "official_field_mismatch" in codes(compare_official_rows(inv, [row(classe_processo="AgRg-REspe")]))


@pytest.mark.parametrize("proclamation, correct", [
    ("O Tribunal, por unanimidade, não conheceu do recurso.", "Não conhecido"),
    ("O Tribunal, por unanimidade, referendou a decisão liminar.", "Referendada"),
])
def test_nonknowledge_and_referendum_are_not_desprovido(proclamation, correct):
    inv = inventory([process(proclamacaoDecisao=proclamation)])
    assert not compare_official_rows(inv, [row(resultado=correct, votacao="Unânime")])
    issues = compare_official_rows(inv, [row(resultado="Desprovido", votacao="Unânime")])
    assert any(i.get("field") == "resultado" for i in issues)


def test_short_number_matches_and_gets_specific_completion_warning():
    result = compare_official_rows(inventory(), [row(numero_processo="0600198-85")])
    assert codes(result) == ["official_incomplete_process_number"]
    assert result[0]["expected"] == CNJ


def test_wrong_full_year_matched_by_core_is_identity_error_not_missing():
    result = compare_official_rows(inventory(), [row(numero_processo="0600198-85.2026.6.02.0000")])
    assert codes(result) == ["official_process_number_mismatch"]
    assert result[0]["severity"] == "error"


def test_duplicate_short_and_full_same_process_detected():
    result = compare_official_rows(inventory(), [row(), row(numero_processo="0600198-85")])
    duplicate = next(x for x in result if x["code"] == "official_duplicate_row")
    assert duplicate["row_indices"] == [0,1]


def test_foreign_row_does_not_satisfy_missing_judgment():
    result = compare_official_rows(inventory(), [row(numero_processo="0600077-57")])
    assert set(codes(result)) == {"official_unexpected_row", "official_missing_judgment"}


def test_same_core_multiple_official_identities_is_ambiguous():
    inv = inventory([process(), process(numeroProcesso="0600198-85.2026.6.02.0000")])
    result = compare_official_rows(inv, [row(numero_processo="0600198-85")])
    assert "official_ambiguous_identity" in codes(result)
    assert codes(result).count("official_missing_judgment") == 2


def test_wrong_row_date_does_not_satisfy_coverage():
    result = compare_official_rows(inventory(), [row(data_sessao="2026-09-15")])
    assert set(codes(result)) == {"official_session_date_mismatch", "official_missing_judgment"}


@pytest.mark.parametrize("overrides", [{"situacaoProcesso":"Retirado de julgamento"}, {"blocoJulgamento":"LISTA 1", "agrupadorOrgaoJulgador":True}])
def test_withdrawn_or_list_row_is_error_but_its_absence_is_normal_info(overrides):
    inv = inventory([process(**overrides)])
    absent = compare_official_rows(inv, [])
    assert codes(absent) == ["official_exclusion"]
    assert absent[0]["severity"] == "info"
    present = compare_official_rows(inv, [row()])
    assert "official_excluded_row" in codes(present)
    assert "official_missing_judgment" not in codes(present)


def test_critical_semantics_detect_suspension_wrong_city_relator_class():
    result = compare_official_rows(inventory(), [row(origem="Pindoba/AL", relator="Min. Estela Aranha", resultado="Suspenso por vista", votacao="Suspenso", classe_processo="PA")])
    assert {x["field"] for x in result} == {"origem", "relator", "resultado", "votacao", "classe_processo"}
    assert all(x["code"] == "official_field_mismatch" for x in result)


def test_known_class_aliases_and_name_accents():
    inv = inventory([process(siglaClasseJudicial="AREspE")])
    assert compare_official_rows(inv, [row(classe_processo="AREspe", relator="Ministro Antonio Carlos Ferreira")]) == []


def test_compound_agravo_and_appeal_not_flattened_but_voting_checked():
    p = process(proclamacaoDecisao="O Tribunal, por unanimidade, deu provimento ao agravo para conhecer parcialmente do recurso especial e negar provimento na parte conhecida.")
    result = compare_official_rows(inventory([p]), [row(resultado="Provido",votacao="Por maioria")])
    assert [x["field"] for x in result] == ["votacao"]


def test_procedural_and_merit_compound_is_not_inferred_from_first_verb():
    p = process(proclamacaoDecisao="O Tribunal, por maioria, acolheu o pedido de desabilitação e, no mérito, indeferiu o pedido de revisão.")
    assert compare_official_rows(inventory([p]), [row(resultado="Procedente")]) == []


def test_devolution_does_not_equal_approval():
    p = process(proclamacaoDecisao="O Tribunal, por unanimidade, determinou a devolução da lista tríplice ao TRE para recomposição.")
    result = compare_official_rows(inventory([p]), [row(resultado="Aprovada",votacao="Unânime")])
    assert result[0]["field"] == "resultado"
    assert result[0]["expected"] == "devolvido"


def test_subordinate_approved_deferment_keeps_main_dispositive():
    p = process(proclamacaoDecisao="O Tribunal, por unanimidade, aprovou o acórdão por meio do qual foi deferido o afastamento.")
    assert compare_official_rows(inventory([p]), [row(resultado="Aprovada",votacao="Unânime")]) == []


def test_secret_case_missing_official_fields_not_invented():
    p = process(segredoJustica=True, origem=None, siglaClasseJudicial=None, classeJudicial=None, proclamacaoDecisao=None, votantes=None)
    assert compare_official_rows(inventory([p]), [row(origem="Jundiá/AL",classe_processo="RvE",resultado="Procedente")]) == []


def test_unknown_official_name_or_class_not_guessed():
    p = process(siglaClasseJudicial="NOVACLASS", classeJudicial="NOVA CLASSE", relator=None)
    assert compare_official_rows(inventory([p]), [row(classe_processo="X",relator="X")]) == []


def test_preserved_historical_composition_overrides_voters_subset():
    p = process(proclamacaoDecisao="O Tribunal, por maioria, indeferiu o pedido.\n\nComposição do julgamento: Ministros (as) Nunes Marques (Presidente), Cármen Lúcia, André Mendonça, Antonio Carlos Ferreira, Ricardo Villas Bôas Cueva, Floriano de Azevedo Marques e Estela Aranha.")
    correct = ["Min. Nunes Marques", "Min. Cármen Lúcia", "Min. André Mendonça", "Min. Antônio Carlos Ferreira", "Min. Ricardo Villas Bôas Cueva", "Min. Floriano de Azevedo Marques", "Min. Estela Aranha"]
    assert compare_official_rows(inventory([p]), [row(composicao=correct)]) == []
    bad = ["Min. Dias Toffoli" if "Cármen" in n else n for n in correct]
    result = compare_official_rows(inventory([p]), [row(composicao=bad)])
    assert [x["field"] for x in result] == ["composicao"]


def test_voters_compare_only_complete_known_valid_names():
    names = ["NUNES MARQUES", "ANDRÉ MENDONÇA", "ANTONIO CARLOS FERREIRA", "RICARDO VILLAS BÔAS CUEVA", "FLORIANO DE AZEVEDO MARQUES", "ESTELA ARANHA"]
    voters = [{"nome":n,"impedido":False,"omisso":False} for n in names]
    p = process(votantes=voters)
    assert any(x.get("field")=="composicao" for x in compare_official_rows(inventory([p]), [row(composicao=[])]))
    p["votantes"][0]["nome"] = "Nome desconhecido"
    assert compare_official_rows(inventory([p]), [row(composicao=[])]) == []
    p["votantes"] = voters[:2]
    assert compare_official_rows(inventory([p]), [row(composicao=[])]) == []
