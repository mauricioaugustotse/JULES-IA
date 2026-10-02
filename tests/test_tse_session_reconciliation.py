"""Regression cases for automatic session repairs and their refusal boundaries."""
from copy import deepcopy

import pytest

from tse_session_reconciliation import reconcile_session_rows


DAY = "2026-09-24"
PANEL = ["Nunes Marques", "André Mendonça", "Dias Toffoli", "Ricardo Villas Bôas Cueva",
         "Sebastião Reis Júnior", "Floriano de Azevedo Marques", "Estela Aranha"]


def process(number="0600957-40.2026.6.07.0000", **values):
    result = {"numeroProcesso": number, "situacaoProcesso": "Julgado", "relator": "DIAS TOFFOLI",
              "siglaClasseJudicial": "RO-El", "origem": "BRASÍLIA - DF",
              "proclamacaoDecisao": "O Tribunal, por maioria, negou provimento ao recurso.\n\nComposição: " + ", ".join(PANEL) + "."}
    result.update(values)
    return result


def inventory(*processes):
    return {"status": "available", "session_date": DAY, "processes": list(processes)}


def row(**values):
    result = {"numero_processo": "0600957-40.2026.6.07.0000", "numero_origem_video": "0600957-40",
              "data_sessao": DAY, "classe_processo": "RO", "origem": "Brasília/DF", "relator": "Min. Dias Toffoli",
              "resultado": "Desprovido", "votacao": "Por maioria", "composicao": ["Min. " + n for n in PANEL],
              "source_start_seconds": 7290, "source_bundle_index": 1, "source_item_index": 1,
              "partes": ["José Roberto Arruda"], "errors": [], "warnings": [],
              "analise_do_conteudo_juridico": "Recurso contra indeferimento de registro de candidatura por sete condenações de improbidade.",
              "punchline": "O TSE manteve o indeferimento do registro.", "raciocinio_juridico": "O relator votou pelo desprovimento."}
    result.update(values)
    return result


def evidence(hint="0600957-40", mentions=None):
    return {"session": {"data_sessao": DAY, "composicao": PANEL,
                        "judgments": [{"title_hint": hint, "start_seconds": 7290,
                                       "mentioned_process_numbers": mentions or [hint]}]},
            "bundles": [{"title_hint": hint, "start_seconds": 7290, "items": [{"composicao": PANEL}]}]}


def test_short_cnj_relator_and_substitute_are_repaired_with_audit_and_no_input_mutation():
    source = row(numero_processo="0600957-40", relator="Min. Floriano de Azevedo Marques",
                 composicao=["Min. Antônio Carlos Ferreira"],
                 errors=["[oficial:official_field_mismatch] Campo relator diverge do registro oficial.", "Bloqueio independente"])
    before = deepcopy(source)
    output, audit = reconcile_session_rows([source], inventory(process()))
    assert source == before
    assert output[0]["numero_processo"] == "0600957-40.2026.6.07.0000"
    assert output[0]["relator"] == "Min. Dias Toffoli"
    assert "Min. Sebastião Reis Júnior" in output[0]["composicao"]
    assert "Min. Antônio Carlos Ferreira" not in output[0]["composicao"]
    assert output[0]["errors"] == ["Bloqueio independente"]
    assert audit["matches"][0]["method"] == "unique_short_cnj"
    assert {"numero_processo", "composicao", "relator"} <= set(audit["matches"][0]["confirmed_fields"])
    assert "0600957-40" in audit["matches"][0]["aliases"]


def test_extra_zero_lt_needs_matching_class_relator_and_origin():
    source = row(numero_processo="", numero_origem_video="0600087771", classe_processo="Lista Tríplice",
                 relator="Min. André Mendonça", origem="Belo Horizonte/MG")
    official = process("0600877-71.2026.6.00.0000", relator="ANDRÉ MENDONÇA", siglaClasseJudicial="LT", origem="BELO HORIZONTE - MG")
    output, audit = reconcile_session_rows([source], inventory(official))
    assert output[0]["numero_processo"] == official["numeroProcesso"]
    assert audit["matches"][0]["method"] == "single_extra_zero"
    for field, wrong in (("relator", "Min. Dias Toffoli"), ("classe_processo", "PA"), ("origem", "João Pessoa/PB")):
        changed = {**source, field: wrong}
        rejected, rejected_audit = reconcile_session_rows([changed], inventory(official))
        assert rejected == [changed]
        assert rejected_audit["matches"] == []


def test_sparse_official_pa_uses_original_scan_number_and_actual_panel():
    source = row(numero_processo="0600113-04.2026.6.00.0000", numero_origem_video="0600113-04.2026.6.00.0000",
                 classe_processo="PA", origem="Doutor Severiano/RN", relator="Min. Nunes Marques",
                 resultado="Deferido", votacao="Unânime", composicao=["Min. Antônio Carlos Ferreira"])
    pa = process("0601130-04.2026.6.20.0000", relator="NUNES MARQUES", origem=None,
                 siglaClasseJudicial=None, proclamacaoDecisao=None, segredoJustica=True)
    output, audit = reconcile_session_rows([source], inventory(pa, process()), evidence=evidence("0600113004"))
    assert output[0]["numero_processo"] == pa["numeroProcesso"]
    assert output[0]["classe_processo"] == "PA"
    assert output[0]["origem"] == "Doutor Severiano/RN"
    assert output[0]["resultado"] == "Deferido"
    assert "Min. Sebastião Reis Júnior" in output[0]["composicao"]
    match = audit["matches"][0]
    assert "0600113004" in match["aliases"]
    assert "resultado" not in match["confirmed_fields"]  # sparse official record cannot confirm it
    assert "classe_processo" not in match["confirmed_fields"]


def test_pa_does_not_guess_a_zero_transposition_without_original_number_evidence():
    source = row(numero_processo="0600113-04.2026.6.00.0000", numero_origem_video="0600113-04.2026.6.00.0000",
                 relator="Min. Nunes Marques")
    output, audit = reconcile_session_rows([source], inventory(process("0601130-04.2026.6.20.0000", relator="NUNES MARQUES", origem=None, siglaClasseJudicial=None)))
    assert output == [source]
    assert not audit["matches"]


def test_ocr_evidence_from_another_session_is_ignored():
    source = row(numero_processo="0600113-04.2026.6.00.0000", numero_origem_video="0600113-04.2026.6.00.0000",
                 relator="Min. Nunes Marques")
    wrong_day = evidence("0600113004")
    wrong_day["session"]["data_sessao"] = "2026-09-17"
    output, audit = reconcile_session_rows([source], inventory(process("0601130-04.2026.6.20.0000", relator="NUNES MARQUES", origem=None, siglaClasseJudicial=None)), evidence=wrong_day)
    assert output == [source]
    assert not audit["matches"]


def test_valid_full_cnj_from_a_different_court_is_not_reassigned():
    digits_without_check = "0600957" + "2025" + "6" + "00" + "0000"
    check = 98 - (int(digits_without_check + "00") % 97)
    number = f"0600957-{check:02d}.2025.6.00.0000"
    source = row(numero_processo=number)
    other_court = number.replace(".6.00.", ".6.07.")
    output, audit = reconcile_session_rows([source], inventory(process(other_court)))
    assert output == [source]
    assert audit["unresolved"][0]["reason"] == "valid_cnj_not_in_session"


def test_ambiguous_duplicate_inventory_is_never_selected():
    source = row(numero_processo="0600957-40")
    output, audit = reconcile_session_rows([source], inventory(process(), process()))
    assert output == [source]
    assert audit["unresolved"][0]["reason"] == "ambiguous"


@pytest.mark.parametrize("change", [{"status": "unavailable"}, {"session_date": "2026-09-17"}, {"processes": None}])
def test_unavailable_or_other_session_cannot_repair(change):
    source = row(numero_processo="0600957-40")
    inv = inventory(process())
    inv.update(change)
    output, audit = reconcile_session_rows([source], inv)
    assert output == [source]
    assert not audit["matches"]


def test_pending_vista_repairs_labels_requester_and_false_collective_prose():
    p = process(situacaoProcesso="Não julgado", motivoRetiradaPauta="Pedido de vista", orgaoJulgadorVista="ANDRÉ MENDONÇA",
                proclamacaoDecisao="Iniciado o julgamento, o Relator votou pelo provimento do recurso.\n\nEm seguida, antecipou pedido de vista o Ministro André Mendonça.")
    source = row(resultado="Provido", pedido_vista="", punchline="O TSE deu provimento e indeferiu o registro.",
                 analise_do_conteudo_juridico="O TSE deu provimento ao recurso.")
    output, audit = reconcile_session_rows([source], inventory(p))
    fixed = output[0]
    assert (fixed["resultado"], fixed["votacao"], fixed["pedido_vista"]) == ("Suspenso por vista", "Suspenso", "Min. André Mendonça")
    assert "O TSE deu provimento" not in fixed["punchline"] + fixed["analise_do_conteudo_juridico"]
    assert "não foi concluído" in fixed["raciocinio_juridico"]
    assert "pedido_vista" in audit["matches"][0]["confirmed_fields"]
    again, again_audit = reconcile_session_rows(output, inventory(p))
    assert again == output
    assert not again_audit["corrections"]


def test_mixed_arruda_disposition_preserves_main_appeal_result_and_voting():
    p = process(proclamacaoDecisao="O Tribunal, por unanimidade, deu parcial provimento ao recurso de Cláudio; por maioria, negou provimento ao recurso de José Roberto Arruda.")
    output, audit = reconcile_session_rows([row()], inventory(p))
    assert output[0]["resultado"] == "Desprovido"
    assert output[0]["votacao"] == "Por maioria"
    assert "resultado" not in audit["matches"][0]["confirmed_fields"]
    assert "votacao" not in audit["matches"][0]["confirmed_fields"]


def test_cited_conviction_is_excluded_with_positive_relationship_evidence():
    main = row()
    citation = row(numero_processo="0013595-14", numero_origem_video="0013595-14", source_item_index=2,
                   analise_do_conteudo_juridico="Recurso contra indeferimento do registro de candidatura, baseado em condenação por improbidade na Secretaria de Educação.")
    original = [main, citation]
    output, audit = reconcile_session_rows(original, inventory(process()), evidence=evidence(mentions=["0600957-40", "0013595-14"]))
    assert len(output) == 1
    assert len(original) == 2
    assert audit["exclusions"][0]["row"] == citation
    assert audit["exclusions"][0]["parent_numero_processo"] == main["numero_processo"]
    assert audit["counts"]["unresolved"] == 0


@pytest.mark.parametrize("change", [
    {"partes": ["Outra pessoa"]},
    {"analise_do_conteudo_juridico": "Outro recurso autônomo de registro de candidatura."},
    {"source_start_seconds": 8000},
    {"source_item_index": 1},
])
def test_same_timestamp_or_missing_from_inventory_alone_never_excludes(change):
    citation = row(numero_processo="0013595-14", numero_origem_video="0013595-14", source_item_index=2,
                   analise_do_conteudo_juridico="Recurso de registro de candidatura baseado em condenação por improbidade.")
    citation.update(change)
    output, audit = reconcile_session_rows([row(), citation], inventory(process()), evidence=evidence(mentions=["0600957-40", "0013595-14"]))
    assert len(output) == 2
    assert not audit["exclusions"]


def test_independent_official_secondary_case_is_never_discarded_as_citation():
    citation = row(numero_processo="0013595-14", numero_origem_video="0013595-14", source_item_index=2,
                   analise_do_conteudo_juridico="Recurso de registro de candidatura baseado em condenação por improbidade.")
    output, audit = reconcile_session_rows([row(), citation], inventory(process(), process("0013595-14.2026.6.07.0000")),
                                          evidence=evidence(mentions=["0600957-40", "0013595-14"]))
    assert len(output) == 2
    assert not audit["exclusions"]


def test_disagreeing_session_panels_cannot_fill_sparse_record():
    source = row(numero_processo="0601130-04.2026.6.20.0000", relator="Min. Nunes Marques", composicao=["original"])
    pa = process("0601130-04.2026.6.20.0000", segredoJustica=True, proclamacaoDecisao=None)
    different = process("0600877-71.2026.6.00.0000", proclamacaoDecisao="Composição: " + ", ".join(PANEL).replace("Sebastião Reis Júnior", "Antônio Carlos Ferreira") + ".")
    output, audit = reconcile_session_rows([source], inventory(pa, process(), different), evidence=evidence())
    assert output[0]["composicao"] == ["original"]
    assert "composicao" not in audit["matches"][0]["confirmed_fields"]


def test_pydantic_rows_are_preserved_as_models():
    from pydantic import BaseModel

    class Row(BaseModel):
        numero_processo: str
        data_sessao: str

    source = Row(numero_processo="0600957-40", data_sessao=DAY)
    output, audit = reconcile_session_rows([source], inventory(process()))
    assert isinstance(output[0], Row)
    assert source.numero_processo == "0600957-40"
    assert output[0].numero_processo == "0600957-40.2026.6.07.0000"


def maceio_case():
    source = row(numero_processo="0601652-32.2026.6.00.0000", numero_origem_video="0601652-32.2026.6.00.0000",
                 classe_processo="PA", origem="Maceió/AL", relator="Min. Ricardo Villas Bôas Cueva", resultado="Procedente",
                 analise_do_conteudo_juridico="O relator, Min. Ricardo Villas Bôas Cueva, deferiu o pedido. O precedente de Ricardo Villas Bôas Cueva foi citado.")
    p = process("0601652-32.2026.6.02.0000", relator="NUNES MARQUES", siglaClasseJudicial="PA", origem="MACEIÓ - AL",
                proclamacaoDecisao="O Tribunal, por unanimidade, deferiu o pedido.")
    return source, p, evidence("PA 060165232", ["0601652-32"])


def test_invalid_suffix_and_relator_repaired_only_with_independent_short_scan_class_and_origin():
    source, p, proof = maceio_case()
    output, audit = reconcile_session_rows([source], inventory(p), evidence=proof)
    fixed = output[0]
    assert (fixed["numero_processo"], fixed["relator"], fixed["resultado"]) == (p["numeroProcesso"], "Min. Nunes Marques", "Deferido")
    assert "O relator, Min. Nunes Marques," in fixed["analise_do_conteudo_juridico"]
    assert "O precedente de Ricardo Villas Bôas Cueva foi citado." in fixed["analise_do_conteudo_juridico"]
    assert audit["matches"][0]["method"] == "invalid_cnj_suffix"
    again, repeat = reconcile_session_rows(output, inventory(p), evidence=proof)
    assert again == output
    assert not repeat["corrections"]


@pytest.mark.parametrize("failure", ["no_scan", "wrong_core", "full_scan", "wrong_date", "wrong_position", "wrong_class", "wrong_origin", "sparse_origin", "ambiguous"])
def test_scan_fallback_refuses_missing_or_conflicting_independent_evidence(failure):
    source, p, proof = maceio_case()
    if failure == "no_scan":
        proof = {}
    elif failure == "wrong_core":
        proof["session"]["judgments"][0]["mentioned_process_numbers"] = ["0601729-67"]
    elif failure == "full_scan":
        proof["session"]["judgments"][0]["mentioned_process_numbers"] = [source["numero_processo"]]
    elif failure == "wrong_date":
        proof["session"]["data_sessao"] = "2026-09-17"
    elif failure == "wrong_position":
        source["source_start_seconds"] += 1
    elif failure == "wrong_class":
        source["classe_processo"] = "RO"
    elif failure == "wrong_origin":
        source["origem"] = "Natal/RN"
    elif failure == "sparse_origin":
        p["origem"] = None
    inv = inventory(p, p) if failure == "ambiguous" else inventory(p)
    output, audit = reconcile_session_rows([source], inv, evidence=proof)
    assert output == [source]
    assert not audit["matches"]


def institutional_case():
    source = row(numero_processo="", numero_origem_video="", classe_processo="", relator="", partes=[], tipo_registro="Julgamento 1")
    proof = evidence("Eleição de Corregedor-Geral Eleitoral")
    proof["session"]["judgments"][0]["mentioned_process_numbers"] = []
    proof["bundles"][0]["items"][0] = {"analise_do_conteudo_juridico": "O TSE realizou a eleição interna para o cargo de Corregedor-Geral Eleitoral."}
    return source, proof


def test_proven_internal_election_is_excluded_and_numbering_closed_with_audit():
    source, proof = institutional_case()
    judgment = row(tipo_registro="Julgamento 2", source_start_seconds=8000)
    output, audit = reconcile_session_rows([source, judgment], inventory(process()), evidence=proof)
    assert len(output) == 1
    assert output[0]["tipo_registro"] == "Julgamento 1"
    assert audit["counts"]["excluded_institutional"] == 1
    assert audit["counts"]["excluded_citations"] == audit["counts"]["unresolved"] == 0
    assert audit["exclusions"][0]["row"] == source
    again, repeat = reconcile_session_rows(output, inventory(process()), evidence=proof)
    assert again == output
    assert not repeat["corrections"]


@pytest.mark.parametrize("failure", ["class", "number", "mentions", "original_number", "no_original_text", "wrong_date", "wrong_position"])
def test_internal_election_exclusion_preserves_possible_processes_and_requires_original_evidence(failure):
    source, proof = institutional_case()
    if failure == "class":
        source["classe_processo"] = "PA"
    elif failure == "number":
        source["numero_processo"] = "0601130-04"
    elif failure == "mentions":
        proof["session"]["judgments"][0]["mentioned_process_numbers"] = ["0601130-04"]
    elif failure == "original_number":
        proof["bundles"][0]["items"][0]["numero_processo"] = "0601130-04"
    elif failure == "no_original_text":
        proof["bundles"][0]["items"][0]["analise_do_conteudo_juridico"] = "Pedido administrativo julgado."
    elif failure == "wrong_date":
        proof["session"]["data_sessao"] = "2026-09-17"
    else:
        source["source_start_seconds"] += 1
    output, audit = reconcile_session_rows([source], inventory(process()), evidence=proof)
    assert len(output) == 1
    assert not audit["exclusions"]
