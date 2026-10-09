"""Regressoes do contrato publico SADP Consulta, conferido em 09/10/2026.

LT 51082 (protocolo 188792015) tem 212 andamentos; a API limita size a 50.
ADVOGADO INDICADO vem com advogado=false e designa integrante da lista triplice.
"""
import copy

import pytest
import requests

import sadp_lookup as sadp


class Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def paged(items, key="listaProcessos", *, size=50):
    total = len(items)
    pages = (total + size - 1) // size
    return [Response({"sucesso": True, "mensagem": None, key: items[i*size:(i+1)*size],
                      "totalElements": total, "totalPages": pages, "page": i, "size": size})
            for i in range(max(1, pages))]


@pytest.fixture(autouse=True)
def no_delay(monkeypatch):
    monkeypatch.setattr(sadp.time, "sleep", lambda _: None)


def process(i):
    return {"numeroUnico": "0000510-82.2015.6.00.0000", "numeroProcesso": "51082",
            "numeroProtocolo": str(i), "siglaClasse": "LT", "nomeCidade": "GOIÂNIA",
            "siglaEstado": "GO", "ultimaSituacao": "Baixado"}


def test_search_keeps_all_protocols_even_when_cnj_repeats():
    session = Session(paged([process(i) for i in range(101)]))
    result = sadp.search_number(session, "51082")
    assert len(result) == 101
    assert {r["nprot"] for r in result} == {str(i) for i in range(101)}
    assert [call[1]["params"] for call in session.calls] == [
        {"page": i, "size": 50} for i in range(3)]


def test_exact_number_search_uses_all_pages_too():
    session = Session(paged([process(i) for i in range(51)]))
    assert len(sadp.search_numunico(session, "0000510-82.2015.6.00.0000")) == 51
    assert "numeroUnico/00005108220156000000" in session.calls[0][0]


def test_successful_empty_search_remains_distinct_from_failure():
    assert sadp.search_number(Session(paged([])), "51082") == []
    with pytest.raises(sadp.SADPQueryError, match="HTTP 503"):
        sadp.search_number(Session([Response({}, status=503)]), "51082")


@pytest.mark.parametrize("failure", [Response({}, status=503), requests.Timeout(),
                                     Response(ValueError("bad json")), Response({"sucesso": False})])
def test_failure_after_first_page_never_returns_partial_candidates(failure):
    responses = paged([process(i) for i in range(51)])
    with pytest.raises(sadp.SADPQueryError) as caught:
        sadp.search_number(Session([responses[0], failure]), "51082")
    assert caught.value.audit["retrieved"] == 50
    assert caught.value.audit["expected_total"] == 51
    assert caught.value.audit["complete"] is False


@pytest.mark.parametrize("change", [
    {"page": 0}, {"totalElements": 52}, {"totalPages": 3},
    {"size": 10}, {"listaProcessos": []}, {"sucesso": False},
])
def test_inconsistent_pagination_fails_closed(change):
    responses = paged([process(i) for i in range(51)])
    responses[1].payload.update(change)
    with pytest.raises(sadp.SADPQueryError):
        sadp.search_number(Session(responses), "51082")


def test_repeated_page_is_not_accepted_as_additional_protocols():
    responses = paged([process(i) for i in range(100)])
    responses[1].payload["listaProcessos"] = copy.deepcopy(responses[0].payload["listaProcessos"])
    with pytest.raises(sadp.SADPQueryError, match="repetida"):
        sadp.search_number(Session(responses), "51082")


def test_partially_overlapping_process_pages_never_return_incomplete_candidates():
    responses = paged([process(i) for i in range(51)])
    responses[1].payload["listaProcessos"] = [process(0)]
    with pytest.raises(sadp.SADPQueryError, match="numeroProtocolo=0") as caught:
        sadp.search_number(Session(responses), "51082")
    assert caught.value.audit["page"] == 1
    assert caught.value.audit["retrieved"] == 50
    assert caught.value.audit["expected_total"] == 51
    assert caught.value.audit["complete"] is False


def test_partially_overlapping_history_pages_are_not_cached_as_complete():
    events = [{"sqAndamento": i, "descricaoAndamento": "Recebimento", "complemento": ""}
              for i in range(51)]
    responses = paged(events, "listaAndamentos")
    responses[1].payload["listaAndamentos"] = [copy.deepcopy(events[0])]
    detail = sadp.fetch_detail_e_publicacoes(Session([detail_response(), *responses]), "188792015")
    assert detail["ok"] is False
    assert detail["andamentos_complete"] is False
    assert "sqAndamento=0" in detail["andamentos_error"]["reason"]
    assert detail["andamentos_error"]["retrieved"] == 50
    assert detail["publicacoes_dje"] == []


def test_page_limit_is_explicit_and_auditable():
    session = Session(paged([process(i) for i in range(151)]))
    with pytest.raises(sadp.SADPQueryError, match="limite") as caught:
        sadp._fetch_all_pages(session, "https://official.example/search", "listaProcessos", max_pages=3)
    assert caught.value.audit["max_pages"] == 3
    assert caught.value.audit["expected_total"] == 151
    assert len(session.calls) == 1


def detail_response():
    return Response({"dadosProcesso": {"numeroUnico": "0000510-82.2015.6.00.0000", "partes": [
        {"nomeParte": "TRIBUNAL REGIONAL ELEITORAL DE GOIÁS", "descricaoParte": "INTERESSADO", "advogado": False},
        {"nomeParte": "MARCELO ARANTES DE MELO BORGES", "descricaoParte": "ADVOGADO INDICADO", "advogado": False},
        {"nomeParte": "OUTRA INDICADA", "descricaoParte": "ADVOGADA INDICADA"},
        {"nomeParte": "PATRONO CONSTITUÍDO", "descricaoParte": "ADVOGADO", "advogado": True},
        {"nomeParte": "PATRONA SEM FLAG", "descricaoParte": "ADVOGADA"},
    ]}})


def test_indicated_lawyers_are_parties_not_counsel():
    detail = sadp.fetch_detail(Session([detail_response()]), "188792015")
    assert detail["partes"] == ["TRIBUNAL REGIONAL ELEITORAL DE GOIÁS",
                                "MARCELO ARANTES DE MELO BORGES", "OUTRA INDICADA"]
    assert detail["advogados"] == ["PATRONO CONSTITUÍDO", "PATRONA SEM FLAG"]


def test_publication_on_last_page_of_212_events_is_recovered():
    events = [{"sqAndamento": i, "descricaoAndamento": "Recebimento", "complemento": ""}
              for i in range(212)]
    events[-1]["complemento"] = "Publicação em 27/10/2017 Diário de justiça eletrônico N. 209 Pag. 74/75. Acórdão de 01/08/2017"
    session = Session([detail_response(), *paged(events, "listaAndamentos")])
    detail = sadp.fetch_detail_e_publicacoes(session, "188792015")
    assert detail["ok"] is True
    assert detail["andamentos_complete"] is True
    assert detail["andamentos_count"] == 212
    assert detail["publicacoes_dje"][0]["data"] == "27/10/2017"
    assert detail["publicacoes_dje"][0]["edicao"] == "209"
    assert len(session.calls) == 6


def test_unavailable_history_is_not_cached_as_no_publications():
    session = Session([detail_response(), Response({}, status=503)])
    detail = sadp.fetch_detail_e_publicacoes(session, "188792015")
    assert detail["partes"]
    assert detail["ok"] is False
    assert detail["andamentos_complete"] is False
    assert detail["andamentos_error"]["reason"] == "HTTP 503"


def test_truly_empty_history_is_complete():
    session = Session([detail_response(), *paged([], "listaAndamentos")])
    detail = sadp.fetch_detail_e_publicacoes(session, "188792015")
    assert detail["ok"] is True
    assert detail["andamentos_count"] == 0
    assert detail["publicacoes_dje"] == []
