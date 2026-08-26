"""Isolamento entre testes.

`_ELENCO_CACHE` e `_MEMBROS_CACHE` sao globais de processo (a apuracao custa dezenas de
consultas e roda uma vez por lote). Sem zerar entre testes, o primeiro que os popula dita
o resultado de todos os seguintes -- e a falha aparece so na suite completa, nunca no teste
isolado, que e o pior modo de falhar.
"""
import pytest

import tse_youtube_notion_core as core


@pytest.fixture(autouse=True)
def _zera_caches_de_composicao():
    core._ELENCO_CACHE = None
    core._MEMBROS_CACHE = None
    yield
    core._ELENCO_CACHE = None
    core._MEMBROS_CACHE = None
