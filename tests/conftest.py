"""Isolamento entre testes.

`_ELENCO_CACHE` e `_MEMBROS_CACHE` sao globais de processo (a apuracao custa dezenas de
consultas e roda uma vez por lote). Sem zerar entre testes, o primeiro que os popula dita
o resultado de todos os seguintes -- e a falha aparece so na suite completa, nunca no teste
isolado, que e o pior modo de falhar.
"""
import pytest

import tse_youtube_notion_core as core
import vistoria_queue


@pytest.fixture(autouse=True)
def _isola_fila_vistoria(tmp_path, monkeypatch):
    """Testes de lote e falhas nunca gravam na fila usada pela GUI real."""
    monkeypatch.setattr(vistoria_queue, "QUEUE_FILE", tmp_path / "vistoria_queue.jsonl")


@pytest.fixture(autouse=True)
def _zera_caches_de_composicao():
    core._ELENCO_CACHE = None
    core._MEMBROS_CACHE = None
    yield
    core._ELENCO_CACHE = None
    core._MEMBROS_CACHE = None
