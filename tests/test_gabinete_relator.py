"""Mapa gabinete (DataJud) -> ministro relator.

Contexto: em 26/08/2026 o mesmo julgamento (lista triplice do TRE-AM, janela 1565-1713s,
identica nas tres rodadas) saiu com tres relatores diferentes, porque o relator herda o
ruido da composicao que o prompt injeta no bloco. O gabinete e a cadeira do relator e nao
depende do audio.
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

import gabinete_relator as gr


class _FakeLookup:
    """lookup_process(numero) -> objeto com orgao_julgador, a partir de um dicionario."""

    def __init__(self, por_numero: dict[str, str], falhar_em: set[str] | None = None):
        self.por_numero = por_numero
        self.falhar_em = falhar_em or set()
        self.chamadas: list[str] = []

    def __call__(self, numero, session=None, **kwargs):
        self.chamadas.append(numero)
        if numero in self.falhar_em:
            raise RuntimeError("timeout do DataJud")
        orgao = self.por_numero.get(numero)
        return SimpleNamespace(orgao_julgador=orgao) if orgao else None


def _pag(numero, relator):
    return {"numero": numero, "relator": relator}


def _cnj(n: int) -> str:
    return f"06000{n:02d}-11.2026.6.00.0000"


def test_gabinete_com_dominancia_clara_entra_no_mapa():
    """STF1 na apuracao real: 25 processos, 96% Nunes Marques."""
    numeros = {_cnj(i): "GABINETE STF1" for i in range(gr.MIN_PROCESSOS + 2)}
    paginas = [_pag(n, "Min. Nunes Marques") for n in numeros]
    paginas[0]["relator"] = "Min. Cármen Lúcia"  # a impureza e o proprio erro de extracao
    mapa, amostra = gr.apurar_mapa(paginas, _FakeLookup(numeros))
    assert mapa["GABINETE STF1"] == "Min. Nunes Marques"
    assert amostra["GABINETE STF1"]["dominancia"] >= gr.MIN_DOMINANCIA


def test_gabinete_em_transicao_de_ocupante_nao_corrige_nada():
    """STJ1 real: Cueva 70% x Isabel Gallotti — corrigir aqui trocaria relator legitimo."""
    numeros = {_cnj(i): "GABINETE STJ1" for i in range(20)}
    paginas = [_pag(n, "Min. Ricardo Villas Bôas Cueva") for n in list(numeros)[:14]]
    paginas += [_pag(n, "Min. Isabel Gallotti") for n in list(numeros)[14:]]
    mapa, amostra = gr.apurar_mapa(paginas, _FakeLookup(numeros))
    assert "GABINETE STJ1" not in mapa
    assert amostra["GABINETE STJ1"]["dominancia"] < gr.MIN_DOMINANCIA


def test_amostra_pequena_nao_entra_mesmo_sendo_unanime():
    """'JUÍZ AUXILIAR 2' com n=1 era 100% e nao pode virar regra."""
    numeros = {_cnj(1): "JUÍZ AUXILIAR 2"}
    mapa, _ = gr.apurar_mapa([_pag(_cnj(1), "Min. Floriano de Azevedo Marques")],
                             _FakeLookup(numeros))
    assert mapa == {}


def test_ignora_numero_curto_relator_vazio_e_falha_de_rede():
    numeros = {_cnj(i): "GABINETE STF3" for i in range(gr.MIN_PROCESSOS + 3)}
    paginas = [_pag(n, "Min. André Mendonça") for n in numeros]
    paginas.append(_pag("0600097-23", "Min. André Mendonça"))       # curto: nem consulta
    paginas.append(_pag(_cnj(90), ""))                              # sem relator: idem
    quebrado = _cnj(3)
    lookup = _FakeLookup(numeros, falhar_em={quebrado})
    mapa, _ = gr.apurar_mapa(paginas, lookup)
    assert mapa["GABINETE STF3"] == "Min. André Mendonça"
    assert "0600097-23" not in lookup.chamadas and _cnj(90) not in lookup.chamadas


def test_relator_do_gabinete_normaliza_caixa_e_espacos():
    mapa = {"GABINETE JURISTA 1": "Min. Estela Aranha"}
    assert gr.relator_do_gabinete(mapa, "gabinete  jurista 1") == "Min. Estela Aranha"
    assert gr.relator_do_gabinete(mapa, "GABINETE STJ1") == ""
    assert gr.relator_do_gabinete(mapa, "") == ""
    assert gr.relator_do_gabinete({}, "GABINETE JURISTA 1") == ""


def test_cache_vencido_reapura_e_falha_de_apuracao_devolve_vazio(tmp_path, monkeypatch):
    monkeypatch.setattr(gr, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(gr, "STATE_DIR", tmp_path)
    esperado = {"GABINETE STF1": "Min. Nunes Marques"}

    chamou = []

    def apurador():
        chamou.append(1)
        return esperado, {}

    assert gr.carregar_mapa(apurador) == esperado
    assert gr.carregar_mapa(apurador) == esperado, "segunda leitura vem do cache"
    assert len(chamou) == 1

    # cache vencido: reapura
    dados = json.loads((tmp_path / "cache.json").read_text(encoding="utf-8"))
    dados["apurado_em"] = time.time() - gr.CACHE_TTL_SECONDS - 10
    (tmp_path / "cache.json").write_text(json.dumps(dados), encoding="utf-8")
    assert gr.carregar_mapa(apurador) == esperado
    assert len(chamou) == 2

    # sem apurador e sem cache valido -> {} (nada e corrigido, que e o seguro)
    (tmp_path / "cache.json").unlink()
    assert gr.carregar_mapa(None) == {}

    def explode():
        raise RuntimeError("Notion fora do ar")

    assert gr.carregar_mapa(explode) == {}


def test_mapa_vazio_nao_e_gravado_no_cache(tmp_path, monkeypatch):
    """Apuracao inconclusiva nao pode congelar {} no cache por 7 dias."""
    monkeypatch.setattr(gr, "CACHE_FILE", tmp_path / "cache.json")
    monkeypatch.setattr(gr, "STATE_DIR", tmp_path)
    assert gr.carregar_mapa(lambda: ({}, {})) == {}
    assert not (tmp_path / "cache.json").exists()
