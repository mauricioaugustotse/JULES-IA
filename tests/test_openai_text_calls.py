"""Regra de 27/09/2026: Gemini só lê/assiste o vídeo e faz pesquisa web; chamada só de texto é OpenAI."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tse_youtube_notion_core as core
from tse_youtube_notion_core import (
    PublishPreviewRow,
    RunArtifacts,
    ThemePunchlineEnricher,
    ThemePunchlineRepairBatchResult,
    ThemePunchlineRepairItem,
    ThemeRepairResult,
)

PUNCHLINE = (
    "A disputa examinou se a divulgação eleitoral de benefícios sociais pela gestão municipal comprometeu a "
    "igualdade da disputa, e o TSE reconheceu o impacto concreto da conduta no desfecho do pleito."
)


@pytest.fixture
def openai_falso(monkeypatch):
    """Troca call_openai_structured por um gravador; o Gemini, se chamado, explode."""
    chamadas = []

    def fake(**kwargs):
        chamadas.append(kwargs)
        modelo = kwargs["response_model"]
        if modelo is ThemePunchlineRepairBatchResult:
            return ThemePunchlineRepairBatchResult(items=[ThemePunchlineRepairItem(
                key="row_001", tema="Uso promocional de programa social em campanha municipal",
                punchline=PUNCHLINE)]), "{}"
        if modelo is ThemeRepairResult:
            return ThemeRepairResult(tema="Uso promocional de programa social em campanha municipal"), "{}"
        return modelo(), "{}"

    def gemini_proibido(**_):
        raise AssertionError("chamada só de texto não pode ir ao Gemini")

    monkeypatch.setattr(core, "call_openai_structured", fake)
    monkeypatch.setattr(core, "call_gemini_generate_content_rest", gemini_proibido)
    monkeypatch.setattr(core, "get_openai_api_key", lambda: "sk-teste")
    return chamadas


def _row():
    return PublishPreviewRow(
        tema="Julgamento", punchline="Recurso provido.", resultado="Provido",
        analise_do_conteudo_juridico=(
            "A controvérsia envolveu a utilização promocional de programa social por agente público durante "
            "a campanha municipal, com debate sobre desequilíbrio eleitoral e alcance da sanção."),
    )


def test_tema_punchline_vai_para_a_openai_com_o_modelo_de_texto(tmp_path, openai_falso):
    [row] = ThemePunchlineEnricher(artifact_store=RunArtifacts(tmp_path)).enrich_rows([_row()])

    assert row.punchline == PUNCHLINE
    [chamada] = openai_falso
    assert chamada["model"] == core.OPENAI_TEXT_MODEL == "gpt-6-luna"
    assert chamada["system_instruction"] == core.THEME_PUNCHLINE_REPAIR_SYSTEM_PROMPT
    assert chamada["api_key"] == "sk-teste"


def test_cache_de_tema_punchline_nao_eterniza_fallback_nem_resposta_de_outro_modelo(tmp_path, openai_falso):
    store = RunArtifacts(tmp_path)
    payload = [core.build_theme_punchline_repair_payload(_row(), key="row_001")]
    # cache antigo: fallback gravado depois de uma falha (sem "parsed") e sem modelo
    store.write_json("04b_theme_punchline_01.json", {
        "payload": payload, "error": "503", "applied": [_row().model_dump(mode="json")]})

    [row] = ThemePunchlineEnricher(artifact_store=store).enrich_rows([_row()])
    assert row.punchline == PUNCHLINE and len(openai_falso) == 1

    ThemePunchlineEnricher(artifact_store=store).enrich_rows([_row()])  # resposta boa do mesmo modelo
    assert len(openai_falso) == 1
    ThemePunchlineEnricher(artifact_store=store, model="outro-modelo").enrich_rows([_row()])
    assert len(openai_falso) == 2


def test_enrich_preview_rows_with_theme_punchline_sem_chave_usa_a_openai(tmp_path, openai_falso):
    rows = core.enrich_preview_rows_with_theme_punchline([_row()], artifact_store=RunArtifacts(tmp_path))
    assert rows[0].punchline == PUNCHLINE
    assert openai_falso[0]["model"] == "gpt-6-luna"


def test_reparo_textual_de_tema_vai_para_a_openai(openai_falso):
    result = core.repair_theme_from_text_context(row=_row(), context_text="contexto do caso")
    assert result.tema.startswith("Uso promocional")
    assert openai_falso[0]["system_instruction"] == core.THEME_REPAIR_SYSTEM_PROMPT
    assert core.repair_theme_from_text_context(row=_row(), context_text="  ") == ThemeRepairResult()
    assert len(openai_falso) == 1


def test_estruturacao_do_texto_pesquisado_vai_para_a_openai(openai_falso):
    enricher = object.__new__(core.GeminiProcessMetadataEnricher)
    parsed = enricher._structure_grounded_text("texto já pesquisado", core.ProcessMetadataResult)
    assert isinstance(parsed, core.ProcessMetadataResult)
    assert openai_falso[0]["prompt"].endswith("texto já pesquisado")
    assert openai_falso[0]["model"] == "gpt-6-luna"


def test_punchline_com_ponto_e_virgula_no_passado_nao_e_tomada_por_citacao():
    row = PublishPreviewRow(tema="Aprovação de lista tríplice para o cargo de juiz substituto do TRE/AC")
    boa = ("Para preencher vaga aberta com o término do primeiro biênio de um dos integrantes, o TSE verificou "
           "a experiência dos indicados e afastou impedimento ligado a cargos efetivos; reconhecidos os "
           "requisitos, aprovou por unanimidade o envio da lista ao Presidente da República.")
    assert core.clean_theme_punchline_punchline(boa, row) == boa
    assert core.clean_theme_punchline_punchline("Art. 73, I, da Lei 9.504/97; Súmula 24 do TSE.", row) == ""


def test_schemas_das_chamadas_openai_aceitam_modo_estrito():
    to_strict = pytest.importorskip("openai.lib._pydantic").to_strict_json_schema
    from fill_pedido_vista_via_grounding import PedidoVistaResult
    from rewrite_notion_tema_punchline import RewriteBatchResult

    for modelo in (core.TeorAnaliseBatchResult, ThemePunchlineRepairBatchResult, ThemeRepairResult,
                   core.ProcessMetadataResult, PedidoVistaResult, RewriteBatchResult):
        schema = to_strict(modelo)
        assert schema["additionalProperties"] is False, modelo.__name__
