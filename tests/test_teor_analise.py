"""Campos analíticos x inteiro teor: a análise não pode ser a ementa crua (27/09/2026).

Caso real: LT 0600317-32.2026.6.00.0000 (TRE/TO, sessão de 09/06/2026), importada do DJE em
04/07/2026 com analise_do_conteudo_juridico = ementa[:1900] e raciocinio_juridico = dispositivo.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tse_youtube_notion_core as core
from tse_youtube_notion_core import (
    TeorAnaliseEnricher,
    PublishPreviewRow,
    RunArtifacts,
    TeorAnaliseBatchResult,
    TeorAnaliseItem,
    campo_analitico_copia_teor,
    fracao_copiada_do_teor,
    parece_copia_do_acordao,
    texto_abre_com_verbetacao,
    validate_preview_row,
)

EMENTA_TO = (
    "LISTA TRÍPLICE. VAGA DE JUIZ TITULAR. CLASSE DOS ADVOGADOS. EXIGÊNCIAS LEGAIS E REGULAMENTARES "
    "ATENDIDAS. PARIDADE DE GÊNERO. COMPOSIÇÃO MISTA. NÃO ATENDIMENTO. RETORNO DOS AUTOS AO TRIBUNAL DE "
    "ORIGEM. I. CASO EM EXAME 1. Lista tríplice para o preenchimento da vaga de juiz titular da classe dos "
    "advogados do TRE/TO, decorrente do término do primeiro biênio de um de seus membros em 14.3.2026. "
    "II. QUESTÕES EM DISCUSSÃO 2. Há duas questões em discussão: (a) se foram atendidos os requisitos "
    "previstos nas normas de regência para o exercício do cargo de juiz titular da classe dos advogados nos "
    "tribunais regionais eleitorais; e (b) se a formação da lista tríplice observa a política de paridade de "
    "gênero prevista na Res.–TSE nº 23.517/2017, com as alterações da Res.–TSE nº 23.746/2025. III. RAZÕES "
    "DE DECIDIR 3. A Assessoria Consultiva desta Corte Superior consignou o preenchimento dos requisitos "
    "objetivos previstos na Res.–TSE nº 23.517/2017. 6. A vaga objeto da presente lista tríplice é de membro "
    "titular da classe dos advogados, e o outro cargo de membro titular é ocupado por um homem. A composição "
    "mista da presente lista tríplice, formada por duas mulheres e um homem, este concorrendo à recondução, "
    "não concretiza o espírito da norma. IV. DISPOSITIVO 8. Lista tríplice devolvida ao TRE/TO, a fim de que "
    "proceda à substituição do nome do advogado indicado, mantidos os nomes das advogadas indicadas."
)
DECISAO_TO = (
    "O Tribunal, por unanimidade, determinou a devolução da lista tríplice ao Tribunal Regional Eleitoral do "
    "Tocantins para que seja providenciada a substituição do Dr. Antonio Paim Broglio, mantendo-se as "
    "indicações da Dra. Bruna Bonilha de Toledo Costa Azevedo e da Dra. Graziela Tavares de Souza Reis, nos "
    "termos do voto do Relator. Acompanharam o Relator os Ministros Ricardo Villas Bôas Cueva, Floriano de "
    "Azevedo Marques, Estela Aranha, André Mendonça, Dias Toffoli e Nunes Marques (Presidente). Composição: "
    "Ministros Nunes Marques (Presidente), André Mendonça, Dias Toffoli, Antonio Carlos Ferreira, Ricardo "
    "Villas Bôas Cueva, Floriano de Azevedo Marques e Estela Aranha."
)
TEOR_TO = f"{EMENTA_TO} {DECISAO_TO}"

ANALISE_REDIGIDA = (
    "O TRE do Tocantins encaminhou ao TSE a lista tríplice para a vaga de juiz titular da classe dos "
    "advogados, aberta com o fim do primeiro biênio de um dos membros em março de 2026. A lista reunia duas "
    "advogadas e um advogado que buscava a recondução. A Assessoria Consultiva atestou os requisitos "
    "objetivos e não houve impugnação aos indicados, mas o outro assento titular da classe já é ocupado por "
    "um homem, o que pôs em debate a política de paridade de gênero das resoluções do TSE. O Tribunal "
    "devolveu a lista ao TRE/TO para substituir o advogado, preservando as duas advogadas."
)
RACIOCINIO_REDIGIDO = (
    "Superados os requisitos formais e constitucionais, o relator examinou a lista sob a ótica da ação "
    "afirmativa da Res.-TSE 23.517/2017, alterada pela Res.-TSE 23.746/2025, que busca ocupação igualitária "
    "dos cargos da classe dos juristas. Como a outra cadeira titular já pertence a um homem, uma lista mista "
    "deixaria aberta a escolha que manteria o desequilíbrio; por isso, apoiado em precedente da Corte sobre "
    "listas mistas, concluiu que só a substituição do advogado garantiria efetividade à norma."
)


def _row(**kwargs):
    base = dict(
        numero_processo="0600317-32.2026.6.00.0000",
        classe_processo="Lista Tríplice",
        data_sessao="2026-06-09",
        tribunal="TSE",
        origem="Palmas/TO",
        relator="Min. Antônio Carlos Ferreira",
        resultado="Devolvida",
        votacao="Unânime",
    )
    base.update(kwargs)
    return PublishPreviewRow(**base)


def _enricher(tmp_path, respostas):
    """Enricher com o Gemini trocado por uma fila de respostas (uma por chamada)."""
    enricher = TeorAnaliseEnricher(
        api_key="x", artifact_store=RunArtifacts(tmp_path / "art"), batch_size=5
    )
    chamadas = []

    def fake_call(*, payload, artifact_name, tentativa):
        chamadas.append({"keys": [item["key"] for item in payload], "tentativa": tentativa})
        itens = respostas.pop(0)
        return TeorAnaliseBatchResult(items=[TeorAnaliseItem(key=item["key"], **itens) for item in payload])

    enricher._call_batch = fake_call
    return enricher, chamadas


# --- detectores --------------------------------------------------------------------------


def test_verbetacao_da_ementa_e_detectada_e_prosa_nao():
    assert texto_abre_com_verbetacao(EMENTA_TO)
    assert texto_abre_com_verbetacao("ELEIÇÕES 2020. AGRAVO INTERNO. RECURSO ESPECIAL. Trata-se de agravo.")
    assert not texto_abre_com_verbetacao(ANALISE_REDIGIDA)
    assert not texto_abre_com_verbetacao("O TSE manteve a cassação do prefeito de Russas/CE.")
    assert not texto_abre_com_verbetacao("AIJE ajuizada pelo MPE contra o prefeito.")


def test_rotulos_e_formulas_do_acordao_sao_copia():
    assert parece_copia_do_acordao("Ementa longa. I. CASO EM EXAME 1. Recurso especial contra acórdão.")
    assert parece_copia_do_acordao("... Tese de julgamento: a fraude se configura quando ...")
    assert parece_copia_do_acordao("ACORDAM os ministros do Tribunal Superior Eleitoral, por unanimidade, em ...")
    assert parece_copia_do_acordao(DECISAO_TO)  # "Composição: Ministros ..."
    assert not parece_copia_do_acordao(RACIOCINIO_REDIGIDO)
    assert not parece_copia_do_acordao("")


def test_fracao_copiada_separa_copia_de_redacao():
    assert fracao_copiada_do_teor(EMENTA_TO[:1900], TEOR_TO) > 0.95
    assert fracao_copiada_do_teor(DECISAO_TO, TEOR_TO) > 0.95
    assert fracao_copiada_do_teor(ANALISE_REDIGIDA, TEOR_TO) < 0.2
    assert fracao_copiada_do_teor(RACIOCINIO_REDIGIDO, TEOR_TO) < 0.2
    assert fracao_copiada_do_teor("curto", TEOR_TO) == 0.0


def test_campo_analitico_copia_teor_pega_trecho_do_meio_sem_forma_de_ementa():
    # Trecho das razões de decidir, sem verbetação nem rótulo: só a comparação com o teor pega.
    trecho = (
        "A vaga objeto da presente lista tríplice é de membro titular da classe dos advogados, e o outro "
        "cargo de membro titular é ocupado por um homem."
    )
    assert not parece_copia_do_acordao(trecho)
    assert campo_analitico_copia_teor(trecho, TEOR_TO)
    assert not campo_analitico_copia_teor(trecho)  # sem teor, só a forma decide
    assert not campo_analitico_copia_teor(ANALISE_REDIGIDA, TEOR_TO)


def test_validate_preview_row_avisa_analise_com_forma_de_ementa():
    row = validate_preview_row(_row(analise_do_conteudo_juridico=EMENTA_TO[:1900]), None)
    assert any(w.startswith("analise_do_conteudo_juridico reproduz o acórdão") for w in row.warnings)
    row = validate_preview_row(row, None)  # revalidação não duplica
    assert sum(w.startswith("analise_do_conteudo_juridico reproduz") for w in row.warnings) == 1
    limpa = validate_preview_row(_row(analise_do_conteudo_juridico=ANALISE_REDIGIDA), None)
    assert not any("reproduz o acórdão" in w for w in limpa.warnings)


def test_call_openai_structured_recusa_resposta_incompleta(monkeypatch):
    import openai

    class Resposta:
        def __init__(self, status, parsed):
            self.status, self.output_parsed, self.output_text = status, parsed, "{}"
            self.incomplete_details = type("D", (), {"reason": "max_output_tokens"})() if status != "completed" else None

    respostas = [Resposta("incomplete", None), Resposta("completed", TeorAnaliseBatchResult())]
    pedidos = []

    class FakeResponses:
        def parse(self, **kwargs):
            pedidos.append(kwargs)
            return respostas.pop(0)

    class FakeOpenAI:
        def __init__(self, **_):
            self.responses = FakeResponses()

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    kwargs = dict(api_key="x", model="gpt-6-luna", system_instruction="s", prompt="p",
                  response_model=TeorAnaliseBatchResult)
    try:
        core.call_openai_structured(**kwargs)
        raise AssertionError("resposta incompleta deveria levantar")
    except RuntimeError as exc:
        assert "max_output_tokens" in str(exc)
    parsed, _ = core.call_openai_structured(**kwargs)
    assert parsed == TeorAnaliseBatchResult()
    assert pedidos[0]["text_format"] is TeorAnaliseBatchResult
    assert pedidos[0]["model"] == "gpt-6-luna" and pedidos[0]["reasoning"] == {"effort": "medium"}


def test_modelo_padrao_do_redator_e_o_gpt_6_luna(tmp_path):
    enricher = TeorAnaliseEnricher(api_key="x", artifact_store=RunArtifacts(tmp_path))
    assert enricher.model == core.OPENAI_TEXT_MODEL == "gpt-6-luna"


# --- enricher ---------------------------------------------------------------------------


def test_enricher_troca_copia_por_redacao(tmp_path):
    enricher, chamadas = _enricher(
        tmp_path,
        [{"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": RACIOCINIO_REDIGIDO}],
    )
    row = _row(analise_do_conteudo_juridico=EMENTA_TO[:1900], raciocinio_juridico=DECISAO_TO)

    [novo] = enricher.enrich_rows([row], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA
    assert novo.raciocinio_juridico == RACIOCINIO_REDIGIDO
    assert len(chamadas) == 1
    assert row.analise_do_conteudo_juridico == EMENTA_TO[:1900]  # entrada intacta


def test_enricher_recusa_proposta_copiada_e_tenta_de_novo(tmp_path):
    enricher, chamadas = _enricher(
        tmp_path,
        [
            {"analise_do_conteudo_juridico": EMENTA_TO[:1500], "raciocinio_juridico": RACIOCINIO_REDIGIDO},
            {"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": "ignorado"},
        ],
    )
    [novo] = enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA
    assert novo.raciocinio_juridico == RACIOCINIO_REDIGIDO  # aceito na 1ª, não reescrito na 2ª
    assert [c["tentativa"] for c in chamadas] == [1, 2]


def test_enricher_deixa_vazio_com_aviso_quando_so_recebe_copia(tmp_path):
    copia = {"analise_do_conteudo_juridico": EMENTA_TO[:1500], "raciocinio_juridico": DECISAO_TO}
    enricher, _ = _enricher(tmp_path, [dict(copia), dict(copia)])
    row = _row(analise_do_conteudo_juridico=EMENTA_TO[:1900], raciocinio_juridico=DECISAO_TO)

    [novo] = enricher.enrich_rows([row], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ""
    assert novo.raciocinio_juridico == ""
    assert sum("não redigido a partir do inteiro teor" in w for w in novo.warnings) == 2


def test_enricher_preserva_texto_do_video_e_ignora_linha_sem_teor(tmp_path):
    enricher, chamadas = _enricher(tmp_path, [])
    do_video = _row(analise_do_conteudo_juridico=ANALISE_REDIGIDA, raciocinio_juridico=RACIOCINIO_REDIGIDO)
    sem_teor = _row(numero_processo="0600001-00.2026.6.00.0000")

    saida = enricher.enrich_rows([do_video, sem_teor], [(EMENTA_TO, DECISAO_TO), ("", "")])

    assert chamadas == []
    assert saida[0].analise_do_conteudo_juridico == ANALISE_REDIGIDA
    assert saida[1].analise_do_conteudo_juridico == ""
    assert not saida[1].warnings


def test_enricher_so_reescreve_o_campo_copiado(tmp_path):
    enricher, _ = _enricher(
        tmp_path,
        [{"analise_do_conteudo_juridico": "outra", "raciocinio_juridico": RACIOCINIO_REDIGIDO}],
    )
    row = _row(analise_do_conteudo_juridico=ANALISE_REDIGIDA, raciocinio_juridico=DECISAO_TO)

    [novo] = enricher.enrich_rows([row], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA
    assert novo.raciocinio_juridico == RACIOCINIO_REDIGIDO


def test_enricher_reaproveita_cache_do_mesmo_payload(tmp_path):
    resposta = {"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": RACIOCINIO_REDIGIDO}
    enricher, chamadas = _enricher(tmp_path, [dict(resposta)])
    enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])
    [novo] = enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])

    assert len(chamadas) == 1
    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA


def test_enricher_sobrevive_a_falha_do_gemini(tmp_path):
    enricher = TeorAnaliseEnricher(api_key="x", artifact_store=RunArtifacts(tmp_path / "art"))

    def explode(**_):
        raise RuntimeError("503")

    enricher._call_batch = explode
    [novo] = enricher.enrich_rows([_row(analise_do_conteudo_juridico=EMENTA_TO[:1900])], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ""
    assert any("não redigido" in w for w in novo.warnings)


# --- import_dje_faltantes (dry-run ponta a ponta, Notion e Gemini falsos) -----------------


def test_import_dje_redige_analise_em_vez_de_copiar_a_ementa(tmp_path, monkeypatch):
    import import_dje_faltantes as imp

    novo_cnj, velho_cnj = "06003173220266000000", "06009999920246270000"
    itens = [
        {"id": "a", "source": "dje", "status": "pending", "data_sessao": "2026-06-09",
         "extra": {"dje": {"cnj20": novo_cnj}}},
        {"id": "b", "source": "dje", "status": "pending", "data_sessao": "2026-06-09",
         "extra": {"dje": {"cnj20": velho_cnj}}},
    ]
    linha = {"textoEmenta": EMENTA_TO, "textoDecisao": DECISAO_TO, "siglaClasse": "LT",
             "relatores": "Min. Antonio Carlos Ferreira", "nomeMunicipio": "", "siglaUF": "TO"}
    existente = {  # página de import anterior: análise = ementa; raciocínio veio do vídeo
        "id": "pagina-velha", "numero_processo": "0600999-99.2024.6.27.0000", "data_sessao": "2026-06-09",
        "tipo_registro": "Julgamento 3", "analise_do_conteudo_juridico": EMENTA_TO[:1900],
        "raciocinio_juridico": RACIOCINIO_REDIGIDO,
    }

    class FakeClient:
        def __init__(self, **_):
            pass

        def fetch_schema(self):
            return None

        def query_data_source(self, cond=None):
            if cond and cond["rich_text"]["contains"] != "0600999-99":
                return []
            return [existente]

        def _extract_property_text(self, page, _schema, field):
            return page.get(field, "")

    monkeypatch.setattr(imp.vistoria_queue, "load_items", lambda _=None: itens)
    monkeypatch.setattr(imp, "load_full_rows", lambda _path, wanted: {c: dict(linha) for c in wanted})
    monkeypatch.setattr(imp, "NotionSessoesClient", FakeClient)
    monkeypatch.setattr(imp, "get_secret", lambda *_: "chave")
    monkeypatch.setattr(imp, "ARTIFACT_ROOT", tmp_path)
    recebidos = []

    def fake_call(self, *, payload, artifact_name, tentativa):
        recebidos.extend(payload)
        return TeorAnaliseBatchResult(items=[
            TeorAnaliseItem(key=p["key"], analise_do_conteudo_juridico=ANALISE_REDIGIDA,
                            raciocinio_juridico=RACIOCINIO_REDIGIDO)
            for p in payload
        ])

    monkeypatch.setattr(TeorAnaliseEnricher, "_call_batch", fake_call)
    monkeypatch.setattr(sys, "argv", ["import_dje_faltantes.py", "--csv", "x.csv", "--skip-theme-enricher"])

    assert imp.main() == 0

    [preview] = list(tmp_path.glob("dje_import/*/import_preview.json"))
    rows = {r["numero_processo"]: r for r in json.loads(preview.read_text(encoding="utf-8"))}
    novo, velho = rows["0600317-32.2026.6.00.0000"], rows["0600999-99.2024.6.27.0000"]
    for row in (novo, velho):
        assert row["analise_do_conteudo_juridico"] == ANALISE_REDIGIDA
        assert not parece_copia_do_acordao(row["analise_do_conteudo_juridico"])
    assert novo["raciocinio_juridico"] == RACIOCINIO_REDIGIDO
    assert velho["page_id"] == "pagina-velha" and velho["action"] == "update"
    assert velho["raciocinio_juridico"] == RACIOCINIO_REDIGIDO  # texto do vídeo preservado
    assert velho["clear_properties"] == ["analise_do_conteudo_juridico"]
    assert {p["ementa"][:13] for p in recebidos} == {"LISTA TRÍPLIC"}
    assert all("Composição: Ministros" in p["decisao"] for p in recebidos)


def test_import_dje_sem_redacao_nunca_grava_a_ementa(tmp_path, monkeypatch):
    import import_dje_faltantes as imp

    itens = [{"id": "a", "source": "dje", "status": "pending", "data_sessao": "2026-06-09",
              "extra": {"dje": {"cnj20": "06003173220266000000"}}}]
    linha = {"textoEmenta": EMENTA_TO, "textoDecisao": DECISAO_TO, "siglaClasse": "LT", "siglaUF": "TO"}

    class FakeClient:
        def __init__(self, **_):
            pass

        def fetch_schema(self):
            return None

        def query_data_source(self, cond=None):
            return []

    monkeypatch.setattr(imp.vistoria_queue, "load_items", lambda _=None: itens)
    monkeypatch.setattr(imp, "load_full_rows", lambda _path, wanted: {c: dict(linha) for c in wanted})
    monkeypatch.setattr(imp, "NotionSessoesClient", FakeClient)
    monkeypatch.setattr(imp, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["import_dje_faltantes.py", "--csv", "x.csv",
                                      "--skip-theme-enricher", "--skip-teor-analise"])

    assert imp.main() == 0

    [preview] = list(tmp_path.glob("dje_import/*/import_preview.json"))
    [row] = json.loads(preview.read_text(encoding="utf-8"))
    assert row["analise_do_conteudo_juridico"] == ""
    assert row["raciocinio_juridico"] == ""
    assert row["votacao"] == "Unânime"  # o dispositivo segue alimentando os selects


def test_enricher_recusa_texto_que_comenta_a_fonte(tmp_path):
    meta = ANALISE_REDIGIDA + " A ementa fornecida não informa o nome do advogado substituto."
    enricher, chamadas = _enricher(
        tmp_path,
        [
            {"analise_do_conteudo_juridico": meta, "raciocinio_juridico": RACIOCINIO_REDIGIDO},
            {"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": "ignorado"},
        ],
    )
    [novo] = enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])

    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA
    assert [c["tentativa"] for c in chamadas] == [1, 2]
    assert core._META_FONTE_RE.search("Não consta do material a data do fato.")
    assert core._META_FONTE_RE.search("O material fornecido não indica a votação.")
    assert not core._META_FONTE_RE.search(RACIOCINIO_REDIGIDO)
    assert not core._META_FONTE_RE.search("Alegou a retirada do material no prazo legal.")


# --- achados da revisão adversarial (27/09/2026) -------------------------------------------


def test_texto_existente_curto_e_legitimo_nao_e_copia_mas_proposta_e():
    # Prosa do vídeo que cita nome completo + cargo + coligação (0600808-20, fração ~0,57).
    teor = ("REGISTRO DE CANDIDATURA. PRESIDENTE DA REPÚBLICA. Trata-se do registro de candidatura de "
            "Guilherme Castro Boulos ao cargo de Presidente da República pela coligação Vamos sem Medo "
            "de Mudar o Brasil, requerido nos termos da resolução.")
    video = ("O processo trata do registro de candidatura de Guilherme Castro Boulos ao cargo de "
             "Presidente da República pela coligação Vamos sem Medo de Mudar o Brasil, com parecer "
             "favorável do MPE na sessão.")
    assert 0.5 <= fracao_copiada_do_teor(video, teor) < 0.8
    assert not campo_analitico_copia_teor(video, teor)
    assert campo_analitico_copia_teor(video, teor, proposta=True)


def test_frase_curta_inteira_do_teor_e_copia_e_frase_literal_longa_derruba_proposta():
    assert campo_analitico_copia_teor("O Tribunal, por unanimidade, determinou a devolução da lista tríplice ao "
                                      "Tribunal Regional Eleitoral do Tocantins.", TEOR_TO)
    literal = ("A vaga objeto da presente lista tríplice é de membro titular da classe dos advogados, e o "
               "outro cargo de membro titular é ocupado por um homem.")
    proposta = ANALISE_REDIGIDA + " " + literal
    assert fracao_copiada_do_teor(proposta, TEOR_TO) < 0.5
    assert core.frase_literal_do_teor(proposta, TEOR_TO)
    assert campo_analitico_copia_teor(proposta, TEOR_TO, proposta=True)
    assert not core.frase_literal_do_teor(ANALISE_REDIGIDA, TEOR_TO)


def test_punchline_do_fallback_sobre_ementa_e_tema_quebrado():
    molde = "LISTA TRÍPLICE. VAGA DE JUIZ TITULAR. O desfecho registrado foi devolvida."
    assert core.punchline_copia_teor(molde, TEOR_TO, analise_copiada=True)
    assert not core.punchline_copia_teor("Punchline editorial sobre paridade. O desfecho registrado foi devolvida.",
                                         TEOR_TO, analise_copiada=False)
    assert core.tema_quebrado(
        "CONDENAÇÃO POR ATO DOLOSO DE IMPROBIDADE ADMINISTRATIVA. O desfecho registrado foi desprovido")
    assert not core.tema_quebrado("Formação de lista tríplice para o cargo de juiz titular da classe dos advogados")
    assert not core.tema_quebrado("Registro de candidatura no TRE-SP (AIJE)")


def test_segunda_tentativa_nao_usa_cache(tmp_path):
    copia = {"analise_do_conteudo_juridico": EMENTA_TO[:1500], "raciocinio_juridico": RACIOCINIO_REDIGIDO}
    enricher, chamadas = _enricher(tmp_path, [dict(copia), dict(copia)])
    enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])
    assert [c["tentativa"] for c in chamadas] == [1, 2]
    # nova execução na MESMA pasta: a 1ª tentativa vem do cache, a 2ª é chamada de novo
    enricher2, chamadas2 = _enricher(
        tmp_path, [{"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": "x"}])
    [novo] = enricher2.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])
    assert [c["tentativa"] for c in chamadas2] == [2]
    assert novo.analise_do_conteudo_juridico == ANALISE_REDIGIDA


def test_cache_muda_com_esforco_do_modelo(tmp_path):
    resposta = {"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": RACIOCINIO_REDIGIDO}
    enricher, chamadas = _enricher(tmp_path, [dict(resposta), dict(resposta)])
    enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])
    enricher.effort = "high"
    enricher.enrich_rows([_row()], [(EMENTA_TO, DECISAO_TO)])
    assert len(chamadas) == 2


def test_enricher_de_tema_marca_punchline_do_fallback():
    enricher = object.__new__(core.ThemePunchlineEnricher)
    row = _row(tema="Paridade de gênero em lista tríplice de TRE", analise_do_conteudo_juridico=ANALISE_REDIGIDA)
    curta = core.ThemePunchlineRepairItem(key="row_001", tema=row.tema, punchline="Lista devolvida.")
    assert core.PUNCHLINE_FALLBACK_WARNING in enricher._apply_repair_item(row, curta).warnings
    boa = core.ThemePunchlineRepairItem(
        key="row_001", tema=row.tema,
        punchline=("Ao devolver a lista do TRE/TO, o TSE sinalizou que a mera presença de advogadas numa "
                   "lista mista não basta quando o outro assento titular já é ocupado por um homem."))
    assert core.PUNCHLINE_FALLBACK_WARNING not in enricher._apply_repair_item(row, boa).warnings


def _import_falso(monkeypatch, tmp_path, existentes, argv, segredo="chave"):
    import import_dje_faltantes as imp

    itens = [{"id": "a", "source": "dje", "status": "pending", "data_sessao": "2026-06-09",
              "extra": {"dje": {"cnj20": "06003173220266000000"}}}]
    linha = {"textoEmenta": EMENTA_TO, "textoDecisao": DECISAO_TO, "siglaClasse": "LT", "siglaUF": "TO"}

    class FakeClient:
        def __init__(self, **_):
            pass

        def fetch_schema(self):
            return None

        def query_data_source(self, cond=None):
            return list(existentes)

        def _extract_property_text(self, page, _schema, field):
            return page.get(field, "")

    def nao_publica(*_a, **_k):
        raise AssertionError("não devia publicar")

    monkeypatch.setattr(imp.vistoria_queue, "load_items", lambda _=None: itens)
    monkeypatch.setattr(imp, "load_full_rows", lambda _path, wanted: {c: dict(linha) for c in wanted})
    monkeypatch.setattr(imp, "NotionSessoesClient", FakeClient)
    monkeypatch.setattr(imp, "get_secret", lambda *_: segredo)
    monkeypatch.setattr(imp, "ARTIFACT_ROOT", tmp_path)
    monkeypatch.setattr(imp, "publish_preview_rows", nao_publica)
    monkeypatch.setattr(sys, "argv", ["import_dje_faltantes.py", "--csv", "x.csv", *argv])
    return imp


def test_import_sem_chave_da_openai_nao_publica(tmp_path, monkeypatch):
    imp = _import_falso(monkeypatch, tmp_path, [], ["--apply"], segredo="")
    assert imp.main() == 2


def test_import_upsert_descarta_punchline_e_tema_montados_sobre_a_ementa(tmp_path, monkeypatch):
    existente = {
        "id": "pagina", "numero_processo": "0600317-32.2026.6.00.0000", "data_sessao": "2026-06-09",
        "tipo_registro": "Julgamento 2", "analise_do_conteudo_juridico": EMENTA_TO[:1900],
        "raciocinio_juridico": DECISAO_TO,
        "punchline": "LISTA TRÍPLICE. VAGA DE JUIZ TITULAR. O desfecho registrado foi devolvida.",
        "tema": "LISTA TRÍPLICE. O desfecho registrado foi devolvida",
    }
    imp = _import_falso(monkeypatch, tmp_path, [existente], ["--skip-theme-enricher", "--skip-teor-analise"])

    assert imp.main() == 0
    [preview] = list(tmp_path.glob("dje_import/*/import_preview.json"))
    [row] = json.loads(preview.read_text(encoding="utf-8"))
    assert row["punchline"] == "" and "O desfecho" not in row["tema"]
    assert set(row["clear_properties"]) >= {"punchline", "analise_do_conteudo_juridico", "raciocinio_juridico"}


# --- script retroativo -----------------------------------------------------------------------


def _retrato(tmp_path, paginas):
    pasta = tmp_path / "retrato"
    pasta.mkdir()
    campos_base = ("numero_processo", "data_sessao", "tipo_registro", "classe_processo", "tema", "punchline",
                   "analise_do_conteudo_juridico", "raciocinio_juridico", "resultado", "votacao", "relator",
                   "origem", "tribunal", "partes", "pedido_vista", "eleicao")
    props, corpos = [], []
    for pid, campos, blocos in paginas:
        props.append({**{k: "" for k in campos_base}, "page_id": pid, "url": f"https://x/{pid}",
                      "created_time": "2026-07-04", **campos})
        corpos.append(json.dumps({"page_id": pid, "blocks": blocos}, ensure_ascii=False))
    (pasta / "props.json").write_text(json.dumps(props, ensure_ascii=False), encoding="utf-8")
    (pasta / "corpos.jsonl").write_text("\n".join(corpos), encoding="utf-8")
    return pasta


def _blocos_teor(ementa, decisao):
    return [{"type": "heading_2", "text": "Inteiro teor (acórdão — DJE)"}, {"type": "heading_3", "text": "Ementa"},
            {"type": "paragraph", "text": ementa}, {"type": "heading_3", "text": "Decisão / Acórdão"},
            {"type": "paragraph", "text": decisao}]


def test_retroativo_usa_a_copia_quando_o_corpo_traz_teor_de_outro_julgamento(tmp_path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manutencao_sessoes"))
    import reescrever_analise_copiada as r

    monkeypatch.setattr(r, "ART", tmp_path)
    embargos = ("EMBARGOS DE DECLARAÇÃO. RECEBIMENTO COMO PEDIDO DE RECONSIDERAÇÃO. O Tribunal, por maioria, "
                "deferiu o pedido para encaminhar a lista ao Poder Executivo, vencidos os Ministros X e Y.")
    copias = {"analise_do_conteudo_juridico": EMENTA_TO[:1900], "raciocinio_juridico": DECISAO_TO}
    pasta = _retrato(tmp_path, [
        ("pag-errada", copias, _blocos_teor("", embargos)),
        ("pag-certa", copias, _blocos_teor(EMENTA_TO, DECISAO_TO)),
        ("pag-vazia", {"analise_do_conteudo_juridico": "", "raciocinio_juridico": RACIOCINIO_REDIGIDO},
         _blocos_teor(EMENTA_TO, DECISAO_TO)),
        ("pag-video", {"analise_do_conteudo_juridico": ANALISE_REDIGIDA, "raciocinio_juridico": RACIOCINIO_REDIGIDO},
         _blocos_teor(EMENTA_TO, DECISAO_TO)),
    ])
    cands = {c["page_id"]: c for c in json.loads(r.detectar(pasta).read_text(encoding="utf-8"))}

    assert set(cands) == {"pag-errada", "pag-certa", "pag-vazia"}
    assert not cands["pag-errada"]["teor_confere"]
    assert r._fonte(cands["pag-errada"]) == (EMENTA_TO[:1900], DECISAO_TO)
    assert cands["pag-certa"]["teor_confere"]
    assert r._fonte(cands["pag-certa"])[0] == EMENTA_TO
    vazia = cands["pag-vazia"]["diag"]
    assert vazia["analise_do_conteudo_juridico"]["reescrever"] and not vazia["raciocinio_juridico"]["reescrever"]
