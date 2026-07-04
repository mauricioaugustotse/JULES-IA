import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_retroactive_judgments as audit


def test_classify_reason_maps_known_messages():
    assert audit.classify_reason(
        ["Item descartado: identificado como precedente citado, não como processo julgado."]
    ) == "precedente_citado"
    assert audit.classify_reason(
        ["Item descartado: densidade informacional insuficiente para representar julgamento autônomo."]
    ) == "densidade_insuficiente"
    assert audit.classify_reason(["Data da sessão ausente para criação."]) == "data_ausente"
    assert audit.classify_reason(["Resultado/votação insuficientes para criação."]) == "resultado_votacao_insuficiente"
    assert audit.classify_reason(["Tema/título vazio."]) == "tema_generico"
    assert audit.classify_reason(["Valor inválido para classe_processo"]) == "valor_invalido"
    assert audit.classify_reason(["mensagem desconhecida"]) == "outro"


def test_is_candidate_lost_judgment_by_cnj_and_category():
    assert audit.is_candidate_lost_judgment("blocked", "tema_generico", "0600263-63.2023.6.00.0000")
    assert audit.is_candidate_lost_judgment("skipped", "precedente_citado", "0600904-54")
    assert not audit.is_candidate_lost_judgment("skipped", "densidade_insuficiente", "")
    assert not audit.is_candidate_lost_judgment("blocked", "data_ausente", "0600904-54")


def test_iter_backlog_entries_parses_07_summary(tmp_path):
    playlist = tmp_path / "2024_PLabc"
    video = playlist / "026_o-PRlDsGUM4"
    video.mkdir(parents=True)
    (video / "07_backfill_summary.json").write_text(
        json.dumps(
            {
                "video_id": "o-PRlDsGUM4",
                "url": "https://www.youtube.com/watch?v=o-PRlDsGUM4",
                "title": "Sessão Plenária do dia 21 de maio de 2024",
                "publish_results": [
                    {"tema": "", "numero_processo": "0600263-63.2023.6.00.0000", "status": "blocked",
                     "errors": ["Tema/título vazio."], "warnings": []},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    entries = list(audit.iter_backlog_video_entries(tmp_path, {2024}))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["video_id"] == "o-PRlDsGUM4"
    assert entry["data_sessao"] == "2024-05-21"
    assert entry["publish_results"][0]["status"] == "blocked"
    # fora do filtro de anos
    assert list(audit.iter_backlog_video_entries(tmp_path, {2016})) == []


def test_iter_batch_entries_reads_rows_and_results(tmp_path):
    video = tmp_path / "20260703_184558" / "01_3N2aH5lgKhw"
    video.mkdir(parents=True)
    (video / "04b_enriched_preview_rows.json").write_text(
        json.dumps([{"data_sessao": "2026-06-23", "numero_processo": "0600904-54"}]),
        encoding="utf-8",
    )
    (video / "05_publish_results.json").write_text(
        json.dumps([
            {"tema": "PA", "numero_processo": "0600904-54", "status": "skipped",
             "errors": [], "warnings": ["Item descartado: identificado como precedente citado, não como processo julgado."]},
        ], ensure_ascii=False),
        encoding="utf-8",
    )
    entries = list(audit.iter_batch_video_entries(tmp_path))
    assert len(entries) == 1
    assert entries[0]["data_sessao"] == "2026-06-23"
    assert entries[0]["video_id"] == "3N2aH5lgKhw"
