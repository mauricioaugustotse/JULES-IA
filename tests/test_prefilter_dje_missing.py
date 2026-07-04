import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prefilter_dje_csv


CSV_HEADER = "numeroUnico,numeroProcesso,dataDecisao,siglaClasse,descricaoClasse,relatores,partes,textoEmenta,siglaTribunalJE\n"


def _write_csv(path: Path) -> None:
    rows = [
        # presente na base (mantida no reduzido)
        '"0600669-77.2024.6.02.0008","0600669-77","23/06/2026","AgRg-REspe","Agravo","Min. X","A x B","Ementa 1","TSE"\n',
        # FALTANTE em data de sessão conhecida
        '"0600354-59.2026.6.00.0000","0600354-59","30/06/2026","LT","Lista Tríplice","Min. Y","C","Ementa 2","TSE"\n',
        # fora de data de sessão conhecida (ignorada)
        '"0600999-99.2024.6.05.0000","0600999-99","02/01/2020","REspe","Recurso","Min. Z","D x E","Ementa 3","TSE"\n',
        # data conhecida mas decisão de TRE (filtrada)
        '"0600888-88.2024.6.05.0000","0600888-88","30/06/2026","REspe","Recurso","Des. W","F","Ementa 4","TRE-BA"\n',
    ]
    path.write_text(CSV_HEADER + "".join(rows), encoding="utf-8")


def test_prefilter_reports_missing_only_for_known_session_dates(tmp_path, monkeypatch):
    src = tmp_path / "bruto.csv"
    _write_csv(src)
    out_dir = tmp_path / "out"

    def fake_index(_dsid):
        return (
            {"06006697720246020008"},
            set(),
            set(),
            {"2026-06-23", "2026-06-30"},
        )

    monkeypatch.setattr(prefilter_dje_csv, "build_base_index", fake_index)
    monkeypatch.setattr(
        sys,
        "argv",
        ["prefilter_dje_csv.py", "--input", str(src), "--out", str(out_dir)],
    )
    assert prefilter_dje_csv.main() == 0

    missing_files = list(out_dir.glob("missing_*.csv"))
    assert len(missing_files) == 1
    lines = missing_files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # header + 1 faltante
    assert "0600354-59" in lines[1]
    summary = json.loads(next(out_dir.glob("missing_*.summary.json")).read_text(encoding="utf-8"))
    assert summary["missing_total"] == 1


def test_queue_from_artifacts_requires_individual_mention(tmp_path, monkeypatch):
    batch_root = tmp_path / "batch_gui"
    video_dir = batch_root / "20260703_000000" / "03_vidid30"
    video_dir.mkdir(parents=True)
    (video_dir / "04b_enriched_preview_rows.json").write_text(
        json.dumps([{"data_sessao": "2026-06-30", "numero_processo": "0602539-85"}]),
        encoding="utf-8",
    )
    (video_dir / "raw_global_response_chunk_01.txt").write_text(
        "lista triplice 0600354-59 de Sao Paulo apregoada em plenario",
        encoding="utf-8",
    )
    queue_file = tmp_path / "queue.jsonl"
    import vistoria_queue

    monkeypatch.setattr(vistoria_queue, "QUEUE_FILE", queue_file)

    missing_rows = [
        {  # mencionado no vídeo → entra na fila
            "numeroUnico": "0600354-59.2026.6.00.0000",
            "cnj20": "06003545920266000000",
            "numeroProcesso": "0600354-59",
            "dataDecisao": "30/06/2026",
        },
        {  # NÃO mencionado → presumido "em lista", fica fora
            "numeroUnico": "0611111-11.2026.6.00.0000",
            "cnj20": "06111111120266000000",
            "numeroProcesso": "0611111-11",
            "dataDecisao": "30/06/2026",
        },
    ]
    queued, presumed = prefilter_dje_csv._queue_missing_with_artifact_evidence(missing_rows, batch_root)
    assert queued == 1
    assert presumed == 1
    items = vistoria_queue.load_items("pending", queue_file)
    assert len(items) == 1
    assert items[0]["disposition"] == "faltante_dje"
    assert items[0]["data_sessao"] == "2026-06-30"
