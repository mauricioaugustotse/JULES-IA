import tse_backfill_2025_notion as backfill
import json
from tse_youtube_notion_core import NotionDataSourceSchema, PublishPreviewRow
from types import SimpleNamespace


def test_normalize_manifest_for_resume_resets_running_entries():
    manifest = {
        "playlist_url": "https://example.com",
        "year": 2025,
        "started_at": "2026-03-21T12:00:00",
        "completed_at": "",
        "videos": {
            "abc": {"status": "running", "title": "Sessão X"},
            "def": {"status": "done", "title": "Sessão Y"},
        },
    }

    normalized = backfill.normalize_manifest_for_resume(manifest)

    assert normalized["videos"]["abc"]["status"] == "pending"
    assert "interrompida" in normalized["videos"]["abc"]["error"]
    assert normalized["videos"]["def"]["status"] == "done"
    assert normalized["completed_at"] == ""


def test_repair_manifest_false_errors_promotes_summary_error_to_done(tmp_path):
    manifest = {
        "videos": {
            "abc": {
                "position": 1,
                "status": "error",
                "last_artifact": "07_backfill_summary.json",
                "finished_at": "",
            }
        },
        "recent_events": [],
    }
    artifact_dir = tmp_path / "001_abc"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "07_backfill_summary.json").write_text('{"ok": true}', encoding="utf-8")

    repaired = backfill.repair_manifest_false_errors(manifest, tmp_path)

    assert repaired["videos"]["abc"]["status"] == "done"
    assert repaired["videos"]["abc"]["summary"] == {"ok": True}
    assert repaired["recent_events"][-1]["type"] == "repair_false_error"


def test_existing_pages_snapshot_roundtrip(tmp_path):
    row = PublishPreviewRow(
        tema="Tema",
        classe_processo="PA",
        tipo_registro="Julgamento 1",
        eleicao="2024",
        origem="Brasília/DF",
        tribunal="TSE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2025-03-20",
        page_id="page-1",
        action="update",
    )
    grouped = {
        "abc123": [
            backfill.ExistingPageRecord(
                page_id="page-1",
                url="https://www.notion.so/page-1",
                video_id="abc123",
                row=row,
            )
        ]
    }

    backfill.dump_existing_pages_snapshot(tmp_path, grouped)
    loaded = backfill.load_existing_pages_snapshot(tmp_path)

    assert list(loaded) == ["abc123"]
    assert loaded["abc123"][0].page_id == "page-1"
    assert loaded["abc123"][0].row.numero_processo == row.numero_processo


def test_schema_snapshot_roundtrip(tmp_path):
    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "classe_processo": {
                    "type": "select",
                    "select": {"options": [{"name": "PA"}]},
                },
            }
        },
    )

    backfill.dump_schema_snapshot(tmp_path, schema)
    loaded = backfill.load_schema_snapshot(tmp_path)

    assert loaded.data_source_id == "ds-123"
    assert loaded.title_property_name == "tema"
    assert loaded.properties["classe_processo"].options == ["PA"]


def test_latest_progress_artifact_ignores_worker_logs(tmp_path):
    (tmp_path / "_worker_stdout.log").write_text("x", encoding="utf-8")
    target = tmp_path / "02_judgment_01.json"
    target.write_text("{}", encoding="utf-8")

    name, _mtime = backfill.latest_progress_artifact(tmp_path)

    assert name == "02_judgment_01.json"


def test_compute_next_worker_target_scales_up_when_healthy():
    target, reason = backfill.compute_next_worker_target(
        current_target=5,
        max_target=20,
        min_target=5,
        pending_videos=40,
        active_workers={},
        healthy_completions_since_scale=2,
        recent_capacity_errors=0,
        seconds_since_last_scale_up=999,
        seconds_since_last_scale_down=999,
        now=1000.0,
    )

    assert target == 6
    assert reason == "scale_up_healthy"


def test_compute_next_worker_target_scales_down_after_capacity_errors():
    target, reason = backfill.compute_next_worker_target(
        current_target=8,
        max_target=20,
        min_target=5,
        pending_videos=40,
        active_workers={},
        healthy_completions_since_scale=0,
        recent_capacity_errors=2,
        seconds_since_last_scale_up=999,
        seconds_since_last_scale_down=999,
        now=1000.0,
    )

    assert target == 7
    assert reason == "scale_down_capacity_errors"


def test_poll_active_worker_treats_summary_file_as_done(tmp_path):
    artifact_dir = tmp_path / "001_video"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "07_backfill_summary.json").write_text('{"ok": true}', encoding="utf-8")

    class DummyProcess:
        pid = 1234

        def poll(self):
            return None

        def kill(self):
            return None

        def communicate(self, timeout=5):
            return ("", "")

    handle = backfill.ActiveWorker(
        video=backfill.PlaylistVideo(position=1, video_id="abc", title="Sessão", url="https://youtu.be/abc"),
        process=DummyProcess(),
        artifact_dir=artifact_dir,
        started_at="2026-03-21T14:00:00",
        started_wall_time=0.0,
        deadline_monotonic=999999999.0,
        last_seen_artifact_name="07_backfill_summary.json",
        last_seen_artifact_mtime=(artifact_dir / "07_backfill_summary.json").stat().st_mtime,
        last_progress_wall_time=0.0,
    )

    result = backfill.poll_active_worker(handle)

    assert result["status"] == "done"
    assert result["summary"] == {"ok": True}


def test_update_manifest_retries_permission_error(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    attempts = {"count": 0}
    original_replace = backfill.Path.replace

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("locked")
        return original_replace(self, target)

    monkeypatch.setattr(backfill.Path, "replace", flaky_replace)
    monkeypatch.setattr(backfill, "MANIFEST_REPLACE_RETRIES", 5)
    monkeypatch.setattr(backfill, "MANIFEST_REPLACE_RETRY_SLEEP_SECONDS", 0.0)

    backfill.update_manifest(manifest_path, {"videos": {}})

    assert attempts["count"] == 3
    assert manifest_path.exists()


def test_build_worker_command_does_not_propagate_removed_ground_origem_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        playlist_url="https://www.youtube.com/playlist?list=x",
        year=2025,
        skip_news=True,
        no_trash_unmatched_precedents=False,
    )
    video = backfill.PlaylistVideo(position=1, video_id="abc123", title="Sessão", url="https://youtu.be/abc123")

    command, _project_dir = backfill.build_worker_command(video, args, tmp_path / "root")

    assert "--ground-origem-with-search" not in command


def test_build_worker_command_embeds_worker_video_id_when_it_starts_with_hyphen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        playlist_url="https://www.youtube.com/playlist?list=x",
        year=2024,
        skip_news=True,
        no_trash_unmatched_precedents=False,
    )
    video = backfill.PlaylistVideo(position=1, video_id="-sCwzKfLVrw", title="Sessão", url="https://youtu.be/-sCwzKfLVrw")

    command, _project_dir = backfill.build_worker_command(video, args, tmp_path / "root")

    assert "--worker-video-id=-sCwzKfLVrw" in command


def test_build_worker_command_prefers_project_venv_python(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    venv_python = tmp_path / ".venv" / "Scripts" / "python.exe"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    args = SimpleNamespace(
        playlist_url="https://www.youtube.com/playlist?list=x",
        year=2024,
        skip_news=True,
        no_trash_unmatched_precedents=False,
    )
    video = backfill.PlaylistVideo(position=1, video_id="abc123", title="Sessão", url="https://youtu.be/abc123")

    command, _project_dir = backfill.build_worker_command(video, args, tmp_path / "root")

    assert command[0] == str(venv_python)


def test_restrict_repaired_row_to_focus_only_keeps_partes_e_advogados_changes():
    original = PublishPreviewRow(
        tema="Tema original",
        punchline="Punchline original",
        classe_processo="PA",
        tipo_registro="Julgamento 3",
        eleicao="2024",
        origem="TSE",
        tribunal="TSE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Unânime",
        data_sessao="2024-03-20",
        partes=[],
        advogados=[],
    )
    repaired = original.model_copy(deep=True)
    repaired.tema = "Tema alterado"
    repaired.origem = "Brasília/DF"
    repaired.partes = ["Parte A"]
    repaired.advogados = ["Advogado A"]

    restricted = backfill._restrict_repaired_row_to_focus(original, repaired, "partes-advogados")

    assert restricted.tema == original.tema
    assert restricted.origem == original.origem
    assert restricted.partes == ["Parte A"]
    assert restricted.advogados == ["Advogado A"]


def test_row_needs_partes_advogados_repair_flags_conjoined_names_and_shared_advogado_suffix():
    row = PublishPreviewRow(
        partes=["José Auricchio Júnior (Prefeito eleito) e Carlos Humberto Seraphim (Vice-prefeito eleito) (Recorrido)"],
        advogados=["Dr. João da Silva e Dra. Maria Souza (pelo recorrente)"],
    )

    assert backfill._row_needs_partes_advogados_repair(row) is True


def test_restrict_repaired_row_to_focus_deterministic_core_only_keeps_blank_fill_fields():
    original = PublishPreviewRow(
        tema="Tema original",
        punchline="Punchline original",
        classe_processo="PA",
        tipo_registro="Julgamento 3",
        eleicao="",
        origem="TSE",
        tribunal="TSE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="",
        resultado="",
        votacao="",
        data_sessao="2024-03-20",
    )
    repaired = original.model_copy(deep=True)
    repaired.tema = "Tema alterado"
    repaired.relator = "Min. Cármen Lúcia"
    repaired.resultado = "Aprovada"
    repaired.votacao = "Unânime"
    repaired.eleicao = "2024"

    restricted = backfill._restrict_repaired_row_to_focus(original, repaired, "deterministic-core")

    assert restricted.tema == original.tema
    assert restricted.relator == "Min. Cármen Lúcia"
    assert restricted.resultado == "Aprovada"
    assert restricted.votacao == "Unânime"
    assert restricted.eleicao == "2024"


def test_restrict_repaired_row_to_focus_schema_core_only_keeps_schema_controlled_changes():
    original = PublishPreviewRow(
        tema="Tema original",
        punchline="Punchline original",
        classe_processo="PA",
        tipo_registro="Julgamento 3",
        eleicao="2024",
        origem="TRE/SE",
        tribunal="TRE-SE",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        relator="",
        pedido_vista="",
        resultado="",
        votacao="",
        data_sessao="2024-03-20",
    )
    repaired = original.model_copy(deep=True)
    repaired.tema = "Tema alterado"
    repaired.relator = "Min. Sérgio Banhos"
    repaired.pedido_vista = "Min. Alexandre de Moraes"
    repaired.resultado = "Desprovido"
    repaired.votacao = "Por maioria"
    repaired.classe_processo = "AgRg-REspe"
    repaired.tribunal = "TRE-MG"
    repaired.origem = "Belo Horizonte/MG"
    repaired.tipo_registro = "Julgamento 7"

    restricted = backfill._restrict_repaired_row_to_focus(original, repaired, "schema-core")

    assert restricted.tema == original.tema
    assert restricted.youtube_link == original.youtube_link
    assert restricted.relator == "Min. Sérgio Banhos"
    assert restricted.pedido_vista == "Min. Alexandre de Moraes"
    assert restricted.resultado == "Desprovido"
    assert restricted.votacao == "Por maioria"
    assert restricted.classe_processo == "AgRg-REspe"
    assert restricted.tribunal == "TRE-MG"
    assert restricted.origem == "Belo Horizonte/MG"
    assert restricted.tipo_registro == "Julgamento 7"


def test_apply_deterministic_blank_completion_from_artifact_only_fills_blank_fields():
    original = PublishPreviewRow(
        numero_processo="0600001-01.2024.6.00.0000",
        relator="",
        resultado="",
        votacao="",
        eleicao="",
    )
    repaired = original.model_copy(deep=True)
    artifact_item = backfill.JudgmentItemExtraction(
        numero_processo="0600001-01.2024.6.00.0000",
        relator="Ministro Benedito Gonçalves",
        resultado_final="Desprovido",
        votacao="Unânime",
        eleicao="2022",
    )

    backfill._apply_deterministic_blank_completion_from_artifact(original, repaired, artifact_item)

    assert repaired.relator == "Min. Benedito Gonçalves"
    assert repaired.resultado == "Desprovido"
    assert repaired.votacao == "Unânime"
    assert repaired.eleicao == "2022"


def test_apply_deterministic_blank_completion_from_artifact_preserves_existing_values():
    original = PublishPreviewRow(
        numero_processo="0600001-01.2024.6.00.0000",
        relator="Min. Cármen Lúcia",
        resultado="Aprovada",
        votacao="Por maioria",
        eleicao="2024",
    )
    repaired = original.model_copy(deep=True)
    artifact_item = backfill.JudgmentItemExtraction(
        numero_processo="0600001-01.2024.6.00.0000",
        relator="Ministro Benedito Gonçalves",
        resultado_final="Desprovido",
        votacao="Unânime",
        eleicao="2022",
    )

    backfill._apply_deterministic_blank_completion_from_artifact(original, repaired, artifact_item)

    assert repaired.relator == "Min. Cármen Lúcia"
    assert repaired.resultado == "Aprovada"
    assert repaired.votacao == "Por maioria"
    assert repaired.eleicao == "2024"


def test_apply_schema_core_rewrite_from_artifact_canonicalizes_dynamic_and_controlled_fields():
    repaired = PublishPreviewRow(
        numero_processo="0000697-22.2016.6.13.0000",
        classe_processo="PA",
        tipo_registro="Julgamento 5",
        origem="TRE/MG",
        tribunal="",
        relator="",
        pedido_vista="",
        resultado="",
        votacao="",
        eleicao="",
        analise_do_conteudo_juridico="Voto-vista do Ministro Alexandre de Moraes.",
    )
    artifact_item = backfill.JudgmentItemExtraction(
        numero_processo="0000697-22.2016.6.13.0000",
        classe_processo="Agravo Regimental no Agravo em Recurso Especial Eleitoral",
        relator="Ministro Sérgio Banhos",
        pedido_vista="",
        resultado_final="Agravo regimental desprovido",
        votacao="por maioria",
        eleicao="2020",
        origem="Belo Horizonte/MG",
        tre="TRE-MG",
        uf="MG",
        composicao=[
            "Ministro Sérgio Banhos (Relator)",
            "Ministro Alexandre de Moraes (Voto-vista)",
        ],
    )

    backfill._apply_schema_core_rewrite_from_artifact(repaired, artifact_item)

    assert repaired.relator == "Min. Sérgio Banhos"
    assert repaired.pedido_vista == "Min. Alexandre de Moraes"
    assert repaired.resultado == "Desprovido"
    assert repaired.votacao == "Por maioria"
    assert repaired.eleicao == "2020"
    assert repaired.classe_processo == "AgRg-REspe"
    assert repaired.tribunal == "TRE-MG"
    assert repaired.origem == "Belo Horizonte/MG"


def test_build_worker_popen_kwargs_uses_replace_for_decode_errors(tmp_path):
    kwargs = backfill.build_worker_popen_kwargs(["python", "worker.py"], tmp_path)

    assert kwargs["args"] == ["python", "worker.py"]
    assert kwargs["cwd"] == str(tmp_path)
    assert kwargs["text"] is True
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_audit_existing_year_flags_origin_class_votacao_and_link_anomalies():
    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0600071-96.2025.6.00.0000",
        classe_processo="PA",
        origem="Tribunal de Justiça de São Paulo/SP",
        resultado="Suspenso por vista",
        votacao="Unânime",
        youtube_link="https://www.youtube.com/watch?v=wrongvideo&t=540",
        analise_do_conteudo_juridico="Agravo em recurso especial eleitoral envolvendo fraude à cota de gênero.",
        data_sessao="2025-02-20",
    )
    grouped = {
        "expectedvideo": [
            backfill.ExistingPageRecord(
                page_id="page-1",
                url="https://www.notion.so/page-1",
                video_id="expectedvideo",
                row=row,
            )
        ]
    }

    summary = backfill.audit_existing_year(grouped)

    assert summary["stats"]["origem_invalid_label"] == 1
    assert summary["stats"]["classe_mismatch"] == 1
    assert summary["stats"]["votacao_inconsistent"] == 1
    assert summary["stats"]["youtube_video_mismatch"] == 1


def test_audit_existing_year_flags_composition_lt6_and_gt7():
    lt6_composition = [
        "Min. Alexandre de Moraes",
        "Min. Cármen Lúcia",
        "Min. Benedito Gonçalves",
        "Min. Raul Araújo",
        "Min. Sérgio Banhos",
    ]
    gt7_composition = [
        "Min. Alexandre de Moraes",
        "Min. Cármen Lúcia",
        "Min. Benedito Gonçalves",
        "Min. Raul Araújo",
        "Min. Sérgio Banhos",
        "Min. Ricardo Lewandowski",
        "Min. Mauro Campbell Marques",
        "Min. Carlos Horbach",
    ]
    grouped = {
        "video-1": [
            backfill.ExistingPageRecord(
                page_id="page-lt6",
                url="https://www.notion.so/page-lt6",
                video_id="video-1",
                    row=PublishPreviewRow(
                        tema="Tema útil",
                        numero_processo="0600001-01.2024.6.00.0000",
                        data_sessao="2024-05-01",
                        composicao=lt6_composition,
                    ),
                ),
                backfill.ExistingPageRecord(
                    page_id="page-gt7",
                    url="https://www.notion.so/page-gt7",
                video_id="video-1",
                    row=PublishPreviewRow(
                        tema="Tema útil",
                        numero_processo="0600002-01.2024.6.00.0000",
                        data_sessao="2024-05-01",
                        composicao=gt7_composition,
                    ),
                ),
            ]
        }

    summary = backfill.audit_existing_year(grouped)

    assert summary["stats"]["composicao_incomplete"] == 2
    assert summary["stats"]["composicao_lt6"] == 1
    assert summary["stats"]["composicao_gt7"] == 1


def test_repair_existing_video_rows_trashes_unproven_records(monkeypatch):
    empty_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: empty_context)

    requests = []

    class DummyNotionClient:
        def _request(self, method, path, json=None):
            requests.append((method, path, json))
            return {}

    row = PublishPreviewRow(
        tema="Tema útil",
        numero_processo="0600001-01.2024.6.00.0000",
        youtube_link="https://www.youtube.com/watch?v=abc123&t=10",
        data_sessao="2024-02-20",
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="abc123",
            row=row,
        )
    ]

    summary = backfill.repair_existing_video_rows(
        video_id="abc123",
        records=records,
        notion_client=DummyNotionClient(),
        notion_schema=None,
        playlist_url="https://www.youtube.com/playlist?list=test",
        year=2024,
        gemini_api_key="",
        model="",
        use_theme_api=False,
        apply_updates=True,
    )

    assert summary["trashed_unproven_pages"] == 1
    assert summary["updated_pages"] == 0
    assert requests == [("PATCH", "/pages/page-1", {"in_trash": True})]


def test_cleanup_notion_schema_residue_removes_probe_property_and_reports_color_state():
    class DummyNotionClient:
        data_source_id = "ds-123"

        def __init__(self):
            self.payload = {
                "properties": {
                    "tema": {"type": "title", "title": {}},
                    "partes": {"type": "multi_select", "multi_select": {"options": [{"name": "A", "color": "red"}]}},
                    "advogados": {"type": "multi_select", "multi_select": {"options": [{"name": "B", "color": "blue"}]}},
                    "probe_expand_default_large": {"type": "multi_select", "multi_select": {"options": [{"name": "X001", "color": "default"}]}},
                }
            }
            self.removed = []
            self.normalized = []

        def _request(self, method, path, json=None):
            if method == "GET":
                return self.payload
            if method == "PATCH":
                for property_name, property_value in (json or {}).get("properties", {}).items():
                    if property_value is None:
                        self.removed.append(property_name)
                        self.payload["properties"].pop(property_name, None)
                return {}
            raise AssertionError((method, path, json))

        def query_data_source(self):
            return [
                {
                    "id": "page-1",
                    "properties": {
                        "partes": {"multi_select": [{"name": "A"}]},
                        "advogados": {"multi_select": [{"name": "B"}]},
                    },
                }
            ]

    client = DummyNotionClient()

    summary = backfill.cleanup_notion_schema_residue(client, normalize_colors=True, apply_changes=True)

    assert "probe_expand_default_large" in summary["removed_properties"]
    assert summary["color_summary"]["updated"] is False
    assert summary["color_summary"]["retro_normalization_supported"] is False
    assert summary["color_summary"]["properties"] == [
        {
            "property": "partes",
            "nondefault_options": 1,
            "used_options": 1,
            "retro_normalization_supported": False,
        },
        {
            "property": "advogados",
            "nondefault_options": 1,
            "used_options": 1,
            "retro_normalization_supported": False,
        },
    ]


def test_repair_existing_video_rows_reorders_and_sanitizes(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2025_PLtest"
    artifact_dir = run_dir / "001_abc123"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2025-03-20",
                "composicao": [
                    "Ministra Cármen Lúcia",
                    "Ministro André Mendonça",
                    "Ministra Isabel Gallotti",
                    "Ministro Kassio Nunes Marques",
                    "Ministro Floriano de Azevedo Marques",
                    "Ministro Alexandre de Moraes",
                    "Ministro Ramos Tavares",
                ],
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "start_seconds": 120,
                "items": [
                    {"numero_processo": "0600999-99.2024.6.00.0000"},
                    {"numero_processo": "0600001-01.2024.6.00.0000"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {
                    "type": "select",
                    "select": {"options": [{"name": "Julgamento 1"}, {"name": "Julgamento 2"}]},
                },
                "origem": {"type": "select", "select": {"options": [{"name": "Brasília/DF"}]}},
                "tribunal": {"type": "select", "select": {"options": [{"name": "TSE"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "partes": {"type": "multi_select", "multi_select": {"options": []}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "punchline": {"type": "rich_text", "rich_text": {}},
                "fundamentacao_normativa": {"type": "rich_text", "rich_text": {}},
                "raciocinio_juridico": {"type": "rich_text", "rich_text": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )

    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="abc123",
            row=PublishPreviewRow(
                tema="Processo 0600001-01.2024.6.00.0000",
                tipo_registro="Julgamento 2",
                origem="Brasília - DF",
                tribunal="TSE",
                numero_processo="0600001-01.2024.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=abc123&t=120",
                partes=["Dr. João da Silva", "Recorrente: Alice"],
                composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
                punchline="Fraude à cota de gênero.",
                raciocinio_juridico="Raciocínio Jurídico Aplicado ao Caso Concreto\nFundamento 1.",
                fundamentacao_normativa="Fundamentação Normativa e Dispositivos Citados\nArt. 10, § 3º.",
                data_sessao="2025-03-20",
                page_id="page-1",
                action="update",
            ),
        ),
        backfill.ExistingPageRecord(
            page_id="page-2",
            url="https://www.notion.so/page-2",
            video_id="abc123",
            row=PublishPreviewRow(
                tema="Processo 0600999-99.2024.6.00.0000",
                tipo_registro="Julgamento 1",
                origem="Brasília/DF",
                tribunal="TSE",
                numero_processo="0600999-99.2024.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=abc123&t=120",
                partes=["Agravante: Bob"],
                composicao=["Min. Cármen Lúcia"],
                punchline="Conduta vedada por uso de bem público.",
                data_sessao="2025-03-20",
                page_id="page-2",
                action="update",
            ),
        ),
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="abc123",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2025,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 2
    by_page = {page_id: row for page_id, row in notion.updated}
    assert by_page["page-2"].tipo_registro == "Julgamento 1"
    assert by_page["page-1"].tipo_registro == "Julgamento 2"
    assert by_page["page-1"].tema == "Fraude à cota de gênero"
    assert by_page["page-2"].tema == "Conduta vedada por uso de bem público"
    assert by_page["page-1"].partes == ["Alice (Recorrente)"]
    assert by_page["page-1"].raciocinio_juridico == "Fundamento 1."
    assert by_page["page-1"].fundamentacao_normativa == "Art. 10, § 3º."
    assert len(by_page["page-2"].composicao) == 7


def test_repair_existing_video_rows_prefers_historical_snapshot_composition(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2025_PLtest"
    artifact_dir = run_dir / "001_pIQzyr3o-PE"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2025-12-19",
                "composicao": [
                    "Ministra Cármen Lúcia (Presidente)",
                    "Ministro Nunes Marques",
                    "Ministro André Mendonça",
                    "Ministro Antônio Carlos Ferreira",
                    "Ministro Villas Bôas Cueva",
                    "Ministro Floriano de Azevedo Marques",
                    "Ministra Estela Aranha",
                    "Ministro Alexandre de Moraes",
                    "Ministro João Paulo",
                    "Procurador-Geral da República Paulo Gonet",
                ],
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600433-71"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / backfill.EXISTING_PAGES_SNAPSHOT_NAME).write_text(
        json.dumps(
            {
                "pIQzyr3o-PE": [
                    {
                        "page_id": "page-1",
                        "url": "https://www.notion.so/page-1",
                        "video_id": "pIQzyr3o-PE",
                        "row": PublishPreviewRow(
                            tema="Tema antigo",
                            numero_processo="0600433-71",
                            youtube_link="https://www.youtube.com/watch?v=pIQzyr3o-PE&t=1470",
                            composicao=[
                                "Min. André Mendonça",
                                "Min. Antônio Carlos Ferreira",
                                "Min. Cármen Lúcia",
                                "Min. Estela Aranha",
                                "Min. Floriano de Azevedo Marques",
                                "Min. Nunes Marques",
                                "Min. Ricardo Villas Bôas Cueva",
                            ],
                            data_sessao="2025-12-19",
                            action="update",
                        ).model_dump(mode="json"),
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "origem": {"type": "select", "select": {"options": []}},
                "tribunal": {"type": "select", "select": {"options": [{"name": "TRE-AL"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "punchline": {"type": "rich_text", "rich_text": {}},
                "fundamentacao_normativa": {"type": "rich_text", "rich_text": {}},
                "raciocinio_juridico": {"type": "rich_text", "rich_text": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="pIQzyr3o-PE",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 1",
                tribunal="TRE-AL",
                numero_processo="0600433-71",
                youtube_link="https://www.youtube.com/watch?v=pIQzyr3o-PE&t=1470",
                composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
                data_sessao="2025-12-19",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="pIQzyr3o-PE",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2025,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    assert len(notion.updated[0][1].composicao) == 7


def test_repair_existing_video_rows_composition_focus_preserves_valid_six_member_row(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2025_PLtest"
    artifact_dir = run_dir / "001_validsix"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2025-05-01",
                "composicao": [
                    "Ministra Cármen Lúcia",
                    "Ministro Nunes Marques",
                    "Ministro André Mendonça",
                    "Ministro Antônio Carlos Ferreira",
                    "Ministro Floriano de Azevedo Marques",
                    "Ministra Estela Aranha",
                    "Ministro Excedente",
                ],
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600433-71"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    valid_six = [
        "Min. Cármen Lúcia",
        "Min. Nunes Marques",
        "Min. André Mendonça",
        "Min. Antônio Carlos Ferreira",
        "Min. Floriano de Azevedo Marques",
        "Min. Estela Aranha",
    ]
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="validsix",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 1",
                numero_processo="0600433-71",
                youtube_link="https://www.youtube.com/watch?v=validsix&t=1470",
                composicao=valid_six,
                data_sessao="2025-05-01",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="validsix",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2025,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
        repair_focus="composition",
    )

    assert summary["updated_pages"] == 0


def test_repair_existing_video_rows_composition_focus_uses_best_session_date_composition(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2022_PLtest"
    artifact_dir = run_dir / "001_onemin"
    artifact_dir.mkdir(parents=True)
    invalid_session = [
        "Min. Alexandre de Moraes",
        "Min. Cármen Lúcia",
        "Min. Nunes Marques",
        "Min. Raul Araújo",
        "Min. Floriano de Azevedo Marques",
        "Min. Ramos Tavares",
        "Min. Isabel Gallotti",
        "Min. Edson Fachin",
    ]
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2024-05-21",
                "composicao": invalid_session,
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "numero_processo": "0600238-92",
                        "composicao": ["Edson Fachin (Presidente)"],
                        "data_sessao": "2024-05-21",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600238-92"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 3"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    valid_seven = [
        "Min. Alexandre de Moraes",
        "Min. Cármen Lúcia",
        "Min. Nunes Marques",
        "Min. Raul Araújo",
        "Min. Floriano de Azevedo Marques",
        "Min. Ramos Tavares",
        "Min. Isabel Gallotti",
    ]
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="onemin",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 3",
                numero_processo="0600238-92",
                youtube_link="https://www.youtube.com/watch?v=onemin&t=1868",
                composicao=["Min. Edson Fachin"],
                data_sessao="2022-05-03",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="onemin",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2022,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
        repair_focus="composition",
        best_composition_by_session_date={"2022-05-03": valid_seven},
    )

    assert summary["updated_pages"] == 1
    assert notion.updated[0][1].composicao == valid_seven


def test_repair_existing_video_rows_composition_focus_uses_valid_artifact_same_date_composition(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2022_PLtest"
    artifact_dir = run_dir / "001_samevideo"
    artifact_dir.mkdir(parents=True)
    valid_seven = [
        "Min. Alexandre de Moraes",
        "Min. Ricardo Lewandowski",
        "Min. Cármen Lúcia",
        "Min. Benedito Gonçalves",
        "Min. Raul Araújo",
        "Min. Sérgio Banhos",
        "Min. Carlos Horbach",
    ]
    invalid_eight = valid_seven + ["Min. Maria Cláudia Bucchianeri"]
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2022-09-22",
                "composicao": invalid_eight,
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "numero_processo": "0601022-69.2022.6.00.0000",
                        "composicao": invalid_eight,
                        "data_sessao": "2022-09-22",
                    },
                    {
                        "numero_processo": "0601101-48.2022.6.00.0000",
                        "composicao": valid_seven,
                        "data_sessao": "2022-09-22",
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0601022-69.2022.6.00.0000"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="samevideo",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 1",
                numero_processo="0601022-69.2022.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=samevideo&t=1080",
                composicao=invalid_eight,
                data_sessao="2022-09-22",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="samevideo",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2022,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
        repair_focus="composition",
        best_composition_by_session_date={},
    )

    assert summary["updated_pages"] == 1
    assert notion.updated[0][1].composicao == valid_seven


def test_repair_existing_video_rows_composition_focus_uses_nearest_valid_session_date(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2022_PLtest"
    artifact_dir = run_dir / "001_nearest"
    artifact_dir.mkdir(parents=True)
    invalid_large = [
        "Min. Luís Roberto Barroso",
        "Min. Edson Fachin",
        "Min. Alexandre de Moraes",
        "Min. Mauro Campbell Marques",
        "Min. Benedito Gonçalves",
        "Min. Sérgio Banhos",
        "Min. Carlos Horbach",
        "Min. Flávio Dino",
    ]
    valid_nearest = [
        "Min. Luís Roberto Barroso",
        "Min. Edson Fachin",
        "Min. Sérgio Banhos",
        "Min. Alexandre de Moraes",
        "Min. Carlos Horbach",
        "Min. Benedito Gonçalves",
        "Min. Mauro Campbell Marques",
    ]
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2024-05-21",
                "composicao": invalid_large,
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "numero_processo": "0601236-02",
                        "composicao": ["Ministro Sérgio Banhos (Relator)"],
                        "data_sessao": "2024-05-21",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0601236-02"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="nearest",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 1",
                numero_processo="0601236-02",
                youtube_link="https://www.youtube.com/watch?v=nearest&t=810",
                composicao=["Min. Sérgio Banhos"],
                data_sessao="2022-02-17",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="nearest",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2022,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
        repair_focus="composition",
        best_composition_by_session_date={"2022-02-15": valid_nearest},
    )

    assert summary["updated_pages"] == 1
    assert notion.updated[0][1].composicao == valid_nearest


def test_repair_existing_video_rows_composition_focus_does_not_downgrade_invalid_to_one_member(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2022_PLtest"
    artifact_dir = run_dir / "001_invalid"
    artifact_dir.mkdir(parents=True)
    invalid_large = [
        "Min. Alexandre de Moraes",
        "Min. Cármen Lúcia",
        "Min. Nunes Marques",
        "Min. Raul Araújo",
        "Min. Floriano de Azevedo Marques",
        "Min. Ramos Tavares",
        "Min. Isabel Gallotti",
        "Min. Edson Fachin",
        "Min. Maria Claudia Bucchianeri",
    ]
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2024-05-21",
                "composicao": invalid_large,
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "numero_processo": "0601236-02",
                        "composicao": ["Ministro Sérgio Banhos (Relator)"],
                        "data_sessao": "2024-05-21",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0601236-02"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "composicao": {"type": "multi_select", "multi_select": {"options": []}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="invalid",
            row=PublishPreviewRow(
                tema="Tema atual",
                tipo_registro="Julgamento 1",
                numero_processo="0601236-02",
                youtube_link="https://www.youtube.com/watch?v=invalid&t=810",
                composicao=invalid_large,
                data_sessao="2022-02-17",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="invalid",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2022,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
        repair_focus="composition",
    )

    assert summary["updated_pages"] == 0
    assert notion.updated == []


def test_repair_existing_video_rows_uses_artifact_fields_for_missing_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2025_PLtest"
    artifact_dir = run_dir / "001_xyz"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "start_seconds": 300,
                "items": [
                    {
                        "numero_processo": "0600001-01.2024.6.25.0000",
                        "classe_processo": "AgRg-REspe",
                        "origem": "Tribunal Regional Eleitoral de Sergipe/SE",
                        "relator": "Ministro Alexandre de Moraes",
                        "votacao": "por unanimidade",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "classe_processo": {"type": "select", "select": {"options": [{"name": "AgRg-REspe"}]}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "TRE/SE"}]}},
                "tribunal": {"type": "select", "select": {"options": [{"name": "TRE-SE"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "relator": {"type": "select", "select": {"options": [{"name": "Min. Alexandre de Moraes"}]}},
                "votacao": {"type": "select", "select": {"options": [{"name": "Unânime"}]}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Tema suficiente",
                origem="",
                tribunal="TRE-SE",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=xyz&t=10",
                data_sessao="2025-03-20",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="xyz",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2025,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.classe_processo == "AgRg-REspe"
    assert repaired.origem == "TRE/SE"
    assert repaired.relator == "Min. Alexandre de Moraes"
    assert repaired.votacao == "Unânime"


def test_repair_existing_video_rows_infers_arespe_from_artifact_resultado_text(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2025_PLtest"
    artifact_dir = run_dir / "001_arespe"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "start_seconds": 1020,
                "items": [
                    {
                        "numero_processo": "0600071-96.2025.6.00.0000",
                        "classe_processo": "",
                        "origem": "Tucuruí - PA",
                        "relator": "Cármen Lúcia",
                        "votacao": "Unânime",
                        "resultado_final": "Parcial provimento aos agravos em recurso especial para afastar a multa imposta na origem.",
                        "analise_do_conteudo_juridico": "Abuso de poder econômico em Tucuruí/PA.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "classe_processo": {"type": "select", "select": {"options": [{"name": "AREspe"}, {"name": "PA"}]}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "Tucuruí/PA"}]}},
                "tribunal": {"type": "select", "select": {"options": [{"name": "TRE-PA"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "relator": {"type": "select", "select": {"options": [{"name": "Min. Cármen Lúcia"}]}},
                "votacao": {"type": "select", "select": {"options": [{"name": "Unânime"}]}},
                "resultado": {"type": "select", "select": {"options": [{"name": "Provido em parte"}]}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="arespe",
            row=PublishPreviewRow(
                tema="Inelegibilidade e aplicação de multa ao titular da chapa",
                origem="Tucuruí/PA",
                tribunal="TRE-PA",
                numero_processo="0600071-96.2025.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=arespe&t=1020",
                data_sessao="2025-04-03",
                resultado="Provido em parte",
                classe_processo="PA",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="arespe",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2025,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.classe_processo == "AREspe"
    assert repaired.origem == "Tucuruí/PA"
    assert repaired.votacao == "Unânime"


def test_repair_existing_video_rows_promotes_special_process_number_from_artifact(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_special"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "start_seconds": 420,
                "items": [
                    {
                        "numero_processo": "ADI 7228",
                        "classe_processo": "ADI",
                        "origem": "Tribunal Superior Eleitoral",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "classe_processo": {"type": "select", "select": {"options": [{"name": "ADI"}]}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "TSE"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="special",
            row=PublishPreviewRow(
                tema="Tema suficiente",
                classe_processo="ADI",
                tipo_registro="Julgamento 1",
                origem="",
                numero_processo="7228",
                youtube_link="https://www.youtube.com/watch?v=special&t=10",
                data_sessao="2024-10-01",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="special",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.numero_processo == "ADI 7228"
    assert repaired.origem == "TSE"


def test_repair_existing_video_rows_overrides_wrong_item_date_with_session_date(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_xyz"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2024-05-23",
                "composicao": [],
                "judgments": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600001-01"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Tema suficiente",
                tipo_registro="Julgamento 1",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=xyz&t=10",
                data_sessao="2023-05-23",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="xyz",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.data_sessao == "2024-05-23"


def test_infer_session_date_from_video_title():
    assert backfill.infer_session_date_from_video_title("Sessão Plenária - 23 de Maio de 2024") == "2024-05-23"
    assert backfill.infer_session_date_from_video_title("Sessão de Encerramento do Ano Judiciário - 19 de Dezembro de 2024") == "2024-12-19"
    assert backfill.infer_session_date_from_video_title("Sessão de Encerramento do Semestre Forense - 1º de Julho de 2024") == "2024-07-01"
    assert backfill.infer_session_date_from_video_title("Sessão Plenária - 13 de Fevereiro 2025") == "2025-02-13"


def test_choose_authoritative_repair_session_date_prefers_title_hint():
    assert backfill._choose_authoritative_repair_session_date(
        "2024-05-21",
        year=2024,
        session_date_hint="2024-08-27",
        artifact_session_date="2024-05-21",
    ) == "2024-08-27"


def test_choose_authoritative_repair_session_date_preserves_current_valid_date_without_hint():
    assert backfill._choose_authoritative_repair_session_date(
        "2024-08-27",
        year=2024,
        session_date_hint="",
        artifact_session_date="2024-05-21",
    ) == "2024-08-27"


def test_row_has_soft_local_association_signal_accepts_publish_summary_process_key():
    row = PublishPreviewRow(numero_processo="0600263-57.2022.6.00.0000", classe_processo="CTA")
    context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys={backfill.canonicalize_numero_processo("0600263-57.2022.6.00.0000")},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    assert backfill._row_has_soft_local_association_signal(row, context)


def test_find_target_video_falls_back_to_manifest_entry_when_playlist_lookup_misses(monkeypatch, tmp_path):
    root_dir = tmp_path / "2024_playlist"
    root_dir.mkdir(parents=True)
    (root_dir / "manifest.json").write_text(
        json.dumps(
            {
                "videos": {
                    "missing123": {
                        "position": 14,
                        "title": "Sessão Plenária - 14 de Maio de 2024",
                        "url": "https://www.youtube.com/watch?v=missing123",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(backfill, "load_playlist_videos", lambda _url: [])

    video = backfill.find_target_video("https://www.youtube.com/playlist?list=PLx", 2024, "missing123", root_dir=root_dir)

    assert video.video_id == "missing123"
    assert video.position == 14
    assert video.url == "https://www.youtube.com/watch?v=missing123"


def test_find_target_video_falls_back_to_main_year_manifest_when_rerun_root_has_no_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    main_root = backfill.BACKFILL_ROOT / "2023_PLx"
    main_root.mkdir(parents=True)
    (main_root / "manifest.json").write_text(
        json.dumps(
            {
                "videos": {
                    "missing456": {
                        "position": 36,
                        "title": "Sessão Plenária de Abertura do 2º Semestre Forense de 2023",
                        "url": "https://www.youtube.com/watch?v=missing456",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    rerun_root = backfill.BACKFILL_ROOT / "_rerun_errors_2023_20260329_081308"
    rerun_root.mkdir(parents=True)
    monkeypatch.setattr(backfill, "load_playlist_videos", lambda _url: [])

    video = backfill.find_target_video("https://www.youtube.com/playlist?list=PLx", 2023, "missing456", root_dir=rerun_root)

    assert video.video_id == "missing456"
    assert video.position == 36
    assert video.url == "https://www.youtube.com/watch?v=missing456"


def test_find_artifact_dir_for_video_prefers_latest_rerun_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLx"
    current_root = backfill.BACKFILL_ROOT / "2020_PLx"
    rerun_root = backfill.BACKFILL_ROOT / "_rerun_errors_2020_20260402_222108"
    (current_root / "033_to-rpGmakME").mkdir(parents=True)
    (rerun_root / "033_to-rpGmakME").mkdir(parents=True)

    selected = backfill.find_artifact_dir_for_video(playlist_url, 2020, "to-rpGmakME")

    assert selected == rerun_root / "033_to-rpGmakME"


def test_run_rerun_error_videos_respects_requested_video_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PL123"
    root_dir = backfill.BACKFILL_ROOT / "2020_PL123"
    root_dir.mkdir(parents=True)
    manifest_path = root_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "videos": {
                    "abc": {
                        "position": 1,
                        "title": "Sessão A",
                        "url": "https://youtu.be/abc",
                        "status": "error",
                    },
                    "def": {
                        "position": 2,
                        "title": "Sessão D",
                        "url": "https://youtu.be/def",
                        "status": "error",
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    class DummyClient:
        def __init__(self, **_kwargs):
            pass

        def fetch_schema(self):
            return SimpleNamespace(raw_payload={})

    processed: list[str] = []

    monkeypatch.setattr(backfill, "build_runtime_context", lambda: {"notion_api_key": "x", "notion_data_source_id": "y"})
    monkeypatch.setattr(backfill, "NotionSessoesClient", DummyClient)
    monkeypatch.setattr(backfill, "load_existing_pages_for_year_with_retry", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(backfill, "dump_schema_snapshot", lambda root, schema: root / "_schema_snapshot.json")
    monkeypatch.setattr(backfill, "dump_existing_pages_snapshot", lambda root, existing: root / "_existing_pages_snapshot.json")
    monkeypatch.setattr(
        backfill,
        "find_target_video",
        lambda playlist_url, year, video_id, root_dir=None: backfill.PlaylistVideo(
            position=1 if video_id == "abc" else 2,
            video_id=video_id,
            title=f"Sessão {video_id}",
            url=f"https://youtu.be/{video_id}",
        ),
    )

    def fake_run_video_worker(*, video, args, root_dir, progress_heartbeat):
        processed.append(video.video_id)
        return {"video_id": video.video_id}

    monkeypatch.setattr(backfill, "run_video_worker", fake_run_video_worker)

    args = SimpleNamespace(
        playlist_url=playlist_url,
        year=2020,
        limit=0,
        video_ids=["def"],
    )

    backfill.run_rerun_error_videos(args)

    assert processed == ["def"]
    updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["videos"]["abc"]["status"] == "error"
    assert updated_manifest["videos"]["def"]["status"] == "done"


def test_iter_backfill_run_dirs_excludes_rerun_from_other_playlist(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PL_CURRENT"
    current_root = backfill.BACKFILL_ROOT / "2020_PL_CURRENT"
    current_root.mkdir(parents=True)

    other_rerun = backfill.BACKFILL_ROOT / "_rerun_errors_2020_20260403_000001"
    other_rerun.mkdir(parents=True)
    (other_rerun / "001_otherVideo").mkdir()
    (other_rerun / "summary.json").write_text(
        json.dumps(
            {
                "playlist_url": "https://www.youtube.com/playlist?list=PL_OTHER",
                "year": 2020,
                "videos": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    same_rerun = backfill.BACKFILL_ROOT / "_rerun_errors_2020_20260403_000002"
    same_rerun.mkdir(parents=True)
    (same_rerun / "001_sameVideo").mkdir()
    (same_rerun / "summary.json").write_text(
        json.dumps(
            {
                "playlist_url": playlist_url,
                "year": 2020,
                "videos": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        backfill,
        "load_playlist_videos",
        lambda _url: [
            backfill.PlaylistVideo(position=1, video_id="sameVideo", title="Sessão", url="https://youtu.be/sameVideo"),
            backfill.PlaylistVideo(position=2, video_id="currentVideo", title="Sessão", url="https://youtu.be/currentVideo"),
        ],
    )

    run_dirs = backfill.iter_backfill_run_dirs(playlist_url, 2020)

    assert same_rerun in run_dirs
    assert current_root in run_dirs
    assert other_rerun not in run_dirs


def test_repair_existing_video_rows_prefers_city_origin_promotes_cnj_and_fills_punchline(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_xyz"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps({"data_sessao": "2024-05-23", "composicao": [], "judgments": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600001-01"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600001-01"}]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "start_seconds": 120,
                "items": [
                    {
                        "numero_processo": "0600001-01.2024.6.02.0001",
                        "origem": "TRE/AL",
                        "punchline": "",
                        "analise_do_conteudo_juridico": "Publicidade institucional em período vedado no município.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "Maceió/AL"}, {"name": "TRE/AL"}]}},
                "tribunal": {"type": "select", "select": {"options": [{"name": "TRE-AL"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "punchline": {"type": "rich_text", "rich_text": {}},
                "analise_do_conteudo_juridico": {"type": "rich_text", "rich_text": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Publicidade institucional em período vedado",
                tipo_registro="Julgamento 1",
                origem="Maceió/AL",
                tribunal="TRE-AL",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=xyz&t=120",
                analise_do_conteudo_juridico="O processo 0600001-01.2024.6.02.0001 trata de publicidade institucional em período vedado no município.",
                data_sessao="2024-05-23",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="xyz",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.numero_processo == "0600001-01.2024.6.02.0001"
    assert repaired.origem == "Maceió/AL"
    assert repaired.punchline == "Publicidade institucional em período vedado no município."


def test_repair_existing_video_rows_uses_bundle_title_hint_and_drops_unvalidated_timestamp(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_xyz"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps({"data_sessao": "2024-05-21", "composicao": [], "judgments": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "title_hint": "Recurso Ordinário Eleitoral 0600557-55.2022.6.05.0000",
                "start_seconds": 810,
                "items": [
                    {
                        "numero_processo": "0600557-55.2022.6.05.0000",
                        "classe_processo": "",
                        "origem": "Tribunal Regional Eleitoral da Bahia (TRE-BA)",
                        "analise_do_conteudo_juridico": "Publicidade institucional em período vedado.",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "classe_processo": {"type": "select", "select": {"options": [{"name": "RO"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "TRE/BA"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Publicidade institucional em período vedado",
                tipo_registro="Julgamento 1",
                classe_processo="",
                origem="TRE/BA",
                numero_processo="0600557-55.2022.6.05.0000",
                youtube_link="https://www.youtube.com/watch?v=xyz&t=9999",
                data_sessao="2024-05-21",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="xyz",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.classe_processo == "RO"
    assert repaired.youtube_link == "https://www.youtube.com/watch?v=xyz&t=810"


def test_repair_existing_video_rows_retries_transient_notion_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(backfill, "NOTION_REPAIR_UPDATE_RETRY_SLEEP_SECONDS", 0.0)
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_xyz"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps({"data_sessao": "2024-05-23", "composicao": [], "judgments": []}, ensure_ascii=False),
        encoding="utf-8",
    )
    (artifact_dir / "07_backfill_summary.json").write_text(
        json.dumps({"publish_results": [{"numero_processo": "0600001-01"}]}, ensure_ascii=False),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "punchline": {"type": "rich_text", "rich_text": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Tema suficiente",
                tipo_registro="Julgamento 1",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=xyz&t=10",
                punchline="",
                analise_do_conteudo_juridico="Publicidade institucional em período vedado no município.",
                data_sessao="2024-05-23",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    class FlakyNotionClient:
        def __init__(self) -> None:
            self.attempts = 0
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("Notion API error 502: bad gateway")
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FlakyNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="xyz",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert notion.attempts == 2
    assert summary["updated_pages"] == 1
    assert summary["failed_pages"] == 0


def test_repair_existing_video_rows_trashes_safe_same_video_duplicates(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")
    playlist_url = "https://www.youtube.com/playlist?list=PLtest"
    run_dir = backfill.BACKFILL_ROOT / "2024_PLtest"
    artifact_dir = run_dir / "001_dupvid"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "01_session_windows.json").write_text(
        json.dumps(
            {
                "data_sessao": "2024-02-20",
                "composicao": [],
                "judgments": [
                    {
                        "title_hint": "AREspe 060065410",
                        "start_seconds": 4150,
                        "end_seconds": 4439,
                        "mentioned_process_numbers": ["0600654-10"],
                        "should_ignore": False,
                        "ignore_reason": "",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (artifact_dir / "02_judgment_01.json").write_text(
        json.dumps(
            {
                "title_hint": "AREspe 060065410",
                "start_seconds": 4150,
                "items": [
                    {
                        "numero_processo": "0600654-10",
                        "classe_processo": "AIJE",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    schema = NotionDataSourceSchema(
        data_source_id="ds-123",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {"type": "select", "select": {"options": [{"name": "Julgamento 1"}]}},
                "classe_processo": {"type": "select", "select": {"options": [{"name": "AIJE"}]}},
                "origem": {"type": "select", "select": {"options": [{"name": "Vila Velha/ES"}]}},
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "resultado": {"type": "select", "select": {"options": [{"name": "Provido"}]}},
                "votacao": {"type": "select", "select": {"options": [{"name": "Unânime"}]}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )
    records = [
        backfill.ExistingPageRecord(
            page_id="page-keep",
            url="https://www.notion.so/page-keep",
            video_id="dupvid",
            row=PublishPreviewRow(
                tema="Fraude à cota de gênero",
                tipo_registro="Julgamento 7",
                classe_processo="AIJE",
                origem="Vila Velha/ES",
                numero_processo="0600654-10",
                youtube_link="https://www.youtube.com/watch?v=dupvid&t=4130",
                resultado="Provido",
                votacao="Unânime",
                data_sessao="2024-02-20",
                page_id="page-keep",
                action="update",
            ),
        ),
        backfill.ExistingPageRecord(
            page_id="page-trash",
            url="https://www.notion.so/page-trash",
            video_id="dupvid",
            row=PublishPreviewRow(
                tema="Fraude à cota de gênero",
                tipo_registro="Julgamento 7",
                classe_processo="AIJE",
                origem="Vila Velha/ES",
                numero_processo="0600654-10",
                youtube_link="https://www.youtube.com/watch?v=dupvid&t=4150",
                resultado="Improcedente",
                votacao="",
                data_sessao="2024-02-20",
                page_id="page-trash",
                action="update",
            ),
        ),
    ]

    class FakeNotionClient:
        def __init__(self) -> None:
            self.updated = []
            self.trashed = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

        def _request(self, method, path, **kwargs):
            self.trashed.append((method, path, kwargs))
            return {}

    notion = FakeNotionClient()
    summary = backfill.repair_existing_video_rows(
        video_id="dupvid",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url=playlist_url,
        year=2024,
        gemini_api_key="token",
        model="gemini-3.1-flash-lite-preview",
        use_theme_api=False,
    )

    assert summary["trashed_duplicates"] == 1
    assert summary["rows_after_dedup"] == 1
    assert notion.trashed[0][0] == "PATCH"
    assert notion.trashed[0][1] == "/pages/page-trash"
    assert notion.updated[0][0] == "page-keep"
    assert notion.updated[0][1].youtube_link == "https://www.youtube.com/watch?v=dupvid&t=4150"


def test_numero_processo_needs_repair_accepts_blank_pa_and_special_stf_classes():
    assert backfill._numero_processo_needs_repair(PublishPreviewRow(numero_processo="", classe_processo="PA")) is False
    assert backfill._numero_processo_needs_repair(PublishPreviewRow(numero_processo="ADO 38", classe_processo="ADO")) is False
    assert backfill._numero_processo_needs_repair(PublishPreviewRow(numero_processo="ADI 7228", classe_processo="ADI")) is False
    assert backfill._numero_processo_needs_repair(PublishPreviewRow(numero_processo="7228", classe_processo="ADI")) is True


def test_safe_normalize_origem_for_repair_preserves_tse_without_appending_uf():
    assert backfill._safe_normalize_origem_for_repair("TSE", "TRE-RS") == "TSE"


def test_video_has_incomplete_composition():
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Tema",
                numero_processo="0600001-01",
                composicao=["Min. Cármen Lúcia", "Min. André Mendonça"],
                page_id="page-1",
                action="update",
            ),
        ),
        backfill.ExistingPageRecord(
            page_id="page-2",
            url="https://www.notion.so/page-2",
            video_id="xyz",
            row=PublishPreviewRow(
                tema="Tema",
                numero_processo="0600002-02",
                composicao=["Min. Cármen Lúcia"] * 7,
                page_id="page-2",
                action="update",
            ),
        ),
    ]

    assert backfill._video_has_incomplete_composition(records) is True


def test_run_repair_existing_year_filters_requested_video_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(backfill, "BACKFILL_ROOT", tmp_path / "artifacts")

    class FakeNotionClient:
        def __init__(self, *args, **kwargs):
            pass

        def fetch_schema(self):
            return NotionDataSourceSchema(
                data_source_id="ds-123",
                raw_payload={"properties": {"tema": {"type": "title", "title": {}}}},
            )

    grouped = {
        "video-a": [
            backfill.ExistingPageRecord(
                page_id="page-a",
                url="https://www.notion.so/page-a",
                video_id="video-a",
                row=PublishPreviewRow(tema="Tema A", numero_processo="0600001-01", page_id="page-a", action="update"),
            )
        ],
        "video-b": [
            backfill.ExistingPageRecord(
                page_id="page-b",
                url="https://www.notion.so/page-b",
                video_id="video-b",
                row=PublishPreviewRow(tema="Tema B", numero_processo="0600002-02", page_id="page-b", action="update"),
            )
        ],
    }
    seen = []

    monkeypatch.setattr(backfill, "build_runtime_context", lambda: {"notion_api_key": "n", "notion_data_source_id": "ds", "gemini_api_key": "g"})
    monkeypatch.setattr(backfill, "NotionSessoesClient", FakeNotionClient)
    monkeypatch.setattr(backfill, "load_playlist_videos", lambda _url: [])
    monkeypatch.setattr(backfill, "load_existing_pages_for_year_with_retry", lambda *_args, **_kwargs: grouped)

    def fake_repair_existing_video_rows(**kwargs):
        seen.append(kwargs["video_id"])
        return {"video_id": kwargs["video_id"], "updated_pages": 0, "failed_pages": 0}

    monkeypatch.setattr(backfill, "repair_existing_video_rows", fake_repair_existing_video_rows)

    args = SimpleNamespace(
        playlist_url="https://www.youtube.com/playlist?list=x",
        year=2024,
        limit=0,
        no_theme_api=True,
        only_composicao_incompleta=False,
        video_ids=["video-b"],
    )
    backfill.run_repair_existing_year(args)

    assert seen == ["video-b"]


def test_expected_tipo_registro_by_page_uses_artifact_ordering():
    records = [
        backfill.ExistingPageRecord(
            page_id="page-a",
            url="https://www.notion.so/page-a",
            video_id="video-1",
            row=PublishPreviewRow(
                tema="Tema A",
                numero_processo="0600002-02",
                youtube_link="https://www.youtube.com/watch?v=abc&t=200",
                page_id="page-a",
                action="update",
            ),
        ),
        backfill.ExistingPageRecord(
            page_id="page-b",
            url="https://www.notion.so/page-b",
            video_id="video-1",
            row=PublishPreviewRow(
                tema="Tema B",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=abc&t=100",
                page_id="page-b",
                action="update",
            ),
        ),
    ]

    expected = backfill._expected_tipo_registro_by_page(
        records,
        {"0600002-02": (10, 1, 1), "0600001-01": (20, 1, 1)},
    )

    assert expected == {"page-a": "Julgamento 1", "page-b": "Julgamento 2"}


def _identity_test_schema() -> NotionDataSourceSchema:
    return NotionDataSourceSchema(
        data_source_id="ds-identity",
        raw_payload={
            "properties": {
                "tema": {"type": "title", "title": {}},
                "tipo_registro": {
                    "type": "select",
                    "select": {
                        "options": [
                            {"name": "Julgamento 1"},
                            {"name": "Julgamento 2"},
                            {"name": "Julgamento 3"},
                        ]
                    },
                },
                "numero_processo": {"type": "rich_text", "rich_text": {}},
                "youtube_link": {"type": "url", "url": {}},
                "data_sessao": {"type": "date", "date": {}},
            }
        },
    )


def test_audit_existing_year_flags_identity_mismatches_and_duplicates(monkeypatch):
    process_key = "0600001-01"
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-02-20",
        session_composicao=[],
        ordering_by_process={process_key: (120, 1, 1)},
        ordering_by_special_process={},
        trusted_ordering_by_process={process_key: (120, 1, 1)},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            process_key: backfill.JudgmentItemExtraction(
                numero_processo="0600001-01.2024.6.00.0000",
                data_sessao="2024-02-20",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)
    monkeypatch.setattr(
        backfill,
        "load_playlist_videos",
        lambda _url: [backfill.PlaylistVideo(position=1, video_id="video-a", title="Sessão Plenária - 20 de fevereiro de 2024", url="https://www.youtube.com/watch?v=video-a")],
    )
    monkeypatch.setattr(backfill, "is_relevant_2025_session", lambda *_args, **_kwargs: True)

    grouped = {
        "video-a": [
            backfill.ExistingPageRecord(
                page_id="page-1",
                url="https://www.notion.so/page-1",
                video_id="video-a",
                row=PublishPreviewRow(
                    tema="Tema 1",
                    numero_processo="0600001-01.2024.6.00.0000",
                    youtube_link="https://www.youtube.com/watch?v=video-a&t=10",
                    data_sessao="2024-02-21",
                    page_id="page-1",
                    action="update",
                ),
            ),
            backfill.ExistingPageRecord(
                page_id="page-2",
                url="https://www.notion.so/page-2",
                video_id="video-a",
                row=PublishPreviewRow(
                    tema="Tema 2",
                    numero_processo="0600001-01.2024.6.00.0000",
                    youtube_link="https://www.youtube.com/watch?v=video-a&t=120",
                    data_sessao="2024-02-20",
                    page_id="page-2",
                    action="update",
                ),
            ),
        ]
    }

    summary = backfill.audit_existing_year(
        grouped,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2024,
    )

    assert summary["stats"]["data_sessao_mismatch"] == 1
    assert summary["stats"]["youtube_timestamp_unvalidated"] == 1
    assert summary["stats"]["identity_duplicate_process"] == 2
    assert summary["stats"]["identity_needs_repair"] == 2


def test_authoritative_video_session_date_prefers_video_title_over_artifact_date():
    assert (
        backfill._authoritative_video_session_date(
            video_title="Sessão Plenária - 03 de fevereiro de 2022",
            year=2022,
            artifact_session_date="2024-05-21",
        )
        == "2022-02-03"
    )


def test_build_identity_repair_universe_uses_video_title_session_date_over_item_and_artifact_dates(monkeypatch):
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-05-21",
        session_composicao=[],
        ordering_by_process={"0000697-22": (1200, 5, 1)},
        ordering_by_special_process={},
        published_process_keys={"0000697-22"},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            "0000697-22": backfill.JudgmentItemExtraction(
                numero_processo="0000697-22.2016.6.13.0000",
                data_sessao="2020-09-08",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
        trusted_ordering_by_process={"0000697-22": (1200, 5, 1)},
        trusted_item_by_process={
            "0000697-22": backfill.JudgmentItemExtraction(
                numero_processo="0000697-22.2016.6.13.0000",
                data_sessao="2020-09-08",
            )
        },
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)

    universe = backfill._build_identity_repair_universe(
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2022,
        video_ids=["NALJtQaMUSs"],
        grouped={},
        playlist_title_by_video={"NALJtQaMUSs": "Sessão Plenária - 03 de fevereiro de 2022"},
    )

    target = universe.target_by_video_process[("NALJtQaMUSs", "0000697-22")]
    assert target.session_date == "2022-02-03"
    assert target.numero_processo == "0000697-22.2016.6.13.0000"


def test_build_identity_repair_universe_drops_untrusted_timestamp_when_item_date_conflicts_with_video_title(monkeypatch):
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2021-06-23",
        session_composicao=[],
        ordering_by_process={"0600378-65": (114, 1, 1)},
        ordering_by_special_process={},
        published_process_keys={"0600378-65"},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            "0600378-65": backfill.JudgmentItemExtraction(
                numero_processo="0600378-65.2020.6.00.0000",
                data_sessao="2021-06-23",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)

    universe = backfill._build_identity_repair_universe(
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2021,
        video_ids=["s9Ts40TfDas"],
        grouped={},
        playlist_title_by_video={"s9Ts40TfDas": "Sessão Plenária do dia 11 de Fevereiro de 2021"},
    )

    assert ("s9Ts40TfDas", "0600378-65") not in universe.target_by_video_process


def test_build_identity_repair_universe_keeps_metadata_proven_association_without_forcing_timestamp(monkeypatch):
    process_key = "0600031-93"
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2021-06-23",
        session_composicao=[],
        ordering_by_process={process_key: (810, 2, 1)},
        ordering_by_special_process={},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            process_key: backfill.JudgmentItemExtraction(
                numero_processo="0600031-93",
                data_sessao="2021-06-23",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
        trusted_item_by_process={
            process_key: backfill.JudgmentItemExtraction(
                numero_processo="0600031-93",
                data_sessao="2021-06-23",
            )
        },
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)

    universe = backfill._build_identity_repair_universe(
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2021,
        video_ids=["s9Ts40TfDas"],
        grouped={},
        playlist_title_by_video={"s9Ts40TfDas": "Sessão Plenária do dia 11 de Fevereiro de 2021"},
    )

    target = universe.target_by_video_process[("s9Ts40TfDas", process_key)]
    assert target.session_date == "2021-02-11"
    assert target.start_seconds == 0
    assert target.timestamp_trusted is False


def test_chunk_support_for_candidate_item_accepts_multi_chunk_process_without_trusting_bad_timestamp():
    item = backfill.JudgmentItemExtraction(numero_processo="0601635-18.2020.6.22.0000")
    chunk_entries = [
        ("raw_global_response_chunk_06.txt", "060163518", 1350),
        ("raw_global_response_chunk_07.txt", "060163518", 1620),
        ("raw_global_response_chunk_08.txt", "060163518", 1890),
    ]

    association_trusted, timestamp_trusted = backfill._chunk_support_for_candidate_item(
        chunk_entries,
        item=item,
        bundle_title_hint="060163518",
        candidate_start_seconds=1120,
    )

    assert association_trusted is True
    assert timestamp_trusted is False


def test_iter_chunk_judgment_entries_accepts_processo_and_processos_plural(tmp_path):
    candidate_dir = tmp_path / "031_r_TMEJe3iIg"
    candidate_dir.mkdir()
    (candidate_dir / "raw_global_response_chunk_10.txt").write_text(
        json.dumps(
            [
                {
                    "julgamentos": [
                        {
                            "processos": ["060213621", "060210598"],
                            "timestamp_inicial": 2679,
                            "should_ignore": False,
                        },
                        {
                            "processo": "0601060-42.2020.6.26.0000",
                            "timestamp_inicial": 4940,
                            "should_ignore": False,
                        },
                        {
                            "processos": ["0600000-00"],
                            "timestamp_inicial": 999,
                            "should_ignore": True,
                        },
                    ]
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    entries = backfill._iter_chunk_judgment_entries(candidate_dir)

    assert ("raw_global_response_chunk_10.txt", "060213621", 2679) in entries
    assert ("raw_global_response_chunk_10.txt", "060210598", 2679) in entries
    assert ("raw_global_response_chunk_10.txt", "06010604220206260000", 4940) in entries
    assert all(chunk_probe != "060000000" for _, chunk_probe, _ in entries)


def test_chunk_support_for_candidate_item_rejects_single_chunk_plain_process():
    item = backfill.JudgmentItemExtraction(numero_processo="0600378-65.2020.6.00.0000")
    chunk_entries = [
        ("raw_global_response_chunk_01.txt", "06003786520206000000", 114),
    ]

    association_trusted, timestamp_trusted = backfill._chunk_support_for_candidate_item(
        chunk_entries,
        item=item,
        bundle_title_hint="0600378-65.2020.6.00.0000",
        candidate_start_seconds=114,
    )

    assert association_trusted is False
    assert timestamp_trusted is False


def test_chunk_support_for_candidate_item_accepts_single_chunk_overlay_case_and_trusts_matching_timestamp():
    item = backfill.JudgmentItemExtraction(numero_processo="0601227-40")
    chunk_entries = [
        ("raw_global_response_chunk_19.txt", "ednapc060122740", 4963),
    ]

    association_trusted, timestamp_trusted = backfill._chunk_support_for_candidate_item(
        chunk_entries,
        item=item,
        bundle_title_hint="ED na PC - 060122740",
        candidate_start_seconds=4963,
    )

    assert association_trusted is True
    assert timestamp_trusted is True


def test_special_process_lookup_key_preserves_overlay_class_for_same_process():
    assert backfill._special_process_lookup_key("262-19", "ED-PC") == "ED-PC 262-19"
    assert backfill._special_process_lookup_key("0600001-01.2024.6.00.0000", "AgRg-REspe") == "AgRg-REspe 0600001-01"


def test_select_identity_target_for_exact_video_prefers_overlay_special_over_plain_process():
    process_key = "262-19"
    plain_target = backfill.IdentityArtifactTarget(
        video_id="video-a",
        numero_processo="262-19",
        process_key=process_key,
        special_key="",
        session_date="2021-02-11",
        start_seconds=1620,
        tipo_registro="Julgamento 3",
    )
    special_target = backfill.IdentityArtifactTarget(
        video_id="video-a",
        numero_processo="262-19",
        process_key=process_key,
        special_key="ED-PC 262-19",
        session_date="2021-02-11",
        start_seconds=1890,
        tipo_registro="Julgamento 4",
    )
    universe = backfill.IdentityRepairUniverse(
        targets_by_process={process_key: [plain_target]},
        targets_by_special={"ED-PC 262-19": [special_target]},
        target_by_video_process={("video-a", process_key): plain_target},
        target_by_video_special={("video-a", "ED-PC 262-19"): special_target},
        existing_page_ids_by_video_process={("video-a", process_key): {"page-1"}},
    )

    row = PublishPreviewRow(numero_processo="262-19", classe_processo="ED-PC")

    target = backfill._select_identity_target_for_exact_video(
        row,
        current_video_id="video-a",
        identity_universe=universe,
    )

    assert target == special_target


def test_audit_existing_year_flags_timestamp_when_link_uses_untrusted_ordering(monkeypatch):
    process_key = "0600378-65"
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2021-06-23",
        session_composicao=[],
        ordering_by_process={process_key: (114, 1, 1)},
        ordering_by_special_process={},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            process_key: backfill.JudgmentItemExtraction(
                numero_processo="0600378-65.2020.6.00.0000",
                data_sessao="2021-06-23",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)
    monkeypatch.setattr(
        backfill,
        "load_playlist_videos",
        lambda _url: [
            backfill.PlaylistVideo(
                position=78,
                video_id="s9Ts40TfDas",
                title="Sessão Plenária do dia 11 de Fevereiro de 2021",
                url="https://www.youtube.com/watch?v=s9Ts40TfDas",
            )
        ],
    )
    monkeypatch.setattr(backfill, "is_relevant_2025_session", lambda *_args, **_kwargs: True)

    grouped = {
        "s9Ts40TfDas": [
            backfill.ExistingPageRecord(
                page_id="page-1",
                url="https://www.notion.so/page-1",
                video_id="s9Ts40TfDas",
                row=PublishPreviewRow(
                    tema="Tema útil",
                    numero_processo="0600378-65.2020.6.00.0000",
                    youtube_link="https://www.youtube.com/watch?v=s9Ts40TfDas&t=114",
                    data_sessao="2021-02-11",
                    page_id="page-1",
                    action="update",
                ),
            )
        ]
    }

    summary = backfill.audit_existing_year(
        grouped,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2021,
    )

    assert summary["stats"]["data_sessao_mismatch"] == 0
    assert summary["stats"]["association_unproven"] == 1
    assert summary["stats"]["youtube_timestamp_unvalidated"] == 1
    assert summary["stats"]["identity_needs_repair"] == 1


def test_row_has_local_association_proof_ignores_published_process_keys_without_trusted_maps():
    process_key = "0600378-65"
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2021-06-23",
        session_composicao=[],
        ordering_by_process={process_key: (114, 1, 1)},
        ordering_by_special_process={},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={
            process_key: backfill.JudgmentItemExtraction(
                numero_processo="0600378-65.2020.6.00.0000",
                data_sessao="2021-06-23",
            )
        },
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )

    assert (
        backfill._row_has_local_association_proof(
            PublishPreviewRow(numero_processo="0600378-65.2020.6.00.0000"),
            artifact_context,
        )
        is False
    )


def test_repair_existing_video_rows_identity_core_corrects_same_video_date_timestamp_and_numero(monkeypatch):
    process_key = "0600001-01"
    schema = _identity_test_schema()
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-02-20",
        session_composicao=[],
        ordering_by_process={process_key: (120, 1, 1)},
        ordering_by_special_process={},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={process_key: backfill.JudgmentItemExtraction(numero_processo="0600001-01.2024.6.00.0000", data_sessao="2024-02-20")},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)

    target = backfill.IdentityArtifactTarget(
        video_id="video-a",
        numero_processo="0600001-01.2024.6.00.0000",
        process_key=process_key,
        special_key="",
        session_date="2024-02-20",
        start_seconds=120,
        tipo_registro="Julgamento 1",
    )
    identity_universe = backfill.IdentityRepairUniverse(
        targets_by_process={process_key: [target]},
        targets_by_special={},
        target_by_video_process={("video-a", process_key): target},
        target_by_video_special={},
        existing_page_ids_by_video_process={("video-a", process_key): {"page-1"}},
    )

    class FakeNotionClient:
        def __init__(self):
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="video-a",
            row=PublishPreviewRow(
                tema="Tema útil",
                tipo_registro="Julgamento 9",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=video-a&t=10",
                data_sessao="2024-02-21",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    summary = backfill.repair_existing_video_rows(
        video_id="video-a",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2024,
        gemini_api_key="",
        model="",
        use_theme_api=False,
        repair_focus="identity-core",
        identity_universe=identity_universe,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.numero_processo == "0600001-01.2024.6.00.0000"
    assert repaired.data_sessao == "2024-02-20"
    assert repaired.youtube_link == "https://www.youtube.com/watch?v=video-a&t=120"
    assert repaired.tipo_registro == "Julgamento 1"


def test_repair_existing_video_rows_identity_core_moves_page_to_unique_target_video(monkeypatch):
    schema = _identity_test_schema()
    empty_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: empty_context)
    process_key = "0600001-01"
    target = backfill.IdentityArtifactTarget(
        video_id="video-b",
        numero_processo="0600001-01.2024.6.00.0000",
        process_key=process_key,
        special_key="",
        session_date="2024-03-01",
        start_seconds=330,
        tipo_registro="Julgamento 2",
    )
    identity_universe = backfill.IdentityRepairUniverse(
        targets_by_process={process_key: [target]},
        targets_by_special={},
        target_by_video_process={("video-b", process_key): target},
        target_by_video_special={},
        existing_page_ids_by_video_process={("video-a", process_key): {"page-1"}},
    )

    class FakeNotionClient:
        def __init__(self):
            self.updated = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

    notion = FakeNotionClient()
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="video-a",
            row=PublishPreviewRow(
                tema="Tema útil",
                tipo_registro="Julgamento 7",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=video-a&t=10",
                data_sessao="2024-03-01",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    summary = backfill.repair_existing_video_rows(
        video_id="video-a",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2024,
        gemini_api_key="",
        model="",
        use_theme_api=False,
        repair_focus="identity-core",
        identity_universe=identity_universe,
    )

    assert summary["updated_pages"] == 1
    repaired = notion.updated[0][1]
    assert repaired.youtube_link == "https://www.youtube.com/watch?v=video-b&t=330"
    assert repaired.data_sessao == "2024-03-01"
    assert repaired.numero_processo == "0600001-01.2024.6.00.0000"
    assert repaired.tipo_registro == "Julgamento 2"


def test_repair_existing_video_rows_identity_core_trashes_ambiguous_short_number(monkeypatch):
    schema = _identity_test_schema()
    empty_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="",
        session_composicao=[],
        ordering_by_process={},
        ordering_by_special_process={},
        published_process_keys=set(),
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: empty_context)
    process_key = "0600001-01"
    target_a = backfill.IdentityArtifactTarget(
        video_id="video-a",
        numero_processo=process_key,
        process_key=process_key,
        special_key="",
        session_date="2024-03-01",
        start_seconds=100,
        tipo_registro="Julgamento 1",
    )
    target_b = backfill.IdentityArtifactTarget(
        video_id="video-b",
        numero_processo=process_key,
        process_key=process_key,
        special_key="",
        session_date="2024-04-01",
        start_seconds=200,
        tipo_registro="Julgamento 1",
    )
    identity_universe = backfill.IdentityRepairUniverse(
        targets_by_process={process_key: [target_a, target_b]},
        targets_by_special={},
        target_by_video_process={("video-a", process_key): target_a, ("video-b", process_key): target_b},
        target_by_video_special={},
        existing_page_ids_by_video_process={("video-x", process_key): {"page-1"}},
    )

    class FakeNotionClient:
        def __init__(self):
            self.trashed = []

        def _request(self, method, path, json=None):
            self.trashed.append((method, path, json))
            return {}

    notion = FakeNotionClient()
    records = [
        backfill.ExistingPageRecord(
            page_id="page-1",
            url="https://www.notion.so/page-1",
            video_id="video-x",
            row=PublishPreviewRow(
                tema="Tema útil",
                numero_processo="0600001-01",
                youtube_link="https://www.youtube.com/watch?v=video-x&t=10",
                data_sessao="2024-05-01",
                page_id="page-1",
                action="update",
            ),
        )
    ]

    summary = backfill.repair_existing_video_rows(
        video_id="video-x",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2024,
        gemini_api_key="",
        model="",
        use_theme_api=False,
        repair_focus="identity-core",
        identity_universe=identity_universe,
    )

    assert summary["trashed_unproven_pages"] == 1
    assert notion.trashed == [("PATCH", "/pages/page-1", {"in_trash": True})]


def test_repair_existing_video_rows_identity_core_trashes_destination_duplicate_and_collapses_same_video_duplicates(monkeypatch):
    schema = _identity_test_schema()
    process_key = "0600001-01"
    artifact_context = backfill.RepairArtifactContext(
        artifact_dir=None,
        session_date="2024-02-20",
        session_composicao=[],
        ordering_by_process={process_key: (120, 1, 1)},
        ordering_by_special_process={},
        published_process_keys={process_key},
        published_special_process_keys=set(),
        theme_text_by_process={},
        theme_text_by_special_process={},
        item_by_process={process_key: backfill.JudgmentItemExtraction(numero_processo="0600001-01.2024.6.00.0000", data_sessao="2024-02-20")},
        item_by_special_process={},
        title_hint_by_process={},
        title_hint_by_special_process={},
    )
    monkeypatch.setattr(backfill, "load_repair_artifact_context", lambda *args, **kwargs: artifact_context)
    target = backfill.IdentityArtifactTarget(
        video_id="video-b",
        numero_processo="0600001-01.2024.6.00.0000",
        process_key=process_key,
        special_key="",
        session_date="2024-02-20",
        start_seconds=120,
        tipo_registro="Julgamento 1",
    )
    identity_universe = backfill.IdentityRepairUniverse(
        targets_by_process={process_key: [target]},
        targets_by_special={},
        target_by_video_process={("video-b", process_key): target},
        target_by_video_special={},
        existing_page_ids_by_video_process={
            ("video-a", process_key): {"page-keep", "page-trash"},
            ("video-b", process_key): {"page-existing"},
        },
    )

    class FakeNotionClient:
        def __init__(self):
            self.updated = []
            self.trashed = []

        def update_row(self, _schema, page_id, row):
            self.updated.append((page_id, row.model_copy(deep=True)))
            return {"id": page_id}

        def _request(self, method, path, json=None):
            self.trashed.append((method, path, json))
            return {}

    notion = FakeNotionClient()
    records = [
        backfill.ExistingPageRecord(
            page_id="page-keep",
            url="https://www.notion.so/page-keep",
            video_id="video-a",
                row=PublishPreviewRow(
                    tema="Tema útil",
                    tipo_registro="Julgamento 2",
                    numero_processo="0600001-01.2024.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=video-a&t=10",
                data_sessao="2024-02-20",
                resultado="Provido",
                votacao="Unânime",
                page_id="page-keep",
                action="update",
            ),
        ),
        backfill.ExistingPageRecord(
            page_id="page-trash",
            url="https://www.notion.so/page-trash",
            video_id="video-a",
                row=PublishPreviewRow(
                    tema="Tema útil",
                    tipo_registro="Julgamento 2",
                    numero_processo="0600001-01.2024.6.00.0000",
                youtube_link="https://www.youtube.com/watch?v=video-a&t=15",
                data_sessao="2024-02-20",
                page_id="page-trash",
                action="update",
            ),
        ),
    ]

    summary = backfill.repair_existing_video_rows(
        video_id="video-a",
        records=records,
        notion_client=notion,
        notion_schema=schema,
        playlist_url="https://www.youtube.com/playlist?list=PLtest",
        year=2024,
        gemini_api_key="",
        model="",
        use_theme_api=False,
        repair_focus="identity-core",
        identity_universe=identity_universe,
    )

    assert summary["trashed_duplicates"] == 1
    assert summary["trashed_unproven_pages"] == 1
    assert len(notion.trashed) == 2
