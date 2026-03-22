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


def test_build_worker_command_propagates_ground_origem_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(
        playlist_url="https://www.youtube.com/playlist?list=x",
        year=2025,
        skip_news=True,
        no_trash_unmatched_precedents=False,
        ground_origem_with_search=True,
    )
    video = backfill.PlaylistVideo(position=1, video_id="abc123", title="Sessão", url="https://youtu.be/abc123")

    command, _project_dir = backfill.build_worker_command(video, args, tmp_path / "root")

    assert "--ground-origem-with-search" in command


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
