import tse_backfill_monitor as monitor


def test_stage_from_artifact_maps_known_steps():
    assert monitor.stage_from_artifact("raw_global_response_chunk_05.json") == "varredura global chunk 5"
    assert monitor.stage_from_artifact("04a_process_metadata_02.json") == "metadados processuais item 2"
    assert monitor.stage_from_artifact("07_backfill_summary.json") == "finalizado, aguardando encerramento"


def test_classify_error_kind_recognizes_false_timeout_after_summary():
    kind = monitor.classify_error_kind(
        "Sem progresso real de artefatos por 300s no vídeo X. Último artefato: 07_backfill_summary.json.",
        "07_backfill_summary.json",
    )
    assert kind == "timeout apos resumo final"


def test_render_snapshot_is_compact_and_readable(tmp_path):
    runner_log = tmp_path / "runner.log"
    runner_log.write_text("", encoding="utf-8")
    manifest = {
        "updated_at": "2026-03-21T13:54:23",
        "eta_seconds": 5721,
        "eta_at": "2026-03-21T15:29:44",
        "avg_video_seconds": 152.6,
        "current_target_workers": 6,
        "initial_workers": 5,
        "max_target_workers": 20,
        "auto_scale_enabled": True,
        "healthy_completions_since_scale": 1,
        "recent_capacity_errors": 0,
        "last_scale_reason": "scale_up_healthy",
        "last_scaled_at": "2026-03-21T13:54:00",
        "recent_events": [
            {
                "at": "2026-03-21T13:53:59",
                "level": "INFO",
                "type": "worker_done",
                "video_id": "abc",
                "message": "abc concluído: rows=1 created=1 updated=0 blocked=0.",
            }
        ],
        "videos": {
            "abc": {
                "status": "done",
                "title": "Sessão A",
                "finished_at": "2026-03-21T13:00:00",
            },
            "def": {
                "status": "running",
                "title": "Sessão B",
                "last_artifact": "raw_start_refinement_09.txt",
                "heartbeat_at": "2026-03-21T13:54:20",
                "worker_pid": 123,
                "attempts": 1,
            },
            "ghi": {
                "status": "error",
                "title": "Sessão C",
                "last_artifact": "07_backfill_summary.json",
                "error": "Sem progresso real de artefatos por 300s no vídeo ghi. Último artefato: 07_backfill_summary.json.",
            },
        },
    }

    output = monitor.render_snapshot(manifest, tmp_path / "manifest.json", runner_log)

    assert "[prod] backfill_2025" in output
    assert "refresh_at:" in output
    assert "manifest_age=" in output
    assert "publicados: 1/3" in output
    assert "saude: atencao" in output
    assert "workers: alvo=6" in output
    assert "running:" in output
    assert "etapa: refino de inicio bloco 9" in output
    assert "timeout apos resumo final" in output
    assert "eventos recentes:" in output
    assert "log recente do runner:" in output
    assert "abc concluído" in output
