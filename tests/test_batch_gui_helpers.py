import tse_youtube_notion_batch_gui as gui


def test_summarize_link_meta_extracts_session_date():
    display = gui.summarize_link_meta(
        "Sessão Plenária Jurisdicional do dia 30 de junho de 2026"
    )
    assert "2026-06-30" in display
    assert "Sessão Plenária" in display


def test_summarize_link_meta_without_date_keeps_title():
    assert gui.summarize_link_meta("Vídeo institucional") == "Vídeo institucional"
    assert gui.summarize_link_meta("") == "(título indisponível)"


def test_build_rito_count_check_reads_report(tmp_path):
    store = gui.RunArtifacts(tmp_path)
    assert gui._build_rito_count_check(store, []) is None
    store.write_json(
        "01b_rito_refinement.json",
        {"transcript_available": True, "apregoamentos_individuais": 3},
    )
    check = gui._build_rito_count_check(store, [object()])
    assert check == {"apregoamentos": 3, "rows": 1, "delta": 2, "verdict": "verificar"}
    check_ok = gui._build_rito_count_check(store, [object()] * 4)
    assert check_ok["verdict"] == "ok"
