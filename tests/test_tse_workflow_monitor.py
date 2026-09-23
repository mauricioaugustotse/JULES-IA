import copy
import json
from types import SimpleNamespace

import pytest

from tse_workflow_monitor import BatchMonitor, reconcile_video, verify_notion_rows


PROCESS_A = "0600814-85.2022.6.00.0000"
PROCESS_B = "0600669-77.2024.6.02.0008"
PROCESS_C = "0600379-88.2024.6.19.0105"


def row_data(number=PROCESS_A, timestamp=600, **overrides):
    values = {
        "numero_processo": number,
        "data_sessao": "2026-06-23",
        "youtube_link": f"https://www.youtube.com/watch?v=abc123&t={timestamp}",
        "tema": "Propaganda eleitoral antecipada em rede social",
    }
    values.update(overrides)
    return values


def analysis_for(*numbers):
    return {
        "session": {
            "judgments": [
                {
                    "start_seconds": 600 + index * 600,
                    "end_seconds": 1100 + index * 600,
                    "mentioned_process_numbers": [number],
                }
                for index, number in enumerate(numbers)
            ]
        },
        "bundles": [
            {
                "start_seconds": 600 + index * 600,
                "items": [{"numero_processo": number}],
            }
            for index, number in enumerate(numbers)
        ],
    }


def issue_codes(report):
    return {issue["code"] for issue in report["issues"]}


def test_case_omission_is_visible_even_when_total_counts_match():
    rows = [row_data(PROCESS_A), row_data(PROCESS_C, 1200)]
    report = reconcile_video(
        analysis_for(PROCESS_A, PROCESS_B), rows,
        scan={"status": "complete"},
        rito={"transcript_available": True, "apregoamentos_individuais": 2},
    )
    assert report["status"] == "pending"
    assert {"scan_process_missing", "detail_process_missing"} <= issue_codes(report)
    assert "rito_gap" not in issue_codes(report)
    assert all(issue["numero_processo"] for issue in report["issues"])
    assert report["counts"]["extracted"] == 2


@pytest.mark.parametrize("scan", [None, {}, {"status": "incomplete", "uncovered_intervals": [[0, 600]]}])
def test_complete_publication_without_scan_evidence_stays_pending(scan):
    report = reconcile_video(
        analysis_for(PROCESS_A), [row_data()],
        [{"status": "created", "page_id": "page-a"}],
        scan=scan, verification=[{"row_index": 0, "status": "verified"}], published=True,
    )
    assert report["status"] == "pending"
    assert "scan_unverified" in issue_codes(report)


@pytest.mark.parametrize("status", ["blocked", "skipped", "error"])
def test_unpublished_row_cannot_be_reported_as_verified(status):
    report = reconcile_video(
        analysis_for(PROCESS_A), [row_data()],
        [{"status": status, "numero_processo": PROCESS_A, "errors": ["Revisar identidade"]}],
        scan={"status": "complete"}, published=True,
    )
    assert report["status"] == "pending"
    issue = next(item for item in report["issues"] if item["code"] == "unpublished_row")
    assert issue["status"] == status
    assert "Revisar identidade" in issue["reasons"]


def test_partial_publication_retains_success_count_and_reports_missing_outcomes():
    report = reconcile_video(
        analysis_for(PROCESS_A, PROCESS_B, PROCESS_C),
        [row_data(PROCESS_A), row_data(PROCESS_B, 1200), row_data(PROCESS_C, 1800)],
        [{"status": "created", "page_id": "page-a"}, {"status": "error", "errors": ["timeout"]}],
        scan={"status": "complete"},
        verification=[{"row_index": 0, "status": "verified"}], published=True,
    )
    assert report["status"] == "pending"
    assert report["counts"]["created"] == 1
    assert report["counts"]["verified"] == 1
    assert {"publication_gap", "unpublished_row"} <= issue_codes(report)


def test_failed_notion_readback_leaves_successful_write_pending():
    report = reconcile_video(
        analysis_for(PROCESS_A), [row_data()],
        [{"status": "created", "page_id": "page-a"}],
        scan={"status": "complete"},
        verification=[{"row_index": 0, "status": "unverified", "error": "Read access denied"}],
        published=True,
    )
    assert report["status"] == "pending"
    assert "notion_unverified" in issue_codes(report)


def test_full_coverage_and_readback_is_verified():
    report = reconcile_video(
        analysis_for(PROCESS_A), [row_data()],
        [{"status": "updated", "page_id": "page-a"}],
        scan={"status": "complete"},
        verification=[{"row_index": 0, "status": "verified"}], published=True,
    )
    assert report["status"] == "verified"
    assert report["issues"] == []


def test_unresolved_detail_recovery_stays_pending_after_successful_publication():
    report = reconcile_video(
        analysis_for(PROCESS_A), [row_data()],
        [{"status": "created", "page_id": "page-a"}],
        scan={"status": "complete"},
        detail={"blocks": [{"index": 0, "status": "pending", "issues": ["Releitura ainda incompleta"]}]},
        verification=[{"row_index": 0, "status": "verified"}], published=True,
    )
    assert report["status"] == "pending"
    assert "detail_unresolved" in issue_codes(report)


def test_preview_without_publication_is_explicit():
    report = reconcile_video(analysis_for(PROCESS_A), [row_data()], scan={"status": "complete"})
    assert report["status"] == "preview"
    assert report["counts"]["verified"] == 0


@pytest.mark.parametrize("counter", ["total_pending", "total_error", "total_unprocessed", "total_stopped"])
def test_batch_monitor_finish_never_hides_incomplete_videos(tmp_path, counter):
    monitor = BatchMonitor(tmp_path, [SimpleNamespace(video_id="abc123", url="https://youtu.be/abc123")])
    monitor.finish({counter: 1, "publish_requested": True})
    state = json.loads((tmp_path / "monitor_status.json").read_text(encoding="utf-8"))
    assert state["status"] == "pending"
    assert "pending" in (tmp_path / "monitor.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("published, expected", [(False, "preview"), (True, "verified")])
def test_batch_monitor_finish_distinguishes_preview_from_verified(tmp_path, published, expected):
    monitor = BatchMonitor(tmp_path, [])
    monitor.finish({"publish_requested": published, "total_pending": 0, "total_error": 0})
    assert json.loads((tmp_path / "monitor_status.json").read_text(encoding="utf-8"))["status"] == expected


def test_batch_monitor_finish_keeps_failed_postpublication_pending(tmp_path):
    monitor = BatchMonitor(tmp_path, [])
    monitor.finish({"publish_requested": True, "post_publish": {"falhas": ["relations"]}})
    assert monitor.state["status"] == "pending"


class FakeNotionReadback:
    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def _request(self, method, path):
        self.requests.append((method, path))
        page = self.pages[path.rsplit("/", 1)[-1]]
        if isinstance(page, Exception):
            raise page
        return page

    def _extract_property_text(self, page, schema, name):
        assert schema == "fake-schema"
        return page["properties"].get(name, "")


def test_notion_readback_checks_each_success_and_checkpoints_progress():
    row = SimpleNamespace(**row_data())
    client = FakeNotionReadback({
        "page-a": {"id": "page-a", "properties": row_data()},
        "page-b": {"id": "page-b", "properties": row_data()},
    })
    checkpoints = []
    checks = verify_notion_rows(
        [row, row, row],
        [{"status": "created", "page_id": "page-a"}, {"status": "blocked"}, {"status": "updated", "page_id": "page-b"}],
        client, "fake-schema", checkpoint=lambda checks: checkpoints.append(copy.deepcopy(checks)),
    )
    assert client.requests == [("GET", "/pages/page-a"), ("GET", "/pages/page-b")]
    assert checks == [
        {"row_index": 0, "page_id": "page-a", "status": "verified"},
        {"row_index": 2, "page_id": "page-b", "status": "verified"},
    ]
    assert checkpoints == [checks[:1], checks]


@pytest.mark.parametrize(
    "page, reason",
    [
        (RuntimeError("Read access denied"), "Read access denied"),
        ({"id": "page-a", "archived": True}, "arquivada"),
        ({"id": "page-a", "in_trash": True}, "lixeira"),
        ({"id": "page-other"}, "Identificador"),
        ({"id": "page-a", "properties": row_data(PROCESS_B)}, "numero_processo"),
        ({"id": "page-a", "properties": row_data("0600814-85.2024.6.00.0000")}, "numero_processo"),
        ({"id": "page-a", "properties": row_data("0600814-85.2022.6.02.0000")}, "numero_processo"),
        ({"id": "page-a", "properties": row_data("0600814-85")}, "numero_processo"),
        ({"id": "page-a", "properties": row_data(data_sessao="2026-06-24")}, "data_sessao"),
        ({"id": "page-a", "properties": row_data(youtube_link="https://youtu.be/other123")}, "youtube_link"),
        ({"id": "page-a", "properties": row_data(tema="Outro julgamento")}, "tema"),
    ],
)
def test_readback_errors_and_identity_mismatches_remain_unverified(page, reason):
    client = FakeNotionReadback({"page-a": page})
    checks = verify_notion_rows(
        [SimpleNamespace(**row_data())], [{"status": "updated", "page_id": "page-a"}],
        client, "fake-schema",
    )
    assert checks[0]["status"] == "unverified"
    assert reason in checks[0]["error"]


def test_readback_without_returned_page_id_does_not_make_request():
    client = FakeNotionReadback({})
    checks = verify_notion_rows(
        [SimpleNamespace(**row_data())], [{"status": "created"}], client, "fake-schema"
    )
    assert checks[0]["status"] == "unverified"
    assert client.requests == []


def test_readback_distinguishes_numberless_judgments_by_video_timestamp():
    expected = row_data(number="", timestamp=600)
    client = FakeNotionReadback({"page-a": {"id": "page-a", "properties": row_data(number="", timestamp=1800)}})
    checks = verify_notion_rows(
        [SimpleNamespace(**expected)], [{"status": "created", "page_id": "page-a"}],
        client, "fake-schema",
    )
    assert checks[0]["status"] == "unverified"
    assert "youtube_link" in checks[0]["error"]


def test_readback_accepts_equivalent_link_with_same_timestamp():
    client = FakeNotionReadback({
        "page-a": {"id": "page-a", "properties": row_data(youtube_link="https://youtu.be/abc123?start=600&feature=share")}
    })
    checks = verify_notion_rows(
        [SimpleNamespace(**row_data())], [{"status": "created", "page_id": "page-a"}],
        client, "fake-schema",
    )
    assert checks[0]["status"] == "verified"


def official_inventory(*, excluded=False):
    from tse_official_session import normalize_official_session
    return normalize_official_session([{
        "id": "test", "tribunal": "TSE", "virtual": False, "dataSessao": "2026-06-23",
        "processos": [{"numeroProcesso": PROCESS_A,
                       "situacaoProcesso": "Retirado de julgamento" if excluded else "Julgado"}],
    }], "2026-06-23")


def test_official_missing_case_detected_when_scan_and_chapters_both_miss_it():
    report = reconcile_video({}, [], scan={"status": "complete"}, official=official_inventory())
    assert "official_missing_judgment" in issue_codes(report)
    assert report["status"] == "pending"


def test_official_unavailable_is_unknown_even_with_complete_scan_and_readback():
    report = reconcile_video(analysis_for(PROCESS_A), [row_data()],
                             [{"status": "created"}], published=True,
                             scan={"status": "complete"}, official={"status": "unavailable"},
                             verification=[{"row_index": 0, "status": "verified"}])
    assert report["status"] == "pending"
    assert "official_inventory_unavailable" in issue_codes(report)
    assert report["official_comparison"][0]["coverage_status"] == "unknown"


def test_official_withdrawal_does_not_become_missing_case_or_precedent():
    report = reconcile_video(analysis_for(PROCESS_A), [], scan={"status": "complete"},
                             published=True, official=official_inventory(excluded=True))
    assert report["issues"] == []
    assert report["status"] == "verified"
    assert report["information"][0]["code"] == "official_exclusion"
    assert report["information"][0]["classification"] == "withdrawn"


def test_official_coverage_never_substitutes_for_whole_video_scan():
    report = reconcile_video(analysis_for(PROCESS_A), [row_data()], official=official_inventory())
    assert "scan_unverified" in issue_codes(report)
    assert report["evidence"]["official_inventory"] == "available"
    assert report["evidence"]["whole_video_scan"] == "unknown"


class PayloadClient(FakeNotionReadback):
    def __init__(self, payload, actual):
        super().__init__({"page-a": {"id": "page-a", "properties": actual}})
        self.payload = payload

    def build_properties_payload(self, schema, row):
        return copy.deepcopy(self.payload)


def property_payload():
    return {"resultado": {"select": {"name": "Indeferido"}},
            "classe_processo": {"select": {"name": "RvE"}},
            "origem": {"rich_text": [{"text": {"content": "Minador do Negrão/AL"}}]},
            "relator": {"select": {"name": "Min. Antônio Carlos Ferreira"}},
            "composicao": {"multi_select": [{"name": "Min. Cármen Lúcia"}, {"name": "Min. Estela Aranha"}]},
            "data_sessao": {"date": {"start": "2026-09-17"}},
            "pedido_vista": {"select": None},
            "fundamentacao_normativa": {"rich_text": [{"text": {"content": "Lei A"}}, {"text": {"content": " e Lei B"}}]}}


@pytest.mark.parametrize("field", ["resultado", "classe_processo", "origem", "relator", "composicao",
                                  "data_sessao", "pedido_vista", "fundamentacao_normativa"])
def test_readback_checks_every_written_field_and_explicit_clears(field):
    payload = property_payload()
    actual = copy.deepcopy(payload)
    del actual[field]
    checks = verify_notion_rows([SimpleNamespace()], [{"status": "updated", "page_id": "page-a"}],
                               PayloadClient(payload, actual), "fake-schema")
    assert checks[0]["status"] == "unverified"


def test_payload_readback_ignores_colors_ids_text_chunking_and_multi_select_order():
    payload = property_payload()
    actual = copy.deepcopy(payload)
    actual["resultado"]["select"].update(id="option-id", color="blue")
    actual["composicao"]["multi_select"].reverse()
    actual["fundamentacao_normativa"]["rich_text"] = [{"plain_text": "Lei A e Lei B"}]
    actual["data_sessao"]["date"].update(end=None, time_zone=None)
    checks = verify_notion_rows([SimpleNamespace()], [{"status": "updated", "page_id": "page-a"}],
                               PayloadClient(payload, actual), "fake-schema")
    assert checks[0]["status"] == "verified"
    assert set(checks[0]["checked_fields"]) == set(payload)
