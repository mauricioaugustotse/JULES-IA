from __future__ import annotations

import json
from pathlib import Path

import run_semantic_bleed_corrective_batch as corrective


def test_collect_candidates_from_audit_report_merges_pages_per_video(tmp_path: Path) -> None:
    audit_root = tmp_path / "audit"
    audit_root.mkdir()
    payload = {
        "flagged_pages": [
            {
                "video_id": "abc123",
                "video_title": "Sessão A",
                "page_id": "page-1",
                "page_url": "https://notion.so/page-1",
                "risk_level": "medium",
                "risk_score": 3,
                "reasons": ["weak_window"],
                "row": {"numero_processo": "0600001-01", "tipo_registro": "Julgamento 1"},
            },
            {
                "video_id": "abc123",
                "video_title": "Sessão A",
                "page_id": "page-2",
                "page_url": "https://notion.so/page-2",
                "risk_level": "high",
                "risk_score": 5,
                "reasons": ["richer_neighbor"],
                "row": {"numero_processo": "0600002-02", "tipo_registro": "Julgamento 2"},
            },
            {
                "video_id": "xyz789",
                "video_title": "Sessão B",
                "page_id": "page-3",
                "page_url": "https://notion.so/page-3",
                "risk_level": "low",
                "risk_score": 1,
                "reasons": ["minor"],
                "row": {"numero_processo": "0600003-03", "tipo_registro": "Julgamento 3"},
            },
        ]
    }
    (audit_root / "2025.json").write_text(json.dumps(payload), encoding="utf-8")

    candidates = corrective._collect_candidates_from_audit_report(audit_root, 2025, min_risk_level="medium")

    assert set(candidates) == {"abc123"}
    assert candidates["abc123"]["risk_level"] == "high"
    assert candidates["abc123"]["risk_score"] == 5
    assert [item["page_id"] for item in candidates["abc123"]["flagged_pages"]] == ["page-2", "page-1"]
