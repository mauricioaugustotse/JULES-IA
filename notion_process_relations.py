"""Relate sessions by complete CNJ without conflating equal short numbers."""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from tse_normalization import cnj_check_digit_is_valid, dedupe_preserve_order


@dataclass(frozen=True)
class ProcessRelationRecord:
    page_id: str
    numero_processo: str
    video_id: str = ""
    data_sessao: str = ""


def full_cnj_relation_key(value: str) -> str:
    """Return all 20 digits only for one complete CNJ with a valid check digit.

    Short, invalid and combined references do not establish process identity.
    The general short-number normalizer intentionally remains unchanged.
    """
    text = re.sub(r"\s+", "", str(value or "")).replace("–", "-").replace("—", "-")
    if re.fullmatch(r"[0-9]{20}", text):
        text = f"{text[:7]}-{text[7:9]}.{text[9:13]}.{text[13]}.{text[14:16]}.{text[16:]}"
    if cnj_check_digit_is_valid(text) is not True:
        return ""
    return text.replace("-", "").replace(".", "")


def distinct_session(left: ProcessRelationRecord, right: ProcessRelationRecord) -> bool:
    """Missing context alone is not evidence of another session."""
    return bool(
        (left.data_sessao and right.data_sessao and left.data_sessao != right.data_sessao)
        or (left.video_id and right.video_id and left.video_id != right.video_id)
    )


def _same_session(left: ProcessRelationRecord, right: ProcessRelationRecord) -> bool:
    return bool(
        left.video_id and right.video_id and left.video_id == right.video_id
        and left.data_sessao and right.data_sessao and left.data_sessao == right.data_sessao
    )


def index_relation_records(records: Iterable[ProcessRelationRecord]) -> dict[str, ProcessRelationRecord]:
    indexed: dict[str, ProcessRelationRecord] = {}
    for record in records:
        if not record.page_id:
            continue
        previous = indexed.get(record.page_id)
        if previous is not None and previous != record:
            # Conflicting copies of one page cannot authorize identity changes.
            indexed[record.page_id] = ProcessRelationRecord(record.page_id, "")
        else:
            indexed[record.page_id] = record
    return indexed


def build_full_cnj_relation_targets(records: Iterable[ProcessRelationRecord]) -> dict[str, list[str]]:
    indexed = index_relation_records(records)
    groups: dict[str, list[ProcessRelationRecord]] = defaultdict(list)
    for record in indexed.values():
        key = full_cnj_relation_key(record.numero_processo)
        if key:
            groups[key].append(record)
    return {
        record.page_id: [candidate.page_id for candidate in group
                         if candidate.page_id != record.page_id and distinct_session(record, candidate)]
        for group in groups.values() for record in group
    }


def reconcile_process_relations(
    record: ProcessRelationRecord,
    current: list[str],
    proposed: list[str],
    indexed: dict[str, ProcessRelationRecord],
) -> list[str]:
    """Preserve manual/out-of-scope links and remove confirmed legacy collisions.

    The previous automatic grouping discarded everything after sequence + DV.
    Only links with that same short prefix and different valid full identities
    can be removed as collisions. Other process links may be intentional manual
    references. Unknown targets/identities are never grounds for removal.
    """
    key = full_cnj_relation_key(record.numero_processo)
    if not key:
        return list(current)
    retained = []
    for target_id in current:
        target = indexed.get(target_id)
        target_key = full_cnj_relation_key(target.numero_processo) if target else ""
        if target_key:
            if target_key[:9] == key[:9] and target_key != key:
                continue
            if target_key == key and (target_id == record.page_id or _same_session(record, target)):
                continue
        retained.append(target_id)
    additions = []
    for target_id in proposed:
        target = indexed.get(target_id)
        if target and target_id != record.page_id and full_cnj_relation_key(target.numero_processo) == key and distinct_session(record, target):
            additions.append(target_id)
    return dedupe_preserve_order(retained + additions)
