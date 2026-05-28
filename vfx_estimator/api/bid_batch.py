"""LVL UP bid CSV ingestion and batch estimate helpers."""

from __future__ import annotations

import base64
import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException
from pydantic import BaseModel, Field

from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.types import (
    BID_DEPT_MAP,
    BID_OUTPUT_COLUMNS,
    BidPreQual,
    ShotEstimate,
    bid_departments_to_internal,
    internal_departments_to_bid,
)

ALLOTMENT_EFFICIENCY = 0.88


def _round_half(x: float) -> float:
    return round(float(x) * 2) / 2


def _norm_header(cell: str) -> str:
    return str(cell or "").strip().upper().replace("\n", " ")


class BidBatchShotItem(BaseModel):
    item_number: Optional[str] = None
    shot_code: Optional[str] = None
    script_description: str = ""
    vfx_notes: str = ""
    vfx_assumptions: str = ""
    client_initial_thoughts: Optional[str] = None
    number_of_shots: int = 1
    # Simple paste / legacy fallback
    description: Optional[str] = None
    shot_number: Optional[str] = None


class BatchEstimateRequest(BaseModel):
    shots: List[BidBatchShotItem]
    day_rate: Optional[float] = None
    project: Optional[str] = None
    mode: Optional[str] = None
    pre_qual: Optional[BidPreQual] = None


@dataclass
class LvlupCsvArtifact:
    """Original bid sheet rows for round-trip CSV export."""

    all_rows: List[List[str]] = field(default_factory=list)
    header_idx: int = 0
    col_map: Dict[str, int] = field(default_factory=dict)
    data_row_indices: List[int] = field(default_factory=list)


def build_shot_description(shot: BidBatchShotItem) -> str:
    if shot.description and shot.description.strip():
        return shot.description.strip()
    parts: List[str] = []
    if shot.script_description and shot.script_description.strip():
        parts.append(shot.script_description.strip())
    if shot.vfx_notes and shot.vfx_notes.strip():
        parts.append(f"VFX: {shot.vfx_notes.strip()}")
    if shot.vfx_assumptions and shot.vfx_assumptions.strip():
        parts.append(f"APPROACH: {shot.vfx_assumptions.strip()}")
    if shot.client_initial_thoughts and shot.client_initial_thoughts.strip():
        parts.append(f"CLIENT: {shot.client_initial_thoughts.strip()}")
    desc = " | ".join(parts)
    return desc.strip(" |")


def _resolve_shot_code(shot: BidBatchShotItem) -> Optional[str]:
    return shot.shot_code or shot.shot_number


def apply_allotment_per_shot_mandays(
    est: ShotEstimate,
    number_of_shots: int,
) -> float:
    n = max(1, int(number_of_shots or 1))
    if n > 1:
        return _round_half(float(est.total_mandays) / (n * ALLOTMENT_EFFICIENCY))
    return float(est.per_shot_mandays)


def scale_internal_depts(dept_days: Dict[str, float], scale: float) -> Dict[str, float]:
    if scale <= 0 or abs(scale - 1.0) < 1e-6:
        return {k: float(v) for k, v in dept_days.items() if float(v or 0) > 0}
    return {
        k: _round_half(float(v) * scale)
        for k, v in dept_days.items()
        if float(v or 0) > 0
    }


def _compute_adjustment_ranges_bid(
    bid_depts: Dict[str, float],
    confidence: float,
    compute_ranges,
) -> Dict[str, Dict]:
    internal = bid_departments_to_internal(bid_depts)
    internal_ranges = compute_ranges(internal, confidence)
    out: Dict[str, Dict] = {}
    for internal_key, spec in internal_ranges.items():
        bid_key = None
        for bcol, ikey in BID_DEPT_MAP.items():
            if ikey == internal_key:
                bid_key = bcol
                break
        if bid_key:
            out[bid_key] = spec
    return out


def _estimate_bid_row(
    est: ShotEstimate,
    shot: BidBatchShotItem,
    *,
    day_rate: float,
    compute_ranges,
) -> Dict[str, Any]:
    n = max(1, int(shot.number_of_shots or 1))
    per_shot_md = apply_allotment_per_shot_mandays(est, n)
    scale = per_shot_md / float(est.per_shot_mandays) if est.per_shot_mandays > 0 else 1.0
    internal_dept = scale_internal_depts(est.dept_days or {}, scale)
    bid_dept = internal_departments_to_bid(internal_dept)
    conf = float(est.confidence)

    return {
        "item_number": shot.item_number,
        "shot_code": _resolve_shot_code(shot),
        "description": build_shot_description(shot),
        "script_description": shot.script_description,
        "vfx_notes": shot.vfx_notes,
        "vfx_assumptions": shot.vfx_assumptions,
        "number_of_shots": n,
        "dept_days": bid_dept,
        "total_mandays": per_shot_md,
        "cost_per_shot": round(per_shot_md * day_rate, 2),
        "confidence": conf,
        "reasoning": est.reasoning or "",
        "adjustment_ranges": _compute_adjustment_ranges_bid(bid_dept, conf, compute_ranges),
        "mode": est.mode,
        "ai_total_mandays": float(est.per_shot_mandays),
    }


def run_bid_batch_estimate(
    svc: EstimatorService,
    shots: List[BidBatchShotItem],
    *,
    project: Optional[str] = None,
    day_rate: Optional[float] = None,
    mode: Optional[str] = None,
    pre_qual: Optional[BidPreQual] = None,
    compute_ranges,
) -> Dict[str, Any]:
    if not shots:
        raise HTTPException(400, "shots required")

    for i, shot in enumerate(shots):
        if not build_shot_description(shot).strip():
            raise HTTPException(400, f"shots[{i}] has no description content")

    rate = float(day_rate if day_rate is not None else svc.settings.day_rate)
    proj = project or (pre_qual.project if pre_qual else None) or "BID"

    rows: List[Dict[str, Any]] = []
    total_mandays = 0.0
    total_cost = 0.0

    for shot in shots:
        n = max(1, int(shot.number_of_shots or 1))
        pq = BidPreQual(
            project=proj,
            allotment_n=n,
        )
        if pre_qual:
            merged = pre_qual.model_dump(exclude_none=True)
            merged["allotment_n"] = n
            merged["project"] = proj
            pq = BidPreQual.model_validate(merged)

        desc = build_shot_description(shot)
        est = svc.estimate(desc, pre_qual=pq, mode=mode)
        row = _estimate_bid_row(est, shot, day_rate=rate, compute_ranges=compute_ranges)
        rows.append(row)
        total_mandays += row["total_mandays"]
        total_cost += row["cost_per_shot"]

    return {
        "count": len(rows),
        "project": proj,
        "day_rate": rate,
        "total_mandays": round(total_mandays, 2),
        "total_cost": round(total_cost, 2),
        "shots": rows,
    }


def _find_col(col_map: Dict[str, int], *names: str) -> Optional[int]:
    for name in names:
        nu = _norm_header(name)
        if nu in col_map:
            return col_map[nu]
        for key, idx in col_map.items():
            if nu in key or key in nu:
                return idx
    return None


def _cell(row: List[str], idx: Optional[int]) -> str:
    if idx is None or idx >= len(row):
        return ""
    return str(row[idx] or "").strip()


def _parse_number_of_shots(raw: str) -> int:
    raw = (raw or "").strip()
    if not raw:
        return 1
    try:
        return max(1, int(float(raw)))
    except ValueError:
        return 1


def parse_lvlup_bid_csv(content: bytes) -> Tuple[List[BidBatchShotItem], LvlupCsvArtifact]:
    text = content.decode("utf-8-sig", errors="replace")
    if not text.strip():
        raise HTTPException(400, "CSV file is empty")

    reader = csv.reader(io.StringIO(text))
    all_rows = [list(r) for r in reader]
    header_idx: Optional[int] = None
    for i, row in enumerate(all_rows):
        if any(_norm_header(c) == "ITEM#" for c in row):
            header_idx = i
            break

    if header_idx is None:
        raise HTTPException(400, 'LVL UP bid CSV must contain a header row with "ITEM#"')

    header = [_norm_header(c) for c in all_rows[header_idx]]
    col_map = {name: idx for idx, name in enumerate(header) if name}

    idx_item = _find_col(col_map, "ITEM#")
    idx_code = _find_col(col_map, "VFX SHOT CODE", "SHOT CODE")
    idx_script = _find_col(col_map, "SCRIPT DESCRIPTIONS", "SCRIPT DESCRIPTION")
    idx_notes = _find_col(col_map, "VFX NOTES")
    idx_assump = _find_col(col_map, "VFX ASSUMPTIONS")
    idx_client = _find_col(col_map, "CLIENT INITIAL THOUGHTS")
    idx_nshots = _find_col(col_map, "NUMBER OF SHOTS")

    shots: List[BidBatchShotItem] = []
    data_row_indices: List[int] = []

    for i in range(header_idx + 1, len(all_rows)):
        row = all_rows[i]
        item = _cell(row, idx_item)
        if not item:
            continue
        data_row_indices.append(i)
        shots.append(
            BidBatchShotItem(
                item_number=item,
                shot_code=_cell(row, idx_code) or None,
                script_description=_cell(row, idx_script),
                vfx_notes=_cell(row, idx_notes),
                vfx_assumptions=_cell(row, idx_assump),
                client_initial_thoughts=_cell(row, idx_client) or None,
                number_of_shots=_parse_number_of_shots(_cell(row, idx_nshots)),
            )
        )

    if not shots:
        raise HTTPException(400, "No shot rows found after ITEM# header (blank item numbers end the list)")

    artifact = LvlupCsvArtifact(
        all_rows=all_rows,
        header_idx=header_idx,
        col_map=col_map,
        data_row_indices=data_row_indices,
    )
    return shots, artifact


def parse_simple_batch_csv(content: bytes) -> List[BidBatchShotItem]:
    """Fallback: description + optional shot_number columns."""
    text = content.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV must include a header row")

    def _col(name: str) -> Optional[str]:
        for f in reader.fieldnames or []:
            if f and f.strip().lower() == name.lower():
                return f
        return None

    desc_col = _col("description") or _col("script descriptions")
    if not desc_col:
        raise HTTPException(400, 'CSV must include a "description" column or LVL UP ITEM# header')

    shot_col = _col("shot_number") or _col("vfx shot code") or _col("shot code")
    shots: List[BidBatchShotItem] = []
    for row in reader:
        desc = (row.get(desc_col) or "").strip()
        if not desc:
            continue
        code = (row.get(shot_col) or "").strip() if shot_col else None
        shots.append(
            BidBatchShotItem(
                description=desc,
                shot_code=code or None,
                shot_number=code or None,
            )
        )
    if not shots:
        raise HTTPException(400, "CSV has no data rows with a description")
    return shots


def parse_bid_csv_upload(content: bytes) -> Tuple[List[BidBatchShotItem], Optional[LvlupCsvArtifact]]:
    try:
        return parse_lvlup_bid_csv(content)
    except HTTPException as exc:
        if exc.status_code != 400 or "ITEM#" not in str(exc.detail):
            raise
    return parse_simple_batch_csv(content), None


def _ensure_output_columns(artifact: LvlupCsvArtifact) -> None:
    header = artifact.all_rows[artifact.header_idx]
    existing = {_norm_header(c) for c in header}
    for col in BID_OUTPUT_COLUMNS:
        if col not in existing:
            header.append(col)
            artifact.col_map[col] = len(header) - 1
    artifact.all_rows[artifact.header_idx] = header


def build_filled_bid_csv(
    artifact: LvlupCsvArtifact,
    results: List[Dict[str, Any]],
    *,
    project: str,
) -> bytes:
    _ensure_output_columns(artifact)
    col = artifact.col_map

    def set_cell(row: List[str], col_name: str, value: Any) -> None:
        idx = col.get(col_name)
        if idx is None:
            return
        while len(row) <= idx:
            row.append("")
        if isinstance(value, float) and value == int(value):
            row[idx] = str(int(value))
        else:
            row[idx] = str(value)

    for row_idx, result in zip(artifact.data_row_indices, results):
        row = artifact.all_rows[row_idx]
        bid_dept = result.get("dept_days") or {}
        for dept_col in BID_DEPT_MAP:
            set_cell(row, dept_col, bid_dept.get(dept_col, 0))
        set_cell(row, "TOTAL MANDAYS", result.get("total_mandays", 0))

    buf = io.StringIO()
    writer = csv.writer(buf)
    for r in artifact.all_rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8-sig")


def build_simple_export_csv(results: List[Dict[str, Any]], project: str) -> bytes:
    header = (
        ["ITEM#", "VFX SHOT CODE", "DESCRIPTION"]
        + list(BID_DEPT_MAP.keys())
        + ["TOTAL MANDAYS"]
    )
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in results:
        bid_dept = row.get("dept_days") or {}
        writer.writerow(
            [
                row.get("item_number") or "",
                row.get("shot_code") or "",
                row.get("description") or "",
                *[bid_dept.get(c, 0) for c in BID_DEPT_MAP],
                row.get("total_mandays", 0),
            ]
        )
    return buf.getvalue().encode("utf-8-sig")


def attach_csv_export(
    payload: Dict[str, Any],
    *,
    artifact: Optional[LvlupCsvArtifact],
    project: str,
    day_rate: float,
) -> Dict[str, Any]:
    results = payload.get("shots") or []
    if artifact:
        csv_bytes = build_filled_bid_csv(artifact, results, project=project)
    else:
        csv_bytes = build_simple_export_csv(results, project)
    payload = dict(payload)
    payload["export_csv_base64"] = base64.b64encode(csv_bytes).decode("ascii")
    payload["export_csv_filename"] = f"{project}_AI_ESTIMATE.csv"
    return payload
