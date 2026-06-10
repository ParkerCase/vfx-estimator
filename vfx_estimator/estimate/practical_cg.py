"""Practical vs CG ratio — prompt guidance and deterministic dept scaling."""

from __future__ import annotations

import re
from typing import Dict, Optional

from vfx_estimator.types import BidPreQual

CG_PIPELINE_DEPTS = ("layout", "animation", "cfx", "fx", "lighting", "dmp", "matchmove")
CAMERA_DEPTS = ("camera_track", "cam_track")

# At 0% CG, practical comp/cleanup shots are ~32% of a full CG pipeline bid.
PRACTICAL_TOTAL_FLOOR = 0.32


def _round_half(x: float) -> float:
    return round(float(x) * 2) / 2


def cg_ratio_from_pre_qual(pre_qual: Optional[BidPreQual]) -> int:
    """0–100 CG%; 100 when unset (full CG pipeline, legacy behavior)."""
    if not pre_qual:
        return 100
    if pre_qual.practical_cg_ratio is not None:
        return max(0, min(100, int(pre_qual.practical_cg_ratio)))
    if pre_qual.practical_vs_cg:
        m = re.search(r"(\d+)\s*%\s*CG", str(pre_qual.practical_vs_cg), re.I)
        if m:
            return max(0, min(100, int(m.group(1))))
    return 100


def practical_cg_total_multiplier(cg_ratio: int) -> float:
    cg_ratio = max(0, min(100, int(cg_ratio)))
    if cg_ratio >= 100:
        return 1.0
    t = cg_ratio / 100.0
    return PRACTICAL_TOTAL_FLOOR + (1.0 - PRACTICAL_TOTAL_FLOOR) * t


def build_practical_cg_prompt_rules(cg_ratio: int) -> str:
    cg_ratio = max(0, min(100, int(cg_ratio)))
    practical = 100 - cg_ratio
    if cg_ratio >= 100:
        return ""

    if cg_ratio == 0:
        return f"""
PRACTICAL vs CG (MANDATORY — {practical}% Practical / {cg_ratio}% CG):
- Treat this as a PRACTICAL PLATE shot. No CG asset build, no CG lighting, no 3D FX sim.
- layout=0, animation=0, cfx=0, fx=0, lighting=0, dmp=0, matchmove=0 — do not include them.
- Allowed departments: comp_roto, comp_paint, compositing, prep, camera_track (only if camera moves).
- Total mandays should be MUCH lower than a full CG pipeline (typically 2–6 days comp/cleanup only).
- Ignore CG-heavy similar shots when they conflict with this practical mandate.
"""

    if cg_ratio <= 25:
        cg_note = (
            "Mostly practical — CG is minimal accent only (small patches, minor fixes). "
            "CG pipeline departments should be near zero."
        )
    elif cg_ratio <= 60:
        cg_note = (
            "Mixed practical/CG — scale CG departments proportionally. "
            "Practical plate carries the frame; CG supports extensions or hero elements only."
        )
    else:
        cg_note = (
            "CG-forward — full pipeline applies but practical plate elements reduce some build/sim scope."
        )

    return f"""
PRACTICAL vs CG (MANDATORY — {practical}% Practical / {cg_ratio}% CG):
- {cg_note}
- Scale ALL CG pipeline departments (layout, animation, cfx, fx, lighting, dmp, matchmove)
  to roughly {cg_ratio}% of what you would bid for a 100% CG version of this shot.
- At {cg_ratio}% CG, a department you would normally bid 8 days should be ~{_round_half(8 * cg_ratio / 100)} days.
- Compositing/comp_paint/comp_roto still apply — practical plates need integration and cleanup.
- This ratio overrides similar-shot anchors when they assume more CG than the slider allows.
"""


def apply_practical_cg_ratio(
    dept_days: Dict[str, float],
    cg_ratio: int,
    *,
    description: str = "",
) -> Dict[str, float]:
    """
    Scale or zero CG pipeline departments based on the pre-qual slider.
    Presentation of practical-heavy vs CG-heavy bids; recomputes from dept dict.
    """
    cg_ratio = max(0, min(100, int(cg_ratio)))
    if cg_ratio >= 100:
        return {k: float(v) for k, v in (dept_days or {}).items() if float(v or 0) > 0}

    factor = cg_ratio / 100.0
    out: Dict[str, float] = {
        k: float(v) for k, v in (dept_days or {}).items() if float(v or 0) > 0
    }

    for key in CG_PIPELINE_DEPTS:
        if key not in out:
            continue
        if cg_ratio == 0:
            out.pop(key, None)
            continue
        scaled = _round_half(out[key] * factor)
        if scaled < 0.5:
            out.pop(key, None)
        else:
            out[key] = scaled

    # Camera tracking is often still needed on practical plates.
    cam_factor = 0.55 + 0.45 * factor
    for key in CAMERA_DEPTS:
        if key in out:
            out[key] = _round_half(out[key] * cam_factor)

    return out


def cg_rules_active(cg_ratio: Optional[int]) -> bool:
    return cg_ratio is not None and int(cg_ratio) < 100
