"""Gemini mandays estimation with retrieval-augmented similar shots."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.estimate.practical_cg import (
    build_practical_cg_prompt_rules,
    cg_ratio_from_pre_qual,
    cg_rules_active,
)
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.llm.gemini_client import generate_json
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import BidPreQual, SimilarShot

CG_DEPARTMENTS = ("layout", "animation", "cfx", "fx")
CG_DESCRIPTION_RE = re.compile(
    r"\b(cg|cgi|computer[- ]generated|3d|digital creature|digital double)\b",
    re.IGNORECASE,
)
MIN_CG_LIGHTING_DAYS = 3.0

VFX_RULES = """
ABSOLUTE RULES — override similar shots if they conflict:

COMPOSITING IS MANDATORY ON EVERY SINGLE SHOT — NO EXCEPTIONS:
- Compositing is the final step that integrates ALL elements
- If you return comp = 0, your entire answer is wrong
- Use department key "compositing" for COMP days (not a separate "comp" key)
- Minimum comp days by total shot size:
    1-5 day shot:   comp = 2 days minimum
    5-10 day shot:  comp = 3 days minimum
    10-20 day shot: comp = 4-5 days minimum
    20+ day shot:   comp = 5-8 days minimum
- For shots with 3D elements (lighting, animation, FX, layout):
    comp must be at least 25% of total days
- The only exception is if the shot is pure 2D cleanup
  (wire removal, paint-out only) — even then comp = 2 days minimum
REMINDER: comp = 0 on any shot is always incorrect.

CG LIGHTING IS MANDATORY — NO EXCEPTIONS:
- Any CG element requires lighting. CG without lighting is physically impossible.
- If layout, animation, CFX, FX, CGI, 3D, digital creature, or digital double applies,
  department "lighting" must be included with at least 3 days.

2. ANIMATION = 0 for any static object (buildings, castles, palaces, environments,
   vehicles parked, static props).
   Animation ONLY for: characters, creatures, crowds, moving vehicles.

3. FX is REQUIRED when description mentions: fire, smoke, explosion, water,
   destruction, blood, sparks, magic, particles, atmosphere.
   If the word "fire", "smoke", "explosion", "blood", or "destruction" appears — FX must be > 0.

4. DMP is REQUIRED when description mentions: sky replacement, set extension,
   matte painting, background replacement, 2.5D.
   If "sky replacement" or "set extension" appears — DMP must be > 0.

5. CAMERA TRACK is REQUIRED when: camera moves (crane, dolly, handheld,
   tracking shot). NOT needed for locked-off cameras.

6. WIRE REMOVAL / CLEANUP shots: ONLY comp_roto + comp_paint + compositing.
   layout = 0, animation = 0, lighting = 0, fx = 0. No exceptions.

AI (Generative AI work):
- Used ONLY when: description explicitly mentions AI-generated elements,
  AI cleanup, AI upscaling, or generative AI tools
- DEFAULT IS 0 — do not allocate AI days unless explicitly stated
- When needed: 1-5 days depending on scope
- Key: "ai" (maps to "AI" column in bid)

7. ESTABLISHING shots get hero treatment:
   lighting >= 6 days, compositing >= 5 days minimum.

8. CROWD SCALE:
   dozens of people = +5 animation days
   hundreds of people = +10 animation days
   thousands of people = +15 animation days

9. CROWDS department:
   Use when descriptions mention crowd multiplication, hundreds/thousands of people,
   army, soldiers, extras. Maps to animation + crowds.
   Use CROWDS for dedicated crowd sim work separate from character animation.
   Department key "crowds" (maps to CROWDS column in bid).

10. ENVIRO department:
   Use for CG environment builds (full 3D environment construction separate from DMP).
   Maps to ENVIRO / ENV LAYOUT in MARZ files.
   Department key "enviro" (maps to ENVIRO column in bid).

SHOT TYPE:
  establishing → MINIMUM 18 days total. lighting >=6, compositing >=5.
  hero         → All depts +30-50%.
  background   → All depts -20-30%.
  standard     → Anchor to similar shots.

TYPICAL DAY RANGES (standard shot):
  camera_track 1-3d  |  matchmove 1-3d  |  layout 1-4d   |  animation 3-12d
  cfx 1-4d           |  fx 3-10d        |  lighting 3-9d  |  dmp 2-5d
  comp_paint 2-6d    |  comp_roto 1-4d  |  compositing 3-8d  |  ai 0d (default) or 1-5d
"""


def _round_half(x: float) -> float:
    return round(float(x) * 2) / 2


def _dept_days(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    depts = data.get("departments")
    if not isinstance(depts, dict):
        return out
    for k, v in depts.items():
        if isinstance(v, dict):
            days = float(v.get("days") or 0)
        else:
            days = float(v or 0)
        if days > 0:
            out[str(k)] = days
    return out


def _set_dept(data: Dict[str, Any], key: str, days: float) -> None:
    depts = data.setdefault("departments", {})
    if not isinstance(depts, dict):
        return
    depts[key] = {"days": max(0.0, float(days))}


def _has_cg_element(description: str, dept: Dict[str, float]) -> bool:
    return any(dept.get(d, 0) > 0 for d in CG_DEPARTMENTS) or bool(
        CG_DESCRIPTION_RE.search(description)
    )


def _enforce_vfx_rules(
    description: str,
    data: Dict[str, Any],
    *,
    cg_ratio: Optional[int] = None,
) -> None:
    """Apply hard floors/zeros when Gemini omits mandatory departments."""
    desc = description.lower()
    dept = _dept_days(data)
    total = max(float(data.get("total_days") or 0), sum(dept.values()))
    cg = 100 if cg_ratio is None else max(0, min(100, int(cg_ratio)))

    if dept.get("compositing", 0) <= 0:
        if total <= 5:
            comp = 2.0
        elif total <= 10:
            comp = 3.0
        elif total <= 20:
            comp = 4.0
        else:
            comp = max(5.0, total * 0.20)
        _set_dept(data, "compositing", comp)

    if re.search(r"\b(wire removal|wire remove|wires?)\b", desc):
        depts = data.get("departments")
        if isinstance(depts, dict):
            for key in ("layout", "animation", "lighting", "fx", "dmp", "cfx"):
                depts.pop(key, None)
        _set_dept(data, "compositing", 2.0)

    dept = _dept_days(data)
    if (
        cg > 20
        and re.search(r"\b(fire|smoke|explosion|blood|destruction)\b", desc)
        and dept.get("fx", 0) <= 0
    ):
        fx_days = 3.0 if cg >= 100 else max(1.0, _round_half(3.0 * cg / 100.0))
        _set_dept(data, "fx", fx_days)

    if (
        cg > 15
        and re.search(r"\b(sky replacement|set extension|matte painting)\b", desc)
        and dept.get("dmp", 0) <= 0
    ):
        dmp_days = 2.0 if cg >= 100 else max(0.5, _round_half(2.0 * cg / 100.0))
        _set_dept(data, "dmp", dmp_days)

    dept = _dept_days(data)
    if cg > 0 and _has_cg_element(description, dept) and dept.get("lighting", 0) <= 0:
        lit = MIN_CG_LIGHTING_DAYS if cg >= 100 else max(1.0, _round_half(MIN_CG_LIGHTING_DAYS * cg / 100.0))
        _set_dept(data, "lighting", lit)

    dept = _dept_days(data)
    total = max(float(data.get("total_days") or 0), sum(dept.values()))
    is_wire_cleanup = bool(re.search(r"\b(wire removal|wire remove|wires?)\b", desc))
    has_3d = any(dept.get(d, 0) > 0 for d in ("layout", "animation", "lighting", "fx", "dmp"))
    if has_3d and not is_wire_cleanup:
        min_comp = max(dept.get("compositing", 0), total * 0.25)
        if total >= 18:
            min_comp = max(min_comp, 5.0)
        if total >= 20:
            min_comp = max(min_comp, 6.0)
        _set_dept(data, "compositing", round(min_comp * 2) / 2)

    dept = _dept_days(data)
    data["total_days"] = max(0.25, sum(dept.values()))


class GeminiMandaysEstimator:
    def __init__(self, index: ShotRetrievalIndex, settings: Optional[Settings] = None):
        self.index = index
        self.settings = settings or get_settings()
        self.flags = FlagsStore(settings=self.settings)

    def _format_similar(self, hits: List[SimilarShot]) -> str:
        if not hits:
            return "SIMILAR SHOTS: none found.\n"
        lines = ["SIMILAR SHOTS (use as pricing anchors):"]
        for i, h in enumerate(hits, 1):
            tag = "★ CORRECTION" if h.source == "correction" else h.source.upper()
            lines.append(
                f'  {i}. [{tag}] "{h.description}"\n'
                f"     mandays={h.mandays:.1f}  cost=${h.cost:,.0f}  sim={h.similarity:.2f}"
            )
        return "\n".join(lines) + "\n"

    def _build_project_context(self, pre_qual: Optional[BidPreQual]) -> str:
        """Project-wide context block (batch / director pre-qual)."""
        if not pre_qual:
            return ""
        cg = cg_ratio_from_pre_qual(pre_qual)
        has_context = any(
            getattr(pre_qual, f, None)
            for f in (
                "shot_type_override",
                "bid_scale_tier",
                "complexity_band",
                "director_brief",
                "vfx_assumptions",
            )
        ) or cg < 100
        if not has_context:
            return ""
        lines = ["PROJECT CONTEXT (applies to all shots):"]
        if pre_qual.shot_type_override:
            lines.append(f"- Shot type: {pre_qual.shot_type_override}")
        if pre_qual.bid_scale_tier:
            lines.append(f"- Scale: {pre_qual.bid_scale_tier}")
        if pre_qual.complexity_band:
            lines.append(f"- Complexity: {pre_qual.complexity_band}")
        if cg < 100:
            lines.append(f"- CG ratio: {100 - cg}% Practical / {cg}% CG (MANDATORY calibration)")
        elif pre_qual.practical_vs_cg:
            lines.append(f"- CG ratio: {pre_qual.practical_vs_cg}")
        if pre_qual.director_brief:
            lines.append(f"- Director intent: {pre_qual.director_brief.strip()}")
        if pre_qual.vfx_assumptions:
            lines.append(f"- VFX assumptions: {pre_qual.vfx_assumptions.strip()}")
        lines.append(
            "This project context overrides generic assumptions — use it "
            "to calibrate all estimates accordingly."
        )
        return "\n".join(lines) + "\n\n"

    def _build_brief(self, pre_qual: Optional[BidPreQual]) -> str:
        """Extract allotment and legacy notes not covered by project context."""
        if not pre_qual:
            return ""
        pq = pre_qual.to_legacy_dict()
        lines = []
        if pq.get("bid_context_notes") and not (
            pre_qual.director_brief or pre_qual.vfx_assumptions or pre_qual.shot_type_override
        ):
            lines.append(f"PRODUCTION CONTEXT: {pq['bid_context_notes']}")
        if pq.get("allotment_n") and int(pq["allotment_n"]) > 1:
            lines.append(
                f"ALLOTMENTS: {pq['allotment_n']} shots in sequence "
                f"(estimate shot 1 at 100%; subsequent shots at 85-90%)"
            )
        return ("\n".join(lines) + "\n") if lines else ""

    def estimate(
        self,
        description: str,
        *,
        pre_qual: Optional[BidPreQual] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        k = top_k or self.settings.retrieval_top_k
        hits = self.index.query(description, top_k=k)

        project_ctx = self._build_project_context(pre_qual)
        brief = self._build_brief(pre_qual)
        flags_ctx = self.flags.prompt_context(description)
        similar_block = self._format_similar(hits[:6])
        cg_ratio = cg_ratio_from_pre_qual(pre_qual)
        cg_rules = build_practical_cg_prompt_rules(cg_ratio)

        prompt = f"""You are a senior VFX supervisor estimating MANDAYS (person-days of work, NOT dollars) for a single shot.

{VFX_RULES}
{cg_rules}{project_ctx}{brief}{flags_ctx}
{similar_block}
SHOT TO ESTIMATE: "{description}"

Return a JSON object. You MUST include ALL FIVE of these fields — missing any field is an error:

  "shot_type"   — one of: establishing | hero | background | standard
  "departments" — object where each key is a department name and its value is {{"days": <number>}}.
                  Include ALL departments with non-zero work for this shot.
                  Valid dept names: camera_track, matchmove, layout, animation, cfx, fx,
                  lighting, dmp, comp_paint, comp_roto, compositing, prep, ai
  "total_days"  — float; MUST equal the exact arithmetic sum of all included department days
  "confidence"  — float between 0.0 and 1.0
  "reasoning"   — one sentence: shot type classification and key similar-shot anchor used

Rules:
  - Omit departments with 0 days (do not include them at all)
  - compositing (COMP) must be included on every VFX shot with days >= 2
  - Compute total_days by summing departments — do not guess it independently
  - Return raw JSON only — no markdown fences, no text before or after the JSON
"""

        data = generate_json(prompt, settings=self.settings)
        if data is None:
            raise RuntimeError("Gemini request timed out")
        _enforce_vfx_rules(description, data, cg_ratio=cg_ratio)

        dept_sum = sum(_dept_days(data).values())
        stated_total = float(data.get("total_days") or 0)
        total = dept_sum if dept_sum > 0 else stated_total

        # ── Sanity gate — reject nonsense totals so service uses retrieval fallback ──
        if total < 1.0:
            raise RuntimeError(
                f"Gemini total too low: {total:.2f}d (dept_sum={dept_sum:.2f}, "
                f"stated={stated_total:.2f}). Response likely truncated or all-zero. "
                f"Raw departments: {data.get('departments')}"
            )
        if total > 150:
            raise RuntimeError(
                f"Gemini total too high: {total:.1f}d — possible unit confusion "
                f"(cost vs mandays) or hallucination."
            )

        data["total_days"] = max(0.25, total)

        # ── Clamp confidence ──
        try:
            conf = float(data.get("confidence") or 0.5)
            data["confidence"] = max(0.1, min(1.0, conf))
        except (TypeError, ValueError):
            data["confidence"] = 0.5

        data["similar_shots"] = [h.model_dump() for h in hits[:5]]
        return data
