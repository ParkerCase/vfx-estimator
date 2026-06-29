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

SHOT_BASELINES: Dict[str, Dict[str, float | str]] = {
    "monitor_insert": {"description": "Screen/monitor replacement, UI insert", "matchmove": 0.5, "compositing": 0.75, "total": 1.9},
    "muzzle_flash": {"description": "Gunfire muzzle flash effect", "matchmove": 0.25, "fx": 0.4, "compositing": 0.75, "total": 1.7},
    "bullet_hits": {"description": "Bullet impact on surface or body", "matchmove": 0.25, "fx": 0.6, "compositing": 0.8, "total": 2.0},
    "wire_removal": {"description": "Safety wire or rig removal from stunt", "comp_paint": 0.8, "comp_roto": 0.4, "compositing": 0.6, "total": 1.9},
    "paint_cleanup": {"description": "Paint out, logo removal, period cleanup", "comp_paint": 1.0, "comp_roto": 0.3, "compositing": 0.5, "total": 1.5},
    "blood_gore": {"description": "Blood spray, wound enhancement", "fx": 0.5, "comp_roto": 0.3, "compositing": 0.75, "total": 1.8},
    "digital_makeup": {"description": "Beauty work, aging, de-aging, scar enhancement", "matchmove": 0.75, "compositing": 2.0, "comp_roto": 0.5, "total": 3.5},
    "sky_replacement": {"description": "Sky plate replacement only", "matchmove": 0.3, "dmp": 0.5, "comp_roto": 0.5, "compositing": 1.0, "total": 2.3},
    "set_extension_simple": {"description": "Simple set extension, DMP-based", "matchmove": 0.5, "layout": 0.5, "dmp": 1.5, "comp_roto": 0.5, "compositing": 1.25, "total": 4.5},
    "matte_painting": {"description": "Full matte painting / DMP environment", "matchmove": 0.5, "dmp": 2.0, "comp_roto": 0.5, "compositing": 1.25, "total": 4.2},
    "smoke_atmosphere": {"description": "Smoke, fog, mist, atmospheric particles", "matchmove": 0.3, "fx": 1.0, "compositing": 1.25, "total": 3.0},
    "fire_enhancement": {"description": "Fire added to practical plate", "matchmove": 0.5, "fx": 1.25, "lighting": 0.75, "compositing": 1.5, "total": 4.5},
    "explosion": {"description": "Explosion, blast, detonation", "matchmove": 0.75, "fx": 2.5, "lighting": 1.25, "compositing": 2.0, "total": 7.1},
    "water_simulation": {"description": "Water FX, ocean, splash, rain simulation", "matchmove": 0.75, "fx": 3.0, "lighting": 1.5, "compositing": 2.0, "total": 7.9},
    "destruction": {"description": "Building collapse, debris, destruction FX", "matchmove": 0.75, "fx": 3.5, "lighting": 1.5, "compositing": 2.5, "total": 9.5},
    "cg_vehicle": {"description": "CG car, truck, ship, aircraft integration", "camera_track": 1.0, "matchmove": 1.0, "layout": 0.75, "animation": 1.0, "lighting": 1.25, "compositing": 1.5, "total": 6.1},
    "cg_creature": {"description": "CG creature, animal, monster (asset excluded)", "camera_track": 1.0, "matchmove": 1.0, "layout": 0.75, "animation": 3.0, "cfx": 0.75, "lighting": 1.75, "compositing": 2.0, "total": 10.1},
    "cg_character": {"description": "Full CG digital double or character", "camera_track": 1.0, "matchmove": 1.25, "layout": 0.75, "animation": 3.5, "cfx": 1.0, "lighting": 2.0, "compositing": 2.5, "total": 12.0},
    "face_replacement": {"description": "Face swap, digital double face, likeness replacement", "matchmove": 1.0, "animation": 1.25, "lighting": 0.75, "compositing": 2.5, "total": 6.1},
    "cg_environment": {"description": "Full CG environment build (3D, no DMP)", "camera_track": 0.75, "layout": 1.0, "lighting": 1.5, "dmp": 1.5, "compositing": 2.0, "total": 7.4},
    "cg_environment_hero": {"description": "Hero CG environment, establishing shot quality", "camera_track": 1.0, "layout": 2.0, "lighting": 5.0, "dmp": 2.0, "compositing": 5.0, "total": 16.0},
    "crowd_replication_dozens": {"description": "Crowd multiplication, dozens of people", "camera_track": 0.5, "matchmove": 0.75, "layout": 0.5, "animation": 2.0, "lighting": 1.0, "compositing": 1.5, "total": 6.3},
    "crowd_replication_hundreds": {"description": "Crowd multiplication, hundreds of people", "camera_track": 1.0, "matchmove": 0.75, "layout": 0.75, "animation": 4.0, "lighting": 1.5, "compositing": 2.0, "total": 10.5},
    "crowd_replication_thousands": {"description": "Massive crowd, thousands of people", "camera_track": 1.5, "matchmove": 1.0, "layout": 1.0, "animation": 8.0, "lighting": 2.5, "compositing": 3.0, "total": 17.5},
    "cloth_hair_sim": {"description": "CFX cloth or hair simulation only", "cfx": 2.0, "lighting": 0.5, "compositing": 1.0, "total": 3.5},
}

COMPLEXITY_MODIFIERS: Dict[str, Dict[str, float]] = {
    "establishing": {"all": 1.5, "lighting": 1.8, "compositing": 1.6},
    "hero_close_up": {"all": 1.4, "animation": 1.5, "compositing": 1.5},
    "background": {"all": 0.7},
    "standard": {"all": 1.0},
    "locked_off": {"camera_track": 0.0, "matchmove": 0.7},
    "handheld": {"camera_track": 1.5, "matchmove": 1.4, "comp_roto": 1.3},
    "crane_dolly": {"camera_track": 1.2, "matchmove": 1.1},
    "night": {"lighting": 1.4},
    "day": {"lighting": 1.0},
    "dusk_dawn": {"lighting": 1.3},
    "single_element": {"all": 1.0},
    "multiple_elements": {"all": 1.3, "compositing": 1.4},
    "short_under_2s": {"all": 0.8},
    "medium_2_5s": {"all": 1.0},
    "long_over_5s": {"all": 1.3},
    "hero_quality": {"all": 1.4},
    "standard_quality": {"all": 1.0},
    "background_quality": {"all": 0.75},
}

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


def build_vfx_rules(custom_baselines: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
    """
    Build the VFX rules prompt with shot baselines.

    Custom baselines are studio-specific overrides loaded from Xata presets.
    """
    baselines: Dict[str, Dict[str, Any]] = {
        key: dict(data) for key, data in SHOT_BASELINES.items()
    }
    if custom_baselines:
        for key, data in custom_baselines.items():
            if isinstance(data, dict):
                baselines[key] = {**baselines.get(key, {}), **data}

    baseline_text = "\n\nSHOT TYPE BASELINE ALLOCATIONS (medium complexity, standard shot):\n"
    baseline_text += "Use these as your starting point, then apply modifiers.\n\n"
    for key, data in baselines.items():
        depts = {
            k: v
            for k, v in data.items()
            if k not in ("description", "total", "source")
        }
        dept_str = ", ".join(
            f"{k}={float(v):g}d"
            for k, v in depts.items()
            if isinstance(v, (int, float)) and float(v) > 0
        )
        total = data.get("total", 0)
        baseline_text += f"  {key}: {data.get('description', key)}\n"
        baseline_text += f"    Base: {dept_str} -> Total: {float(total or 0):g}d\n\n"

    modifier_text = """
COMPLEXITY MODIFIERS (multiply baseline days by):
  establishing shot:    all depts x1.5, lighting x1.8, comp x1.6
  hero/close-up:        all depts x1.4
  background/distant:   all depts x0.7
  handheld camera:      camera_track x1.5, matchmove x1.4
  locked camera:        camera_track = 0
  night scene:          lighting x1.4
  multiple CG elements: all x1.3, comp x1.4
  long shot (5s+):      all x1.3

ESTIMATION PROCESS:
1. Identify the closest shot type baseline from the list above
2. Note which complexity modifiers apply
3. Apply modifiers to the baseline days
4. Check against similar shots from training data
5. If training data strongly disagrees with baseline, weight training data at 60%, baseline at 40%
6. Apply the absolute rules below (COMP minimum, etc.)
7. Return final adjusted department days
"""
    return baseline_text + modifier_text + "\n\n" + VFX_RULES


def _load_studio_presets(settings: Settings) -> Dict[str, Dict[str, Any]]:
    """Load custom shot presets from Xata if any exist."""
    try:
        from vfx_estimator.integrations.xata import load_presets

        return load_presets(settings.resolved_xata_postgres_url())
    except Exception:
        return {}


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
        rules = build_vfx_rules(_load_studio_presets(self.settings))

        prompt = f"""You are a senior VFX supervisor estimating MANDAYS (person-days of work, NOT dollars) for a single shot.

{rules}
{cg_rules}{project_ctx}{brief}{flags_ctx}
{similar_block}
SHOT TO ESTIMATE: "{description}"

Return a JSON object. You MUST include these fields — missing core fields is an error:

  "shot_type"   — one of: establishing | hero | background | standard
  "departments" — object where each key is a department name and its value is {{"days": <number>}}.
                  Include ALL departments with non-zero work for this shot.
                  Valid dept names: camera_track, matchmove, layout, animation, cfx, fx,
                  lighting, dmp, comp_paint, comp_roto, compositing, prep, crowds, enviro, ai
  "total_days"  — float; MUST equal the exact arithmetic sum of all included department days
  "confidence"  — float between 0.0 and 1.0
  "reasoning"   — one sentence: shot type classification and key similar-shot anchor used
  "baseline_used" — closest baseline key, e.g. cg_creature
  "modifiers_applied" — array of modifier keys applied, e.g. ["hero_close_up", "night"]
  "baseline_days" — baseline total before modifiers
  "adjusted_days" — total after modifiers before similar-shot blending

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
