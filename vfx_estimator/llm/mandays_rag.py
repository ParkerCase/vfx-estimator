"""Gemini mandays estimation with retrieval-augmented similar shots."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.llm.gemini_client import generate_json
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import BidPreQual, SimilarShot

VFX_RULES = """
ABSOLUTE RULES — override similar shots if they conflict:

1. COMP (compositing) is REQUIRED on EVERY VFX shot without exception.
   Minimum: 2 days. Hero/establishing shots: 5-8 days.
   If COMP = 0, your answer is WRONG.
   Use department key "compositing" for COMP days (not a separate "comp" key).

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

7. ESTABLISHING shots get hero treatment:
   lighting >= 6 days, compositing >= 5 days minimum.

8. CROWD SCALE:
   dozens of people = +5 animation days
   hundreds of people = +10 animation days
   thousands of people = +15 animation days

SHOT TYPE:
  establishing → MINIMUM 18 days total. lighting >=6, compositing >=5.
  hero         → All depts +30-50%.
  background   → All depts -20-30%.
  standard     → Anchor to similar shots.

TYPICAL DAY RANGES (standard shot):
  camera_track 1-3d  |  matchmove 1-3d  |  layout 1-4d   |  animation 3-12d
  cfx 1-4d           |  fx 3-10d        |  lighting 3-9d  |  dmp 2-5d
  comp_paint 2-6d    |  comp_roto 1-4d  |  compositing 3-8d
"""


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


def _enforce_vfx_rules(description: str, data: Dict[str, Any]) -> None:
    """Apply hard floors/zeros when Gemini omits mandatory departments."""
    desc = description.lower()
    dept = _dept_days(data)

    if dept.get("compositing", 0) <= 0:
        _set_dept(data, "compositing", 2.0)

    if re.search(r"\b(wire removal|wire remove|wires?)\b", desc):
        depts = data.get("departments")
        if isinstance(depts, dict):
            for key in ("layout", "animation", "lighting", "fx", "dmp", "cfx"):
                depts.pop(key, None)

    if re.search(r"\b(fire|smoke|explosion|blood|destruction)\b", desc) and dept.get("fx", 0) <= 0:
        _set_dept(data, "fx", 3.0)

    if re.search(r"\b(sky replacement|set extension|matte painting)\b", desc) and dept.get("dmp", 0) <= 0:
        _set_dept(data, "dmp", 2.0)

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

    def _build_brief(self, pre_qual: Optional[BidPreQual]) -> str:
        """Extract all pre-qualification context into a single brief block."""
        if not pre_qual:
            return ""
        pq = pre_qual.to_legacy_dict()
        lines = []
        if pq.get("bid_context_notes"):
            lines.append(f"PRODUCTION CONTEXT: {pq['bid_context_notes']}")
        if pq.get("complexity_band"):
            lines.append(f"COMPLEXITY BAND: {pq['complexity_band']}")
        if pq.get("bid_scale_tier"):
            lines.append(f"BID SCALE: {pq['bid_scale_tier']}")
        if pq.get("practical_vs_cg"):
            lines.append(f"PRACTICAL VS CG: {pq['practical_vs_cg']}")
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

        brief = self._build_brief(pre_qual)
        flags_ctx = self.flags.prompt_context(description)
        similar_block = self._format_similar(hits[:6])

        prompt = f"""You are a senior VFX supervisor estimating MANDAYS (person-days of work, NOT dollars) for a single shot.

{VFX_RULES}
{brief}{flags_ctx}
{similar_block}
SHOT TO ESTIMATE: "{description}"

Return a JSON object. You MUST include ALL FIVE of these fields — missing any field is an error:

  "shot_type"   — one of: establishing | hero | background | standard
  "departments" — object where each key is a department name and its value is {{"days": <number>}}.
                  Include ALL departments with non-zero work for this shot.
                  Valid dept names: camera_track, matchmove, layout, animation, cfx, fx,
                  lighting, dmp, comp_paint, comp_roto, compositing, prep
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
        _enforce_vfx_rules(description, data)

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
