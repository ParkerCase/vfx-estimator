"""Gemini mandays estimation with retrieval-augmented similar shots."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.learning.flags import FlagsStore
from vfx_estimator.llm.gemini_client import generate_json
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import BidPreQual, SimilarShot

# Concise domain rules — shorter guide = better instruction-following on flash models.
# The full verbose guide caused gemini-2.0-flash to truncate output and echo templates.
VFX_RULES = """
SHOT TYPE:
  establishing → wide hero vista, full CG environment. MINIMUM 15 days total. Lighting +40%, comp +30%.
  hero         → close-up / primary frame. All depts +30-50%.
  background   → CG not primary focus. All depts -20-30%.
  standard     → baseline; anchor to similar shots.

DEPARTMENT RULES (non-negotiable):
  Wire / stunt removal   → comp_roto + comp_paint ONLY. All 3D depts must be 0.
  CG environment / build → layout + lighting + comp_paint. animation = 0 for static.
  CG character / creature→ layout + animation + cfx + lighting + comp_paint.
  FX shot (fire/smoke)   → fx + lighting + comp_paint — all three required.
  Crowd hundreds         → animation +10d. Crowd thousands → animation +15d.
  Greenscreen            → comp_roto + matchmove (moving cam) + lighting (grade).

TYPICAL DAY RANGES (standard shot; scale for shot-type multiplier above):
  camera_track 1-3d  |  matchmove 1-3d  |  layout 1-4d   |  animation 3-12d
  cfx 1-4d           |  fx 3-10d        |  lighting 3-9d  |  dmp 1-4d
  comp_paint 2-7d    |  comp_roto 2-6d  |  prep 0.5-2d
"""


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
                  lighting, dmp, comp_paint, comp_roto, prep
  "total_days"  — float; MUST equal the exact arithmetic sum of all included department days
  "confidence"  — float between 0.0 and 1.0
  "reasoning"   — one sentence: shot type classification and key similar-shot anchor used

Rules:
  - Omit departments with 0 days (do not include them at all)
  - Compute total_days by summing departments — do not guess it independently
  - Return raw JSON only — no markdown fences, no text before or after the JSON
"""

        data = generate_json(prompt, settings=self.settings)

        # ── Recompute total from dept sums (guards against missing total_days) ──
        dept_sum = 0.0
        if isinstance(data.get("departments"), dict):
            for v in data["departments"].values():
                if isinstance(v, dict):
                    try:
                        dept_sum += float(v.get("days") or 0)
                    except (TypeError, ValueError):
                        pass
                else:
                    try:
                        dept_sum += float(v or 0)
                    except (TypeError, ValueError):
                        pass

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
