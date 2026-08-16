"""Gemini-backed CG asset build-day estimator."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from vfx_estimator.config import Settings
from vfx_estimator.llm.gemini_client import generate_json

ASSET_BASELINES: Dict[str, Dict[str, Any]] = {
    "hero_creature": {
        "description": "Hero creature, full pipeline",
        "modelling": 12,
        "texturing": 6,
        "rigging": 12,
        "lookdev": 4,
        "cfx": 5,
        "fx": 3,
        "comp_dev": 2,
        "total": 44,
    },
    "background_creature": {
        "description": "Background/distant creature",
        "modelling": 6,
        "texturing": 3,
        "rigging": 6,
        "lookdev": 2,
        "cfx": 1,
        "comp_dev": 1,
        "total": 19,
    },
    "cg_vehicle_hero": {
        "description": "Hero CG vehicle (car, ship, aircraft)",
        "modelling": 10,
        "texturing": 5,
        "rigging": 3,
        "lookdev": 4,
        "fx": 1,
        "comp_dev": 2,
        "total": 25,
    },
    "cg_vehicle_background": {
        "description": "Background CG vehicle",
        "modelling": 4,
        "texturing": 2,
        "rigging": 1,
        "lookdev": 1,
        "comp_dev": 1,
        "total": 9,
    },
    "cg_environment_hero": {
        "description": "Hero CG environment (castle, city, interior)",
        "modelling": 14,
        "texturing": 6,
        "lookdev": 5,
        "dmp": 2,
        "comp_dev": 2,
        "total": 29,
    },
    "cg_environment_background": {
        "description": "Background CG environment",
        "modelling": 7,
        "texturing": 3,
        "lookdev": 3,
        "dmp": 1,
        "comp_dev": 1,
        "total": 15,
    },
    "digital_double": {
        "description": "Digital double of actor",
        "modelling": 10,
        "texturing": 6,
        "rigging": 14,
        "lookdev": 5,
        "cfx": 6,
        "comp_dev": 3,
        "total": 44,
    },
    "hero_prop": {
        "description": "Hero CG prop (featured, close camera)",
        "modelling": 4,
        "texturing": 2,
        "lookdev": 2,
        "comp_dev": 1,
        "total": 9,
    },
    "background_prop": {
        "description": "Background CG prop",
        "modelling": 2,
        "texturing": 1,
        "lookdev": 0.5,
        "comp_dev": 0.5,
        "total": 4,
    },
}

ASSET_PROMPT = """You are a VFX asset estimator.
Estimate BUILD days per department for this CG asset.
Exclude per-shot work (lighting shots, compositing shots) -- this is ONE-TIME build effort only.

ASSET DEPARTMENTS:
  modelling:  Building 3D geometry (0-15d)
  texturing:  Surfacing and material creation (0-8d)
  lookdev:    Look development / lighting test renders (0-6d)
  rigging:    Character or creature rig and controls (0-15d)
  cfx:        Cloth/hair/fur simulation setup (0-8d)
  fx:         FX development for asset (fire, destruction sim, etc) (0-10d)
  lighting:   Asset-specific lighting rig setup (0-5d) — ONLY when the asset
              needs custom build-time lighting, such as a glowing creature,
              light-emitting prop, or environment with lighting pre-baked as
              part of the asset. Do not include ordinary per-shot lighting.
  dmp:        Matte painting elements for asset, rare (0-5d)
  comp_dev:   Compositing development / integration tests (0-4d)

ASSET BASELINES (one-time build, medium quality):
{baselines}

COMPLEXITY MODIFIERS:
  hero / featured:     all x1.5
  secondary:           all x1.0 (baseline)
  background/distant:  all x0.5-0.7

IMPORTANT — VARIATIONS:
  This asset has {variations} variation(s).
  Estimate the BASE BUILD days (for the first/hero version).
  The system will automatically calculate additional variation
  days at 30% of base per variation.
  Do NOT multiply by variation count in your estimate.
  Just return the single-asset base build days.

ASSET: "{name}"
DESCRIPTION: "{description}"
VARIATIONS: {variations}
TIER: "{tier}"
CONTEXT: "{notes}"

Identify the closest baseline, apply modifiers, return final per-department build days.

Return JSON only:
{{
  "asset_tier": "hero|secondary|background",
  "baseline_used": "hero_creature",
  "departments": {{
    "modelling": {{"days": 12}},
    "texturing": {{"days": 6}},
    "rigging": {{"days": 12}},
    "lookdev": {{"days": 4}},
    "cfx": {{"days": 5}},
    "lighting": {{"days": 0}},
    "dmp": {{"days": 0}},
    "comp_dev": {{"days": 2}}
  }},
  "total_days": 41,
  "variations_note": "First variation 41d. Each additional ~10d.",
  "confidence": 0.75,
  "reasoning": "Hero creature baseline x1.0 (secondary quality)"
}}
"""


def _baseline_text() -> str:
    out = ""
    for key, data in ASSET_BASELINES.items():
        depts = {k: v for k, v in data.items() if k not in ("description", "total")}
        dept_str = ", ".join(f"{k}={v}d" for k, v in depts.items() if float(v or 0) > 0)
        out += f"  {key}: {data['description']}\n"
        out += f"    {dept_str} -> {data['total']}d\n\n"
    return out


def _dept_days(data: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    departments = data.get("departments") or {}
    if not isinstance(departments, dict):
        return out
    for key, raw in departments.items():
        try:
            days = float(raw.get("days") if isinstance(raw, dict) else raw)
        except (TypeError, ValueError):
            continue
        if days > 0:
            out[str(key)] = days
    return out


def _set_dept(data: Dict[str, Any], key: str, days: float) -> None:
    departments = data.setdefault("departments", {})
    if not isinstance(departments, dict):
        departments = {}
        data["departments"] = departments
    departments[key] = {"days": max(0.0, float(days))}


def _enforce_asset_dmp(asset: Any, result: Dict[str, Any]) -> None:
    text = " ".join(
        str(part or "")
        for part in (
            getattr(asset, "asset_name", ""),
            getattr(asset, "description", ""),
            result.get("baseline_used", ""),
            result.get("reasoning", ""),
        )
    ).lower()
    needs_dmp = bool(
        re.search(
            r"\b(environment|castle|city|set extension|matte|dmp|background plate|vista|landscape|establishing)\b",
            text,
        )
    )
    creature_or_character = bool(
        re.search(r"\b(creature|dragon|character|digital double|vehicle|prop)\b", text)
    )
    if not needs_dmp or creature_or_character and not re.search(r"\b(environment|castle|city|matte|dmp)\b", text):
        return
    dept = _dept_days(result)
    if dept.get("dmp", 0) > 0:
        return
    days = 2.0 if re.search(r"\b(hero|establishing|castle|city|matte|dmp)\b", text) else 1.0
    _set_dept(result, "dmp", days)


def _enforce_asset_lighting(asset: Any, result: Dict[str, Any]) -> None:
    """Add build-time lighting only when the asset explicitly requires it."""
    text = " ".join(
        str(part or "")
        for part in (
            getattr(asset, "asset_name", ""),
            getattr(asset, "description", ""),
            result.get("baseline_used", ""),
            result.get("reasoning", ""),
        )
    ).lower()
    custom_lighting = re.search(
        r"\b(pre[- ]?baked (?:interior )?lighting|custom lighting(?: rig)?|"
        r"light[- ]emitting|emissive|self[- ]illuminat(?:ed|ing)|"
        r"glowing|bioluminescent)\b",
        text,
    )
    if not custom_lighting:
        return
    dept = _dept_days(result)
    if dept.get("lighting", 0) > 0:
        return
    days = 3.0 if re.search(r"\b(environment|interior|pre[- ]?baked)\b", text) else 2.0
    _set_dept(result, "lighting", days)


def estimate_single_asset(asset: Any, asset_context: Optional[Any], settings: Settings) -> Dict[str, Any]:
    if asset_context is None:
        tier, notes = "", ""
    else:
        tier = getattr(asset_context, "tier", "") or ""
        notes = getattr(asset_context, "notes", "") or ""
    prompt = ASSET_PROMPT.format(
        baselines=_baseline_text(),
        name=asset.asset_name,
        description=asset.description,
        variations=max(1, int(asset.variations or 1)),
        tier=tier or "auto-detect",
        notes=notes or "none",
    )
    print(f"[asset_estimate] Calling Gemini for: {asset.asset_name}", flush=True)
    print(f"[asset_estimate] Prompt length: {len(prompt)} chars", flush=True)
    try:
        result = generate_json(
            prompt,
            settings=settings,
            debug_label="asset_estimate",
            timeout_sec=45,
        )
    except Exception as exc:
        print(f"[asset_estimate] FAILED — generate_json raised {type(exc).__name__}: {exc}", flush=True)
        raise
    if not result:
        print("[asset_estimate] FAILED — generate_json returned None", flush=True)
        print("[asset_estimate] Check: API key valid? Rate limited? Prompt too long?", flush=True)
        return {
            "asset_tier": "secondary",
            "departments": {},
            "total_days": 0,
            "confidence": 0.2,
            "reasoning": "Estimation failed -- no Gemini response",
        }
    print(f"[asset_estimate] SUCCESS — got keys: {list(result.keys())}", flush=True)
    _enforce_asset_dmp(asset, result)
    _enforce_asset_lighting(asset, result)
    dept = _dept_days(result)
    if dept:
        result["total_days"] = round(sum(dept.values()) * 2) / 2
    return result
