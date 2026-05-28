"""Shared types for estimates and pre-qual."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

BID_DEPT_MAP: Dict[str, str] = {
    "CAMERA": "camera_track",
    "MATCHMOVE": "matchmove",
    "LAYOUT": "layout",
    "ANIM": "animation",
    "CFX": "cfx",
    "FX": "fx",
    "LGT": "lighting",
    "DMP": "dmp",
    "COMP PAINT": "comp_paint",
    "COMP ROTO": "comp_roto",
    "COMP": "compositing",
}

INTERNAL_TO_BID: Dict[str, str] = {v: k for k, v in BID_DEPT_MAP.items()}

BID_OUTPUT_COLUMNS: List[str] = list(BID_DEPT_MAP.keys()) + ["TOTAL MANDAYS"]


def bid_departments_to_internal(dept_days: Dict[str, float]) -> Dict[str, float]:
    """Map bid column names (or internal keys) to internal department keys."""
    out: Dict[str, float] = {}
    for key, val in dept_days.items():
        if val is None:
            continue
        k = str(key).strip()
        ku = k.upper()
        internal = BID_DEPT_MAP.get(ku) or BID_DEPT_MAP.get(k)
        if internal is None and k in INTERNAL_TO_BID:
            internal = k
        if internal is None:
            internal = k
        out[internal] = out.get(internal, 0.0) + float(val)
    return out


def internal_departments_to_bid(dept_days: Dict[str, float]) -> Dict[str, float]:
    """Map internal department keys to bid column names; missing columns default to 0."""
    bid = {col: 0.0 for col in BID_DEPT_MAP}
    for internal, val in dept_days.items():
        if not val:
            continue
        col = INTERNAL_TO_BID.get(internal) or INTERNAL_TO_BID.get(str(internal))
        if col:
            bid[col] = round(float(bid[col]) + float(val), 2)
    return bid


class DeptDays(BaseModel):
    camera_track: float = 0.0
    layout: float = 0.0
    animation: float = 0.0
    fx: float = 0.0
    lighting: float = 0.0
    dmp: float = 0.0
    comp_paint: float = 0.0
    comp_roto: float = 0.0
    matchmove: float = 0.0
    obj_track: float = 0.0
    cfx: float = 0.0
    prep: float = 0.0

    def total(self) -> float:
        return sum(self.model_dump().values())

    def to_dict(self) -> Dict[str, float]:
        return self.model_dump()


class BidPreQual(BaseModel):
    project: Optional[str] = None
    shot_type_override: Optional[str] = None
    complexity_band: Optional[str] = None
    primary_depts: List[str] = Field(default_factory=list)
    bid_scale_tier: Optional[str] = None
    budget_band: Optional[str] = None
    practical_vs_cg: Optional[str] = None
    practical_cg_ratio: Optional[int] = None
    allotment_n: int = 1
    bid_context_notes: Optional[str] = None
    director_brief: Optional[str] = None
    vfx_assumptions: Optional[str] = None
    screenplay_full_text: Optional[str] = None
    screenplay_text_path: Optional[str] = None
    screenplay_fdx_path: Optional[str] = None
    calibration_anchors: List[Dict[str, Any]] = Field(default_factory=list)
    dept_calibration: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("director_brief", mode="before")
    @classmethod
    def _coerce_director_brief(cls, v: Any) -> Optional[str]:
        if v is None or v == "":
            return None
        if isinstance(v, dict):
            parts = [str(x).strip() for x in v.values() if x]
            return "\n".join(parts) if parts else None
        text = str(v).strip()
        return text or None

    def to_legacy_dict(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if not d.get("primary_depts"):
            d.pop("primary_depts", None)
        if self.practical_cg_ratio is not None:
            cg = int(self.practical_cg_ratio)
            d["practical_vs_cg"] = f"{cg}% CG / {100 - cg}% Practical"
        notes: List[str] = []
        if self.shot_type_override:
            notes.append(f"SHOT TYPE OVERRIDE: {self.shot_type_override}.")
        if self.director_brief:
            notes.append(self.director_brief.strip())
        if self.vfx_assumptions:
            notes.append(self.vfx_assumptions.strip())
        if self.bid_context_notes:
            notes.append(self.bid_context_notes.strip())
        if notes:
            d["bid_context_notes"] = "\n\n".join(notes)
        return d


class SimilarShot(BaseModel):
    description: str
    mandays: float
    cost: float
    similarity: float
    source: Literal["training", "correction", "xata"] = "training"
    project: Optional[str] = None


class ShotEstimate(BaseModel):
    description: str
    per_shot_mandays: float
    total_mandays: float
    cost: float
    confidence: float = 0.5
    mode: str = "hybrid"
    reasoning: str = ""
    dept_days: Dict[str, float] = Field(default_factory=dict)
    similar_shots: List[SimilarShot] = Field(default_factory=list)
    screenplay_scene_matches: List[Dict[str, Any]] = Field(default_factory=list)
    numeric_mandays: Optional[float] = None
    gemini_mandays: Optional[float] = None
    retrieval_median_mandays: Optional[float] = None


FLAG_TYPES = [
    "wrong_shot_type",        # AI classified shot type incorrectly
    "wrong_departments",      # Wrong set of depts allocated (e.g. animation on static CG)
    "missing_department",     # A dept was omitted that's required
    "cost_anchor_wrong",      # Similar shots retrieved were wrong price bracket
    "overestimated",          # AI came in too high overall
    "underestimated",         # AI came in too low overall
    "wrong_complexity_band",  # Hero/standard/background misread
    "other",
]


class UserFlag(BaseModel):
    """Supervisor flag: captures systematic estimation errors for RAG prompt injection."""
    description: str
    flag_type: str  # one of FLAG_TYPES
    notes: str = ""
    user_id: str = "default"
    ai_total_days: Optional[float] = None
    ai_shot_type: Optional[str] = None
    ai_departments: Dict[str, float] = Field(default_factory=dict)


class UserCorrection(BaseModel):
    description: str
    final_total_days: float
    final_departments: Dict[str, float] = Field(default_factory=dict)
    user_id: str = "default"
    notes: str = ""
    ai_total_days: Optional[float] = None
