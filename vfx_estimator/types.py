"""Shared types for estimates and pre-qual."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


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
    complexity_band: Optional[str] = None
    primary_depts: List[str] = Field(default_factory=list)
    bid_scale_tier: Optional[str] = None
    budget_band: Optional[str] = None
    practical_vs_cg: Optional[str] = None
    allotment_n: int = 1
    bid_context_notes: Optional[str] = None
    director_brief: Dict[str, str] = Field(default_factory=dict)
    screenplay_full_text: Optional[str] = None
    screenplay_text_path: Optional[str] = None
    screenplay_fdx_path: Optional[str] = None
    calibration_anchors: List[Dict[str, Any]] = Field(default_factory=list)
    dept_calibration: Dict[str, Any] = Field(default_factory=dict)

    def to_legacy_dict(self) -> Dict[str, Any]:
        d = self.model_dump(exclude_none=True)
        if not d.get("primary_depts"):
            d.pop("primary_depts", None)
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
