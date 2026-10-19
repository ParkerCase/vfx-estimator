"""Rank methodology suggestions so the best-fit option is first."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "into", "onto",
    "shot", "vfx", "using", "used", "over", "under", "then", "than",
    "are", "was", "were", "been", "being", "have", "has", "had",
    "not", "but", "its", "via", "per", "all", "any", "only",
}

_DEMANDING = (
    "hero",
    "complex",
    "intricate",
    "difficult",
    "challenging",
    "destruction",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HERO_NAME_RE = re.compile(r"\b(?:hero|complex)\b")
_COMPLEX_METHOD_RE = re.compile(
    r"\bhero\b|\bcomplex\b|full\s*3d|full\s*cg|simulation|destruction|creature"
)
_SIMPLE_METHOD_RE = re.compile(
    r"\b2d\b|comp(?:osite)?[\s-]*only|paint|roto|cleanup|simple|plate-based"
)


def _tokens(text: str) -> set:
    words = _TOKEN_RE.findall((text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def _shot_is_demanding(description: str) -> bool:
    text = (description or "").lower()
    if any(word in text for word in _DEMANDING):
        return True
    return len(text) > 280


def _shot_band(description: str) -> str:
    text = description or ""
    if _shot_is_demanding(text):
        return "complex"
    if len(text.split()) <= 14:
        return "simple"
    return "standard"


def _method_band(label: str, methodology: str) -> str:
    blob = f"{label} {methodology}".lower()
    if _COMPLEX_METHOD_RE.search(blob):
        return "complex"
    if _SIMPLE_METHOD_RE.search(blob):
        return "simple"
    return "standard"


_MATCH = {
    ("simple", "simple"): 0.40,
    ("simple", "standard"): 0.18,
    ("simple", "complex"): -0.28,
    ("standard", "simple"): 0.08,
    ("standard", "standard"): 0.28,
    ("standard", "complex"): -0.08,
    ("complex", "simple"): -0.12,
    ("complex", "standard"): 0.10,
    ("complex", "complex"): 0.40,
}

_BAND_RANK = {"simple": 0, "standard": 1, "complex": 2}


def score_methodology_suggestion(description: str, suggestion: Dict[str, Any]) -> float:
    """Return a 0–1 fit score. Higher means a better match for this shot."""
    label = str(suggestion.get("label") or "")
    methodology = str(suggestion.get("methodology") or "")
    shot_tokens = _tokens(description)
    method_tokens = _tokens(f"{label} {methodology}")
    overlap = (
        len(shot_tokens & method_tokens) / len(shot_tokens)
        if shot_tokens and method_tokens
        else 0.0
    )
    match = _MATCH[(_shot_band(description), _method_band(label, methodology))]
    penalty = 0.0
    if _HERO_NAME_RE.search(label.lower()) and not _shot_is_demanding(description):
        penalty = 0.34
    raw = (0.50 * overlap) + (0.55 * (0.5 + match)) - penalty
    return round(max(0.0, min(1.0, raw)), 4)


def rank_methodology_suggestions(
    description: str,
    suggestions: List[Any],
) -> List[Dict[str, Any]]:
    """Score and sort suggestions. Only the top result is marked best match."""
    ranked: List[Dict[str, Any]] = []
    for item in suggestions or []:
        if not isinstance(item, dict):
            continue
        copy = dict(item)
        copy["score"] = score_methodology_suggestion(description, copy)
        ranked.append(copy)
    ranked.sort(
        key=lambda item: (
            -float(item.get("score") or 0),
            _BAND_RANK[_method_band(str(item.get("label") or ""), str(item.get("methodology") or ""))],
        )
    )
    for index, item in enumerate(ranked):
        item["is_best_match"] = index == 0
    return ranked
