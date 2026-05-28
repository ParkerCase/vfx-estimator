# Copied from apps/breakdown/src/ml/screenplay_scene_match.py (keep in sync for slug/FDX behavior).
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

_SLUG = re.compile(r"^(INT\.|EXT\.|I/E\.|INT\s*/\s*EXT\.)\s+(.+?)\s*$", re.IGNORECASE)
_USER_SCENE_NUM = re.compile(r"(?:^|\s)(?:scene|sc)\s*#?\s*(\d{1,3})(?:\s|$|[^\d])", re.IGNORECASE)
_TRAIL_PAIR = re.compile(r"(\d{1,3})\s+(\d{1,3})\s*$")


def _norm_tokens(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9\s]", " ", (s or "").lower())
    return [t for t in s.split() if len(t) > 1]


def fdx_xml_to_plaintext(fdx_xml: str) -> str:
    raw = (fdx_xml or "").strip()
    if not raw:
        return ""
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return ""
    out: List[str] = []
    for para in root.findall(".//Paragraph"):
        parts = [((t.text or "").strip()) for t in para.findall(".//Text")]
        line = " ".join([p for p in parts if p]).strip()
        if not line:
            line = "".join(para.itertext()).strip()
        if line:
            out.append(line)
    return "\n".join(out)


def _overlap_score(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / (len(sa | sb) or 1)


@dataclass
class ScreenplayScene:
    order_index: int
    heading: str
    body: str
    slug_tokens: Tuple[str, ...]
    trailing_pair: Optional[Tuple[int, int]] = None

    def excerpt(self, max_chars: int = 1200) -> str:
        t = f"{self.heading.strip()}\n{self.body.strip()}".strip()
        return t if len(t) <= max_chars else t[: max_chars - 1] + "…"


def index_screenplay_plaintext(raw: str, *, max_body_per_scene: int = 8000) -> List[ScreenplayScene]:
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    scenes: List[ScreenplayScene] = []
    current_heading: Optional[str] = None
    current_body: List[str] = []
    order = 0

    def flush() -> None:
        nonlocal current_heading, current_body, order
        if not current_heading:
            current_body = []
            return
        order += 1
        body = "\n".join(current_body).strip()
        if len(body) > max_body_per_scene:
            body = body[:max_body_per_scene] + "\n…"
        slug_line = current_heading
        tp: Optional[Tuple[int, int]] = None
        m2 = _TRAIL_PAIR.search(slug_line.strip())
        if m2:
            try:
                tp = (int(m2.group(1)), int(m2.group(2)))
            except ValueError:
                tp = None
            slug_line = slug_line[: m2.start()].strip()
        tok = tuple(_norm_tokens(slug_line + " " + body[:400]))
        scenes.append(
            ScreenplayScene(order_index=order, heading=slug_line, body=body, slug_tokens=tok, trailing_pair=tp)
        )
        current_body = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_heading:
                current_body.append("")
            continue
        if _SLUG.match(stripped):
            flush()
            current_heading = stripped
            continue
        if current_heading is None:
            continue
        current_body.append(line.rstrip())
    flush()
    return scenes


def match_scenes(query: str, scenes: Sequence[ScreenplayScene], *, top_k: int = 3):
    q = (query or "").strip()
    if not q or not scenes:
        return []
    qtok = _norm_tokens(q)
    want = parse_user_scene_number(q)
    ranked = []
    for sc in scenes:
        score = 0.0
        reasons: List[str] = []
        ov = _overlap_score(qtok, sc.slug_tokens)
        if ov > 0:
            score += 0.55 * ov
            reasons.append(f"text_overlap={ov:.2f}")
        if want is not None:
            if sc.order_index == want:
                score += 0.95
                reasons.append(f"order_index=={want}")
            if sc.trailing_pair and sc.trailing_pair[0] == want:
                score += 0.45
                reasons.append(f"slug_lead_num=={want}")
        ranked.append((sc, score, "; ".join(reasons) if reasons else "weak"))
    ranked.sort(key=lambda x: -x[1])
    out = ranked[:top_k]
    if out and out[0][1] <= 0 and len(scenes) <= 5:
        return [(scenes[0], 0.01, "fallback_first_scene")]
    return [t for t in out if t[1] > 0.001] or ranked[:1]


def parse_user_scene_number(query: str) -> Optional[int]:
    m = _USER_SCENE_NUM.search(query or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def screenplay_augment_with_metadata(
    shot_description: str,
    screenplay_text: str,
    *,
    augment_top_k: int = 1,
    return_matches_k: int = 3,
    max_augment_chars: int = 1000,
    max_excerpt_chars: int = 1200,
) -> Tuple[str, List[Dict[str, Any]]]:
    q = (shot_description or "").strip()
    scenes = index_screenplay_plaintext(screenplay_text)
    k = max(int(augment_top_k), int(return_matches_k), 1)
    hits = match_scenes(q, scenes, top_k=k)
    meta: List[Dict[str, Any]] = []
    for sc, scv, why in hits[: max(1, int(return_matches_k))]:
        meta.append(
            {
                "order_index": sc.order_index,
                "heading": sc.heading.strip(),
                "score": round(float(scv), 4),
                "match_reason": why,
                "excerpt": sc.excerpt(max_chars=max_excerpt_chars),
            }
        )
    if not hits:
        return q, []
    parts = [q]
    for sc, scv, why in hits[: max(1, int(augment_top_k))]:
        ex = sc.excerpt(max_chars=max_augment_chars)
        parts.append(f"[screenplay_scene order={sc.order_index} score={scv:.2f} {why}]\n{ex}")
    return "\n\n".join(parts), meta
