"""Minimal Gemini REST client (stdlib-friendly, no google-generativeai dep)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from vfx_estimator.config import Settings, get_settings

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _find_json_object(text: str) -> str:
    """Find the first complete, balanced JSON object via brace-depth tracking.

    More reliable than rfind('}') which breaks on nested objects.
    """
    in_string = False
    escaped = False
    depth = 0
    start: Optional[int] = None

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start : i + 1]

    return ""


def _repair_truncated_json(text: str) -> str:
    """Close unclosed braces in a response truncated mid-generation."""
    in_string = False
    escaped = False
    last_safe = 0

    for i, ch in enumerate(text):
        if escaped:
            escaped = False
            continue
        if ch == "\\" and in_string:
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string and ch in ",{":
            last_safe = i

    if in_string:
        text = text[:last_safe].rstrip().rstrip(",")

    opens = text.count("{") - text.count("}")
    text += "}" * max(0, opens)
    return text


def _extract_json(text: str) -> str:
    """Pull the first complete JSON object from a Gemini response."""
    text = text.strip()
    if not text:
        return ""

    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # Try to find a complete balanced JSON object
    candidate = _find_json_object(text)
    if candidate:
        return candidate

    # No complete object — attempt repair on whatever we have
    start = text.find("{")
    if start == -1:
        return ""
    return _repair_truncated_json(text[start:])


def _needs_thinking_disabled(model_id: str) -> bool:
    """Gemini 2.5+ uses thinking tokens that eat into the output budget.

    For structured JSON tasks we disable thinking so the model uses its
    full token budget for the actual response instead of internal reasoning.
    """
    return "2.5" in model_id or "flash-thinking" in model_id.lower()


def generate_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
    max_output_tokens: int = 8192,
    temperature: float = 0.2,
    max_retries: int = 2,
    disable_thinking: Optional[bool] = None,
) -> Dict[str, Any]:
    """Call Gemini and return a parsed JSON dict.

    Key design decisions:
    - No responseMimeType=application/json (causes empty bodies on complex prompts)
    - thinkingBudget=0 for Gemini 2.5 models (thinking tokens consume output budget)
    - max_output_tokens=8192 (2.5-flash with thinking disabled needs room)
    - Retries with back-off on empty/malformed responses
    """
    settings = settings or get_settings()
    key = (api_key or settings.resolved_gemini_key()).strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY required")
    model_id = (model or settings.resolved_gemini_mandays_model()).strip()
    url = API_URL.format(model=model_id) + f"?key={key}"

    gen_config: Dict[str, Any] = {
        "temperature": temperature,
        "maxOutputTokens": max_output_tokens,
    }

    # Disable thinking for Gemini 2.5 — thinking tokens silently consume the
    # output budget, causing truncation mid-JSON (e.g. stops at "total_days": )
    should_disable = disable_thinking if disable_thinking is not None else _needs_thinking_disabled(model_id)
    if should_disable:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }

    last_error: Exception = RuntimeError("No attempts made")

    for attempt in range(max_retries + 1):
        if attempt > 0:
            time.sleep(1.5 * attempt)

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini HTTP {e.code}: {err_body[:400]}")
            continue
        except Exception as e:
            last_error = e
            continue

        # Collect text across all candidate parts
        text = ""
        finish_reason = ""
        for cand in payload.get("candidates") or []:
            finish_reason = cand.get("finishReason", "")
            for part in (cand.get("content") or {}).get("parts") or []:
                if "text" in part:
                    text += part["text"]

        if not text.strip():
            last_error = RuntimeError(
                f"Empty Gemini response (finish={finish_reason!r}). "
                "Likely safety filter, rate limit, or thinkingConfig not supported by this model."
            )
            continue

        raw = _extract_json(text)
        if not raw:
            last_error = RuntimeError(
                f"No JSON found in response ({len(text)} chars): {text[:300]!r}"
            )
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = RuntimeError(
                f"JSON parse failed: {e}\n"
                f"  raw ({len(raw)} chars): {raw[:300]!r}\n"
                f"  text ({len(text)} chars): {text[:200]!r}"
            )
            continue

    raise last_error
