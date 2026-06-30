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

_JSON_RE = re.compile(r"\{[\s\S]*\}", re.DOTALL)


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of JSON truncated mid-response.

    Strips trailing partial strings/keys, then closes any unclosed
    braces/brackets so json.loads has a chance of succeeding.
    """
    text = text.rstrip()
    if not text:
        return text
    # Strip trailing partial string value: ..."key": "text cut off here
    text = re.sub(r'"[^"]*$', '""', text)
    text = text.rstrip().rstrip(",")
    # Strip trailing partial key: ..., "partial_ke
    text = re.sub(r',?\s*"[^"]*$', "", text).rstrip()
    # Close unclosed arrays
    for _ in range(max(0, text.count("[") - text.count("]"))):
        text += "]"
    # Close unclosed objects
    for _ in range(max(0, text.count("{") - text.count("}"))):
        text += "}"
    return text


def _extract_json(text: str) -> str:
    """Pull the first JSON object from a response that may contain prose."""
    text = text.strip()
    if not text:
        return ""
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # Fast path: starts with {
    if text.startswith("{"):
        last = text.rfind("}")
        if last != -1:
            return text[: last + 1]
        # No closing brace — truncated; attempt repair
        return _repair_truncated_json(text)
    # Slower path: find embedded JSON object
    m = _JSON_RE.search(text)
    return m.group(0) if m else ""


def generate_json(
    prompt: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    settings: Optional[Settings] = None,
    max_output_tokens: int = 4096,
    temperature: float = 0.2,
    max_retries: int = 2,
    debug_label: Optional[str] = None,
    timeout_sec: float = 15,
) -> Optional[Dict[str, Any]]:
    """Call Gemini and return a parsed JSON dict.

    Retries up to max_retries times on empty/malformed responses.
    Does NOT use responseMimeType=application/json — that causes gemini-2.5-flash
    to return an empty body when it can't guarantee clean JSON output.
    We extract and optionally repair JSON from the text response instead.
    """
    settings = settings or get_settings()
    key = (api_key or settings.resolved_gemini_key()).strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY required")
    model = (model or settings.resolved_gemini_mandays_model()).strip()
    url = API_URL.format(model=model) + f"?key={key}"
    if debug_label:
        print(
            f"[{debug_label}] Gemini model={model} api=v1beta "
            f"prompt_len={len(prompt)} max_output_tokens={max_output_tokens} timeout={timeout_sec}s",
            flush=True,
        )

    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
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
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except TimeoutError:
            if debug_label:
                print(f"[{debug_label}] Gemini request timed out", flush=True)
            return None
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini HTTP {e.code}: {err_body[:400]}")
            if debug_label:
                print(f"[{debug_label}] Gemini HTTP {e.code}: {err_body[:500]}", flush=True)
            continue
        except urllib.error.URLError as e:
            if isinstance(getattr(e, "reason", None), TimeoutError):
                if debug_label:
                    print(f"[{debug_label}] Gemini URL timeout", flush=True)
                return None
            last_error = e
            if debug_label:
                print(f"[{debug_label}] Gemini URL error: {e}", flush=True)
            continue
        except Exception as e:
            last_error = e
            if debug_label:
                print(f"[{debug_label}] Gemini request error: {e}", flush=True)
            continue

        # Collect text from all candidate parts
        text = ""
        finish_reason = ""
        for cand in payload.get("candidates") or []:
            finish_reason = cand.get("finishReason", "")
            content = cand.get("content") or {}
            for part in content.get("parts") or []:
                if "text" in part:
                    text += part["text"]
        if debug_label:
            print(f"[{debug_label}] RAW Gemini response: {text[:500]!r}", flush=True)

        raw = _extract_json(text)
        if not raw:
            last_error = RuntimeError(
                f"Empty/unparseable response (finish={finish_reason!r}). "
                f"Text ({len(text)} chars): {text[:200]!r}"
            )
            if debug_label:
                print(f"[{debug_label}] JSON extraction failed: {last_error}", flush=True)
            continue

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try repairing before giving up on this attempt
            repaired = _repair_truncated_json(raw)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError as e2:
                last_error = RuntimeError(
                    f"JSON parse failed after repair: {e2} | raw={raw[:200]!r}"
                )
                if debug_label:
                    print(f"[{debug_label}] JSON parse failed: {last_error}", flush=True)
                continue

    raise last_error
