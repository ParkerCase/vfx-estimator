"""Google Identity Services token verification."""

from __future__ import annotations

from typing import Dict

from google.auth.transport import requests as grequests
from google.oauth2 import id_token

from vfx_estimator.config import get_settings


def verify_google_token(credential: str) -> Dict[str, str]:
    """Verify a Google ID token and return normalized user info."""
    settings = get_settings()
    client_id = (settings.google_client_id or "").strip()
    if not client_id:
        raise ValueError("GOOGLE_CLIENT_ID is not configured")
    try:
        info = id_token.verify_oauth2_token(
            credential,
            grequests.Request(),
            client_id,
        )
        return {
            "id": str(info["sub"]),
            "email": str(info["email"]),
            "name": str(info.get("name") or ""),
            "picture": str(info.get("picture") or ""),
        }
    except Exception as e:
        raise ValueError(f"Invalid Google token: {e}") from e
