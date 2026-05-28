#!/usr/bin/env python3
"""Run FastAPI server."""

from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfx_estimator.api.app import create_app
from vfx_estimator.config import get_settings


def main() -> None:
    s = get_settings()
    uvicorn.run(create_app(), host=s.api_host, port=s.api_port, reload=False)


if __name__ == "__main__":
    main()
