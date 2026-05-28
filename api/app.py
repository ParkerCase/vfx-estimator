"""Vercel / ASGI entrypoint — exposes top-level ``app`` for the Python runtime."""

from vfx_estimator.api.app import create_app

app = create_app()
