"""Tests for bid history API endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from vfx_estimator.api.app import create_app


@pytest.fixture
def client():
    return TestClient(create_app())


class TestBidHistoryEndpoints:
    @patch("vfx_estimator.api.app.save_bid_history")
    def test_save_bid(self, mock_save, client):
        mock_save.return_value = 42
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.post(
                "/bid-history",
                json={
                    "project_name": "ANDROMEDA",
                    "user_id": "supervisor",
                    "shots": [
                        {
                            "shot_code": "AND_001",
                            "description": "Hero shot",
                            "total_mandays": 6.0,
                            "dept_days": {"LGT": 3.0, "COMP": 3.0},
                        }
                    ],
                    "pre_qual": {"project": "ANDROMEDA"},
                },
            )
            assert r.status_code == 200
            data = r.json()
            assert data["ok"] is True
            assert data["bid_id"] == 42
            mock_save.assert_called_once()
        finally:
            svc_patch.stop()

    @patch("vfx_estimator.api.app.list_bid_history")
    def test_list_bids(self, mock_list, client):
        mock_list.return_value = [
            {
                "id": 1,
                "project_name": "IRON VEIL",
                "user_id": "supervisor",
                "shot_count": 12,
                "total_mandays": 88.5,
                "created_at": datetime(2026, 6, 19, tzinfo=timezone.utc),
                "notes": "",
            }
        ]
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.get("/bid-history?limit=10")
            assert r.status_code == 200
            data = r.json()
            assert data["count"] == 1
            assert data["bids"][0]["project_name"] == "IRON VEIL"
            assert "2026-06-19" in data["bids"][0]["created_at"]
        finally:
            svc_patch.stop()

    @patch("vfx_estimator.api.app.get_bid_history")
    def test_get_bid(self, mock_get, client):
        mock_get.return_value = {
            "id": 7,
            "project_name": "ANDROMEDA",
            "user_id": "supervisor",
            "shot_count": 2,
            "total_mandays": 10.0,
            "shots": [{"shot_code": "AND_001", "total_mandays": 6.0}],
            "pre_qual": {},
            "created_at": datetime(2026, 6, 19, tzinfo=timezone.utc),
            "notes": "",
        }
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.get("/bid-history/7")
            assert r.status_code == 200
            assert r.json()["project_name"] == "ANDROMEDA"
            assert len(r.json()["shots"]) == 1
        finally:
            svc_patch.stop()

    @patch("vfx_estimator.api.app.get_bid_history")
    def test_get_bid_not_found(self, mock_get, client):
        mock_get.return_value = None
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.get("/bid-history/999")
            assert r.status_code == 404
        finally:
            svc_patch.stop()

    @patch("vfx_estimator.api.app.delete_bid_history")
    def test_delete_bid(self, mock_delete, client):
        mock_delete.return_value = True
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.delete("/bid-history/3")
            assert r.status_code == 200
            assert r.json()["deleted"] == 3
        finally:
            svc_patch.stop()

    def test_save_bid_requires_postgres(self, client):
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = ""

        try:
            r = client.post(
                "/bid-history",
                json={"project_name": "X", "shots": [{"total_mandays": 1}]},
            )
            assert r.status_code == 503
        finally:
            svc_patch.stop()

    def test_save_bid_requires_shots(self, client):
        svc_patch = patch("vfx_estimator.api.app.get_service")
        mock_svc = svc_patch.start()
        mock_svc.return_value.settings.resolved_xata_postgres_url.return_value = "postgresql://test"

        try:
            r = client.post("/bid-history", json={"project_name": "X", "shots": []})
            assert r.status_code == 400
        finally:
            svc_patch.stop()
