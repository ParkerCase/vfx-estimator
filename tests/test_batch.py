"""Tests for bid batch estimate and LVL UP CSV parsing."""

from __future__ import annotations

import base64
import csv
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from vfx_estimator.api.app import create_app
from vfx_estimator.api.bid_batch import (
    ALLOTMENT_EFFICIENCY,
    BidBatchShotItem,
    apply_allotment_per_shot_mandays,
    build_shot_description,
    parse_bid_csv_upload,
    parse_lvlup_bid_csv,
    run_bid_batch_estimate,
)
from vfx_estimator.types import BidPreQual, ShotEstimate, bid_departments_to_internal, internal_departments_to_bid


class TestBidDeptMapping:
    def test_internal_to_bid_uses_lvl_names(self):
        bid = internal_departments_to_bid({"lighting": 7.0, "animation": 3.0, "comp_paint": 2.0})
        assert bid["LGT"] == 7.0
        assert bid["ANIM"] == 3.0
        assert bid["COMP PAINT"] == 2.0

    def test_bid_to_internal(self):
        internal = bid_departments_to_internal({"LGT": 3.0, "COMP": 4.0, "COMP ROTO": 1.0})
        assert internal["lighting"] == 3.0
        assert internal["compositing"] == 4.0
        assert internal["comp_roto"] == 1.0


class TestBuildDescription:
    def test_combines_script_vfx_approach(self):
        shot = BidBatchShotItem(
            script_description="Party Girl staggers",
            vfx_notes="digital distortion",
            vfx_assumptions="practical plate",
        )
        desc = build_shot_description(shot)
        assert "Party Girl" in desc
        assert "VFX:" in desc
        assert "APPROACH:" in desc


class TestAllotmentEfficiency:
    def test_single_shot_unchanged(self):
        est = ShotEstimate(
            description="x",
            per_shot_mandays=10.0,
            total_mandays=10.0,
            cost=7000.0,
        )
        assert apply_allotment_per_shot_mandays(est, 1) == 10.0

    def test_multi_shot_divides_by_n_times_efficiency(self):
        est = ShotEstimate(
            description="x",
            per_shot_mandays=10.0,
            total_mandays=30.0,
            cost=21000.0,
        )
        expected = round((30.0 / (3 * ALLOTMENT_EFFICIENCY)) * 2) / 2
        assert apply_allotment_per_shot_mandays(est, 3) == expected


class TestParseLvlupCsv:
    def test_finds_item_header_and_rows(self):
        csv_text = """Project,ANDROMEDA,,,,,,,,,,,
,,,,,,,,,,,,
ITEM#,VFX SHOT CODE,SCRIPT DESCRIPTIONS,VFX NOTES,VFX ASSUMPTIONS,NUMBER OF SHOTS,CAMERA,MATCHMOVE,TOTAL MANDAYS,COST PER SHOT
1,AND_004_001,Party Girl staggers,Add distortion,Practical plate,1,,,,
2,AND_004_002,Presence looms,Digital creature,Practical elements,2,,,,
"""
        shots, artifact = parse_lvlup_bid_csv(csv_text.encode("utf-8"))
        assert len(shots) == 2
        assert shots[0].shot_code == "AND_004_001"
        assert shots[0].number_of_shots == 1
        assert shots[1].number_of_shots == 2
        assert len(artifact.data_row_indices) == 2

    def test_stops_at_blank_item(self):
        csv_text = """ITEM#,VFX SHOT CODE,SCRIPT DESCRIPTIONS,VFX NOTES,VFX ASSUMPTIONS,NUMBER OF SHOTS
1,AND_001,desc,notes,assump,1
,,,,,
"""
        shots, _ = parse_lvlup_bid_csv(csv_text.encode("utf-8"))
        assert len(shots) == 1


class TestBatchEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(create_app())

    def test_batch_empty_shots_400(self, client):
        r = client.post("/estimate/batch", json={"shots": []})
        assert r.status_code == 400

    @patch("vfx_estimator.api.app.get_service")
    def test_batch_bid_format_response(self, mock_get_service, client):
        svc = MagicMock()
        svc.settings.day_rate = 750
        svc.gemini = None
        svc.estimate.return_value = ShotEstimate(
            description="combined",
            per_shot_mandays=6.0,
            total_mandays=6.0,
            cost=4500.0,
            confidence=0.72,
            dept_days={"lighting": 3.0, "comp_paint": 1.0, "compositing": 2.0},
            reasoning="hero work",
        )
        mock_get_service.return_value = svc

        r = client.post(
            "/estimate/batch",
            json={
                "shots": [
                    {
                        "item_number": "1",
                        "shot_code": "AND_004_001",
                        "script_description": "Party Girl staggers",
                        "vfx_notes": "distortion",
                        "vfx_assumptions": "practical",
                        "number_of_shots": 1,
                    }
                ],
                "day_rate": 750,
                "project": "ANDROMEDA",
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["project"] == "ANDROMEDA"
        row = data["shots"][0]
        assert row["dept_days"]["LGT"] == 3.0
        assert row["dept_days"]["COMP PAINT"] == 1.0
        assert row["dept_days"]["COMP"] == 2.0
        assert "lighting" not in row["dept_days"]
        assert row["total_mandays"] == 6.0
        assert row["cost_per_shot"] == 4500.0

    @patch("vfx_estimator.api.app.get_service")
    def test_batch_csv_returns_base64_export(self, mock_get_service, client):
        svc = MagicMock()
        svc.settings.day_rate = 750
        svc.gemini = None
        svc.estimate.return_value = ShotEstimate(
            description="x",
            per_shot_mandays=4.0,
            total_mandays=4.0,
            cost=3000.0,
            dept_days={"comp_paint": 4.0},
        )
        mock_get_service.return_value = svc

        csv_body = """ITEM#,VFX SHOT CODE,SCRIPT DESCRIPTIONS,VFX NOTES,VFX ASSUMPTIONS,NUMBER OF SHOTS,LGT,TOTAL MANDAYS,COST PER SHOT
1,AND_001,Hero shot,notes,assump,1,,,
"""
        r = client.post(
            "/estimate/batch/csv",
            files={"file": ("andromeda.csv", csv_body, "text/csv")},
            data={"day_rate": "750", "project": "ANDROMEDA"},
        )
        assert r.status_code == 200
        data = r.json()
        assert "export_csv_base64" in data
        assert data["export_csv_filename"] == "ANDROMEDA_AI_ESTIMATE.csv"
        decoded = base64.b64decode(data["export_csv_base64"]).decode("utf-8-sig")
        assert "TOTAL MANDAYS" in decoded
        assert "COMP PAINT" in decoded
        assert "COST PER SHOT" not in decoded.split("\n")[2]  # filled row has no cost column value

    @patch("vfx_estimator.api.app.get_service")
    def test_corrections_batch_translates_bid_depts(self, mock_get_service, client):
        svc = MagicMock()
        svc.corrections.load.return_value = []
        mock_get_service.return_value = svc

        r = client.post(
            "/corrections/batch",
            json={
                "corrections": [
                    {
                        "description": "Hero",
                        "final_total_days": 8,
                        "final_departments": {"LGT": 3, "COMP": 4, "COMP ROTO": 1},
                        "ai_total_days": 6,
                        "user_id": "che",
                    }
                ]
            },
        )
        assert r.status_code == 200
        assert r.json()["saved"] == 1
        call_args = svc.record_correction.call_args[0][0]
        assert call_args.final_departments["lighting"] == 3.0
        assert call_args.final_departments["compositing"] == 4.0
        assert call_args.final_departments["comp_roto"] == 1.0


class TestRunBidBatch:
    @patch("vfx_estimator.api.bid_batch.EstimatorService")
    def test_passes_batch_pre_qual_to_estimate(self, _mock_cls):
        svc = MagicMock()
        svc.settings.day_rate = 750
        svc.gemini = None
        svc.estimate.return_value = ShotEstimate(
            description="d",
            per_shot_mandays=8.0,
            total_mandays=8.0,
            cost=6000.0,
            dept_days={"lighting": 8.0},
        )

        shot = BidBatchShotItem(script_description="castle vista")
        pre_qual = BidPreQual(
            bid_scale_tier="premium_tv",
            complexity_band="high",
            director_brief="Photoreal",
        )
        run_bid_batch_estimate(
            svc,
            [shot],
            project="ANDROMEDA",
            pre_qual=pre_qual,
            compute_ranges=lambda d, c: {},
        )
        pq = svc.estimate.call_args.kwargs["pre_qual"]
        assert pq.bid_scale_tier == "premium_tv"
        assert pq.complexity_band == "high"
        assert pq.director_brief == "Photoreal"

    @patch("vfx_estimator.api.bid_batch.EstimatorService")
    def test_passes_allotment_to_pre_qual(self, _mock_cls):
        svc = MagicMock()
        svc.settings.day_rate = 750
        svc.gemini = None
        svc.estimate.return_value = ShotEstimate(
            description="d",
            per_shot_mandays=8.0,
            total_mandays=16.0,
            cost=12000.0,
            dept_days={"lighting": 8.0},
        )

        shot = BidBatchShotItem(
            script_description="seq",
            number_of_shots=2,
        )
        result = run_bid_batch_estimate(
            svc,
            [shot],
            project="ANDROMEDA",
            day_rate=750,
            compute_ranges=lambda d, c: {},
        )
        pq = svc.estimate.call_args.kwargs["pre_qual"]
        assert pq.allotment_n == 2
        assert result["shots"][0]["number_of_shots"] == 2
