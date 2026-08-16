from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from scripts.migrate_xata_corrections import _load_reference_presets
from vfx_estimator.llm.mandays_rag import build_vfx_rules
from vfx_estimator.retrieval.index import load_preset_training_rows


def test_reference_preset_seed_has_all_valid_workbook_rows():
    presets = _load_reference_presets()

    assert len(presets) == 490
    assert len({preset["shot_type"] for preset in presets}) == 490
    assert all(float(preset["total"]) > 0 for preset in presets)
    assert presets[0]["shot_type"] == "cleanup_low_sh0001"
    assert presets[-1]["shot_type"] == "destruction_hero_sh0500"


def test_prompt_groups_reference_presets_by_full_category():
    rules = build_vfx_rules(
        {
            "digital_human_hero_sh0001": {
                "description": "Hero digital human",
                "animation": 5,
                "compositing": 4,
                "total": 9,
            }
        }
    )

    assert "DIGITAL HUMAN:" in rules
    assert "digital_human_hero_sh0001" in rules
    assert "animation=5d" in rules


def test_presets_are_retrieval_rows_with_training_weight():
    settings = SimpleNamespace(
        resolved_xata_postgres_url=lambda: "postgresql://example",
        day_rate=700,
    )
    db_presets = {
        "destruction_hero_sh0040": {
            "description": "Building collapse with debris",
            "fx": 18,
            "lighting": 6,
            "compositing": 10,
            "total": 55,
        }
    }

    with patch("vfx_estimator.integrations.xata.load_presets", return_value=db_presets):
        rows = load_preset_training_rows(settings)

    assert rows == [
        {
            "description": "Building collapse with debris",
            "mandays": 55.0,
            "cost": 38500.0,
            "source": "preset",
            "weight": 1.0,
            "project": "destruction_hero_sh0040",
            "dept_days": {"FX": 18.0, "LGT": 6.0, "COMP": 10.0},
        }
    ]
