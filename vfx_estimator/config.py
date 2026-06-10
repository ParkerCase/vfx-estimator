"""Environment and tunable settings."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from dotenv import load_dotenv
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PKG_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _PKG_ROOT.parent


def _load_env() -> None:
    load_dotenv(_REPO_ROOT / ".env")
    load_dotenv(_REPO_ROOT / "apps" / "breakdown" / ".env")
    load_dotenv(_PKG_ROOT / ".env", override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    legacy_breakdown_root: Path = Field(
        default=_REPO_ROOT / "apps" / "breakdown",
        validation_alias="VFX_LEGACY_BREAKDOWN_ROOT",
    )
    data_dir: Path = Field(default=_PKG_ROOT / "data", validation_alias="VFX_ESTIMATOR_DATA_DIR")
    training_json: Optional[Path] = Field(default=None, validation_alias="VFX_TRAINING_JSON")
    shot_training_jsonl: Optional[Path] = Field(default=None, validation_alias="VFX_SHOT_TRAINING_JSONL")
    byzantine_csv: Optional[Path] = Field(default=None, validation_alias="VFX_BYZANTINE_CSV")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", validation_alias="GEMINI_MODEL")
    gemini_mandays_model: str = Field(default="", validation_alias="GEMINI_MANDAYS_MODEL")

    xata_api_key: str = Field(default="", validation_alias="XATA_API_KEY")
    xata_postgres_url: str = Field(default="", validation_alias="XATA_POSTGRES_URL")
    xata_database_url: str = Field(
        default="",
        validation_alias=AliasChoices("XATA_DATABASE_URL", "DATABASE_URL"),
    )
    xata_branch: str = Field(default="main", validation_alias="XATA_BRANCH")
    xata_table: str = Field(default="vfx_historical_shots", validation_alias="XATA_TABLE")
    xata_corrections_table: str = Field(default="vfx_corrections", validation_alias="XATA_CORRECTIONS_TABLE")

    estimate_mode: Literal["numeric_only", "gemini_rag", "hybrid"] = Field(
        default="hybrid", validation_alias="VFX_ESTIMATE_MODE"
    )
    blend_numeric_weight: float = Field(default=0.55, validation_alias="VFX_BLEND_NUMERIC_WEIGHT")
    blend_gemini_weight: float = Field(default=0.45, validation_alias="VFX_BLEND_GEMINI_WEIGHT")
    correction_boost: float = Field(default=2.0, validation_alias="VFX_CORRECTION_BOOST")
    retrieval_top_k: int = Field(default=10, validation_alias="VFX_RETRIEVAL_TOP_K")
    day_rate: float = Field(default=700.0, validation_alias="VFX_DAY_RATE")
    use_legacy_numeric: bool = Field(default=True, validation_alias="VFX_USE_LEGACY_NUMERIC")

    api_host: str = Field(default="127.0.0.1", validation_alias="VFX_API_HOST")
    api_port: int = Field(default=8090, validation_alias="VFX_API_PORT")

    def resolved_gemini_key(self) -> str:
        return (self.gemini_api_key or self.google_api_key or "").strip()

    def resolved_gemini_mandays_model(self) -> str:
        return (self.gemini_mandays_model or self.gemini_model).strip()

    def resolved_xata_postgres_url(self) -> str:
        for candidate in (self.xata_postgres_url, self.xata_database_url):
            url = (candidate or "").strip()
            if url.startswith("postgresql://") or url.startswith("postgres://"):
                return url
        return ""

    def resolved_xata_rest_base(self) -> str:
        from vfx_estimator.integrations.xata import resolve_xata_rest_base

        url = (self.xata_database_url or "").strip()
        if not url:
            return ""
        return resolve_xata_rest_base(url, self.xata_branch)

    def resolved_training_json(self) -> Path:
        if self.training_json and self.training_json.exists():
            return self.training_json
        for candidate in (
            self.data_dir / "training" / "retraining_bundle.json",
            self.data_dir / "retraining_bundle.json",
            self.data_dir / "xata_full_export.json",
        ):
            if candidate.is_file():
                return candidate
        legacy = self.legacy_breakdown_root / "data" / "processed" / "retraining_bundle.json"
        if legacy.is_file():
            return legacy
        return self.legacy_breakdown_root / "data" / "xata_full_export.json"

    def resolved_byzantine_csv(self) -> Path:
        if self.byzantine_csv and self.byzantine_csv.exists():
            return self.byzantine_csv
        for name in ("byzantine_actual_expanded.csv", "byzantine_actual.csv"):
            p = self.legacy_breakdown_root / "data" / name
            if p.exists():
                return p
        raise FileNotFoundError("Byzantine CSV not found; set VFX_BYZANTINE_CSV")

    def corrections_path(self) -> Path:
        return self.data_dir / "corrections.jsonl"

    def tuning_path(self) -> Path:
        return self.data_dir / "tuning.json"

    def resolved_dept_rates(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        from vfx_estimator.rates import build_dept_rates

        tuning = self.load_tuning_overrides()
        return build_dept_rates(
            fallback=self.day_rate,
            tuning=tuning.get("dept_rates"),
            overrides=overrides,
        )

    def load_tuning_overrides(self) -> Dict[str, Any]:
        path = self.tuning_path()
        if not path.exists():
            defaults = _PKG_ROOT / "data" / "tuning.defaults.json"
            if defaults.exists():
                with open(defaults, encoding="utf-8") as f:
                    return json.load(f)
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def apply_tuning(self) -> None:
        t = self.load_tuning_overrides()
        if not t:
            return
        for key, val in t.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)


@lru_cache
def get_settings() -> Settings:
    _load_env()
    s = Settings()
    s.apply_tuning()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    (s.data_dir / "reports").mkdir(parents=True, exist_ok=True)
    return s
