"""Persist human-in-the-loop corrections for retrieval boost."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.types import UserCorrection

logger = logging.getLogger(__name__)


class CorrectionsStore:
    def __init__(self, path: Optional[Path] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        default_path = self.settings.corrections_path()
        self._path = path or default_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._xata_store = None
        use_xata = bool(self.settings.resolved_xata_postgres_url()) and (
            path is None or self._path.resolve() == default_path.resolve()
        )
        if use_xata:
            try:
                from vfx_estimator.integrations.xata import XataCorrectionsStore

                self._xata_store = XataCorrectionsStore(settings=self.settings)
            except Exception as e:
                logger.warning("Xata corrections unavailable, using local JSONL: %s", e)

    @property
    def path(self) -> Path:
        return self._path

    @path.setter
    def path(self, value: Path) -> None:
        self._path = value
        default = self.settings.corrections_path().resolve()
        if value.resolve() != default:
            self._xata_store = None

    @property
    def storage_backend(self) -> str:
        return "postgres" if self._xata_store else "local_jsonl"

    def load(self) -> List[UserCorrection]:
        if self._xata_store:
            return self._xata_store.load()
        if not self._path.exists():
            return []
        out: List[UserCorrection] = []
        with open(self._path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(UserCorrection.model_validate(json.loads(line)))
        return out

    def append(self, correction: UserCorrection) -> None:
        if self._xata_store:
            self._xata_store.append(correction)
        row = correction.model_dump()
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def as_training_rows(self) -> List[dict]:
        rows = []
        for c in self.load():
            rows.append(
                {
                    "description": c.description,
                    "mandays": c.final_total_days,
                    "cost": c.final_total_days * self.settings.day_rate,
                    "source": "correction",
                    "dept_days": c.final_departments,
                    "user_id": c.user_id,
                }
            )
        return rows
