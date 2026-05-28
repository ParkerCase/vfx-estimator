"""Persist supervisor flags (systematic error reports) for RAG prompt injection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.types import UserFlag


class FlagsStore:
    def __init__(self, path: Optional[Path] = None, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.path = path or (self.settings.data_dir / "flags.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> List[UserFlag]:
        if not self.path.exists():
            return []
        out: List[UserFlag] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(UserFlag.model_validate(json.loads(line)))
                except Exception:
                    pass  # skip malformed lines
        return out

    def append(self, flag: UserFlag) -> None:
        row = flag.model_dump()
        row["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def prompt_context(self, description: str, max_flags: int = 5) -> str:
        """Return a formatted block of relevant flags for injection into the Gemini prompt.

        Finds flags whose description shares keywords with the current shot,
        plus the most recent global flags, to warn the model about known pitfalls.
        """
        all_flags = self.load()
        if not all_flags:
            return ""

        desc_tokens = set(description.lower().split())

        def relevance(f: UserFlag) -> float:
            flag_tokens = set(f.description.lower().split())
            overlap = len(desc_tokens & flag_tokens) / max(len(desc_tokens | flag_tokens), 1)
            return overlap

        # Score by relevance; fall back to recency (last N) for low-overlap flags
        scored = sorted(all_flags, key=lambda f: -relevance(f))
        top = scored[:max_flags]

        lines = ["SUPERVISOR FLAGS — known estimation pitfalls to avoid:\n"]
        for f in top:
            excerpt = f.description[:80] + ("…" if len(f.description) > 80 else "")
            flag_label = f.flag_type.replace("_", " ").upper()
            lines.append(f'  [{flag_label}] "{excerpt}"')
            if f.notes:
                lines.append(f"    → Supervisor note: {f.notes}")
            if f.ai_shot_type:
                lines.append(f"    → AI classified as: {f.ai_shot_type} (was wrong)")
        lines.append("")
        return "\n".join(lines)

    def stats(self) -> dict:
        flags = self.load()
        if not flags:
            return {"count": 0, "by_type": {}}
        by_type: dict = {}
        for f in flags:
            by_type[f.flag_type] = by_type.get(f.flag_type, 0) + 1
        return {
            "count": len(flags),
            "by_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        }
