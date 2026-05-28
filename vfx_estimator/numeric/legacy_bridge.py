"""Bridge to vendored or external generalized_mandays_pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from vfx_estimator.config import Settings, get_settings
from vfx_estimator.types import BidPreQual

_NUMERIC_PKG = Path(__file__).resolve().parent
_BUNDLED = _NUMERIC_PKG / "bundled"


def resolve_legacy_roots(settings: Optional[Settings] = None) -> Optional[Tuple[Path, Path]]:
    """Return (scripts_dir, src_dir) if a legacy pipeline is available."""
    settings = settings or get_settings()
    candidates = [(_BUNDLED / "scripts", _BUNDLED / "src")]
    root = Path(settings.legacy_breakdown_root)
    if str(root).strip() and str(root) != ".":
        candidates.append((root / "scripts", root / "src"))
    for scripts, src in candidates:
        if (scripts / "generalized_mandays_pipeline.py").is_file():
            return scripts, src
    return None


class LegacyNumericEstimator:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._loaded = False
        self._predict = None
        self._index = None
        self._using_fallback = False

    def set_retrieval_index(self, index: Any) -> None:
        """Attach retrieval index for fallback predictor when pipeline is not vendored."""
        self._index = index
        if self._using_fallback:
            self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        roots = resolve_legacy_roots(self.settings)
        if roots is not None:
            scripts, src = roots
            for p in (str(scripts), str(src)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import os

            os.environ.setdefault("BREAKDOWN_TRAINING_JSON", str(self.settings.resolved_training_json()))
            from generalized_mandays_pipeline import get_cached_training, predict_with_prequal  # noqa: WPS433

            get_cached_training()
            self._predict = predict_with_prequal
            self._using_fallback = False
            self._loaded = True
            return

        if self._index is not None:
            from vfx_estimator.numeric.fallback_numeric import make_predictor

            self._predict = make_predictor(self._index)
            self._using_fallback = True
            self._loaded = True
            return

        raise FileNotFoundError(
            "No numeric pipeline found. Either:\n"
            "  1) Run: python -m scripts.vendor_legacy_pipeline --source /path/to/apps/breakdown\n"
            "  2) Set VFX_LEGACY_BREAKDOWN_ROOT to a valid apps/breakdown checkout\n"
            f"Checked bundled: {_BUNDLED}\n"
            f"Checked env root: {self.settings.legacy_breakdown_root}"
        )

    def predict(self, description: str, pre_qual: Optional[BidPreQual] = None) -> Dict[str, Any]:
        if not self.settings.use_legacy_numeric:
            return {"per_shot_mandays": 0.0, "total_mandays": 0.0, "cost": 0.0}
        self._ensure_loaded()
        pq = pre_qual.to_legacy_dict() if pre_qual else None
        return self._predict(description.strip(), pre_qual=pq, day_rate=self.settings.day_rate)

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback
