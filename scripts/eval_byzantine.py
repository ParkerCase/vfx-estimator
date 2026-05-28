#!/usr/bin/env python3
"""Evaluate all modes on Byzantine holdout (labeled per-shot mandays)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfx_estimator.config import get_settings
from vfx_estimator.data.loaders import load_byzantine_holdout
from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.metrics import metrics_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shots", type=int, default=0, help="0 = all")
    ap.add_argument("--modes", default="retrieval_median,numeric_only,hybrid,gemini_rag")
    args = ap.parse_args()

    settings = get_settings()
    holdout = load_byzantine_holdout(settings)
    if args.max_shots > 0:
        holdout = holdout[: args.max_shots]

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    svc = EstimatorService(settings)
    acts = [float(r["actual_per_shot_mandays"]) for r in holdout]

    print(f"Byzantine n={len(holdout)}")
    print("=" * 72)
    results = {}
    for mode in modes:
        if mode == "gemini_rag" and not settings.resolved_gemini_key():
            print(f"{mode}: SKIP (no Gemini key)")
            continue
        preds = []
        skipped = 0
        for row in holdout:
            if mode == "retrieval_median":
                p = svc.index.median_mandays(row["description"])
                preds.append(max(0.25, round(p * 2) / 2))
            else:
                try:
                    preds.append(float(svc.estimate(row["description"], mode=mode).per_shot_mandays))
                except RuntimeError as e:
                    skipped += 1
                    preds.append(max(0.25, round(svc.index.median_mandays(row["description"]) * 2) / 2))
                    if skipped == 1:
                        print(f"  warn: {mode} fallback on some shots ({e})")
        if skipped:
            print(f"  ({skipped} shot(s) used retrieval fallback)")
        m = metrics_dict(preds, acts)
        results[mode] = m
        hit50 = m["within_20pct"] >= 50.0
        print(
            f"{mode:18s} ±20%={m['within_20pct']:5.1f}%  MAE={m['mae']:.3f}  "
            f"±10%={m['within_10pct']:5.1f}%  {'✓ >=50%' if hit50 else '  <50%'}"
        )

    out_path = settings.data_dir / "reports" / f"byzantine_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
