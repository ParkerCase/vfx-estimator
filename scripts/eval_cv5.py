#!/usr/bin/env python3
"""
5-repeat random holdout CV on training shots (rebuilds retrieval index per fold).

Compares: retrieval_median | numeric_only | gemini_rag | hybrid

Usage:
  cd vfx-estimator
  python -m scripts.eval_cv5 --repeats 5 --max-test 200
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vfx_estimator.config import get_settings
from vfx_estimator.data.loaders import load_training_shots
from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.metrics import metrics_dict
from vfx_estimator.retrieval.index import ShotRetrievalIndex


def run_fold(train_desc, test_rows, mode: str, settings, train_shots):
    index = ShotRetrievalIndex(train_shots, settings=settings)
    svc = EstimatorService(settings)
    svc.training = train_shots
    svc.index = index
    if svc.gemini:
        svc.gemini.index = index

    preds, acts = [], []
    for row in test_rows:
        acts.append(float(row["mandays"]))
        if mode == "retrieval_median":
            p = index.median_mandays(row["description"])
            preds.append(max(0.25, round(p * 2) / 2))
        else:
            est = svc.estimate(row["description"], mode=mode)
            preds.append(float(est.per_shot_mandays))
    return metrics_dict(preds, acts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed-base", type=int, default=42)
    ap.add_argument("--max-test", type=int, default=250)
    ap.add_argument("--modes", default="retrieval_median,numeric_only,hybrid,gemini_rag")
    args = ap.parse_args()

    settings = get_settings()
    shots = load_training_shots(settings)
    rows = [{"description": s.description, "mandays": s.mandays} for s in shots]
    n = len(rows)
    n_test = max(50, int(n * args.test_frac))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    print(f"Training shots: {n} | holdout ~{n_test} | modes: {modes}")
    print("=" * 72)

    summary = {m: [] for m in modes}
    for r in range(args.repeats):
        rng = np.random.RandomState(args.seed_base + r)
        perm = rng.permutation(n)
        te = set(int(i) for i in perm[:n_test])
        train_shots = [shots[i] for i in range(n) if i not in te]
        test_rows = [rows[i] for i in range(n) if i in te]
        if len(test_rows) > args.max_test:
            pick = rng.choice(len(test_rows), size=args.max_test, replace=False)
            test_rows = [test_rows[int(i)] for i in pick]

        print(f"\nFold {r+1}/{args.repeats} (train={len(train_shots)} test={len(test_rows)})")
        for mode in modes:
            if mode == "gemini_rag" and not settings.resolved_gemini_key():
                print(f"  {mode:18s} SKIP (no Gemini key)")
                continue
            m = run_fold(train_shots, test_rows, mode, settings, train_shots)
            summary[mode].append(m)
            print(
                f"  {mode:18s} ±20%={m['within_20pct']:5.1f}%  MAE={m['mae']:.3f}  "
                f"±10%={m['within_10pct']:5.1f}%  ≤1d={m['within_1day']:5.1f}%"
            )

    print("\n" + "=" * 72)
    print("MEAN across folds:")
    for mode in modes:
        if not summary[mode]:
            continue
        w20 = np.mean([x["within_20pct"] for x in summary[mode]])
        mae = np.mean([x["mae"] for x in summary[mode]])
        print(f"  {mode:18s} ±20%={w20:5.1f}%  MAE={mae:.3f}")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_training": n,
        "repeats": args.repeats,
        "summary": {m: summary[m] for m in modes if summary[m]},
    }
    report = settings.data_dir / "reports" / f"cv5_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {report}")


if __name__ == "__main__":
    main()
