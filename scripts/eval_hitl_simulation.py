#!/usr/bin/env python3
"""
Simulate human-in-the-loop: add holdout corrections to retrieval, re-score test rows.

Usage:
  python -m scripts.eval_hitl_simulation --folds 5 --correction-frac 0.15 --mode hybrid
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vfx_estimator.config import get_settings
from vfx_estimator.data.loaders import load_training_shots
from vfx_estimator.estimate.service import EstimatorService
from vfx_estimator.learning.corrections import CorrectionsStore
from vfx_estimator.metrics import metrics_dict
from vfx_estimator.retrieval.index import ShotRetrievalIndex
from vfx_estimator.types import UserCorrection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--correction-frac", type=float, default=0.15)
    ap.add_argument("--max-test", type=int, default=200)
    ap.add_argument("--mode", default="hybrid")
    args = ap.parse_args()

    settings = get_settings()
    shots = load_training_shots(settings)
    n = len(shots)
    rng = np.random.RandomState(args.seed)
    before_all, after_all = [], []

    for fold in range(args.folds):
        perm = rng.permutation(n)
        n_test = max(40, n // 5)
        te_idx = set(int(i) for i in perm[:n_test])
        train_shots = [shots[i] for i in range(n) if i not in te_idx]
        test_shots = [shots[i] for i in range(n) if i in te_idx]
        if len(test_shots) > args.max_test:
            pick = rng.choice(len(test_shots), size=args.max_test, replace=False)
            test_shots = [test_shots[int(i)] for i in pick]

        base_index = ShotRetrievalIndex(train_shots, settings=settings)
        svc_before = EstimatorService(settings)
        svc_before.training = train_shots
        svc_before.index = base_index
        if svc_before.gemini:
            svc_before.gemini.index = base_index

        preds_b, acts = [], []
        for s in test_shots:
            acts.append(float(s.mandays))
            preds_b.append(float(svc_before.estimate(s.description, mode=args.mode).per_shot_mandays))

        n_corr = max(5, int(len(test_shots) * args.correction_frac))
        corr_pick = rng.choice(len(test_shots), size=min(n_corr, len(test_shots)), replace=False)
        corr_path = settings.data_dir / f"_hitl_sim_fold{fold}.jsonl"
        corr_path.unlink(missing_ok=True)
        store = CorrectionsStore(path=corr_path, settings=settings)
        for j in corr_pick:
            s = test_shots[int(j)]
            store.append(
                UserCorrection(
                    description=s.description,
                    final_total_days=float(s.mandays),
                    ai_total_days=float(preds_b[int(j)]),
                )
            )

        index_after = ShotRetrievalIndex(train_shots, settings=settings, corrections=store)
        svc_after = EstimatorService(settings)
        svc_after.training = train_shots
        svc_after.index = index_after
        if svc_after.gemini:
            svc_after.gemini.index = index_after

        preds_a = [
            float(svc_after.estimate(s.description, mode=args.mode).per_shot_mandays) for s in test_shots
        ]

        mb = metrics_dict(preds_b, acts)
        ma = metrics_dict(preds_a, acts)
        before_all.append(mb["within_20pct"])
        after_all.append(ma["within_20pct"])
        print(
            f"Fold {fold+1}: before ±20%={mb['within_20pct']:.1f}%  "
            f"after ±20%={ma['within_20pct']:.1f}%  "
            f"(+{ma['within_20pct']-mb['within_20pct']:.1f} pp)  corrections={len(corr_pick)}"
        )
        corr_path.unlink(missing_ok=True)

    print("=" * 72)
    print(f"Mean ±20% before: {np.mean(before_all):.1f}%")
    print(f"Mean ±20% after:  {np.mean(after_all):.1f}%")
    print(f"Mean gain:        {np.mean(after_all)-np.mean(before_all):+.1f} pp")


if __name__ == "__main__":
    main()
