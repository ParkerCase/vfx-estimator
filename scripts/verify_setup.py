#!/usr/bin/env python3
"""
End-to-end setup verification for vfx-estimator.

Run from project root (with venv activated):

  python -m scripts.verify_setup
  python -m scripts.verify_setup --quick      # skip slow legacy + API checks
  python -m scripts.verify_setup --with-gemini # also call Gemini (costs $)

Exit code 0 = all required checks passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_PATHS = [
    "SLIMDOWN_MASTER.md",
    "README.md",
    ".env.example",
    "requirements.txt",
    "pyproject.toml",
    "setup.cfg",
    "data/tuning.defaults.json",
    "data/reports/.gitkeep",
    "vfx_estimator/__init__.py",
    "vfx_estimator/config.py",
    "vfx_estimator/metrics.py",
    "vfx_estimator/types.py",
    "vfx_estimator/data/loaders.py",
    "vfx_estimator/retrieval/index.py",
    "vfx_estimator/learning/corrections.py",
    "vfx_estimator/llm/gemini_client.py",
    "vfx_estimator/llm/mandays_rag.py",
    "vfx_estimator/numeric/legacy_bridge.py",
    "vfx_estimator/numeric/fallback_numeric.py",
    "vfx_estimator/numeric/bundled/README.md",
    "scripts/vendor_legacy_pipeline.py",
    "vfx_estimator/estimate/service.py",
    "vfx_estimator/integrations/xata.py",
    "vfx_estimator/screenplay/scene_match.py",
    "vfx_estimator/api/app.py",
    "scripts/eval_cv5.py",
    "scripts/eval_byzantine.py",
    "scripts/eval_hitl_simulation.py",
    "scripts/run_api.py",
    "tests/test_metrics.py",
    "tests/test_screenplay.py",
]

Check = Tuple[str, Callable[[], None]]


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}")
    raise AssertionError(msg)


def check_files() -> None:
    missing = [p for p in REQUIRED_PATHS if not (ROOT / p).exists()]
    if missing:
        _fail(f"Missing {len(missing)} file(s), e.g. {missing[0]}")
    _ok(f"All {len(REQUIRED_PATHS)} required paths exist")


def check_imports() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    for mod in (
        "vfx_estimator",
        "vfx_estimator.config",
        "vfx_estimator.estimate.service",
        "vfx_estimator.retrieval.index",
        "vfx_estimator.numeric.legacy_bridge",
    ):
        importlib.import_module(mod)
    _ok("Core Python packages import")


def check_env_and_data() -> None:
    from vfx_estimator.config import get_settings

    get_settings.cache_clear()
    s = get_settings()
    train = s.resolved_training_json()
    if not train.is_file():
        _fail(f"Training JSON missing: {train}")
    _ok(f"Training JSON ({train.name})")

    try:
        byz = s.resolved_byzantine_csv()
    except FileNotFoundError as e:
        _fail(str(e))
    if not byz.is_file():
        _fail(f"Byzantine CSV missing: {byz}")
    _ok(f"Byzantine CSV ({byz.name})")

    from vfx_estimator.numeric.legacy_bridge import resolve_legacy_roots

    roots = resolve_legacy_roots(s)
    if roots:
        _ok(f"Legacy pipeline ({roots[0].parent.name}/scripts)")
    else:
        _ok("Legacy pipeline: retrieval fallback (run vendor_legacy_pipeline for full numeric)")


def check_training_load() -> None:
    from vfx_estimator.data.loaders import load_training_shots

    shots = load_training_shots()
    if len(shots) < 500:
        _fail(f"Expected >=500 training shots, got {len(shots)}")
    _ok(f"Loaded {len(shots)} training shots")


def check_retrieval_and_screenplay() -> None:
    from vfx_estimator.data.loaders import load_training_shots
    from vfx_estimator.retrieval.index import ShotRetrievalIndex
    from vfx_estimator.screenplay.scene_match import index_screenplay_plaintext, match_scenes

    shots = load_training_shots()
    idx = ShotRetrievalIndex(shots)
    hits = idx.query("greenscreen comp wire removal", top_k=3)
    if not hits:
        _fail("Retrieval returned no neighbors")
    _ok(f"Retrieval top hit sim={hits[0].similarity:.2f}")

    raw = "EXT. A - DAY\n\nINT. CHURCH - NIGHT\nHal enters.\n"
    scenes = index_screenplay_plaintext(raw)
    m = match_scenes("Scene 2 INT church", scenes, top_k=1)
    if not m or m[0][0].order_index != 2:
        _fail("Screenplay scene match failed")
    _ok("Screenplay scene index + match")


def check_numeric_legacy() -> None:
    from vfx_estimator.estimate.service import EstimatorService

    svc = EstimatorService()
    est = svc.estimate("wire removal cleanup plate", mode="numeric_only")
    if est.per_shot_mandays <= 0:
        _fail("numeric_only returned zero mandays")
    src = "fallback" if svc.legacy.using_fallback else "legacy"
    _ok(f"numeric_only ({src}) = {est.per_shot_mandays} mandays")


def check_corrections_roundtrip() -> None:
    from vfx_estimator.config import get_settings
    from vfx_estimator.estimate.service import EstimatorService
    from vfx_estimator.types import UserCorrection

    path = get_settings().data_dir / "_verify_correction.jsonl"
    path.unlink(missing_ok=True)
    svc = EstimatorService()
    svc.corrections.path = path
    before = len(svc.corrections.load())
    svc.record_correction(
        UserCorrection(
            description="VERIFY test shot greenscreen comp",
            final_total_days=4.5,
            ai_total_days=2.0,
        )
    )
    loaded = svc.corrections.load()
    if len(loaded) != before + 1:
        _fail(f"Expected {before + 1} corrections after append, got {len(loaded)}")
    if svc.corrections.storage_backend == "local_jsonl":
        if not path.is_file() or path.stat().st_size < 10:
            _fail("Corrections JSONL file not written")
    svc.reload_corrections()
    path.unlink(missing_ok=True)
    _ok(f"Corrections write + reload ({svc.corrections.storage_backend})")


def check_api_app() -> None:
    from vfx_estimator.api.app import create_app

    app = create_app()
    routes = {getattr(r, "path", None) for r in app.routes}
    for path in ("/health", "/estimate", "/corrections", "/tuning"):
        if path not in routes:
            _fail(f"Missing route {path}")
    _ok("FastAPI routes registered")


def check_pytest() -> None:
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        _fail(f"pytest failed:\n{r.stdout}\n{r.stderr}")
    _ok("pytest tests/ passed")


def check_byzantine_smoke() -> None:
    from vfx_estimator.config import get_settings
    from vfx_estimator.data.loaders import load_byzantine_holdout
    from vfx_estimator.estimate.service import EstimatorService
    from vfx_estimator.metrics import metrics_dict

    holdout = load_byzantine_holdout()[:40]
    svc = EstimatorService(get_settings())
    preds = [float(svc.estimate(r["description"], mode="numeric_only").per_shot_mandays) for r in holdout]
    acts = [float(r["actual_per_shot_mandays"]) for r in holdout]
    m = metrics_dict(preds, acts)
    _ok(f"Byzantine smoke n=40 numeric_only ±20%={m['within_20pct']}% MAE={m['mae']}")


def check_gemini_optional() -> None:
    from vfx_estimator.config import get_settings
    from vfx_estimator.estimate.service import EstimatorService

    s = get_settings()
    if not s.resolved_gemini_key():
        _fail("GEMINI_API_KEY / GOOGLE_API_KEY not set (--with-gemini)")
    est = EstimatorService(s).estimate("hero cg creature closeup", mode="gemini_rag")
    if est.per_shot_mandays <= 0:
        _fail("gemini_rag returned zero mandays")
    _ok(f"gemini_rag estimate = {est.per_shot_mandays} mandays")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify vfx-estimator setup")
    ap.add_argument("--quick", action="store_true", help="Skip pytest, legacy numeric, Byzantine smoke")
    ap.add_argument("--with-gemini", action="store_true", help="Require Gemini API key and one live call")
    args = ap.parse_args()

    checks: List[Check] = [
        ("Project layout", check_files),
        ("Python imports", check_imports),
        ("Env + data paths", check_env_and_data),
        ("Load training shots", check_training_load),
        ("Retrieval + screenplay", check_retrieval_and_screenplay),
        ("Corrections store", check_corrections_roundtrip),
        ("FastAPI app", check_api_app),
    ]
    if not args.quick:
        checks.extend(
            [
                ("pytest unit tests", check_pytest),
                ("Legacy numeric estimate", check_numeric_legacy),
                ("Byzantine holdout smoke", check_byzantine_smoke),
            ]
        )
    if args.with_gemini:
        checks.append(("Gemini live call", check_gemini_optional))

    print("VFX Estimator setup verification")
    print(f"Root: {ROOT}\n")

    failed = 0
    for name, fn in checks:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            failed += 1
            print(f"  FAIL  {e}")
        print()

    if failed:
        print(f"RESULT: {failed} check group(s) FAILED")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")
    if not args.with_gemini:
        print("Tip: run with --with-gemini after setting GEMINI_API_KEY in .env")


if __name__ == "__main__":
    main()
