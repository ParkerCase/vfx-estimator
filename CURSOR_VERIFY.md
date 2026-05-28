# Handoff: verify `vfx-estimator` is complete

Give this file to a **new Cursor chat** opened in the `vfx-estimator` folder (or repo root). The agent should run the commands below and fix anything that fails.

---

## Prompt to paste into Cursor

```text
Read vfx-estimator/SLIMDOWN_MASTER.md and CURSOR_VERIFY.md.

Verify the vfx-estimator standalone project is complete and working.

1. Read SLIMDOWN_MASTER.md, CURSOR_VERIFY.md, and README.md in vfx-estimator/.
2. From vfx-estimator/, ensure .env exists (copy from .env.example if needed) with correct absolute paths to MuseAI-xata apps/breakdown data.
3. Create venv, pip install -r requirements.txt && pip install -e .
4. Run: python -m scripts.verify_setup
5. If all pass, run: python -m scripts.eval_byzantine --max-shots 50 --modes retrieval_median,numeric_only
6. Report: pass/fail per step, training shot count, Byzantine ±20% for numeric_only, and any missing files.

Do not modify apps/breakdown unless a path in .env is wrong. Fix only vfx-estimator issues.
```

---

## Expected layout (must exist)

```
vfx-estimator/
  README.md
  CURSOR_VERIFY.md
  .env.example
  requirements.txt
  pyproject.toml
  data/tuning.defaults.json
  vfx_estimator/          # Python package
    config.py
    metrics.py
    types.py
    data/loaders.py
    retrieval/index.py
    learning/corrections.py
    llm/gemini_client.py
    llm/mandays_rag.py
    numeric/legacy_bridge.py
    estimate/service.py
    integrations/xata.py
    screenplay/scene_match.py
    api/app.py
  scripts/
    verify_setup.py       # main gate
    eval_cv5.py
    eval_byzantine.py
    eval_hitl_simulation.py
    run_api.py
  tests/
    test_metrics.py
    test_screenplay.py
```

---

## One-command gate (required)

```bash
cd vfx-estimator
source .venv/bin/activate   # after venv + pip install
python -m scripts.verify_setup
```

**Success:** last line is `RESULT: ALL CHECKS PASSED` and exit code `0`.

**Quick mode** (skip slow checks):

```bash
python -m scripts.verify_setup --quick
```

**With Gemini** (optional; needs `GEMINI_API_KEY` in `.env`):

```bash
python -m scripts.verify_setup --with-gemini
```

---

## Manual checklist (if script is unavailable)

| # | Check | Command / criterion |
|---|--------|---------------------|
| 1 | Files present | 30+ paths in `scripts/verify_setup.py` → `REQUIRED_PATHS` all exist |
| 2 | `.env` paths | `VFX_LEGACY_BREAKDOWN_ROOT` points to `.../apps/breakdown` |
| 3 | Training file | `retraining_bundle.json` or `xata_full_export.json` loads |
| 4 | Byzantine CSV | `byzantine_actual_expanded.csv` exists |
| 5 | Unit tests | `pytest -q` → 3 passed |
| 6 | Numeric estimate | `numeric_only` returns mandays > 0 for a test line |
| 7 | API routes | `/health`, `/estimate`, `/corrections`, `/tuning` |
| 8 | Byzantine accuracy | `python -m scripts.eval_byzantine --max-shots 50` → numeric_only ±20% often **≥ 40%** (≥50% on full set is target) |

---

## Optional deeper tests

```bash
# 5-fold CV (slow)
python -m scripts.eval_cv5 --repeats 5 --max-test 150

# Human-in-the-loop simulation
python -m scripts.eval_hitl_simulation --folds 3 --max-test 100

# API
python -m scripts.run_api
curl -s http://127.0.0.1:8090/health | python -m json.tool
```

---

## Common failures

| Symptom | Fix |
|---------|-----|
| Training JSON missing | Set `VFX_TRAINING_JSON` in `.env` to real path |
| Legacy pipeline not found | Set `VFX_LEGACY_BREAKDOWN_ROOT` to `apps/breakdown` |
| `pytest` fails | `pip install pytest`; run from `vfx-estimator/` |
| Gemini check skipped | Normal without API key; use `--with-gemini` only when keyed |
| Import errors | `pip install -e .` from `vfx-estimator/` |

---

## Success criteria for “everything is there”

1. `verify_setup` passes (required).
2. `eval_byzantine` runs without crash (required).
3. `numeric_only` within ±20% on Byzantine sample is **documented** (not necessarily ≥50% on every 50-shot slice).
4. README documents env vars, API, tuning, and separate-repo copy instructions.
