# VFX Estimator — complete handoff (slimdown master doc)

This is the **single checklist** for everything built to reach the current state, what still ties you to `MuseAI-xata`, and how to make `vfx-estimator/` the **only** product codebase.

**Verified baseline (your run):**

| Gate | Result |
|------|--------|
| `python -m scripts.verify_setup` | **ALL CHECKS PASSED** |
| Training shots loaded | **4,287** |
| Byzantine n=50 `numeric_only` ±20% | **52%** (≥50% bar) |
| Byzantine n=50 `retrieval_median` ±20% | **26%** |
| Live Xata Postgres | **Yes** — shot search + `vfx_corrections` table |

---

## 1. What “done” means (product scope)

The **perfect VFX estimator** (this repo folder) should do:

1. **Ingest** a shot line (+ optional bid prequal: scale, complexity, screenplay, anchors, dept sliders).
2. **Retrieve** similar historical shots (TF-IDF + boosted user corrections).
3. **Predict** mandays + department split (`numeric_only` / `gemini_rag` / `hybrid`).
4. **Explain** similar shots, screenplay scene matches, optional Gemini reasoning.
5. **Learn** from supervisor corrections (JSONL, no LLM fine-tune required).
6. **Evaluate** with CV + Byzantine holdout scripts.
7. **Serve** via FastAPI (`/estimate`, `/corrections`, `/tuning`).

It should **not** own: full MuseAI frontend, Supabase auth, script intelligence UI, OpenBID-style spreadsheets — only estimation.

---

## 2. Architecture (today)

```mermaid
flowchart TB
  subgraph client [Clients]
    API[FastAPI /estimate]
    CLI[eval scripts]
  end

  subgraph vfx [vfx-estimator]
    SVC[EstimatorService]
    RET[ShotRetrievalIndex]
    GEM[GeminiMandaysEstimator]
    SCR[screenplay scene_match]
    COR[CorrectionsStore JSONL]
    XATA_OPT[XataShotSearch optional]
  end

  subgraph legacy [Still required today]
    GMP[generalized_mandays_pipeline.py]
    TRAIN[retraining_bundle.json]
    TIER[tiered_complexity_byzantine.py]
  end

  API --> SVC
  CLI --> SVC
  SVC --> RET
  SVC --> GEM
  SVC --> SCR
  SVC --> COR
  SVC --> XATA_OPT
  SVC --> GMP
  RET --> TRAIN
  GMP --> TRAIN
  GMP --> TIER
```

**Modes:**

| Mode | Source of mandays number |
|------|---------------------------|
| `retrieval_median` | Median of top-k TF-IDF neighbors only |
| `numeric_only` | Legacy `predict_with_prequal` (main accuracy today) |
| `gemini_rag` | Gemini JSON + retrieval neighbors in prompt |
| `hybrid` | Weighted blend: `blend_numeric_weight` + `blend_gemini_weight` |

---

## 3. Every file in `vfx-estimator/` (source tree)

```
vfx-estimator/
├── SLIMDOWN_MASTER.md          ← this document
├── CURSOR_VERIFY.md            ← paste-prompt for new Cursor chats
├── README.md
├── .env.example
├── .env                        ← local only; never commit secrets
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── setup.cfg
├── data/
│   ├── tuning.defaults.json    ← default tuning knobs
│   ├── corrections.jsonl       ← created at runtime (human feedback)
│   └── reports/                ← eval JSON outputs
├── vfx_estimator/
│   ├── config.py               ← all env + tuning
│   ├── types.py                ← BidPreQual, ShotEstimate, UserCorrection
│   ├── metrics.py              ← ±10/±20/MAE
│   ├── data/loaders.py          ← training JSON + Byzantine CSV
│   ├── retrieval/index.py      ← TF-IDF + correction boost
│   ├── learning/corrections.py
│   ├── llm/gemini_client.py
│   ├── llm/mandays_rag.py
│   ├── numeric/legacy_bridge.py ← imports apps/breakdown pipeline
│   ├── screenplay/scene_match.py
│   ├── estimate/service.py     ← orchestrator
│   ├── integrations/xata.py    ← optional live Xata REST
│   └── api/app.py              ← FastAPI
├── scripts/
│   ├── verify_setup.py         ← required gate
│   ├── eval_cv5.py
│   ├── eval_byzantine.py
│   ├── eval_hitl_simulation.py
│   └── run_api.py
└── tests/
    ├── test_metrics.py
    └── test_screenplay.py
```

---

## 4. What you still need from `MuseAI-xata` (cannot delete yet)

### 4.1 Runtime code (imported at estimate time)

| Monorepo path | Why |
|---------------|-----|
| `apps/breakdown/scripts/generalized_mandays_pipeline.py` | **Core mandays model** (`predict_with_prequal`) |
| `apps/breakdown/scripts/tiered_complexity_byzantine.py` | Imported by pipeline for complexity scoring |
| `apps/breakdown/src/ml/` (and deps) | Pipeline imports from `ml.*` when prequal/dept patterns used |

Set via: `VFX_LEGACY_BREAKDOWN_ROOT=/path/to/MuseAI-xata/apps/breakdown`

### 4.2 Data files (read at startup / eval)

| File | Purpose | Env var |
|------|---------|---------|
| `apps/breakdown/data/processed/retraining_bundle.json` | **4,287** labeled shots (primary training) | `VFX_TRAINING_JSON` |
| `apps/breakdown/data/xata_full_export.json` | Fallback if bundle missing | auto in `config.py` |
| `apps/breakdown/data/byzantine_actual_expanded.csv` | Holdout eval only | `VFX_BYZANTINE_CSV` |
| `apps/breakdown/data/processed/shot_training_rows.jsonl` | Optional JSONL training (not required for verify) | `VFX_SHOT_TRAINING_JSONL` |

### 4.3 Optional test screenplay (monorepo root)

| File | Purpose |
|------|---------|
| `data/test-script-1-09-26.txt` | Screenplay scene-match demos |

### 4.4 What you do **not** need for the estimator API

These can be archived or deleted **after** you confirm no other product depends on them:

- `frontend/` (React MuseAI app)
- `apps/breakdown/demo/` (old Flask spreadsheet UI)
- Most `apps/breakdown/scripts/*` except pipeline + tiered complexity
- Hundreds of `apps/breakdown/data/*.md` status reports
- `supabase/` migrations (unless you still ship that app)

---

## 5. Environment variables (complete)

### 5.1 `vfx-estimator/.env` (this service)

| Variable | Required? | Purpose |
|----------|-----------|---------|
| `VFX_LEGACY_BREAKDOWN_ROOT` | **Yes** (today) | Path to `apps/breakdown` for numeric pipeline |
| `VFX_TRAINING_JSON` | **Yes** | Labeled shots JSON |
| `VFX_BYZANTINE_CSV` | For eval | Holdout labels |
| `VFX_SHOT_TRAINING_JSONL` | Optional | Alternate training format |
| `VFX_ESTIMATOR_DATA_DIR` | Optional | Default `./data` (corrections, reports, tuning) |
| `GEMINI_API_KEY` or `GOOGLE_API_KEY` | For `hybrid` / `gemini_rag` | Gemini REST |
| `GEMINI_MODEL` | Optional | Default `gemini-2.5-flash` |
| `GEMINI_MANDAYS_MODEL` | Optional | Overrides model for mandays RAG |
| `XATA_API_KEY` | Optional | Live Xata search |
| `XATA_DATABASE_URL` | Optional | Xata workspace DB URL |
| `XATA_BRANCH` | Optional | Default `main` |
| `XATA_TABLE` | Optional | Default `vfx_historical_shots` |
| `VFX_ESTIMATE_MODE` | Optional | `hybrid` / `numeric_only` / `gemini_rag` |
| `VFX_BLEND_NUMERIC_WEIGHT` | Optional | Default `0.55` |
| `VFX_BLEND_GEMINI_WEIGHT` | Optional | Default `0.45` |
| `VFX_CORRECTION_BOOST` | Optional | Default `2.0` (corrections in retrieval) |
| `VFX_RETRIEVAL_TOP_K` | Optional | Default `10` |
| `VFX_DAY_RATE` | Optional | Default `700` |
| `VFX_USE_LEGACY_NUMERIC` | Optional | `1` = use legacy pipeline |
| `VFX_API_HOST` / `VFX_API_PORT` | Optional | API bind |

**Copy from repo root:** `GEMINI_API_KEY` can be copied from monorepo `.env` into `vfx-estimator/.env`.  
**Xata:** `vfx-estimator` does **not** read `XATA_POSTGRES_URL` from monorepo `.env` today — only REST `XATA_API_KEY` + `XATA_DATABASE_URL` if you enable live search.

### 5.2 Tuning file (no redeploy)

Edit `vfx-estimator/data/tuning.json` (or `PUT /tuning`):

```json
{
  "estimate_mode": "hybrid",
  "blend_numeric_weight": 0.55,
  "blend_gemini_weight": 0.45,
  "correction_boost": 2.0,
  "retrieval_top_k": 10,
  "use_legacy_numeric": true
}
```

Defaults: `data/tuning.defaults.json`.

---

## 6. Setup from zero (new machine or new repo)

```bash
# 1) Get the folder (inside monorepo or its own git repo)
cd /path/to/vfx-estimator

# 2) Python
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
pip install pytest

# 3) Configure
cp .env.example .env
# Edit .env — absolute paths to MuseAI-xata apps/breakdown + data files

# 4) Gate
python -m scripts.verify_setup

# 5) Accuracy smoke
python -m scripts.eval_byzantine --max-shots 50 --modes retrieval_median,numeric_only,hybrid

# 6) Optional API
python -m scripts.run_api
curl -s http://127.0.0.1:8090/health | python -m json.tool
```

---

## 7. Slimdown plan (phases)

### Phase A — Use vfx-estimator as product (now)

- [x] Standalone package + API + eval scripts
- [x] Verify gate passes
- [ ] Point product UI (new thin client) at `http://127.0.0.1:8090`
- [ ] Copy `GEMINI_API_KEY` into `vfx-estimator/.env`
- [ ] Decide production mode: `numeric_only` vs `hybrid`

### Phase B — Copy data into vfx-estimator (drop path dependency)

```bash
mkdir -p vfx-estimator/data/training
cp apps/breakdown/data/processed/retraining_bundle.json vfx-estimator/data/training/
cp apps/breakdown/data/byzantine_actual_expanded.csv vfx-estimator/data/
```

Update `.env`:

```bash
VFX_TRAINING_JSON=/path/to/vfx-estimator/data/training/retraining_bundle.json
VFX_BYZANTINE_CSV=/path/to/vfx-estimator/data/byzantine_actual_expanded.csv
```

You still need **legacy code** until Phase C.

### Phase C — Port numeric pipeline into vfx-estimator

Move (or vendor) into `vfx_estimator/numeric/`:

- `generalized_mandays_pipeline.py`
- `tiered_complexity_byzantine.py`
- Required `src/ml/*` modules the pipeline imports

Then set `VFX_USE_LEGACY_NUMERIC=1` but **remove** `VFX_LEGACY_BREAKDOWN_ROOT` dependency.

**Effort:** ~1–2 weeks careful port + re-run `verify_setup` + `eval_byzantine`.

### Phase D — Own git repo + delete monorepo bloat

```bash
cp -r vfx-estimator ~/vfx-estimator-product
cd ~/vfx-estimator-product
git init && git add . && git commit -m "VFX estimator v0.1"
```

Archive `MuseAI-xata` or keep only:

- `vfx-estimator/` (or the new repo)
- Raw data exports / Xata sync scripts (if any)

---

## 8. Xata: connected or not?

| Connection | Status |
|------------|--------|
| **Historical training data** | Local `retraining_bundle.json` (+ optional live Postgres `vfx_historical_shots`) |
| **Live shot search** | `XATA_POSTGRES_URL` → Postgres ILIKE on `vfx_historical_shots` |
| **Supervisor corrections** | `vfx_corrections` table (dual-write to `data/corrections.jsonl` backup) |

Enable in `vfx-estimator/.env`:

```bash
XATA_POSTGRES_URL=postgresql://xata:...@HOST:5432/?sslmode=require
XATA_TABLE=vfx_historical_shots
XATA_CORRECTIONS_TABLE=vfx_corrections
```

One-time migration:

```bash
python -m scripts.migrate_xata_corrections
```

`/health` reports `"xata_corrections": "postgres"` when connected. Estimates show `"source": "xata"` in `similar_shots` when Postgres search hits.

**Refreshing training data from Xata** (monorepo, separate from live search):

```bash
cd apps/breakdown
# Use your existing export script if still maintained, e.g.:
# npx tsx scripts/export-all-xata-shots.ts
# → updates data/processed/ or xata_full_export.json
```

Then point `VFX_TRAINING_JSON` at the new export.

---

## 9. API contract (for a thin UI)

### `POST /estimate`

```json
{
  "description": "Scene 2 INT church — wire removal and cleanup",
  "mode": "hybrid",
  "pre_qual": {
    "bid_scale_tier": "mid",
    "complexity_band": "medium",
    "screenplay_text_path": "/optional/path/to/script.txt",
    "calibration_anchors": [{ "description": "similar anchor shot", "mandays": 4 }],
    "dept_calibration": { "comp_paint": { "gain": 1.1 } }
  }
}
```

Response includes: `per_shot_mandays`, `dept_days`, `similar_shots`, `screenplay_scene_matches`, `numeric_mandays`, `gemini_mandays`, `reasoning`, `confidence`.

### `POST /corrections`

Supervisor adjustment → appended to `data/corrections.jsonl` → boosts future retrieval.

### `GET/PUT /tuning`

Runtime knobs without code changes.

---

## 10. Evaluation commands (regression suite)

```bash
# Required gate
python -m scripts.verify_setup

# 5-fold CV (different withheld sets)
python -m scripts.eval_cv5 --repeats 5 --max-test 200

# Byzantine full holdout
python -m scripts.eval_byzantine

# Human-in-the-loop simulation
python -m scripts.eval_hitl_simulation --folds 5 --mode hybrid

# Gemini live (costs API $)
python -m scripts.verify_setup --with-gemini
```

Success targets (pragmatic):

| Metric | Target |
|--------|--------|
| `verify_setup` | Exit 0 |
| Byzantine `numeric_only` ±20% | **≥ 50%** on full holdout (you hit 52% on n=50) |
| After corrections | Improving on **similar** shots (see hitl sim) |

---

## 11. Features built in monorepo that **fed** this (reference)

These were developed in `apps/breakdown` and are **wrapped** by vfx-estimator, not reimplemented:

| Feature | Monorepo location | Exposed in vfx-estimator |
|---------|-------------------|---------------------------|
| Tier + pattern mandays | `generalized_mandays_pipeline.py` | `numeric_only` / `hybrid` |
| Prequal (scale, complexity, depts) | same | `BidPreQual` → API |
| Screenplay scene excerpt | `src/ml/screenplay_scene_match.py` | `vfx_estimator/screenplay/` |
| Screenplay in API response | `demo/app.py` | `screenplay_scene_matches` in estimate |
| Dept calibration sliders | pipeline + demo | `dept_calibration` in pre_qual |
| Anchor calibration | pipeline | `calibration_anchors` in pre_qual |
| Gemini structured breakdown | `src/llm/gemini_breakdown.py` | Not merged yet — only mandays RAG in vfx-estimator |
| Byzantine / CV eval | `scripts/random_subset_cv.py`, etc. | `scripts/eval_*.py` |

---

## 12. New Cursor chat — paste this

```text
Read vfx-estimator/SLIMDOWN_MASTER.md and CURSOR_VERIFY.md.

Goal: keep only vfx-estimator as the VFX bidding product. Do not expand scope.

1. Run verify_setup and eval_byzantine.
2. List remaining dependencies on MuseAI-xata/apps/breakdown.
3. Propose minimal Phase B/C steps to copy data and port numeric pipeline.
4. Do not modify frontend/ or unrelated monorepo folders unless asked.
```

---

## 13. Security

- **Never commit** `vfx-estimator/.env` or monorepo `.env` (API keys).
- Rotate any keys that were ever committed to git history.
- `GEMINI_API_KEY` in repo-root `.env` is separate from `vfx-estimator/.env` — copy explicitly.

---

## 14. One-page answers

| Question | Answer |
|----------|--------|
| Can I delete MuseAI-xata now? | **No** — still need `apps/breakdown` for numeric + training JSON paths. |
| Is vfx-estimator “complete”? | **Yes** as a service shell; **no** as fully independent of monorepo. |
| Is Xata connected? | **Offline export yes**; **live API only if you set `XATA_*` in vfx-estimator/.env**. |
| What hits 50%+ today? | **`numeric_only`** (legacy pipeline), not retrieval-only. |
| What improves over time? | **`data/corrections.jsonl`** + higher `correction_boost`. |

This document + `verify_setup` + `eval_byzantine` are the full acceptance kit for the slim estimator codebase.
