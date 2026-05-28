# VFX Estimator (standalone)

Clean estimator service combining:

- **ML retrieval** over historical shots (TF-IDF + tunable weights)
- **Legacy numeric pipeline** from `apps/breakdown` (tiers, patterns, dept split, prequal)
- **Gemini RAG** (optional) for reasoning + department proposals
- **Human corrections** (JSONL) that boost retrieval for continuous learning
- **Xata** (optional) live similar-shot search
- **Screenplay** scene matching (txt / FDX)

This folder is designed to live **inside or outside** the MuseAI-xata monorepo.

**Full handoff:** see [`SLIMDOWN_MASTER.md`](SLIMDOWN_MASTER.md) (architecture, monorepo dependencies, slimdown phases, API contract).

---

## 1. Quick start (separate codebase)

```bash
cd vfx-estimator

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

cp .env.example .env
# Edit .env — at minimum set VFX_LEGACY_BREAKDOWN_ROOT and training paths
```

### Required paths (if repo layout is standard)

```bash
# In .env — adjust if you moved the repo
VFX_LEGACY_BREAKDOWN_ROOT=/path/to/MuseAI-xata/apps/breakdown
VFX_TRAINING_JSON=/path/to/MuseAI-xata/apps/breakdown/data/processed/retraining_bundle.json
VFX_BYZANTINE_CSV=/path/to/MuseAI-xata/apps/breakdown/data/byzantine_actual_expanded.csv
```

### Gemini (optional but needed for `hybrid` / `gemini_rag`)

```bash
GEMINI_API_KEY=your_key
# or GOOGLE_API_KEY=...
GEMINI_MODEL=gemini-2.0-flash
```

### Xata (optional)

```bash
XATA_API_KEY=
XATA_DATABASE_URL=https://your-workspace.xata.io/db/your-db
XATA_BRANCH=main
XATA_TABLE=vfx_historical_shots
```

---

## 2. Verify setup

```bash
python -m scripts.verify_setup
```

Quick mode (skip slow checks):

```bash
python -m scripts.verify_setup --quick
```

See also `CURSOR_VERIFY.md` for a handoff prompt for new Cursor chats.

---

## 3. Run evaluations

### 5-fold CV on training corpus (different holdouts each repeat)

```bash
python -m scripts.eval_cv5 --repeats 5 --max-test 200
```

Modes compared: `retrieval_median`, `numeric_only`, `hybrid`, `gemini_rag`

Report JSON written to `data/reports/cv5_*.json`.

### Byzantine holdout (labeled script)

```bash
python -m scripts.eval_byzantine --max-shots 0
```

Look for **±20% ≥ 50%** in the output. This is the “off the bat” bar you asked about.

### Human-in-the-loop simulation

```bash
python -m scripts.eval_hitl_simulation --folds 5 --mode hybrid
```

Adds a fraction of test rows as “corrections” and re-scores. Expect a **modest gain** on similar shots; real production gain depends on correction volume and quality.

---

## 4. API server

```bash
python -m scripts.run_api
# http://127.0.0.1:8090/health
```

### Estimate one shot

```bash
curl -s http://127.0.0.1:8090/estimate -H 'Content-Type: application/json' -d '{
  "description": "CG castle establishing shot — hero lighting",
  "pre_qual": { "bid_scale_tier": "mid", "complexity_band": "hero" },
  "mode": "hybrid"
}' | python -m json.tool
```

### Record a correction (active learning)

```bash
curl -s http://127.0.0.1:8090/corrections -H 'Content-Type: application/json' -d '{
  "description": "CG castle establishing shot — hero lighting",
  "final_total_days": 21,
  "final_departments": { "layout": 3, "lighting": 9, "comp_paint": 9 },
  "notes": "Che adjusted hero lighting"
}'
```

### Tune without redeploy

```bash
curl -s http://127.0.0.1:8090/tuning
curl -s -X PUT http://127.0.0.1:8090/tuning -H 'Content-Type: application/json' -d '{
  "blend_numeric_weight": 0.6,
  "blend_gemini_weight": 0.4,
  "correction_boost": 2.5,
  "estimate_mode": "hybrid"
}'
```

Overrides persist in `data/tuning.json` (merged over `data/tuning.defaults.json`).

---

## 5. Modes

| Mode | Behavior |
|------|----------|
| `numeric_only` | Legacy `predict_with_prequal` from apps/breakdown |
| `gemini_rag` | Gemini JSON estimate using retrieval neighbors only |
| `hybrid` | Weighted blend of numeric + Gemini (tunable) |
| `retrieval_median` | Baseline: median mandays of top-k similar shots |

Env: `VFX_ESTIMATE_MODE=hybrid`

---

## 6. Fine-tuning / tuning knobs

**Runtime (env or `data/tuning.json`):**

- `blend_numeric_weight` / `blend_gemini_weight`
- `correction_boost` — multiplier for user-corrected rows in retrieval
- `retrieval_top_k`
- `VFX_USE_LEGACY_NUMERIC=0|1`

**Continuous learning:**

- Corrections append to `data/corrections.jsonl`
- Reload via API `/corrections` or restart service
- No automatic fine-tune of LLM weights — corrections act as **weighted exemplars** (production-safe)

**Future:** export corrections to retraining bundle JSON on a schedule.

---

## 7. Tests

```bash
pytest -q
```

---

## 8. Expectations (honest)

| Question | Typical answer with current data |
|----------|----------------------------------|
| 50%+ within ±20% on Byzantine immediately? | **Maybe** on `hybrid` with Gemini; **unlikely** on retrieval-only. Run `eval_byzantine`. |
| Better with human feedback? | **Yes for similar shots** once corrections exist; run `eval_hitl_simulation`. |
| Replace legacy numeric? | Not yet — hybrid uses legacy for stability. |

---

## 9. Moving to a fully separate git repo

```bash
cp -r vfx-estimator ~/vfx-estimator
cd ~/vfx-estimator
git init
git add .
git commit -m "Initial vfx-estimator service"
```

Point `VFX_LEGACY_BREAKDOWN_ROOT` at a checkout of `apps/breakdown`, or copy `retraining_bundle.json` into `vfx-estimator/data/` and set `VFX_TRAINING_JSON` locally so the monorepo is not required at runtime.
# vfx-estimator
