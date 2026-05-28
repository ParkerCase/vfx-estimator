# Bundled legacy numeric pipeline

Populated by:

```bash
python -m scripts.vendor_legacy_pipeline --source /path/to/MuseAI-xata/apps/breakdown
```

Requires `generalized_mandays_pipeline.py` and `tiered_complexity_byzantine.py` plus any `src/ml` imports they need.

If this folder is empty, `vfx-estimator` uses a **retrieval fallback** for `numeric_only` (lower Byzantine accuracy than the full pipeline).
