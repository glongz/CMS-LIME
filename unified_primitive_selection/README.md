# Unified primitive selection

Extract primitives from the **current** sample and score importance immediately, so there is no train-to-test primitive matching step.

```
sample → extract primitives → score importance → keep high-importance units
```

## Run

```bash
python unified_primitive_selection/run_unified_selection.py
```

```python
from unified_primitive_selection import (
    UnifiedPrimitiveSelection,
    create_default_config,
)

config = create_default_config()
framework = UnifiedPrimitiveSelection(config=config, model=your_model.predict_proba)
sample_biomarkers = framework.process_samples(X_samples, y_samples)
top_biomarkers = framework.get_top_biomarkers(top_k=20)
framework.save_results("output_folder")
```

## Config (high level)

- Stage-1 style priors for shapelet / microstate / time–frequency counts
- `n_perturbations` (default 10)
- `stability_iterations` (default 3)
- `importance_threshold` (default 0.01)
- `top_k_biomarkers` (default 20)

Outputs include `config.json`, `all_biomarkers.json`, `top_biomarkers.json`, `sample_biomarkers.json`, and `statistics.json`.
