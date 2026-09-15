# Two-stage primitive selection

Extract candidate EEG primitives (stage 1), then score them with a LIME-style causal perturbation (stage 2).

## Layout

```
two_stage_selection/
├── prior_knowledge_config.py
├── primitive_selection_stage1.py    # extract candidates
├── primitive_selection_stage2.py    # perturbation importance
├── two_stage_primitive_selection.py
├── run_two_stage_selection.py       # CHB-MIT driver
└── utils/visualization.py
```

## Run

From the repository root:

```bash
python two_stage_selection/run_two_stage_selection.py
python two_stage_selection/test_imports.py
```

As a package:

```python
from two_stage_selection import (
    TwoStagePrimitiveSelection,
    create_default_prior_config,
)

prior_config = create_default_prior_config()
framework = TwoStagePrimitiveSelection(
    prior_config=prior_config,
    model=your_model.predict_proba,
    n_perturbations=10,
    importance_threshold=0.1,
    top_k_biomarkers=20,
)
framework.fit(X_train, y_train)
biomarkers = framework.explain(X_sample, y_sample)
framework.save_results("output_folder")
```

Depends on root modules `shapelet_analysis.py`, `microstate_analysis.py`, `timefreq_analysis.py`, and `cms_lime_chb_analysis_final.py`.
