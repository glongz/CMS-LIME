# Setup

Code for two-stage primitive selection lives in `two_stage_selection/`.

Run from the repository root:

```bash
python two_stage_selection/run_two_stage_selection.py
python two_stage_selection/test_imports.py
```

In-package imports use relative paths (`from .module import ...`). Parent-directory modules are added to `sys.path` automatically.
