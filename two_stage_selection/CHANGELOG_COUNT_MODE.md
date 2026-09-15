# Count-mode changelog

Stage 1 now defaults to **count mode** instead of hard quality thresholds, so each primitive family keeps a target number of units.

New keys in `prior_knowledge_config.py`:

- `target_shapelet_count` (default 100)
- `target_microstate_count` (default 100)
- `target_timefreq_count` (default 100)
- `use_count_mode` (default True)

In count mode, a weak quality floor is applied and the top-N units are kept. Quality mode still uses the original strict thresholds.
