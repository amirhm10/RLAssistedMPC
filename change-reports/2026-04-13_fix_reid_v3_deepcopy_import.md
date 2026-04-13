## Summary

Fixed the v3 re-identification ablation study launcher crash caused by a missing `deepcopy` import.

## Root Cause

`experiments/run_reid_batch_ablation_study.py` called `deepcopy(...)` inside
`run_reid_batch_ablation_study(...)` but did not import it.

## Change

Added:

```python
from copy import deepcopy
```

to `experiments/run_reid_batch_ablation_study.py`.

## Validation

- `python -m py_compile experiments/run_reid_batch_ablation_study.py`
- imported `run_reid_batch_ablation_study` and loaded `get_polymer_notebook_defaults("reid_batch_v3_study")` under `rl-env`
