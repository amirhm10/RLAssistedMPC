## Summary

Updated the polymer core study helper and notebook to default to three seeds (`7, 11, 23`) and support method-level parallel execution through a process pool.

## Changes

- Reduced `utils/polymer_multiseed_core_study.py` default seeds from five to three.
- Added `parallel_methods` and `max_method_workers` controls to the shared study runner.
- Refactored the runner to execute one method family across all requested seeds as a single worker task.
- Preserved `continue_on_error` behavior while collecting successful results and failed-run metadata across worker boundaries.
- Added a parallel all-method launcher plus a sequential fallback cell to `polymer_five_seed_core_study.ipynb`.

## Validation

- Parsed the notebook JSON after edits.
- Compiled the touched Python helper.
- Ran focused synthetic smoke checks for sequential and parallel aggregation paths without launching the full long-running study.
