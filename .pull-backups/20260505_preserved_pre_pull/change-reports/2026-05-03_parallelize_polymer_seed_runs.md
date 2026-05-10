## Summary

Extended the polymer core study parallelism so seed runs can execute concurrently as well, instead of only parallelizing method groups.

## Changes

- Added `parallel_seeds` and `max_run_workers` controls to `utils/polymer_multiseed_core_study.py`.
- Added a single-run worker path so a method+seed pair can be submitted as its own spawned process.
- Updated the parallel all-method notebook launcher to run both methods and seeds concurrently by default.
- Updated the single-method notebook helper to parallelize the three configured seeds for that method.
- Added notebook-visible runtime summary fields for submitted task count and effective worker count.

## Validation

- Re-parsed the notebook JSON after edits.
- Recompiled the shared study helper.
- Ran focused synthetic checks confirming:
  - all-method mode can submit method+seed tasks concurrently
  - single-method mode can fan out seeds concurrently
  - sequential fallback still preserves the three-seed default
