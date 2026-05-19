## Summary

Added detached notebook launcher scripts so long-running experiments can be started outside the VS Code window lifecycle.

## Files changed

- `tools/start_detached_notebook_run.ps1`
- `tools/start_distillation_markov_ls_only_detached.ps1`

## Notes

- The generic launcher executes a notebook via `python -m jupyter nbconvert --execute` in a separate process.
- Outputs are written to a timestamped run directory under `.detached-notebook-runs/`.
- The convenience wrapper targets `distillation_RL_assisted_MPC_markov_ls_only_unified.ipynb`.
