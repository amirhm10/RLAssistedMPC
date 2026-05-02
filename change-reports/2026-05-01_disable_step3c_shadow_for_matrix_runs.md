## Summary

Disabled `mpc_dual_cost_shadow` at runtime for both scalar and structured matrix supervisors.

## Reason

Matrix runs should stay on the guarded `Step 4G` path even if a notebook carries stale executed state or a copied config that still marks `Step 3C shadow` as enabled.

## Scope

- `utils/matrix_runner.py`
- `utils/structured_matrix_runner.py`

## Notes

The shared notebook defaults already set `mpc_dual_cost_shadow.enabled = False` for polymer and distillation matrix studies. This change makes the runner behavior match that intended policy in all cases.
