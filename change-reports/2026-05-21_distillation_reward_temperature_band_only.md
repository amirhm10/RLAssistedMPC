# Distillation Reward Temperature Band-Only Ablation

## Summary

Changed the shared distillation RL reward default so the temperature output uses the older weight while preserving the tightened temperature reward band.

## Change

- `systems/distillation/config.py`
  - `Q_diag` changed from `[37000, 5000]` to `[37000, 1500]`.
  - `k_rel` remains `[0.3, 0.01]`.
  - `band_floor_phys` remains `[0.003, 0.2]`.

## Rationale

This isolates the effect of tightening the temperature reward band from the previous simultaneous increase in the temperature penalty weight. The next distillation runs can therefore test whether the band tightening alone is enough to improve temperature behavior, or whether the higher `Q_temp = 5000` was also necessary.

## Validation

- Lightweight import/default check.
- Python compile check for the touched config path and the active distillation Markov entrypoint.
