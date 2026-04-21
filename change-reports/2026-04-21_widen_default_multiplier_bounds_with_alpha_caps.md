# 2026-04-21 Widen Default Multiplier Bounds With Alpha Caps

- widened polymer and distillation default matrix-multiplier ranges to `0.75` to `1.25`
- capped the `A` multiplier upper bound by the analyzed system-specific spectral limits:
  - polymer `alpha_max = 1.0566`
  - distillation `alpha_max = 1.1929`
- kept `B`-side defaults at the full widened `0.75` to `1.25` range
- added structured-matrix A/B bound overrides so structured defaults follow the same rule:
  - `A` side uses `0.75` to `alpha_max`
  - `B` side uses `0.75` to `1.25`
- updated the wide-range report generator and markdown to explain `B` selection in more detail and reflect the widened defaults
