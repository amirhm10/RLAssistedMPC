# 2026-04-21 Polymer Wide-Range Report Extension

- extended `report/generate_polymer_wide_range_report.py` to analyze the actual polymer RL reward, not the earlier simplified `Q=[5,1]` proxy
- added reward-geometry diagnostics showing setpoint-dependent band widths, edge slopes, and implied equalization targets
- added mathematical multiplier-range analysis for the matrix family and a sampled admissibility frontier for the structured block family
- added residual-style gating diagnostics based on `tracking_error_raw` to show how low-tracking multiplier authority can be reduced
- regenerated `report/polymer_wide_range_matrix_structured_report.md` and the new embedded figures:
  - `wide_range_reward_balance.png`
  - `wide_range_admissibility_frontier.png`
  - `wide_range_authority_gate.png`
