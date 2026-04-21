# RL State Scaling Diagnostics Report

Expanded the reproducible RL-state scaling analysis for polymer and distillation bundles.

Files updated:

- [report/generate_rl_state_scaling_report.py](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/generate_rl_state_scaling_report.py)
- [report/rl_state_scaling_diagnostics.md](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/rl_state_scaling_diagnostics.md)
- generated figures and CSVs under [report/rl_state_scaling_diagnostics](/c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/report/rl_state_scaling_diagnostics)

Scope:

- kept the 6-run distillation and 6-run polymer cross-family scaling sample
- switched the representative distillation residual TD3 run to the latest saved bundle
- added a dedicated comparison of the last distillation residual evaluation episode against baseline MPC, the previous TD3 residual run, and the SAC residual run
- added a method-surface audit from the shared notebook defaults so the report now distinguishes shared runner code from family-specific supervisory semantics
- extended the normalization section with fixed min-max and running-z-score mathematics plus a per-state width/sensitivity figure
- added a transform-comparison section for `innovation` and `tracking_error`, including real-evaluation traces for hard clipping, tanh, signed-log, and `VecNormalize`-style running normalization
- added figures for:
  - unified method feature matrix
  - combined-supervisor default mode matrix
  - residual-family parameter comparison across polymer and distillation
  - mismatch-scale parameter comparison across polymer and distillation
- extended the `rho` section with candidate smoother mappings, episode-level diagnostics, and literature-backed alternatives based on shielding and predictive safety filtering, now scoped explicitly to the residual family rather than the entire repo
- refreshed the markdown report so it now ties the local figures and CSVs to specific conclusions about:
  - polymer observer-state compression
  - distillation mismatch-feature saturation
  - the latest distillation residual offset behavior
  - when `VecNormalize` helps and when it is not enough by itself
  - which supervisory ideas should stay family-specific even though the code is unified
  - why the current clipped `rho` is a coarse residual gate, not a repo-wide supervisory abstraction
