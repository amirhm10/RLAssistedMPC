# 2026-04-15 add Van de Vusse nonlinear plant

- added `VanDeVusseCSTR` to `Simulation/system_functions.py` as the first plant-only seed for the future `vandevusse` case study
- mirrored the existing `PolymerCSTR` lifecycle:
  - constructor with `params`, `design_params`, `ss_inputs`, `delta_t`, `deviation_form`
  - `odes(...)`
  - `odes_deviation(...)`
  - `ss_params(...)`
  - `step(...)`
- kept the plant output compact as `[c_B, T]`
- kept the implementation limited to the nonlinear plant layer only; no system-id, MPC, RL, notebook, or package scaffolding was added in this pass

Validation:
- `py_compile` for `Simulation/system_functions.py`
- instantiated `VanDeVusseCSTR` with literature-style nominal parameters and `ss_inputs = [14.19, -1113.5]`
- confirmed finite steady state, nonnegative concentrations, physically reasonable temperatures, and successful one-step simulation in both absolute and deviation form
- confirmed `PolymerCSTR` still imports from the same module
