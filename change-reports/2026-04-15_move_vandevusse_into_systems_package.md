# 2026-04-15 move Van de Vusse into systems package

- moved the Van de Vusse plant implementation out of `Simulation/system_functions.py`
- added a new canonical case-study seed package at `systems/vandevusse`
- added the first package pieces for that new case study:
  - `config.py`
  - `data_io.py`
  - `labels.py`
  - `plant.py`
  - `__init__.py`
- kept the phase limited to plant-layer scaffolding only; no notebooks, system-id, MPC, RL, or artifact generation were added

Validation:
- `py_compile` for `Simulation/system_functions.py` and the new `systems/vandevusse/*.py` files
- instantiated `systems.vandevusse.plant.VanDeVusseCSTR` with the package defaults
- confirmed finite steady state, nonnegative concentrations, physically reasonable temperatures, and successful one-step simulation in both absolute and deviation form
