# 2026-04-15 add Van de Vusse system-identification surface

- extended `systems/vandevusse` from a plant-only seed into a notebook-facing case-study package
- added `systems/vandevusse/notebook_params.py` with `get_vandevusse_notebook_defaults("system_identification")`
- added `systems/vandevusse/system_id.py` for Van de Vusse step-test orchestration, artifact export, and validation plotting
- added `vandevusse_systemIdentification_unified.ipynb` as the canonical root notebook entrypoint
- added Python-only delayed FOPDT to discrete-time state-space helpers for the Van de Vusse workflow in `Simulation/sys_ids.py`
- kept the existing MATLAB-backed distillation/polymer helpers callable and unchanged as the active path for those older workflows
- added `report/08_vandevusse_system_identification.tex` for the new plant-plus-ID case-study stage

Validation:
- `py_compile` for the updated `systems/vandevusse/*.py` files, `Simulation/sys_ids.py`, and `utils/notebook_setup.py`
- notebook JSON parse for `vandevusse_systemIdentification_unified.ipynb`
- instantiated the Van de Vusse plant at the benchmark-ID inputs and confirmed finite steady state plus one-step simulation in absolute and deviation form
- ran the new Van de Vusse Python identification workflow end-to-end on the canonical local step tests and verified finite fitted channels, finite identified `A/B/C/D`, saved CSV/pickle/json artifacts, and validation plots
