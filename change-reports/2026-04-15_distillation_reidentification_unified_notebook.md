# Distillation Reidentification Unified Notebook

- added a canonical `distillation_RL_assisted_MPC_reidentification_unified.ipynb` entrypoint
- added `get_distillation_notebook_defaults("reidentification")`
- generalized the shared reidentification helper and runner so the low-rank dual-eta workflow now supports both polymer and distillation
- added distillation low-rank basis caching under `Distillation/Data`
- assigned the distillation Aspen family `reidentification` to `C2S_SS_simulation10.dynf` and kept the generic one-family-per-method mapping rule in `AGENTS.md`
