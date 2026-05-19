# 2026-05-17 TD3-priority Markov shared runner

## What changed

- added shared TD3-priority fallback support to `utils/markov_runner.py`
- enabled TD3-priority fallback in polymer and distillation Markov defaults
- disabled post-warm-start LS-target behavioral cloning in active Markov defaults
- updated shared distillation reward defaults to the temperature-emphasis settings:
  - `Q_diag = [37000, 5000]`
  - `k_rel = [0.3, 0.01]`
  - `band_floor_phys = [0.003, 0.2]`
- kept TD3 gamma defaults unchanged at `0.995`
- removed extra polymer Markov notebook variants, leaving `RL_assisted_MPC_markov_unified.ipynb` as the active polymer Markov entrypoint
- updated `report/distillation_markov_td3_decision_authority_2026_05_17.md` with implementation status

## Runtime behavior

The unified Markov notebooks now keep nominal MPC as the backbone while making TD3 the primary post-warm-start decision path.

TD3 candidates are executed unless they fail a phased catastrophic check. LS remains available as an emergency fallback, and nominal MPC remains the final fallback. Prediction score remains logged but is not a positive-score hard veto in TD3-priority mode.

The no-safeguard path is unchanged: `force_td3_execute = True` still bypasses fallback screening.

## Scope

- no Aspen run
- no notebook execution
- no raw result folders changed
- `distillation_RL_assisted_MPC_markov_td3_only_no_safeguard_unified.ipynb` was not edited because it may be running

