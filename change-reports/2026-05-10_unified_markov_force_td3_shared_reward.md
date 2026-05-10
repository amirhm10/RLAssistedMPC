## Summary

Retuned the unified polymer Markov notebook path for a new A/B experiment:

- restored the shared unified reward function in the unified notebook
- removed the legacy-reward branch from the unified notebook entrypoint
- switched the unified nominal reference solve back to the legacy lifted-`G0` path
- returned unified `z_bound` to `0.05`
- disabled LS fallback in the unified defaults
- added a unified-only `force_td3_execute` switch so the shared Markov runner can execute TD3-requested corrections directly without LS or nominal fallback

## Notes

- The separate restored legacy notebook and legacy reward helper were left intact for historical comparison and report analysis.
- In the force-TD3 mode, the unified runner begins using TD3 from the first Markov-eligible step (`step >= predict_h`), while replay training still starts at the configured warm-start step.

## Verification

- Loaded `polymer_markov_corrected_mpc_unified.ipynb` as valid JSON.
- Confirmed unified Markov defaults resolve to:
  - `nominal_solver_mode = "lifted_g0_prototype"`
  - `z_bound = 0.05`
  - `rl_fallback_to_ls = False`
  - `force_td3_execute = True`
- Confirmed the unified notebook no longer contains the legacy reward branch.
- Imported `utils.markov_runner` successfully after the force-TD3 runner change.
