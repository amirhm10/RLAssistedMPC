# Polymer Markov RL Activation

## Summary

Activated TD3 as the default proposal policy for the polymer Markov correction prototype. The runner now uses constrained LS as the warm-start teacher and fallback path, while TD3 proposes bounded Markov correction coordinates after the warm-start boundary.

## Implementation Notes

- Defaults now use `run_rl_proposal=True`, `agent_kind="td3"`, `warm_start=10`, and `decision_interval=1` while preserving the disturbed polymer run setup and MPC penalties.
- The Markov runner stores executed normalized actions in replay, logs requested and executed TD3/Markov actions, and saves a TD3 checkpoint with each saved Markov run.
- The report and notebook wording now describe the RL-active setup and avoid claiming closed-loop superiority from a smoke run.

## Validation

- `python -m py_compile report/scripts/generate_polymer_markov_correction_assets.py` passed with the local Conda `rl` interpreter.
- A capped `--max-steps 30` smoke run saved outputs under `Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/` and comparison plots under `Polymer/Results/polymer_markov_compare_disturb/20260508_121849/`.
- The smoke config confirmed `run_rl_proposal=True`, `agent_kind="td3"`, `warm_start=10`, `n_tests=200`, `set_points_len=400`, `predict_h=9`, and `cont_h=3`.

