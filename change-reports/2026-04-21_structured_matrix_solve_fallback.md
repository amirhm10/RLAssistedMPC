# Structured Matrix Solve Fallback

- updated `utils/structured_matrix_runner.py` so structured-matrix runs no longer crash immediately when a wide candidate model makes the MPC solve fail
- the runner now:
  - tries the RL-selected structured prediction model first
  - falls back to the nominal structured model for that step if the candidate model is non-finite or the assisted MPC solve fails
  - only raises if the nominal fallback solve also fails
- added runtime diagnostics to the result bundle:
  - `prediction_fallback_on_solve_failure`
  - `structured_prediction_fallback_count`
  - `prediction_fallback_active_log`
  - `prediction_fallback_reason_log`
  - attempted vs effective action/multiplier logs
  - candidate vs effective model-deviation and spectral-radius logs
- the replay buffer now stores the effective action used for the step, so training data matches the control decision that actually generated the transition
