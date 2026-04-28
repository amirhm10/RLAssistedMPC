# Matrix Prediction Acceptance And Fallback Review

## Scope

This note explains what was meant by:

> "most likely from the accepted assisted prediction model / fallback behavior late in the run, not from `A_aug/B_aug/C_aug` being mutated in the observer path"

The note applies to:

- scalar matrix supervisor
- structured matrix supervisor
- the matrix branch inside combined

for both polymer and distillation.

## 1. What Is Fixed And What Is Changing

The current matrix methods are split into two separate model roles:

1. Fixed nominal estimator
2. RL-assisted prediction model for MPC only

That split is explicit in the runners.

### Scalar matrix runner

In [utils/matrix_runner.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py):

- nominal model copy is created once at [utils/matrix_runner.py:175](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:175)
- estimator matrices are fixed at:
  - [utils/matrix_runner.py:177](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:177)
  - [utils/matrix_runner.py:178](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:178)
- nominal observer gain is computed once at:
  - [utils/matrix_runner.py:179](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:179)
- the observer update uses only the fixed nominal estimator at:
  - [utils/matrix_runner.py:336](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:336)

### Structured matrix runner

In [utils/structured_matrix_runner.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py):

- nominal model copy is created once at [utils/structured_matrix_runner.py:186](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:186)
- estimator matrices are fixed at:
  - [utils/structured_matrix_runner.py:188](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:188)
  - [utils/structured_matrix_runner.py:189](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:189)
- nominal observer gain is computed once at:
  - [utils/structured_matrix_runner.py:190](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:190)
- the observer update uses only the fixed nominal estimator at:
  - [utils/structured_matrix_runner.py:394](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:394)

### Combined runner

The same fixed-estimator idea is used in the matrix branch of [utils/combined_runner.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/combined_runner.py), and the result bundle labels that mode explicitly:

- `estimator_mode = "fixed_nominal"` at [utils/combined_runner.py:834](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/combined_runner.py:834)
- `matrix_prediction_model_mode = "rl_assisted"` at [utils/combined_runner.py:835](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/combined_runner.py:835)

## 2. What "Accepted Assisted Prediction Model" Means

At each control step, the matrix method builds a candidate prediction model from the RL action.

### Scalar matrix

The candidate is built from the nominal model at:

- [utils/matrix_runner.py:238](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:238)
- [utils/matrix_runner.py:239](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:239)

### Structured matrix

The candidate is built from the structured update helper at:

- [utils/structured_matrix_runner.py:309](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:309)
- [utils/structured_matrix_runner.py:310](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:310)

That candidate is then screened by [utils/robust_matrix_prediction.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/robust_matrix_prediction.py), specifically:

- `validate_prediction_candidate(...)` at [utils/robust_matrix_prediction.py:105](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/robust_matrix_prediction.py:105)

The candidate is accepted only if all of these pass:

1. finite check
2. mapped-action bounds check
3. relative Frobenius norm check on `A` and `B`
4. short-horizon prediction-deviation check against the nominal model

The prediction-deviation check is based on:

- `prediction_deviation_metric(...)` at [utils/robust_matrix_prediction.py:74](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/robust_matrix_prediction.py:74)

It compares nominal vs candidate output rollouts over a short probe horizon using the current estimated state and a held input probe sequence.

## 3. What "Fallback Behavior" Means

Once the candidate is accepted or rejected, the solver follows this sequence in:

- `solve_prediction_mpc_with_fallback(...)` at [utils/robust_matrix_prediction.py:203](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/robust_matrix_prediction.py:203)

The solve order is:

1. `assisted_tight`
2. `nominal_tight`
3. `nominal_full`

Source labels are defined at:

- [utils/robust_matrix_prediction.py:5](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/robust_matrix_prediction.py:5)

Meaning:

- `assisted_tight`
  - the RL-assisted prediction model was accepted and MPC solved with tightened input bounds
- `nominal_tight`
  - the RL-assisted model was rejected, or the assisted solve failed, so MPC solved with the nominal model and tightened bounds
- `nominal_full`
  - the tightened nominal solve also failed, so MPC fell back to the nominal model with the original input bounds

So "accepted assisted prediction model / fallback behavior" means:

- sometimes MPC is optimizing against the RL-assisted model
- sometimes it is optimizing against the nominal model
- sometimes it is also switching between tightened and full bounds

That can happen step by step during one run.

## 4. Why This Can Create A Real Tracking Mismatch

This is the important part.

The mismatch can be real even though the estimator is nominal and fixed.

### Case A: the candidate is accepted, but still not good enough

The acceptance logic is local and short-horizon:

- norm threshold on `A` and `B`
- short probe rollout
- held current input probe

That means a candidate can pass the screen and still lead MPC to compute a move that is not actually best for the real plant over the full prediction horizon.

In that case:

- the estimator remains nominal
- the plant output is real
- MPC chooses a move from the assisted prediction model
- the result can drift away from the setpoint near the end of the run

### Case B: switching between assisted and nominal changes the control law

If the accepted/rejected status changes late in the run, the optimizer may alternate between:

- assisted prediction with tightened bounds
- nominal prediction with tightened bounds
- nominal prediction with full bounds

That means the closed-loop move can change because of the model-selection logic itself, not because the plant or observer changed.

### Case C: tightened bounds reduce available authority

Even when the model is rejected and the solver falls back to nominal, it may still be using tightened input bounds first.

So a late mismatch can come from:

- valid fallback to nominal
- but less control authority than the full-bound nominal controller

## 5. Why This Is Not An Observer-Mutation Issue

The matrix mismatch is not coming from the observer matrices drifting online.

The evidence is:

- `A_est` and `B_est` are fixed copies of the nominal matrices in both matrix runners
- `L_nom` is computed once and reused
- `C_aug` is used unchanged for the measurement prediction
- the observer recursion uses the nominal estimator matrices, not the RL candidate matrices

So if the final output does not match the setpoint, that mismatch is not because `A_aug/B_aug/C_aug` were being silently rewritten inside the observer path.

## 6. Which Logged Signals To Check In A Saved Run

If a distillation matrix run ends far from the setpoint, the first signals to inspect are:

### Scalar matrix bundle

Saved in [utils/matrix_runner.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py):

- `model_accepted_log` at [utils/matrix_runner.py:463](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:463)
- `model_source_log` at [utils/matrix_runner.py:464](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:464)
- `assisted_model_used_log` at [utils/matrix_runner.py:466](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:466)
- `prediction_max_dev_log` at [utils/matrix_runner.py:467](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:467)
- `first_move_delta_vs_nominal_log` at [utils/matrix_runner.py:469](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:469)
- rejection counts at:
  - [utils/matrix_runner.py:474](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:474)
  - [utils/matrix_runner.py:475](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:475)
  - [utils/matrix_runner.py:476](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:476)
  - [utils/matrix_runner.py:477](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/matrix_runner.py:477)

### Structured matrix bundle

Saved in [utils/structured_matrix_runner.py](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py):

- `model_accepted_log` at [utils/structured_matrix_runner.py:558](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:558)
- `model_source_log` at [utils/structured_matrix_runner.py:559](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:559)
- `assisted_model_used_log` at [utils/structured_matrix_runner.py:561](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:561)
- `prediction_max_dev_log` at [utils/structured_matrix_runner.py:562](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:562)
- `first_move_delta_vs_nominal_log` at [utils/structured_matrix_runner.py:564](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:564)
- rejection counts at:
  - [utils/structured_matrix_runner.py:569](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:569)
  - [utils/structured_matrix_runner.py:570](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:570)
  - [utils/structured_matrix_runner.py:571](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:571)
  - [utils/structured_matrix_runner.py:572](c:/Users/HAMEDI/Desktop/RL_assisted_MPC_polymer/utils/structured_matrix_runner.py:572)

## 7. Why This Part May Need Revision

This part is a reasonable revision candidate because the current screen is still fairly weak for a nonlinear plant:

1. the prediction check horizon is short
2. the probe input is just the current held input
3. acceptance is based on model similarity, not actual setpoint-tracking quality
4. a candidate can pass the screen but still shift the MPC solution in an unhelpful direction
5. fallback may introduce controller switching late in the run

## 8. Practical Revision Directions

If this mechanism is revised, the main options are:

1. make the prediction-acceptance test stricter
   - longer probe horizon
   - tighter `eps_y_pred_scaled`
   - stronger norm thresholds

2. base acceptance on the nominal-vs-assisted first move
   - reject candidates that change the first move too much

3. freeze the matrix update near convergence
   - once output error is small, stop accepting new assisted matrices

4. use nominal prediction near the end of each sub-episode
   - keep the assisted model for learning, but hand control back to nominal MPC near the target

5. add hysteresis to the acceptance logic
   - avoid assisted/nominal switching every few samples

## Bottom Line

The current matrix methods are already fixed-estimator.

The likely issue is not observer mutation. The likely issue is that:

- the RL-assisted prediction model is sometimes accepted
- MPC then optimizes against that assisted model
- or the solver switches between assisted and nominal fallback modes
- and that changes the real closed-loop trajectory enough to produce the final setpoint mismatch

So if the end-of-run mismatch looks suspicious, the part that likely needs revision is the assisted-model acceptance and fallback policy, not the observer update.
