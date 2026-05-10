# Polymer Markov Correction Progress

## Objective

This note tracks the polymer-only TD3-assisted prototype for prediction-error-validated Markov-parameter correction of offset-free MPC. The prototype keeps the existing nonlinear polymer plant, nominal observer, scaling, disturbance schedule, and standard comparison plotting workflow unchanged. It tests whether finite-horizon input-output Markov corrections can reduce recent plant prediction error before they are allowed to affect the MPC action, while TD3 learns to propose bounded correction coordinates after the warm-start period.

## Files inspected

- `polymer_markov_correction_plan.md`
- `Simulation/mpc.py`
- `Simulation/system_functions.py`
- `systems/polymer/notebook_params.py`
- `systems/polymer/data_io.py`
- `utils/helpers.py`
- `utils/plotting.py`
- `utils/plotting_core.py`
- `utils/rewards.py`

## What the existing method was doing

The existing matrix-supervisor family perturbs the state-space model through low-dimensional multipliers, such as scaled versions of $A$ and $B$. Earlier Step 3D-style logic relied heavily on internal MPC cost comparisons. For polymer, the hard usefulness gate accepted essentially no live candidates in the prior analysis, so the method collapsed toward nominal MPC behavior.

## Why Step 3D is not reused as the main gate

The new prototype uses recent plant prediction error as the primary evidence. The old gate compared quantities such as:

$$ J_0(U_z^\star) \quad \text{and} \quad J_z(U_0^\star)-J_z(U_z^\star). $$

The adjusted Markov method first tests:

$$ \|Y^{\mathrm{meas}}-Y^0\|_2^2 - \|Y^{\mathrm{meas}}-Y^z\|_2^2. $$

The nominal MPC cost check remains only as a loose guard against catastrophic corrected actions.

## Mathematical formulation

The nominal offset-free model is:

$$ x_{a,k+1} = A_a x_{a,k} + B_a u_k^{\mathrm{dev}}, \qquad y_k^{\mathrm{dev}} = C_a x_{a,k}. $$

For prediction horizon $P$ and control horizon $M$, the lifted prediction is:

$$ Y_k = Y_{\mathrm{free},k} + G_z(P,M)U_k. $$

The correction changes finite-horizon Markov blocks rather than the state-space matrices:

$$ M_i(z_k)=M_{i,0}+\sum_{j=1}^r z_{j,k}M_{i,j}^{\mathrm{basis}}. $$

The prediction-error score is:

$$ S_{\mathrm{pred}}(z)=\sum_\tau \left(\|W_y(Y_\tau^{\mathrm{meas}}-Y_\tau^0)\|_2^2-\|W_y(Y_\tau^{\mathrm{meas}}-Y_\tau^z)\|_2^2\right)-\lambda_z\|z\|_2^2. $$

## Step-by-step algorithm

1. Load the existing polymer unified configuration. The default case is disturbed polymer with `n_tests=200`, `set_points_len=400`, `warm_start=10`, `predict_h=9`, `cont_h=3`, and the existing MPC penalties. The TD3 hyperparameters come from `POLYMER_MATRIX_DEFAULTS["td3_agent"]`.

2. Load the identified polymer model, scaling data, input bounds, steady states, observer poles, and canonical baseline MPC result path through the existing polymer helper layer. The nonlinear plant rollout still uses `PolymerCSTR`.

3. Build the offset-free augmented model and the lifted Markov representation. The code computes nominal Markov blocks $M_{i,0}=C_aA_a^{i-1}B_a$ and then builds the Toeplitz prediction matrix $G_0(P,M)$ using the same absolute scaled input-deviation convention as `MpcSolverGeneral`.

4. Validate the lifted prediction implementation before any live correction is allowed. A random input sequence is simulated both with the state-space recursion and with the lifted Toeplitz form. Live Markov correction is permitted only if the maximum absolute difference is below `1e-8`.

5. Run a nominal closed-loop rollout. At every time step, nominal MPC solves with $G_0$, the nonlinear polymer plant advances, the nominal observer updates, and the resulting trajectory becomes the history used by shadow scoring and LS diagnostics.

6. Score shadow Markov candidates without applying them. Candidate $z$ values are evaluated only on already observed windows, so future plant measurements are not used. This step estimates whether any Markov basis direction would have improved recent finite-horizon prediction error.

7. Fit the constrained LS teacher. The LS problem searches for a bounded $z_{\mathrm{LS}}$ that reduces recent prediction error while paying the regularization penalty $\lambda_z\|z\|_2^2$. The LS candidate is accepted only if its prediction score is positive enough and its lifted gain drift remains below the configured limit.

8. Build the TD3 Markov state for the live rollout. The state vector concatenates the nominal observer state, current tracking error, innovation, previous input deviation, previous executed correction $z$, current accepted LS teacher correction, LS prediction score, and LS gain drift.

9. Select a bounded TD3 correction action. During warm start, the baseline action is the LS teacher action. After `warm_start_step`, TD3 proposes a raw action in $[-1,1]^r$, which is mapped to the Markov correction by $z_{\mathrm{TD3}}=z_{\mathrm{bound}}a_{\mathrm{TD3}}$.

10. Safety-filter the TD3 proposal. The runner solves corrected MPC with the TD3-corrected lifted matrix and accepts it only when the nominal solve succeeds, the corrected solve succeeds, the prediction score exceeds `s_pred_min`, gain drift is below `gain_drift_max`, and the loose nominal-cost guard passes.

11. Fall back in a fixed order if TD3 is not accepted. If TD3 fails the filter, the controller tries the accepted LS correction. If LS is unavailable or fails, the controller applies nominal MPC. The replay action is the executed action, not merely the requested TD3 action.

12. Advance the nonlinear plant and update replay. The selected first input move is applied to `PolymerCSTR`, the nominal observer updates, the existing closed-loop reward convention is computed, and the transition is pushed to TD3 replay on train steps. TD3 training starts only after the configured warm-start boundary.

13. Save artifacts in the polymer result tree. The run writes `input_data.pkl`, summary tables, verification tables, Markov diagnostic figures, RL diagnostic logs, and the TD3 checkpoint under `Polymer/Results/polymer_markov_corrected_mpc/<timestamp>/`. Standard MPC comparison plots are generated with `compare_mpc_rl_from_dirs()` under `Polymer/Results/polymer_markov_compare_disturb/<timestamp>/`.

14. Interpret results conservatively. The method is not considered successful from prediction score alone. A successful full run must pass lifted equivalence, show meaningful prediction-error improvement, avoid material output-MAE degradation, avoid excessive input movement, and produce acceptable reward relative to nominal MPC.

## Phase 1: lifted-prediction equivalence validation

The notebook validates the lifted absolute input-deviation convention against the state-space rollout. The pass threshold is `max_abs_error < 1e-8`.

Observed max absolute error: `1.734723e-18`.

## Phase 2: shadow prediction-error scoring

Candidate Markov corrections are scored on nominal closed-loop history without executing corrected actions. The fraction of steps with positive best shadow score is `0.9929`.

## Phase 3: adaptive LS Markov correction

The adaptive constrained LS correction estimates $z_k$ with bounds and regularization, then accepts it only when prediction improvement and gain-drift checks pass. The accepted fraction is `0.9904`.

## Phase 4: corrected MPC with loose safety guard

The live corrected controller solves both nominal and corrected lifted MPC. It executes the corrected first input only when prediction-error validation, gain-drift, and the loose nominal-cost guard pass. The live accepted fraction is `0.9893`.

## Phase 5: TD3 Markov proposal

TD3 is enabled by default through `run_rl_proposal=True` and proposes normalized Markov correction coordinates in $[-1,1]$. The runner maps the raw action to $z_k$, stores the executed action in replay, and uses the existing closed-loop reward convention. Constrained LS remains the warm-start teacher and safety fallback. The TD3 accepted fraction is `0.3938`, the LS fallback fraction is `0.5466`, and the nominal fallback fraction is `0.0106`. This run pushed `159200` replay transitions and recorded `151200` TD3 critic updates.

## Result summary

| Check                               | Value                  | Pass  |
| ----------------------------------- | ---------------------- | ----- |
| Lifted equivalence max error        | 1.734723475976807e-18  | True  |
| Any positive shadow S_pred fraction | 0.992875               | True  |
| Adaptive LS accepted fraction       | 0.990425               | True  |
| Live corrected accepted fraction    | 0.989325               | True  |
| TD3 accepted action fraction        | 0.39375625             | True  |
| Reward delta mean                   | -5.286738904764695     | False |
| Output MAE delta                    | -0.000978472944889175  | True  |
| Input movement delta                | 2.1630681706200083e-05 | True  |

Result bundle: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/input_data.pkl`

Comparison directory: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260510_144133`

## Smoke-run interpretation

This run has `nFE=160000` and `warm_start_step=8000`. If the run is shorter than or equal to the warm-start boundary, TD3 is configured, checkpointed, and populated with replay data, but post-warm-start TD3 action acceptance and gradient updates are not expected. In that case, accepted Markov moves mainly validate the LS teacher and safety-gated execution path rather than TD3 closed-loop superiority.

## Figures

- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase1_lifted_equivalence.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase2_prediction_score_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase2_candidate_selection_histogram.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase3_z_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase3_prediction_error_improvement.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase4_outputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase4_inputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase4_reward_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase4_acceptance_and_gain_drift.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase5_rl_action_source_and_norm.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260510_123506/phase4_prediction_improvement_vs_reward.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260510_144133/compare_inputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260510_144133/compare_outputs_full.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260510_144133/compare_outputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260510_144133/compare_rewards.png`

## Bugs, inconsistencies, or risks found

- Online prediction-error validation only uses windows whose full prediction horizon has already been observed. This avoids future leakage, but delays correction evidence by $P$ steps.
- The notebook keeps the observer nominal. If Markov corrections become large, prediction and observer dynamics can diverge.
- The full default run uses the existing polymer disturbed setting with `n_tests=200` and `set_points_len=400`, so a complete run may be long.

## Limitations

This is now an RL-active polymer Markov prototype: the TD3 training path is enabled by default, constrained LS remains a safety fallback, and shared RL/MPC modules are reused rather than modified. Closed-loop superiority is not claimed unless a saved full run shows improved or neutral output MAE, controlled input movement, and acceptable reward relative to nominal MPC.

## Next experiment

Run the full disturbed polymer default after the smoke path passes, then compare output-wise MAE, input movement, TD3 accepted fraction, LS fallback fraction, and reward delta against nominal MPC. If TD3 is mostly filtered out, inspect the replay losses and test whether the LS teacher action should be behavior-cloning weighted during early training.

## Remaining uncertainty

The method should be considered successful only if the saved metrics show lifted equivalence, meaningful prediction-error improvement, no material output MAE degradation, and controlled input movement.
