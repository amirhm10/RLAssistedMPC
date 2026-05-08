# Polymer Markov Correction Progress

## Objective

This note tracks the polymer-only prototype for prediction-error-validated Markov-parameter correction of offset-free MPC. The prototype keeps the existing observer and nonlinear polymer plant workflow unchanged, and tests whether finite-horizon input-output Markov corrections reduce recent plant prediction error before they are allowed to affect the MPC action.

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

## Phase 1: lifted-prediction equivalence validation

The notebook validates the lifted absolute input-deviation convention against the state-space rollout. The pass threshold is `max_abs_error < 1e-8`.

Observed max absolute error: `1.734723e-18`.

## Phase 2: shadow prediction-error scoring

Candidate Markov corrections are scored on nominal closed-loop history without executing corrected actions. The fraction of steps with positive best shadow score is `0.7000`.

## Phase 3: adaptive LS Markov correction

The adaptive constrained LS correction estimates $z_k$ with bounds and regularization, then accepts it only when prediction improvement and gain-drift checks pass. The accepted fraction is `0.7000`.

## Phase 4: corrected MPC with loose safety guard

The live corrected controller solves both nominal and corrected lifted MPC. It executes the corrected first input only when prediction-error validation, gain-drift, and the loose nominal-cost guard pass. The live accepted fraction is `0.7000`.

## Phase 5: TD3 Markov proposal

TD3 is enabled by default through `run_rl_proposal=True` and proposes normalized Markov correction coordinates in $[-1,1]$. The runner maps the raw action to $z_k$, stores the executed action in replay, and uses the existing closed-loop reward convention. Constrained LS remains the warm-start teacher and safety fallback. The TD3 accepted fraction is `0.0000`, the LS fallback fraction is `0.0000`, and the nominal fallback fraction is `0.0000`. This run pushed `30` replay transitions and recorded `0` TD3 critic updates.

## Result summary

| Check                               | Value                  | Pass |
| ----------------------------------- | ---------------------- | ---- |
| Lifted equivalence max error        | 1.734723475976807e-18  | True |
| Any positive shadow S_pred fraction | 0.7                    | True |
| Adaptive LS accepted fraction       | 0.7                    | True |
| Live corrected accepted fraction    | 0.7                    | True |
| TD3 accepted action fraction        | 0.0                    | True |
| Reward delta mean                   | 0.08320262290208046    | True |
| Output MAE delta                    | 0.00015205034328885647 | True |
| Input movement delta                | 0.001539019552175308   | True |

Result bundle: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/input_data.pkl`

Comparison directory: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_121849`

## Smoke-run interpretation

This run has `nFE=30` and `warm_start_step=20000`. Because the run is shorter than the warm-start boundary, TD3 is configured, checkpointed, and populated with replay data, but post-warm-start TD3 action acceptance and gradient updates are not expected. The accepted Markov moves in this smoke run validate the LS teacher and safety-gated execution path rather than TD3 closed-loop superiority.

## Figures

- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase1_lifted_equivalence.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase2_prediction_score_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase2_candidate_selection_histogram.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase3_z_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase3_prediction_error_improvement.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase4_outputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase4_inputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase4_reward_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase4_acceptance_and_gain_drift.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase5_rl_action_source_and_norm.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_121846/phase4_prediction_improvement_vs_reward.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_121849/compare_inputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_121849/compare_outputs_full.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_121849/compare_outputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_121849/compare_rewards.png`

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
