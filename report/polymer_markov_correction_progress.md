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

## Phase 5: RL proposal scaffold

The notebook stores requested and executed $z$, fallback status, prediction score, and gain drift. Full TD3 proposal training remains disabled by default through `run_rl_proposal=False`.

## Result summary

This section summarizes the existing smoke run saved at `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/`. The run used `max_steps=30`, so it is an implementation and diagnostic check, not a full scientific validation of the disturbed polymer case.

| Check                               | Value                  | Pass |
| ----------------------------------- | ---------------------- | ---- |
| Lifted equivalence max error        | 1.734723475976807e-18  | True |
| Any positive shadow S_pred fraction | 0.7                    | True |
| Adaptive LS accepted fraction       | 0.7                    | True |
| Live corrected accepted fraction    | 0.7                    | True |
| Reward delta mean                   | 0.08320262290208046    | True |
| Output MAE delta                    | 0.00015205034328885647 | True |
| Input movement delta                | 0.001539019552175308   | True |

Result bundle: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/input_data.pkl`

Comparison directory: `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_115809`

## Analysis of existing smoke run

The smoke run confirms that the lifted prediction machinery is internally consistent and that the prediction-error gate can become active. However, the closed-loop tracking evidence is mixed at the output level. The Markov-corrected controller slightly improved viscosity tracking in scaled and physical units, but slightly worsened temperature tracking. The mean scaled MAE change across the two outputs is small and positive, so the present run should be treated as neutral rather than a demonstrated improvement.

| Output | Nominal MAE, scaled | Markov MAE, scaled | Delta, scaled | Nominal MAE, physical | Markov MAE, physical | Delta, physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| $\eta$ (L/g) | 1.771923 | 1.762272 | -0.009650 | 0.392752 | 0.390613 | -0.002139 |
| $T$ (K) | 0.712346 | 0.722300 | 0.009955 | 0.556725 | 0.564505 | 0.007780 |

The gate was inactive for the first prediction horizon and then accepted every eligible step. This is useful as a proof that the gate is not collapsed to nominal MPC, but it is also a warning sign: for a longer run, the gate should reject some corrections when prediction evidence weakens or gain drift rises. In this smoke run the gain drift stayed at the configured bound of approximately `0.05`, below the `0.10` limit.

| Quantity | Value |
| --- | ---: |
| Steps in run | 30 |
| Eligible steps after prediction horizon | 21 |
| Accepted steps | 21 |
| Overall acceptance fraction | 0.7000 |
| Eligible acceptance fraction | 1.0000 |
| Positive `S_pred` fraction, eligible steps | 1.0000 |
| Mean eligible `S_pred` | 0.04657 |
| Max eligible gain drift | 0.05000 |

The reward delta was positive in the smoke run, with mean Markov-minus-nominal reward of `0.0832`. Because the output-level MAE was split between the two controlled variables and input movement increased slightly, this reward improvement should not be interpreted as sufficient evidence by itself.

| Input | Mean abs move nominal, scaled | Mean abs move Markov, scaled | Delta, scaled | Total variation nominal, physical | Total variation Markov, physical | Delta, physical |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| $Q_c$ (L/h) | 0.037842 | 0.037792 | -0.000050 | 45.410684 | 45.350425 | -0.060259 |
| $Q_m$ (L/h) | 0.107414 | 0.109395 | 0.001981 | 128.897301 | 131.274104 | 2.376803 |

The executed Markov corrections frequently reached the configured bound in this short run. The `y1_u2`, `y2_u1`, and `y2_u2` basis coefficients hit the approximate `0.05` bound on 70 percent of all steps, while `y1_u1` hit it on 63.3 percent of steps. Candidate selection was also concentrated: candidate index `12` accounted for about 90.5 percent of valid shadow selections. These are not failures in a smoke test, but they suggest that the next run should monitor whether LS is fitting at the boundary rather than finding an interior correction.

Additional analysis assets saved in the run folder:

- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_output_metrics_by_output.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_input_movement_by_input.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_acceptance_prediction_summary.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_candidate_selection_counts.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_z_summary.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_summary.json`

## Figures

- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase1_lifted_equivalence.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase2_prediction_score_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase2_candidate_selection_histogram.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase3_z_trace.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase3_prediction_error_improvement.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase4_outputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase4_inputs_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase4_reward_compare.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase4_acceptance_and_gain_drift.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/phase4_prediction_improvement_vs_reward.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_115809/compare_inputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_115809/compare_outputs_full.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_115809/compare_outputs_last_episode.png`
- `C:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/RL_assisted_MPC/Polymer/Results/polymer_markov_compare_disturb/20260508_115809/compare_rewards.png`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_prediction_gate_timeline.png`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_output_mae_by_output.png`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_scaled_error_trace_by_output.png`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_prediction_score_vs_reward_delta.png`
- `Polymer/Results/polymer_markov_corrected_mpc/20260508_115807/analysis_executed_z_trace_with_bounds.png`

## Figure audit

The existing comparison plots provide the standard notebook-style view against MPC, including full output comparison, last-episode output comparison, last-episode input comparison, and reward comparison. The added analysis figures support the mechanism-level claims: the gate timeline shows when prediction evidence activates, the output MAE and error-trace plots show the split between viscosity and temperature, the `S_pred` versus reward scatter checks whether prediction improvement aligns with reward, and the executed-$z$ trace checks whether the correction is operating near its configured bounds.

## Bugs, inconsistencies, or risks found

- Online prediction-error validation only uses windows whose full prediction horizon has already been observed. This avoids future leakage, but delays correction evidence by $P$ steps.
- The notebook keeps the observer nominal. If Markov corrections become large, prediction and observer dynamics can diverge.
- The full default run uses the existing polymer disturbed setting with `n_tests=200` and `set_points_len=400`, so a complete run may be long.
- In the smoke run, every eligible step was accepted. This is acceptable for a smoke test, but a full run should show whether the gate can reject corrections when prediction evidence weakens.
- The output-level result is mixed: viscosity tracking improved slightly, while temperature tracking worsened slightly.
- The executed Markov correction often sat on the configured coefficient bounds, so the present `z_bound=0.05` may be acting as an active limiter rather than a loose prior.

## Limitations

This is a first-pass polymer prototype. It does not modify shared RL/MPC code, does not prove closed-loop superiority, and does not train a TD3 proposal policy by default.

## Next experiment

1. Run a medium-length diagnostic before the full default, for example `max_steps=400` or one setpoint block. Purpose: check whether the 100 percent eligible acceptance rate persists after the first transient. Metric to watch: eligible acceptance fraction should not be interpreted as better unless output-wise MAE improves.
2. Run the full disturbed polymer default with the current `io_pair_gain` basis. Purpose: determine whether the smoke-run behavior scales to the actual `n_tests=200`, `set_points_len=400` setting. Metrics to compare: output-wise MAE, RMSE, reward delta, input movement, accepted fraction, and gain drift.
3. If temperature remains worse while viscosity improves, test `basis_family="input_channel_gain"` with the same bounds. Purpose: reduce output-pair degrees of freedom and see whether the correction becomes less output-specific. Confirming result: both outputs improve or the temperature degradation disappears without excessive input movement.
4. If the full run accepts nearly every eligible correction, add a stricter diagnostic run with larger `s_pred_min` or a minimum relative prediction-improvement threshold. Purpose: verify that the prediction-error gate has real selectivity. Confirming result: accepted fraction decreases while tracking does not degrade materially.
5. If LS traces hit bounds frequently, reduce `z_bound` or add a step-to-step rate limit on $z$. Purpose: distinguish useful Markov adaptation from bound-saturated fitting. Confirming result: lower bound-hit fraction with similar or better output-wise MAE.

## Remaining uncertainty

The method should be considered successful only if the full-length saved metrics show lifted equivalence, meaningful prediction-error improvement, no material output MAE degradation on either controlled output, and controlled input movement. The existing 30-step smoke run is useful implementation evidence, but it is not enough to claim controller improvement.
