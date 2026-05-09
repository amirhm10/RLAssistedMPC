# Polymer Markov Latest-Run Analysis

Date: 2026-05-09

Latest unified run analyzed:
`Polymer/Results/td3_markov_disturb/20260509_023119/input_data.pkl`

Latest unified comparison directory:
`Polymer/Results/disturb_compare_td3_markov/20260509_023133/`

Previous prototype run analyzed:
`Polymer/Results/polymer_markov_corrected_mpc/20260508_123902/input_data.pkl`

Canonical nominal baseline used by the latest unified workflow:
`Polymer/Data/mpc_results_dist.pickle`

Generated analysis artifacts:
`report/figures/polymer_markov_latest_run_20260509/`

## Objective

This note answers four questions:

1. What the Markov-corrected RL-assisted MPC method is, step by step, in mathematical form.
2. What the latest unified run actually achieved.
3. Why the previous prototype looked better near the end.
4. Which code and logic changes separate the previous prototype from the latest unified implementation.

## Files inspected

- `polymer_markov_corrected_mpc_unified.ipynb`
- `utils/markov_runner.py`
- `utils/rewards.py`
- `utils/plotting.py`
- `utils/plotting_core.py`
- `systems/polymer/notebook_params.py`
- `report/polymer_markov_correction_progress.md`
- `report/scripts/generate_polymer_markov_correction_assets.py`
- Prototype reference from git history:
  `git show ce34e86^:report/scripts/generate_polymer_markov_correction_assets.py`

## Methodology

### 1. Scaled deviation coordinates

The unified polymer workflow uses scaled deviation variables around the steady state:

$$ \tilde{u}_k = S_u(u_k) - \bar{u}, \qquad \tilde{y}_k = S_y(y_k) - \bar{y}. $$

Here, $S_u(\cdot)$ and $S_y(\cdot)$ are the min-max scalers from the identified polymer dataset, and $(\bar{u}, \bar{y})$ are the steady-state scaled inputs and outputs.

### 2. Offset-free prediction model

The controller uses the augmented linear offset-free model:

$$ x_{k+1} = A_a x_k + B_a \tilde{u}_k + L(\tilde{y}_k - C_a x_k), \qquad \hat{y}_k = C_a x_k. $$

The observer gain $L$ is computed from the unified observer helper. The nonlinear plant that is actually stepped online is still the polymer CSTR model.

### 3. Nominal lifted MPC

For prediction horizon $P$ and control horizon $M$, the nominal lifted prediction is:

$$ Y_{k|k}^{\mathrm{nom}} = F x_k + G_0 U_{k|k}, $$

with

$$ U_{k|k} = [\tilde{u}_{k|k}^\top,\tilde{u}_{k+1|k}^\top,\dots,\tilde{u}_{k+M-1|k}^\top]^\top. $$

The nominal MPC problem is:

$$ \min_{U_{k|k}} \sum_{i=1}^{P} \lVert y_{k+i|k} - y_{sp,k} \rVert_Q^2 + \sum_{i=0}^{M-1} \lVert \Delta u_{k+i|k} \rVert_R^2. $$

In the latest unified implementation, the nominal step uses the shared `MpcSolverGeneral.mpc_opt_fun(...)` path.

### 4. Markov correction parameterization

The Markov correction does not replace the MPC structure. It perturbs the lifted Toeplitz matrix through a bounded basis expansion:

$$ G(z) = G_0 + \sum_{j=1}^{r} z_j \Delta G_j, \qquad z_j \in [-z_{\max}, z_{\max}]. $$

For both the previous prototype and the latest unified run:

- `basis_family = "io_pair_gain"`
- `z_max = 0.05`
- `prediction_window = 20`
- `lambda_z = 10^{-3}`
- `s_pred_min = 10^{-6}`
- `gain_drift_max = 0.1`

So the main Markov hyperparameters did not change across these two runs.

### 5. Prediction-improvement score

The correction is screened through a lifted prediction score over a recent window $\mathcal{T}_k$:

$$ S_{\mathrm{pred}}(z) = \sum_{\tau \in \mathcal{T}_k} \left( \lVert W_y e_{\tau}^{\mathrm{nom}} \rVert_2^2 - \lVert W_y e_{\tau}^{z} \rVert_2^2 \right) - \lambda_z \lVert z \rVert_2^2. $$

Positive score means the corrected lifted model explains the recent measured trajectories better than the nominal lifted model after the $z$-penalty is applied.

### 6. LS teacher

The constrained least-squares teacher chooses:

$$ z_k^{\mathrm{LS}} = \arg\min_{z \in [-z_{\max}, z_{\max}]^r} \sum_{\tau \in \mathcal{T}_k} \lVert W_y e_{\tau}^{z} \rVert_2^2 + \lambda_z \lVert z \rVert_2^2. $$

This teacher is used during warm start and as the fallback action when TD3 fails the filter.

### 7. TD3 state and action

The TD3 supervisor sees the state

$$ s_k = \left[ x_k^\top,\; (\tilde{y}_k - y_{sp,k})^\top,\; (\tilde{y}_k - \hat{y}_k)^\top,\; \tilde{u}_{k-1}^\top,\; z_{k-1}^\top,\; (z_k^{\mathrm{LS}})^\top,\; S_{\mathrm{pred}}(z_k^{\mathrm{LS}}),\; d_k^{\mathrm{LS}} \right]^\top, $$

where the gain-drift metric is

$$ d_k = \frac{\lVert G(z_k) - G_0 \rVert_F}{\lVert G_0 \rVert_F + \varepsilon}. $$

The TD3 actor outputs a normalized action:

$$ a_k \in [-1,1]^r, \qquad z_k^{\mathrm{TD3}} = z_{\max} a_k. $$

### 8. Safety filter and fallback logic

The TD3 proposal is accepted only if all of the following hold:

- the nominal solve succeeds
- the corrected solve succeeds
- $S_{\mathrm{pred}}(z_k^{\mathrm{TD3}}) > s_{\min}$
- $d_k^{\mathrm{TD3}} \le d_{\max}$
- the corrected candidate passes the loose nominal-cost guard

If TD3 fails, the controller falls back in this order:

1. accepted LS teacher correction
2. nominal MPC

### 9. Reward definitions

This is the biggest code-level difference between the previous prototype and the latest unified run.

The previous prototype used a legacy reward of the form

$$ r_k^{\mathrm{legacy}} = - e_k^\top Q e_k - \Delta u_k^\top R \Delta u_k + 1000 e^{-\bar{p}_k} \mathbf{1}\{ p_{k,i} \le 5\% \ \forall i \}, $$

where $p_{k,i}$ is the percentage tracking error relative to the setpoint-like denominator used in the prototype script.

The latest unified run uses the shared relative-band reward from `utils/rewards.py`. The main ingredients are:

$$ b_i(y_{sp}) = \max(k_i^{\mathrm{rel}} |y_{sp,i}|,\; b_i^{\mathrm{floor}}), \qquad s_i = \sigma \left( \frac{b_i - |e_i|}{\tau b_i} \right), $$

$$ w_{\mathrm{in}} = \left( \prod_i s_i \right)^{1/n_y}, $$

and

$$ r_k^{\mathrm{unified}} = \alpha \left( -J_{\mathrm{quad}} - J_{\Delta u} - J_{\mathrm{lin,out}} - J_{\mathrm{lin,in}} + J_{\mathrm{bonus}} \right). $$

This newer reward is smoother near the setpoint band and is exactly the same reward used by the shared unified polymer RL notebooks.

## Latest unified run results

The latest run is not a failure, but it is almost a nominal-MPC-equivalent controller rather than a strongly outperforming controller.

### Key metrics

| Quantity | Latest unified run |
| --- | ---: |
| Mean reward delta vs canonical MPC | `-0.0259` |
| Last-20-episode reward delta | `-0.0295` |
| Fraction of episodes with better reward than MPC | `0.0450` |
| TD3 accepted fraction | `0.4808` |
| LS fallback fraction | `0.4524` |
| Nominal fallback fraction | `0.0190` |
| Any-coordinate `z` saturation fraction | `0.8067` |
| Input-movement delta vs MPC | `-0.0180` |
| Last-20 input-movement delta vs MPC | `-0.0261` |

Interpretation:

- The latest controller is slightly worse than canonical MPC on the unified reward almost everywhere.
- It is smoother than canonical MPC in terms of input movement.
- It still uses TD3 often, so the near-nominal behavior is not because TD3 never activates.
- It remains heavily constrained by the `z` box, although less than the previous prototype.

### Output tracking

Using mean absolute tracking error in the scaled deviation coordinates:

| Output metric | Full run delta | Last-20 delta |
| --- | ---: | ---: |
| Viscosity MAE, Markov minus MPC | `-0.0035` | `-0.0035` |
| Temperature MAE, Markov minus MPC | `0.0009` | `0.0026` |

Interpretation:

- The latest controller slightly improves viscosity tracking.
- It slightly worsens temperature tracking, especially in the last 20 episodes.
- This is why it looks almost the same as nominal MPC in the output plots.

### Figures

The latest run reward delta stays close to zero and usually slightly below zero:

![Reward delta compare](figures/polymer_markov_latest_run_20260509/reward_delta_compare.png)

The final-episode output traces confirm that the latest unified controller stays close to the nominal response:

![Tail output compare](figures/polymer_markov_latest_run_20260509/tail_output_compare.png)

The action mix shows that TD3 and LS are both active, but the resulting corrections remain conservative:

![Action mix and saturation](figures/polymer_markov_latest_run_20260509/action_mix_and_saturation.png)

The coordinate-wise $z$ usage is still strong, but not as saturated as in the prototype:

![Z usage compare](figures/polymer_markov_latest_run_20260509/z_usage_compare.png)

## Why the previous prototype looked better near the end

The short version is:

- under the prototype's own legacy reward, the previous run looked clearly better at the end
- under the unified reward, that late advantage disappears
- and the previous run was not compared against the same nominal reference used now

### Previous prototype metrics under its original reward

| Quantity | Previous prototype |
| --- | ---: |
| Mean reward delta vs its own nominal MPC | `-7.0366` |
| Last-20-episode reward delta | `25.7074` |
| Fraction of episodes with better reward than nominal | `0.1850` |
| Last-20 fraction with better reward than nominal | `0.9500` |
| TD3 accepted fraction | `0.3169` |
| LS fallback fraction | `0.6229` |
| Nominal fallback fraction | `0.0112` |
| Any-coordinate `z` saturation fraction | `0.9519` |

So the old prototype really did show a strong late reward advantage in its own metric, but that came with much heavier saturation and much more LS usage.

### Fair rescoring with the unified reward

When the previous prototype trajectories are rescored with the latest unified reward, the picture changes:

| Quantity | Previous prototype, rescored by unified reward |
| --- | ---: |
| Mean reward delta vs previous nominal | `0.0029` |
| Last-20 reward delta vs previous nominal | `-0.0258` |
| Fraction of episodes with better reward than previous nominal | `0.5600` |
| Last-20 fraction with better reward than previous nominal | `0.2000` |

This is the most important comparison in the whole note. The prototype's large late reward win is not robust to the reward definition. Under the shared unified reward, that late superiority disappears.

The figure below makes that point directly:

![Reward rescoring compare](figures/polymer_markov_latest_run_20260509/reward_rescoring_compare.png)

### What the prototype was actually doing at the end

The previous prototype's last-20-episode window had:

- reward advantage under the legacy reward
- higher LS fallback than the latest unified run
- larger correction magnitudes
- more frequent boundary-hitting behavior earlier in training
- slightly higher input movement than its nominal comparator

That makes the prototype's late-stage behavior look more aggressive, not necessarily more correct.

## What changed between the previous prototype and the latest unified implementation

This section separates cosmetic migration changes from the changes that actually affect the conclusion.

### Changes that matter scientifically

| Area | Previous prototype | Latest unified implementation | Why it matters |
| --- | --- | --- | --- |
| Online reward | Local `legacy_mpc_reward(...)` with a large exponential inside-band bonus | Shared `make_reward_fn_relative_QR(...)` reward | This is the biggest driver of the behavioral difference and of the changed late-stage interpretation |
| Comparison reward | Local `make_compare_reward_fn(...)` | Exactly the same shared reward function passed into the runner and into `compare_mpc_rl_from_dirs(...)` | Removes reward inconsistency between online training and offline comparison |
| Nominal comparison reference | Internally rerun nominal trajectory saved inside the prototype bundle | Canonical saved baseline pickle `Polymer/Data/mpc_results_dist.pickle` | Makes the latest comparison stricter and more consistent with the unified notebooks |
| Default experiment path | Prototype script owned config, rollout, report generation, and comparison | Notebook + shared runner + shared plotting path | Reduces experiment drift and makes the Markov method follow the same rules as the other unified polymer notebooks |

### Changes that mainly improve consistency

| Area | Previous prototype | Latest unified implementation | Likely effect |
| --- | --- | --- | --- |
| Nominal step solve | Prototype-local control flow | Shared `MpcSolverGeneral.mpc_opt_fun(...)` solve pattern | Aligns Markov nominal behavior with the standard polymer notebooks |
| Disturbance stepping | Prototype-local runtime ownership | Shared disturbance stepping helper path | Better consistency with canonical disturbed polymer runs |
| Observer and action-selection flow | Prototype-local runtime ownership | Shared observer/action helper path | Better alignment with unified RL notebooks |
| Plotting | Report-local figures plus legacy comparison wrapper | Shared RL plotter + shared compare plotter | Better auditability, but not the main cause of the control difference |

### One more important comparison issue

The previous prototype and the latest unified workflow do not even use the same nominal reference trajectory.

The prototype's internally rerun nominal MPC and the canonical baseline differ by:

| Quantity | Difference |
| --- | ---: |
| Shared-reward delta, old nominal minus canonical baseline, full run | `0.3629` |
| Shared-reward delta, old nominal minus canonical baseline, last 20 | `0.6085` |
| Max absolute output trajectory difference | `0.9675` |
| Max absolute input trajectory difference | `265.18` |

So a visual comparison between "old Markov vs old nominal" and "new Markov vs canonical nominal" is not an apples-to-apples comparison.

## Why the latest run looks almost nominal while the prototype did not

The best evidence from this analysis is:

1. The Markov hyperparameters did not materially change. The `z` range is still `[-0.05, 0.05]`, so the behavior change is not because the latest code suddenly shrank the correction authority.
2. The reward definition changed a lot. The prototype's late reward win is largely a consequence of the old reward construction and does not survive rescoring with the unified reward.
3. The comparison baseline changed. The latest workflow is compared against the canonical baseline pickle, while the old workflow compared against its own nominal rerun.
4. The latest controller is more conservative in its executed corrections. TD3 is accepted more often than before, but with smaller average $z$ magnitude and much lower overall saturation than the previous prototype.
5. The latest controller trades away some aggressiveness for smoother inputs. That is why it looks similar to nominal MPC rather than decisively better.

So the latest run is best interpreted as a conservative, nearly nominal, smoother-input Markov corrector, not as a broken controller and not as clear proof of superiority.

## Is `z` range the problem?

Not as the main explanation for the current difference between the two implementations.

Evidence:

- both runs used `z_max = 0.05`
- both runs hit the `z` wall often
- the prototype was actually more saturated than the latest unified run

So `z_max` is still a valid tuning target, but it is not the main reason the latest implementation looks close to nominal while the prototype looked more different.

The stronger explanation is:

- reward definition
- nominal comparison reference
- unified runtime alignment

## Recommended next experiments

1. Run a unified ablation with the old legacy reward but the new unified runtime. This isolates reward logic from runner logic.
2. Sweep `z_max` over `0.02`, `0.03`, `0.05`, and `0.07` under the unified reward.
3. Sweep `lambda_z` together with `z_max`, because a lower-saturation policy may need both changes together.
4. Re-score all candidate runs with the same unified reward and compare only against the canonical baseline pickle.
5. Keep reporting full-run and last-20 metrics together. The prototype looked good at the end while still being worse on the full run under its own metric.

## Artifacts generated for this note

- `report/figures/polymer_markov_latest_run_20260509/reward_delta_compare.png`
- `report/figures/polymer_markov_latest_run_20260509/tail_output_compare.png`
- `report/figures/polymer_markov_latest_run_20260509/action_mix_and_saturation.png`
- `report/figures/polymer_markov_latest_run_20260509/z_usage_compare.png`
- `report/figures/polymer_markov_latest_run_20260509/reward_rescoring_compare.png`
- `report/figures/polymer_markov_latest_run_20260509/comparison_summary.csv`
- `report/figures/polymer_markov_latest_run_20260509/window_metrics.csv`
- `report/figures/polymer_markov_latest_run_20260509/baseline_reference_difference.csv`
