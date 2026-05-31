# Distillation Five-Runner Analysis After Temperature Reward Increase And Probation Removal

Date: 2026-05-31

## Objective

This report analyzes the latest five active distillation runners after two major default changes:

- Reward collapse probation was disabled for weights, standard horizon, dueling horizon, residual, and Markov.
- The temperature tracking reward weight was increased from `Q2 = 5.0e3` to `Q2 = 2.0e4`, with `Q1 = 3.7e4` unchanged.

The goal is to explain not only which runner won, but why each runner behaved the way it did, what the safety layers actually did during the run, and what the next experiment should test.

## Files Inspected

Configuration and implementation:

- `systems/distillation/config.py`
- `systems/distillation/notebook_params.py`
- `utils/rewards.py`
- `utils/weights_runner.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/residual_runner.py`
- `utils/markov_runner.py`

Latest result bundles:

- `Distillation/Results/distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260530_220604/input_data.pkl`
- `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260530_223346/input_data.pkl`
- `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260530_225843/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260530_230956/input_data.pkl`
- `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified/20260530_231105/input_data.pkl`
- `Distillation/Results/distillation_compare_residual_td3_disturb_fluctuation/20260530_231118/input_data.pkl`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`

Previous Markov comparison bundles:

- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified/20260518_091937/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260518_184548/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260529_220507/input_data.pkl`

Generated analysis artifacts:

- `report/scripts/analyze_distillation_post_reward_no_probation_20260531.py`
- `report/figures/distillation_post_reward_no_probation_20260531/current_summary_metrics.csv`
- `report/figures/distillation_post_reward_no_probation_20260531/current_vs_baseline_metrics.csv`
- `report/figures/distillation_post_reward_no_probation_20260531/current_vs_previous_metrics.csv`
- `report/figures/distillation_post_reward_no_probation_20260531/current_horizon_pair_counts.csv`
- `report/figures/distillation_post_reward_no_probation_20260531/markov_reference_z_metrics.csv`

## Figure Evidence Map

The analysis is supported by the figure set generated from saved bundles by `report/scripts/analyze_distillation_post_reward_no_probation_20260531.py`. These figures are deliberately diagnostic rather than polished presentation figures. Their job is to make the mechanism visible.

### Figure 1: Reward Trajectories

This figure supports the main ranking and shows the transient/tail split. Residual is unstable early but becomes dominant late. Markov peaks early and then collapses. Weights improves late without matching residual.

![Reward trajectories](figures/distillation_post_reward_no_probation_20260531/fig_reward_trajectories.png)

### Figure 2: Tail Reward Ranking

The tail ranking makes the final ordering explicit: residual, weights, OF-MPC, standard horizon, dueling horizon, and Markov.

![Tail reward ranking](figures/distillation_post_reward_no_probation_20260531/fig_tail_reward_ranking.png)

### Figure 3: Tail Physical Tracking Errors

This is the key evidence that residual is not only winning the scalar reward. It also has the best temperature tracking and improves composition relative to OF-MPC. Weights improves temperature but worsens composition.

![Tail physical tracking errors](figures/distillation_post_reward_no_probation_20260531/fig_tail_tracking_errors.png)

### Figure 4: Safety Intervention Rates

This figure supports the claim that reward probation and most tail interventions are not responsible for the late results. The only persistent tail intervention is Markov z projection. The other runners are mostly running with live learned authority in the tail.

![Safety intervention rates](figures/distillation_post_reward_no_probation_20260531/fig_safety_intervention_rates.png)

### Figure 5: Continuous Action Diagnostics

This figure connects each continuous runner to its action mechanism. Weights uses the widened multiplier space, residual corrections remain small in norm after release, and Markov z norm is pinned at the active vector cap.

![Continuous action diagnostics](figures/distillation_post_reward_no_probation_20260531/fig_continuous_action_norms.png)

### Figure 6: Horizon Pair Usage

This figure supports the horizon diagnosis. Standard DDQN spreads tail decisions across many horizon pairs, while dueling is more concentrated but still chooses pairs that do not improve the temperature-sensitive objective.

![Horizon tail pair usage](figures/distillation_post_reward_no_probation_20260531/fig_horizon_tail_pair_usage.png)

### Figure 7: Current Versus Previous Tail Reward

This figure separates the effect of the latest settings from the previous batch. Residual improves further, while horizons and Markov worsen substantially under the no-probation, higher-temperature-reward setup.

![Current versus previous tail reward](figures/distillation_post_reward_no_probation_20260531/fig_current_vs_previous_tail20.png)

### Figure 8: Residual Early Release Zoom

This figure supports the residual next-step recommendation. The residual policy has the best late result, but the early release window still needs protection. The lower panel shows that the safe-start mechanisms are active around release, but the reward crash still occurs.

![Residual early release zoom](figures/distillation_post_reward_no_probation_20260531/fig_residual_release_zoom.png)

### Figure 9: Markov z Mechanism

This is the strongest visual evidence for the Markov diagnosis. Earlier successful Markov runs used varied z directions. The latest run drives a nearly fixed corner action and z-safety projects it to the vector norm cap. This explains the observed `abs(z_i) = 0.03` behavior and shows why widening z is not the right next move.

![Markov z mechanism](figures/distillation_post_reward_no_probation_20260531/fig_markov_z_mechanism.png)

## Method Snapshot

The distillation controlled outputs are tray-24 ethane composition and tray-85 temperature. The manipulated inputs are reflux flow and reboiler duty. All five RL runners sit above the same offset-free MPC baseline and are evaluated on the disturbance fluctuation profile.

The reward used in the latest batch is the relative-band reward in `utils/rewards.py`. For scaled output error `e`, scaled input movement `du`, and physical setpoint `y_sp`, each output receives a physical tolerance band

$$ b_i = \max(k_{\mathrm{rel},i} |y_{\mathrm{sp},i}|, b_{\mathrm{floor},i}). $$

The reward combines quadratic tracking cost, move cost, linear inside/outside band terms, and an in-band bonus:

$$ r = (-(J_e + J_u + J_{\mathrm{out}} + J_{\mathrm{in}}) + B_{\mathrm{in}}) s_r. $$

The key current parameters are:

- `Q_diag = [3.7e4, 2.0e4]`
- `R_diag = [2.5e3, 2.5e3]`
- `k_rel = [0.3, 0.01]`
- `band_floor_phys = [0.003, 0.2]`
- `reward_scale = 1.0`

This means the latest scalar reward is much more temperature-sensitive than the previous batch. Comparisons against the current OF-MPC baseline are fair because the baseline reward was read from the latest compare bundle. Comparisons against May 29 reward values are useful for direction, but the scalar reward itself is not a clean apples-to-apples quantity because the reward weight changed.

## Safety State In The Latest Batch

Reward probation is inactive in every latest bundle. All trigger counts are zero, and all tail probation-active fractions are zero.

The remaining safety layers stayed active:

- Weights: identity warm start, post-warm handoff, multiplier deviation cap ramp, identity fallback on invalid action or solve failure, and shadow identity-MPC diagnostics.
- Horizon and dueling: default `(6, 3)` warm start, wider 263-action grid, post-warm release filter, projection to allowed release sets, and shadow default-MPC diagnostics.
- Residual: zero residual warm start, post-warm handoff, residual cap ramp, physical headroom clipping, nonfinite zero fallback, and shadow rho/deadband/direction logs. Rho remains inactive in execution and state.
- Markov: TD3-priority fallback, LS and nominal emergency fallback, z-safety coordinate caps, z vector-norm cap, BC handoff, and diagnostic-only release gate.

The important practical finding is that tail safety intervention is mostly inactive except for Markov z-safety:

| Runner | Tail active intervention evidence |
| --- | --- |
| Weights | Tail cap projection `0.000`, fallback `0.000`, TD3 accepted `1.000` |
| Horizon DDQN | Tail projection `0.000`, cooldown `0.000`, accepted `1.000` |
| Dueling horizon | Tail projection `0.000`, cooldown `0.000`, accepted `1.000` |
| Residual TD3 | Tail residual cap projection `0.000`, zero fallback `0.000` |
| Markov TD3 | Tail fallback `0.000`, z projection `1.000`, vector projection `1.000` |

So the poor late results for horizons and Markov are not caused by reward cooldown. The agents are mostly running with live authority.

## Current Ranking

Tail metrics are computed over the final 20 subepisodes.

| Rank | Runner | Tail-20 reward | Final reward | First-20 min reward | Comp MAE | Temp MAE | Outside-band frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TD3 residual | `27.890` | `31.970` | `-31.460` | `0.001087` | `0.0675` | `0.0900` |
| 2 | TD3 weights | `9.794` | `13.857` | `-3.787` | `0.002648` | `0.1306` | `0.1936` |
| 3 | OF-MPC | `6.391` | `6.926` | `-2.477` | `0.001545` | `0.1921` | `0.1633` |
| 4 | Horizon DDQN | `-0.862` | `4.372` | `-32.235` | `0.001160` | `0.2360` | `0.1623` |
| 5 | Dueling horizon | `-0.908` | `-9.342` | `-58.375` | `0.001338` | `0.2353` | `0.1698` |
| 6 | TD3 Markov | `-24.383` | `-19.762` | `-18.784` | `0.005026` | `0.4605` | `0.7438` |

The result is very sharp:

- Residual is the only runner that clearly beats OF-MPC on both outputs and scalar reward.
- Weights beats OF-MPC in scalar reward by improving temperature, but it sacrifices composition and outside-band frequency.
- Horizon and dueling do not improve the new temperature-sensitive objective.
- Markov is currently the main failure mode. It is not nominal copy-paste. It is live TD3 choosing a bad saturated direction.

## Current Versus OF-MPC

| Runner | Tail reward delta | Final reward delta | Comp MAE delta | Temp MAE delta | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| TD3 weights | `+3.403` | `+6.931` | `+71.3%` | `-32.0%` | Reward win comes from temperature improvement, not balanced tracking. |
| Horizon DDQN | `-7.253` | `-2.554` | `-24.9%` | `+22.9%` | Composition improves, but the new temperature objective exposes worse thermal behavior. |
| Dueling horizon | `-7.299` | `-16.268` | `-13.4%` | `+22.5%` | Similar to horizon, with worse final scalar reward. |
| TD3 residual | `+21.499` | `+25.044` | `-29.6%` | `-64.9%` | Strong true improvement. It improves both outputs and reward. |
| TD3 Markov | `-30.774` | `-26.688` | `+225.3%` | `+139.7%` | Large degradation on both outputs. |

The reward change achieved its purpose in one sense: temperature matters much more now. But it also revealed that not all runners can exploit that signal without damaging composition or stability.

## Runner-by-Runner Diagnosis

### TD3 Residual

Residual is the current best runner. It reached tail-20 reward `27.890`, final reward `31.970`, temperature MAE `0.0675`, and composition MAE `0.001087`. Relative to OF-MPC, it reduced tail band-normalized MAE by `57.1%` and outside-band fraction by `44.9%`.

The mechanism is consistent with what we hoped residual would do. The nominal MPC remains the structural controller, while TD3 learns a small additive correction. In the tail, the executed residual norm mean is only `0.00629`, with q95 `0.0265`. The residual cap is not clipping in the tail, which means the late improvement is not an artifact of the safety layer. It is the learned residual policy.

The safety logs are also informative:

- Tail `td3_accepted` fraction is effectively zero because almost every TD3 residual is counted as `projected_td3`.
- Tail cap projection flag is `0.000`, so that source code name mostly reflects the post-safety path rather than active clipping in the final window.
- Early residual cap projection is `0.350`, so the safe-start layer did intervene during release.
- Tail zero fallback is `0.000`.
- Shadow rho projection is `1.000`, with mean shadow rho effect `0.376`. This supports keeping rho inactive for now, because the rho layer would have changed almost every tail action.

The main problem is early release. The first-20 minimum reward is `-31.460`, much worse than OF-MPC's `-2.477`. This means the current safe-start stack protects the tail objective but does not sufficiently protect the first release window.

Conclusion: keep residual as the primary research lead. Do not add active rho yet. The next residual experiment should target early-release safety only.

### TD3 Weights

Weights is second by scalar reward and first among the non-residual families. It beats OF-MPC by `+3.403` tail reward and `+6.931` final reward. The reason is temperature: tail temperature MAE is `0.1306`, a `32.0%` improvement over OF-MPC.

But the improvement is not balanced. Composition MAE is `0.002648`, which is `71.3%` worse than OF-MPC, and the outside-band fraction increases by `18.6%`.

The weight logs show why:

- Tail mean multipliers are approximately `[Q1, Q2, R1, R2] = [1.012, 1.739, 1.414, 1.557]`.
- The tail boundary fraction is `0.504`, so the widened `[0.5, 2.5]` range is being used aggressively.
- Tail TD3 accepted fraction is `1.000`.
- Tail cap projection and fallback are both `0.000`.
- Early cap projection is `0.401`, so the safe-start layer did work during release but stopped influencing the final policy.

The policy is therefore not being suppressed by safety in the tail. It has learned that, under the new reward, temperature performance is worth paying for with composition error and more frequent band exits.

Conclusion: weights is promising, but it now needs a balance guard. It should not be made safer by returning to identity more often. It should be made more balanced by discouraging boundary-heavy multipliers or adding a composition-protection term.

### Standard Horizon DDQN

Standard horizon performs poorly under the new reward, with tail-20 reward `-0.862` and first-20 minimum `-32.235`. This is not due to release projection in the tail:

- Tail accepted fraction is `1.000`.
- Tail projection is `0.000`.
- Tail cooldown is `0.000`.
- Early projection is only `0.023`.

The action distribution shows non-convergence:

- Tail mean prediction horizon is `14.67`.
- Tail mean control horizon is `6.58`.
- Tail unique horizon pairs are `183` out of `263`.
- Tail entropy is `4.60`.
- The most common pair, `(4, 1)`, appears only `3.5%` of the tail.
- Tail switch-step fraction is `0.230`.

This is a very broad policy distribution, not a learned stable horizon schedule. The wider grid gave the agent more options than it can use reliably in this training budget.

Interestingly, composition MAE improves by `24.9%` relative to OF-MPC, but temperature MAE worsens by `22.9%`. So the horizon mechanism is not useless. It is just optimizing the wrong trade-off under the new temperature-sensitive reward.

Conclusion: the standard horizon runner should not keep the full `263`-action space as the next default experiment. It needs either a medium grid, a dwell constraint, or stronger value confidence before switching.

### Dueling Horizon DDQN

Dueling horizon is not better than standard horizon in this batch. It has tail-20 reward `-0.908`, final reward `-9.342`, and the worst early crash at `-58.375`.

The dueling policy is more concentrated than standard horizon:

- Tail unique horizon pairs are `139`, not `183`.
- Tail entropy is `3.43`, not `4.60`.
- The top pair `(7, 1)` appears `16.3%` of the tail.
- Top tail pairs include `(7, 1)`, `(6, 4)`, `(9, 9)`, `(18, 3)`, and `(2, 2)`.

This means the dueling architecture is producing sharper action preferences, but the preferences are not translating into better temperature tracking. The selected pairs include low-control and aggressive combinations that can change the closed-loop transient without consistently improving thermal settling.

Conclusion: dueling is a better candidate than standard horizon if we keep one horizon architecture, but the next horizon experiment should narrow or phase the action space. A sharper Q-decomposition does not solve an over-wide action set by itself.

### TD3 Markov

Markov is the clearest failure. Tail-20 reward is `-24.383`, tail temperature MAE is `0.4605`, and outside-band fraction is `0.7438`.

The failure is not because TD3 was blocked into nominal behavior:

- Tail TD3 accepted fraction is `1.000`.
- Tail fallback fraction is `0.000`.
- Tail LS fallback fraction is `0.000`.
- Tail nominal fallback fraction is `0.000`.

The failure is also not because the configured `z_bound = 0.04` became `0.01`. The current tail is saturating the vector trust region:

- Raw TD3 action tail is essentially `[-1, -1, -1, +1]`.
- Before vector projection, the requested z norm is `0.080`.
- After projection, the executed z norm is exactly `0.060`.
- Executed z is approximately `[-0.03, -0.03, -0.03, +0.03]`.
- Coordinate q95 abs z is `0.030`.
- z projection and vector projection are both active for `100%` of tail steps.

This explains the apparent `0.03` ceiling. With four coordinates at `0.04`, the vector norm would be `0.08`. The active vector cap is `0.06`, so the projection scale is `0.75`, and each coordinate becomes `0.03`. The bound is not stuck at `0.01`. The actor is saturating all four coordinates, and the vector cap is scaling that corner.

The more serious issue is candidate quality:

- Tail requested cost-guard pass fraction is only `0.0163`.
- Tail requested prediction-score mean is `-0.0867`.
- Tail requested cost margin mean is `0.00162`.
- Since `score_hard_min = None`, negative prediction scores are logged but not used to block TD3.
- The broad full-phase cost caps permit many candidates that the old diagnostic cost guard considers poor.

Previous successful Markov runs looked very different:

| Run | Tail-20 reward | Markov setup | Tail z behavior |
| --- | ---: | --- | --- |
| 20260518 TD3-only no safeguard | `21.978` | `force_td3_execute=True`, no priority fallback, no z-safety | z varied, mean near zero, q95 abs about `0.05` |
| 20260518 priority fallback | `21.247` | priority fallback enabled, BC disabled, reward probation enabled | z varied, no vector projection saturation |
| 20260529 restored safety | `-8.673` | priority fallback, BC active, z-safety, reward probation enabled | z held near `[+0.01, +0.01, -0.01, -0.01]` |
| 20260530 current | `-24.383` | priority fallback, BC active, z-safety, reward probation disabled | z saturated at `[-0.03, -0.03, -0.03, +0.03]` |

The key change from the successful runs is not simply larger or smaller z. It is loss of useful action direction. The successful runs used varied Markov corrections. The current run drives a nearly constant corner action.

There is also a configuration risk: the current Markov bundle confirms BC is enabled with `target_mode = ls_action`, `active_subepisodes = 10`, and `start_after_warm_start = False`. That means warm-start BC is active toward LS action. This does not explain the tail corner by itself, but it differs from the cleanest successful priority-fallback run, where BC was disabled.

Conclusion: do not widen z. The current actor already asks for the largest possible direction. The next Markov experiment must reject or shrink bad directions based on prediction score, cost guard, or direction-risk diagnostics.

## What Changed Relative To Earlier Strong Markov Runs

The earlier successful Markov behavior came from a different regime:

1. The reward was less temperature-dominant. The latest `Q2` is four times larger than the immediate previous default.
2. Reward probation is now disabled. This removed the old `cooldown_scale = 0.25`, so the actor can express full post-ramp authority.
3. z-safety now uses both coordinate caps and a vector cap. This is why saturated four-coordinate actions become about `0.03` per coordinate.
4. The latest actor saturates to a fixed action corner. Older successful runs had varied z signals with small means.
5. The priority fallback does not currently use prediction score as a hard gate because `score_hard_min = None`.
6. Current Markov BC is active during warm start toward LS action, while one of the strong priority-fallback runs had BC disabled.

So the current Markov problem is not that the safety layer is too conservative. It is that the safety layer is mostly a magnitude projector. It clips a bad direction into a smaller bad direction.

## Bugs, Inconsistencies, And Risks

1. Markov BC configuration should be revisited.
   The current bundle shows active LS-action BC during warm start. This is not the same as a live hard gate, but it may bias the actor and should be separated from the z-safety experiment.

2. Previous-run scalar reward comparisons must be handled carefully.
   The latest reward has `Q2 = 2.0e4`, while earlier runs used lower temperature weighting. Tracking metrics are comparable. Stored scalar rewards across reward definitions are not fully comparable.

3. Horizon action space is probably too wide for the current training budget.
   Standard horizon used `183` unique pairs in the tail. That is not healthy exploitation.

4. Residual safe start is active but not sufficient.
   Early cap projection occurred, but the first release crash remained severe. This means magnitude clipping alone is not enough for residual release.

5. Weight widening is useful but boundary-heavy.
   Tail boundary usage is about `50%`. This is a sign that the widened range is being used, but also a sign that the policy may be overfitting the scalar reward trade-off.

## Recommended Next Experiment

I recommend the next experiment be a targeted five-runner correction, not another blind full repeat.

### Priority 1: residual release-only protection

Purpose: keep the excellent late residual performance while reducing the first-20 crash.

Change:

- Keep rho inactive.
- Keep reward probation disabled.
- Keep the final residual authority unchanged.
- Add an early-release direction or objective guard for the first 20 post-warm subepisodes.
- If a residual candidate worsens the predicted first move or shadow objective beyond a small tolerance, shrink the residual toward zero rather than falling back for the whole episode.

Success metric:

- Tail-20 reward remains above `25`.
- First-20 minimum improves from `-31.46` to better than `-10`.
- Tail temperature MAE remains below `0.08`.

### Priority 2: Markov score-gated shrink instead of wider z

Purpose: stop the constant corner action without returning to nominal copy-paste.

Change:

- Keep `z_bound = 0.04`.
- Keep vector cap initially.
- Disable active Markov BC imitation or restrict it to diagnostic-only for the Markov runner.
- Add an active candidate-quality shrink gate:
  - if prediction score is negative or diagnostic cost guard fails, shrink z toward zero or last-good z
  - avoid immediate LS fallback except for nonfinite or failed solve

Success metric:

- Tail z projection fraction decreases from `1.000`.
- Tail requested prediction-score mean becomes nonnegative or much closer to zero.
- Tail reward beats OF-MPC or at least recovers above zero.
- Tail fallback remains low enough to prove TD3 is still active.

### Priority 3: horizon medium-grid restart

Purpose: test whether horizon can learn a useful thermal trade-off with a smaller action space.

Change:

- Use a medium grid such as `Np = 3..18`, `Nc = 1..10`, with `Nc <= Np`.
- Keep standard and dueling separate.
- Prefer dueling first if only one horizon run is affordable, because it at least concentrated its tail actions.
- Add a dwell or switch penalty only if medium-grid entropy remains high.

Success metric:

- Tail unique horizon pairs drops materially from `183` standard or `139` dueling.
- Tail switch-step fraction drops below `0.10`.
- Temperature MAE no longer exceeds OF-MPC by more than `5%`.

### Priority 4: weights composition guard

Purpose: keep the temperature gain while preventing composition sacrifice.

Change:

- Keep multiplier range `[0.5, 2.5]`.
- Keep reward probation disabled.
- Add either a soft boundary regularizer around identity or a composition minimum-performance guard.
- Do not restore identity fallback as a frequent safety action.

Success metric:

- Tail reward remains above OF-MPC.
- Temperature MAE remains below `0.15`.
- Composition MAE returns below `0.0020`.
- Boundary fraction drops below `0.30`.

## Recommended Experiment Order

Run these in this order:

1. Residual release-only guard, because residual is already the strongest result and only needs early protection.
2. Markov score-gated shrink, because Markov currently has a clear logged failure mode.
3. Dueling medium-grid horizon, because horizon needs action-space simplification.
4. Weights with composition guard, because weights is promising but trade-off biased.

Do not make rho active yet. Do not widen Markov z. Do not re-enable reward probation. Do not interpret a nominal-looking Markov recovery as success unless TD3 tail action fraction remains high.

## Remaining Uncertainty

- The exact best Markov gate threshold is not identified yet. The logs strongly indicate that negative prediction score matters, but the threshold should be tested rather than assumed.
- The current horizon failures may be due to action-space size, reward timing, replay distribution, or DQN exploration. The wide action distribution points most strongly to action-space size.
- The residual early crash needs step-level inspection around the first post-warm release to determine whether the bad move is direction, cap size, or handoff slope.
- Scalar reward comparisons to pre-temperature-change runs should be treated as contextual only. Tracking metrics are the stable comparison basis.
