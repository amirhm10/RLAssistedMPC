# Distillation Five-Runner Analysis After Wider Search And Safety Restoration

Date: 2026-05-30  
Run batch analyzed: 2026-05-29  
Case study: Aspen C2 splitter distillation column  
Scenario: `run_mode = "disturb"`, `disturbance_profile = "fluctuation"`  
Baseline reward source: `Distillation/Results/distillation_compare_markov_td3_disturb_fluctuation/20260529_220519/input_data.pkl`

## Executive Summary

This report analyzes the latest saved simulations for all five active distillation RL-assisted MPC runners:

- TD3 weights
- standard horizon DDQN
- dueling horizon DDQN
- TD3 residual
- TD3 Markov correction

The current batch is a safety-restoration and wider-search batch, not a pure performance-maximization batch. The main result is:

**Residual is still the only runner that clearly outperforms OF-MPC in the latest May 29 batch. Markov is much safer than the forced-TD3 May 28 run, but still below baseline. Weights, horizon, and dueling horizon lost tail reward after widening and safety restoration, mostly because the new safety/cooldown layers are active often enough to bias execution back toward conservative/default behavior.**

The tail-20 ranking is:

| Runner | Tail-20 reward | Final reward | Worst first-20 reward | Tail band-normalized MAE | Outside-band fraction | Main interpretation |
|---|---:|---:|---:|---:|---:|---|
| TD3 residual | 24.429 | 27.413 | -9.045 | 0.264 | 0.074 | Best late result, much safer start than previous residual, but not fully implementing intended residual-safety config |
| OF-MPC | 13.898 | 13.136 | 8.175 | 0.583 | 0.163 | Robust reference |
| Dueling horizon | 12.946 | 12.484 | -9.098 | 0.594 | 0.155 | Slightly below reward baseline, better composition, worse temperature |
| TD3 weights | 12.222 | 13.508 | 8.175 | 0.673 | 0.199 | Safe start, but active cap/probation suppresses useful tail adaptation |
| Horizon DDQN | 11.374 | 13.598 | -2.577 | 0.633 | 0.165 | Wider grid plus cooldown causes default-heavy tail behavior |
| TD3 Markov | -8.673 | -6.025 | 2.278 | 0.879 | 0.167 | Large recovery from May 28, but candidate quality is still poor |

![Reward trajectories](figures/distillation_wider_safety_20260530/fig_reward_trajectories.png)

![Tail reward ranking](figures/distillation_wider_safety_20260530/fig_tail_reward_ranking.png)

## Files Inspected

Latest result bundles:

| Runner | Bundle |
|---|---|
| TD3 weights | `Distillation/Results/distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260529_211201/input_data.pkl` |
| Horizon DDQN | `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260529_213627/input_data.pkl` |
| Dueling horizon | `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260529_213847/input_data.pkl` |
| TD3 residual | `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified/20260529_213534/input_data.pkl` |
| TD3 Markov | `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260529_220507/input_data.pkl` |
| OF-MPC baseline trajectory | `Distillation/Data/mpc_results_disturb_fluctuation.pickle` |

Previous-run anchors:

| Runner | Previous bundle |
|---|---|
| TD3 weights | `Distillation/Results/distillation_weights_td3_disturb_fluctuation_mismatch_unified/20260528_194904/input_data.pkl` |
| Horizon DDQN | `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260528_201659/input_data.pkl` |
| Dueling horizon | `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260528_202519/input_data.pkl` |
| TD3 residual | `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_no_rho_unified/20260528_213902/input_data.pkl` |
| TD3 Markov | `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260528_230949/input_data.pkl` |

Implementation files inspected:

| File | Use in this report |
|---|---|
| `systems/distillation/notebook_params.py` | Active default safety/search settings |
| `distillation_RL_assisted_MPC_residual_unified.py` | Verified that `residual_safety` is currently not passed into `residual_cfg` |
| `utils/weights_runner.py` | Weight cap, probation, fallback, shadow identity logs |
| `utils/horizon_runner.py` | Standard horizon safety and shadow default-MPC logs |
| `utils/horizon_runner_dueling.py` | Dueling horizon safety and shadow default-MPC logs |
| `utils/residual_runner.py` | Residual cap ramp, probation hooks, action-source logs |
| `utils/markov_runner.py` | Markov z-safety, priority fallback, probation, source logs |

Analysis artifacts created:

| Artifact | Purpose |
|---|---|
| `report/scripts/analyze_distillation_wider_safety_20260530.py` | Recomputes all metrics and figures from saved bundles |
| `report/figures/distillation_wider_safety_20260530/current_summary_metrics.csv` | Main current-batch metrics |
| `report/figures/distillation_wider_safety_20260530/current_vs_baseline_metrics.csv` | Current runner deltas against OF-MPC |
| `report/figures/distillation_wider_safety_20260530/current_vs_previous_metrics.csv` | May 29 versus May 28 deltas |
| `report/figures/distillation_wider_safety_20260530/current_horizon_pair_counts.csv` | Tail horizon-pair usage |
| `report/figures/distillation_wider_safety_20260530/analysis_summary.json` | Machine-readable summary |

## Method Reconstruction

The plant outputs are tray-24 ethane composition and tray-85 temperature:

$$ y_k=[x_{24,\mathrm{C2H6},k},T_{85,k}]^\top. $$

The manipulated inputs are reflux flow and reboiler duty:

$$ u_k=[F_{\mathrm{reflux},k},Q_{\mathrm{reb},k}]^\top. $$

The offset-free linear MPC model is used as the base controller:

$$ x_{\mathrm{aug},k+1}=A_{\mathrm{aug}}x_{\mathrm{aug},k}+B_{\mathrm{aug}}\Delta u_k,\qquad y_k=C_{\mathrm{aug}}x_{\mathrm{aug},k}. $$

Nominal MPC minimizes tracking and input movement over prediction and control horizons:

$$ J_k=\sum_{j=1}^{N_p} e_{k+j}^\top Q e_{k+j}+\sum_{j=0}^{N_c-1}\Delta u_{k+j}^\top R\Delta u_{k+j}. $$

The five RL supervisors modify this nominal MPC in different action spaces:

| Runner | RL action |
|---|---|
| Weights | Four MPC penalty multipliers around identity, now widened to `[0.5, 2.5]` |
| Horizon | Discrete pair `(Np, Nc)` from `Np = 2..24`, `Nc = 1..16`, filtered by `Nc <= Np`, giving 263 actions |
| Dueling horizon | Same widened 263-action horizon set, with dueling value architecture |
| Residual | Additive scaled-input residual correction after nominal MPC |
| Markov | Four lifted response correction coordinates `z` used by the Markov-adjusted MPC model |

The safety-restoration logic is also method-specific:

| Runner | Intended safety role |
|---|---|
| Weights | Identity warm start, post-warm handoff, multiplier-deviation cap, reward probation, identity fallback only for invalid candidates |
| Horizon | Default `(6, 3)` warm start, post-warm release filter, reward probation to default, shadow default-MPC diagnostics |
| Dueling horizon | Same horizon safety stack as standard horizon |
| Residual | Zero-residual warm start, post-warm handoff, residual cap ramp, reward probation, zero fallback, shadow rho/deadband/risk logs |
| Markov | LS warm start, TD3-priority fallback, reward probation, LS and nominal emergency fallback, dynamic z trust region |

The general safe-start pattern is:

$$ a_{\mathrm{exec},k}=(1-\alpha_k)a_{\mathrm{safe},k}+\alpha_k a_{\theta,k}. $$

For DQN horizon runners, the equivalent operation is projection into the currently allowed horizon set, followed by replay on the executed action.

## Overall Result Versus OF-MPC

![Tail tracking errors](figures/distillation_wider_safety_20260530/fig_tail_tracking_errors.png)

| Runner | Tail reward delta vs OF-MPC | Band MAE change | Outside-band change | Composition MAE change | Temperature MAE change |
|---|---:|---:|---:|---:|---:|
| TD3 weights | -1.676 | +15.3 percent | +21.6 percent | -8.8 percent | +21.7 percent |
| Horizon DDQN | -2.525 | +8.5 percent | +1.2 percent | -17.4 percent | +18.0 percent |
| Dueling horizon | -0.952 | +1.9 percent | -4.9 percent | -25.8 percent | +13.8 percent |
| TD3 residual | +10.531 | -54.7 percent | -54.6 percent | -61.2 percent | -50.0 percent |
| TD3 Markov | -22.571 | +50.7 percent | +2.5 percent | +84.4 percent | +45.3 percent |

This table is the most important one in the report. It shows that residual is not just better on reward. It improves composition, temperature, band-normalized error, and outside-band fraction at the same time. The horizon runners improve composition but lose enough temperature performance to drop below the reward baseline. Weights improves final reward slightly but loses tail reward and tail tracking. Markov is still far below baseline even after a large safety recovery.

## Effect Of The New Changes Versus May 28

![Current versus previous tail reward](figures/distillation_wider_safety_20260530/fig_current_vs_previous_tail20.png)

| Runner | May 29 tail reward | May 28 tail reward | Tail delta | Worst first-20 delta | Main effect |
|---|---:|---:|---:|---:|---|
| TD3 weights | 12.222 | 18.742 | -6.519 | +6.484 | Safer start, worse tail due active cap/probation |
| Horizon DDQN | 11.374 | 15.370 | -3.996 | +5.202 | Safer early standard horizon, but cooldown/default bias hurts tail |
| Dueling horizon | 12.946 | 15.828 | -2.882 | -17.274 | Widening made dueling less stable early and weaker late |
| TD3 residual | 24.429 | 23.580 | +0.849 | +46.339 | Cap ramp/handoff preserved late upside and removed most early crash severity |
| TD3 Markov | -8.673 | -27.582 | +18.909 | +15.968 | Safety restored a large fraction of the failure, but not enough |

The new safety layers did what safety layers are supposed to do in the first release window: they reduced the worst damage for weights, standard horizon, residual, and Markov. The cost is that several runners now spend too much time in protected or default-biased behavior.

## Safety Intervention Evidence

![Safety intervention rates](figures/distillation_wider_safety_20260530/fig_safety_intervention_rates.png)

| Runner | Tail intervention metric | Tail value | Early value | Interpretation |
|---|---:|---:|---:|---|
| Weights | multiplier cap projection | 0.300 | 0.399 | Cap/probation is still active late |
| Horizon | horizon projection or cooldown | 0.100 projection, 0.400 cooldown | 0.041 projection | Tail is default-heavy |
| Dueling horizon | horizon projection or cooldown | 0.049 projection, 0.200 cooldown | 0.045 projection | Less protected than standard horizon, but still affected |
| Residual | residual cap projection | 0.000 | 0.350 | Cap ramp acted early, then released |
| Markov | fallback fraction | 0.000 tail, 0.457 early | 0.457 early | Fallback helped early, tail relies on scaled TD3 during probation |

This is the central safety lesson. Residual has the healthiest profile because safety is active early and inactive in the tail. Weights and horizons still have substantial tail safety activity, so they do not fully test their widened search spaces. Markov uses early fallback, but in the tail it accepts TD3 candidates while probation remains active and scales the correction.

## Runner 1: TD3 Weights

### What It Did

Weights used the widened multiplier range `[0.5, 2.5]` with identity warm start, post-warm BC handoff, multiplier-deviation cap, reward probation, and shadow identity-MPC diagnostics.

Tail multiplier means were:

$$ \bar m_{\mathrm{tail}}=[1.614,\;1.176,\;1.560,\;1.339]. $$

Additional diagnostics:

| Metric | Value |
|---|---:|
| Tail multiplier boundary fraction | 0.297 |
| Tail cap projection fraction | 0.300 |
| Early cap projection fraction | 0.399 |
| Reward probation trigger count | 34 |
| Tail probation fraction | 0.300 |
| Tail fallback fraction | 0.000 |
| Tail TD3 accepted fraction | 0.700 |
| Tail projected-TD3 fraction | 0.300 |
| Shadow selected-minus-identity objective delta | 0.000257 |
| Shadow first-move delta norm | 0.00425 |

### Interpretation

Weights was safer at the beginning than May 28, but the price was a strong late performance loss. Tail reward fell from 18.742 to 12.222. Against OF-MPC, it improves composition MAE by 8.8 percent but worsens temperature MAE by 21.7 percent and outside-band fraction by 21.6 percent.

The most plausible mechanism is over-active safety/probation. Thirty percent of tail steps are projected and thirty percent are in probation. The actor is still exploring boundary multipliers, but the executed multiplier is frequently compressed around identity. The shadow identity-MPC objective difference is tiny, so the selected weighted-MPC candidates are not clearly better than identity in the saved diagnostics.

### What Safety Helped

The cap and probation prevented the bad boundary behavior seen in earlier uncontrolled weight runs. The worst first-20 reward improved by 6.48 relative to May 28. That is real safety value.

### What Safety Hurt

The same cap/probation stayed active in the tail. That likely blocked the high-performing behavior from the May 28 batch. For weights, the old issue was not "too much TD3 everywhere." It was "untrusted boundary TD3." The next safety layer should be candidate-quality based rather than always shrinking deviations.

### Recommendation

Keep identity fallback and shadow identity diagnostics. Relax the tail cap once the candidate passes objective and first-move checks. Add a rule that distinguishes:

- saturated but objectively better multiplier candidates
- saturated and worse-than-identity candidates

Only the second group should be softened strongly.

## Runner 2: Standard Horizon DDQN

### What It Did

The standard horizon runner used the widened action set:

$$ N_p\in\{2,\dots,24\},\qquad N_c\in\{1,\dots,16\},\qquad N_c\leq N_p. $$

The saved bundle contains 263 valid horizon recipes. The safety layer used default `(6, 3)` during warm start, protected and ramp allowed sets after warm start, and reward-probation cooldown to `(6, 3)`.

Tail horizon statistics:

| Metric | Value |
|---|---:|
| Mean prediction horizon | 10.318 |
| Mean control horizon | 5.165 |
| Prediction horizon standard deviation | 5.485 |
| Control horizon standard deviation | 3.394 |
| Unique tail horizon pairs | 63 |
| Tail horizon entropy | 3.025 |
| Top pair fraction | 0.404 |
| Tail projection fraction | 0.100 |
| Tail cooldown fraction | 0.400 |
| Reward probation trigger count | 46 |
| Tail switch-step fraction | 0.125 |

Top tail pairs:

| Rank | Pair `(Np, Nc)` | Tail fraction |
|---:|---|---:|
| 1 | `(6, 3)` | 0.4035 |
| 2 | `(12, 5)` | 0.0540 |
| 3 | `(5, 2)` | 0.0205 |
| 4 | `(12, 6)` | 0.0180 |
| 5 | `(4, 3)` | 0.0170 |

![Horizon tail pair usage](figures/distillation_wider_safety_20260530/fig_horizon_tail_pair_usage.png)

### Interpretation

The widened horizon search did not yet pay off. Tail reward is 2.525 below OF-MPC and 3.996 below the May 28 horizon run. The top pair is the fallback/default `(6, 3)` for 40.35 percent of tail steps, and cooldown is active for 40 percent of tail steps.

This suggests the runner is not truly exploiting the 263-action grid in the tail. It is frequently returning to the default. Since the action dimension changed substantially, old policies are not directly reusable. The current behavior is consistent with a cold or under-trained DQN operating under a conservative safety wrapper.

### What Safety Helped

Worst first-20 reward improved by 5.20 compared with the previous horizon run. The release filter and cooldown did reduce early bad excursions.

### What Safety Hurt

Cooldown dominates the tail. If 40 percent of the last 20 subepisodes are forced to `(6, 3)`, then the widened grid is mostly a diagnostic object, not a fully exploited controller.

### Recommendation

Keep the widened grid, but reduce tail default bias:

- make reward probation phase-aware so it is strict during the first 20 to 30 subepisodes and weaker later
- use selected-versus-default shadow objective to decide cooldown, not reward collapse alone
- add a horizon dwell-time penalty only after verifying that switching causes reward dips

## Runner 3: Dueling Horizon DDQN

### What It Did

Dueling horizon used the same 263-action widened grid and the same horizon safety layer as standard horizon. The dueling architecture should help separate state value from action advantage, which can matter when many horizon pairs are similar.

Tail horizon statistics:

| Metric | Value |
|---|---:|
| Mean prediction horizon | 10.622 |
| Mean control horizon | 4.750 |
| Prediction horizon standard deviation | 4.484 |
| Control horizon standard deviation | 3.587 |
| Unique tail horizon pairs | 81 |
| Tail horizon entropy | 2.901 |
| Top pair fraction | 0.214 |
| Tail projection fraction | 0.049 |
| Tail cooldown fraction | 0.200 |
| Reward probation trigger count | 24 |
| Tail switch-step fraction | 0.151 |

Top tail pairs:

| Rank | Pair `(Np, Nc)` | Tail fraction |
|---:|---|---:|
| 1 | `(6, 3)` | 0.2140 |
| 2 | `(9, 1)` | 0.1345 |
| 3 | `(14, 10)` | 0.0835 |
| 4 | `(10, 6)` | 0.0620 |
| 5 | `(10, 1)` | 0.0450 |

### Interpretation

Dueling is less default-dominated than standard horizon, but it still falls below OF-MPC in reward. It has a better outside-band fraction than OF-MPC and a large composition MAE improvement, but temperature MAE worsens by 13.8 percent. This means dueling is learning a real tradeoff, just not the one the reward prefers overall.

The widened grid made dueling less stable than its May 28 version. Tail reward dropped by 2.882 and worst first-20 reward dropped by 17.274. The likely explanation is a new action space with many more poor or weakly distinguished actions, plus cold DQN learning.

### What Safety Helped

Tail projection is only 4.9 percent and cooldown is 20 percent, so dueling is less safety-suppressed than standard horizon. It also keeps outside-band fraction slightly better than OF-MPC.

### What Safety Hurt Or Failed To Prevent

The release filter did not prevent a worse early minimum than May 28. Because the widened grid includes aggressive and unusual pairs such as `(24, 9)`, the release policy may need better early action priors or action masking based on controller condition, not only `Np` and `Nc` ranges.

### Recommendation

Keep dueling as the preferred horizon architecture, but do not judge the widened grid from one cold run. Next:

- initialize replay with default and historically good pairs
- log Q-value entropy or action-gap confidence
- compare a narrower medium grid against the 263-action grid
- keep `(6, 3)` as fallback but avoid letting cooldown dominate tail learning

## Runner 4: TD3 Residual

### What It Did

Residual remains the strongest method. It executes:

$$ u_{\mathrm{exec},k}=u_{\mathrm{MPC},k}+\Delta u_{\mathrm{res},k}. $$

The run used no rho state and no active rho authority:

- `append_rho_to_state = False`
- `authority_use_rho = False`
- `residual_authority_enabled = False`

Tail residual diagnostics:

| Metric | Value |
|---|---:|
| Tail raw residual norm mean | 0.00566 |
| Tail executed residual norm mean | 0.00566 |
| Tail executed residual norm q95 | 0.02790 |
| Tail raw-executed difference norm mean | 0.000000044 |
| Tail material projection fraction | 0.000 |
| Early residual cap projection fraction | 0.350 |
| Tail residual cap projection fraction | 0.000 |
| Residual probation trigger count | 0 |
| Tail zero-fallback fraction | 0.000 |

![Continuous action norms](figures/distillation_wider_safety_20260530/fig_continuous_action_norms.png)

### Why Residual Is Still Best

Residual is successful because it is the most local and physically direct RL correction. It does not change the full predictive model like Markov, and it does not indirectly change controller preference like weights. It adds a small first-move trim on top of a feasible nominal MPC action.

The late residual norm is small enough to stay near the MPC solution, but large enough to correct persistent mismatch and disturbance effects. That is exactly the use case where residual RL usually works well: the baseline controller is competent, and the learned policy only needs to repair systematic local error.

The no-rho setting also matters. Earlier rho-authority variants compressed useful residual authority near the target. The latest residual tail executes almost exactly what the actor proposes, and that preserves the high late reward.

### What Safety Helped

Compared with May 28, tail reward improved from 23.580 to 24.429, and worst first-20 reward improved from -55.383 to -9.045. The early residual cap projection fraction is 35.0 percent, while the tail cap projection fraction is zero. That is the desired safety profile: protect release, then get out of the way.

### Important Implementation Inconsistency

The saved residual bundle reports:

- `residual_safety_enabled = False`
- `residual_safety = {}`

This means the intended residual reward-probation and shadow rho/deadband/risk diagnostics were not active in this run. The active pieces that did appear are BC handoff and TD3 authority cap ramp. Inspection of `distillation_RL_assisted_MPC_residual_unified.py` shows that `residual_cfg` passes `behavioral_cloning` and `td3_authority_ramp`, but does not pass `residual_safety`.

So the correct conclusion is:

**Residual improved because warm zero, handoff, and the cap ramp were active. The run does not yet test the full planned residual safe-start stack.**

### Recommendation

Fix the residual entrypoint so it passes:

`"residual_safety": dict(NB["residual_safety"])`

Then rerun residual only. The success target should be:

- tail-20 reward remains above 23
- worst first-20 reward improves above -5, ideally above 0
- residual probation and shadow rho logs are populated
- tail cap projection remains close to zero

Do not add rho to the actor state yet. The evidence still says rho is useful as a shadow or execution safety concept, not necessarily as an actor feature.

## Runner 5: TD3 Markov

### What It Did

Markov restored the main safety stack:

- `force_td3_execute = False`
- `td3_priority_fallback.enabled = True`
- reward probation enabled inside priority fallback
- LS and nominal fallback paths available
- z-safety enabled with protected, ramp, full, and probation caps
- vector-norm cap enabled

Current Markov safety diagnostics:

| Metric | Value |
|---|---:|
| Tail TD3 accepted fraction | 1.000 |
| Tail fallback fraction | 0.000 |
| Early fallback fraction | 0.457 |
| Probation trigger count | 188 |
| Tail probation fraction | 1.000 |
| Tail z 2-norm mean | 0.0200 |
| Tail z 2-norm q95 | 0.0200 |
| Tail q95 abs z coordinate | 0.0100 |
| Tail requested cost-guard pass fraction | 0.205 |
| Tail requested prediction score mean | -0.0154 |
| Tail requested cost margin mean | 0.000180 |

### Why It Improved But Still Failed

Markov improved dramatically over May 28:

- tail reward improved by 18.909
- final reward improved by 21.145
- worst first-20 reward improved by 15.968

That confirms the restored safety stack is helping. The earlier forced-TD3 Markov run was catastrophic because bad lifted-model candidates were sent to the plant. The current run avoids most of that early damage.

But Markov remains below baseline because candidate quality is still weak. In the tail, only 20.5 percent of requested candidates pass the cost guard, and the average prediction score is still negative. The controller accepts TD3 in the tail while probation remains active. Probation scales the correction, which explains why the executed `z` remains around `[-0.01, 0.01]` per coordinate even though the configured z safety band allows larger values in later phases. With four coordinates at about 0.01 magnitude, the z 2-norm sits near 0.02.

This is better than the May 28 forced bad-candidate behavior, but it is not enough to become a good controller. Magnitude control and probation scaling reduce harm. They do not make a bad Markov model correction useful.

### Recommendation

Markov needs a stricter candidate-quality decision in the tail:

- if requested cost-guard fails during probation, fallback to LS or nominal instead of accepting scaled TD3
- keep z-safety as a magnitude trust region, but do not treat it as a quality filter
- use prediction score and cost margin as active acceptance gates, not only diagnostics
- log separate rates for rejected by cost, rejected by prediction score, rejected by nonfinite, and accepted after scaling

The next Markov target is not to reach residual-level reward immediately. The first target is to get above OF-MPC tail reward while keeping early fallback below 50 percent and tail fallback below 20 percent.

## Cross-Runner Interpretation

The five runners now separate into three categories.

### Category 1: Residual Is Working

Residual has the strongest control interpretation:

- it preserves nominal MPC feasibility
- it corrects local mismatch directly
- safety is active early and inactive late
- it improves both composition and temperature

The remaining task is to fix the missing residual-safety config pass-through and prove the same result with probation and shadow diagnostics populated.

### Category 2: Weights And Horizons Need Less Tail Conservatism

Weights and horizons are not failing catastrophically. They are being pushed back toward conservative behavior:

- weights has 30 percent tail cap projection and 30 percent tail probation
- standard horizon has 40 percent tail cooldown to `(6, 3)`
- dueling has 20 percent tail cooldown to `(6, 3)`

Their current single-run results should not be read as "wider search is bad." The better interpretation is:

**The search was widened faster than the learning and release logic could exploit.**

### Category 3: Markov Is Safer But Still Candidate-Misaligned

Markov safety restoration worked in the limited sense that it prevented the extreme May 28 failure. But the requested candidates still look bad by the existing cost and prediction diagnostics. This is a model-correction action-space problem, not only a safety-cap problem.

## Scientific Risks And Inconsistencies

| Issue | Evidence | Why it matters |
|---|---|---|
| Residual safety config not active | Saved bundle has `residual_safety_enabled=False` and `residual_safety={}` | The latest residual run is not the full planned safe-start stack |
| Baseline reward must come from compare bundle | Raw baseline pickle stores a different reward series | Using the raw pickle would falsely rank OF-MPC near zero reward |
| Horizon widening changed action dimension | 263 actions versus older smaller grids | Previous DQN behavior is not directly comparable as a warm continuation |
| Tail cooldown is too frequent for horizon | 40 percent standard, 20 percent dueling | It biases tail behavior toward default and can hide learning |
| Markov tail accepts candidates with weak diagnostics | Cost-guard pass only 20.5 percent, prediction score negative | Probation scaling reduces damage but does not make candidates good |
| Residual action-source code is overly sensitive | Tail source is almost all `projected_td3`, but material raw-executed difference is about `4.4e-8` | Reported source flags should distinguish numerical projection from real intervention |

## Literature Connections

The results are consistent with the MPC-RL view that learning MPC parameters can improve a structured controller without replacing constraint handling. A relevant reference is Gros and Zanon, "Data-driven Economic NMPC using Reinforcement Learning," which supports treating MPC parameters as learnable policy variables.

The residual result matches the residual-RL idea: a learned correction can be more effective and easier to learn than a full replacement policy when the nominal controller is already competent.

The Markov result is aligned with safety-filter and model predictive safety certification work. Clipping action magnitude is not equivalent to checking whether the candidate action improves predicted closed-loop behavior.

No new citations or BibTeX entries were added in this pass because this report is based on local simulation bundles and previously discussed literature connections.

## Recommended Next Experiments

### Experiment 1: Fix And Rerun Residual Safe-Start

Purpose: test the actual intended residual safe-start stack.

Code target:

- `distillation_RL_assisted_MPC_residual_unified.py`

Change:

- pass `residual_safety` into `residual_cfg`

Success metric:

- tail-20 reward above 23
- worst first-20 reward above -5
- `residual_safety_enabled=True` in the saved bundle
- probation and shadow logs populated

### Experiment 2: Weights Tail-Cap Relaxation

Purpose: recover May 28 tail reward while keeping May 29 early safety.

Change:

- keep warm identity and handoff
- keep identity fallback for invalid candidates
- reduce or phase out tail probation after stable windows
- accept large multiplier deviations when shadow identity objective and first-move checks are favorable

Success metric:

- tail-20 reward above 17
- tail cap projection below 10 percent
- outside-band fraction no worse than OF-MPC

### Experiment 3: Horizon Medium Grid Versus Full Grid

Purpose: determine whether the 263-action grid is too wide for the current DQN training budget.

Variants:

- full grid: 263 actions
- medium grid: for example `Np = 3..18`, `Nc = 1..10`
- historical grid: previous smaller action space

Success metric:

- dueling tail reward above OF-MPC
- cooldown below 10 percent in the tail
- selected-versus-default shadow objective nonpositive on average

### Experiment 4: Markov Candidate-Quality Gate

Purpose: convert Markov safety from magnitude-only protection to candidate-quality control.

Change:

- use requested cost-guard and prediction score as active tail gates during probation
- fallback to LS or nominal if candidate quality fails
- keep z-safety for magnitude only

Success metric:

- Markov tail reward above 13.9
- requested cost-guard pass fraction above 50 percent, or rejected candidates logged as fallback
- prediction score nonnegative on average

### Experiment 5: Multi-Seed Final Batch

Purpose: turn single-run evidence into defensible research evidence.

Run at least three seeds for:

- OF-MPC reference
- residual fixed safe-start
- weights relaxed tail safety
- dueling medium grid
- Markov candidate-quality gate

Report:

- tail-20 reward mean and standard deviation
- worst first-20 reward
- band-normalized MAE
- outside-band fraction
- safety intervention fraction
- action-source fractions

## Bottom Line

The latest batch is not a final performance batch. It is a very useful diagnostic batch.

The strongest conclusion is that residual should become the leading method, but only after the residual-safety pass-through bug is fixed and the intended probation/shadow diagnostics are verified. Markov safety restoration clearly helped, but the Markov actor still proposes poor candidates too often. Weights and horizons need less tail conservatism, or they need candidate-quality safety instead of broad cooldown/default behavior.

The next phase should be a targeted follow-up, not another broad blind rerun: fix residual safety pass-through, relax weights tail caps based on shadow identity diagnostics, compare medium versus full horizon grids, and make Markov candidate-quality checks active.
