# Distillation Horizon DQN and Dueling DQN Diagnosis

Date: 2026-06-02

## Objective

This note reviews the latest distillation horizon-selection results for the standard DDQN and dueling DDQN supervisors. The purpose is to explain why the current results look poor compared with earlier good distillation horizon runs, separate exploration issues from reward issues, and define the next safety-gated DQN experiment.

The latest available runs are:

- Standard DDQN horizon: `Distillation/Results/distillation_horizon_disturb_fluctuation_mismatch_unified/20260602_144031/input_data.pkl`
- Dueling DDQN horizon: `Distillation/Results/distillation_dueling_horizon_disturb_fluctuation_mismatch_unified/20260602_144134/input_data.pkl`
- OF-MPC baseline: `Distillation/Data/mpc_results_disturb_fluctuation.pickle`

## Executive Summary

The latest standard DDQN result is weak rather than fully failed. It beats OF-MPC on tail-20 reward by `+0.945`, but it does so with worse temperature MAE than OF-MPC and a very unstable horizon schedule. The latest dueling DDQN result is a clear failure under the current reward. It is `-2.398` below OF-MPC on tail-20 reward and has worse temperature tracking.

The latest bad behavior is not explained by the old `50k` replay-buffer issue. Both June 2 runs used `150k` capacity and stored about `79k` transitions. It is also not simply that exploration is off. Both agents use NoisyNet exploration, tail epsilon is expected to be zero, and the saved noisy-layer sigma traces are still nonzero.

The strongest mechanism is policy instability under the current reward geometry. The standard DDQN tail uses `86` of `87` horizon recipes and switches on `22.2%` of tail steps. Earlier good standard DDQN runs used far fewer recipes and often concentrated on `(6, 3)`. The current learner has not formed a confident horizon ranking.

There is no SG-DQN implementation or SG-DQN result bundle visible in the current tree. The current horizon safety block is disabled in the latest configs, so these are live DQN and dueling DQN supervisors, not safety-gated DQN supervisors.

## Files Inspected

- `report/distillation_horizon_agents_failure_analysis_2026_06_01.md`
- `report/scripts/analyze_distillation_horizon_agents_20260601.py`
- `report/scripts/analyze_distillation_horizon_dqn_dueling_20260602.py`
- `report/figures/distillation_horizon_dqn_dueling_20260602/latest_horizon_pair_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/all_horizon_runs_current_reward_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/focus_horizon_runs_current_reward_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/ofmpc_current_reward_summary.csv`
- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `systems/distillation/notebook_params.py`
- `systems/distillation/config.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`
- `utils/agent_step_runtime.py`
- `utils/state_features.py`
- `utils/rewards.py`
- `DQN/dqn_agent.py`
- `DuelingDQN/dueling_dqn_agent.py`

I also searched for local SG-DQN naming patterns and found no matching implementation or result folder.

## Analysis Artifacts

The reproducible local analysis command was:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_horizon_dqn_dueling_20260602.py
```

Generated local artifacts:

- `report/figures/distillation_horizon_dqn_dueling_20260602/all_horizon_runs_current_reward_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/focus_horizon_runs_current_reward_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/latest_horizon_pair_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/ofmpc_current_reward_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/fig_horizon_current_reward_history.png`
- `report/figures/distillation_horizon_dqn_dueling_20260602/fig_latest_horizon_vs_ofmpc_metrics.png`

![Latest horizon metrics](figures/distillation_horizon_dqn_dueling_20260602/fig_latest_horizon_vs_ofmpc_metrics.png)

![Horizon reward history](figures/distillation_horizon_dqn_dueling_20260602/fig_horizon_current_reward_history.png)

## Current Method

The plant is the distillation column with controlled output

$$ y_k = [x_{\mathrm{C2},24,k}, T_{85,k}]^\top. $$

The manipulated input is

$$ u_k = [L_k, Q_{\mathrm{reb},k}]^\top, $$

where the first channel is reflux flow and the second channel is reboiler duty. The baseline controller is offset-free MPC with an augmented linear model and observer. The horizon agent does not change the plant model, MPC weights, residual input, or setpoint. It only chooses the MPC prediction and control horizon pair.

The discrete action is

$$ a_k = (N_{p,k}, N_{c,k}) \in \mathcal{A}_H, \quad \mathcal{A}_H = \{(N_p,N_c): N_p \in \{4,\ldots,14\}, N_c \in \{2,\ldots,13\}, N_c \le N_p\}. $$

The default OF-MPC-like recipe is `(6, 3)`. The active grid has `87` feasible recipes. The horizon decision interval is `4` plant steps. The latest runs use `state_mode = mismatch`, so the RL state is the base augmented state, setpoint, and input feature vector plus normalized innovation and normalized tracking-error features.

The reward is computed from scaled output error and scaled input movement after the selected horizon has been applied:

$$ e_k = y_{k+1,\mathrm{scaled}} - y_{\mathrm{sp},k,\mathrm{scaled}}, \quad \Delta u_k = u_{k,\mathrm{scaled}} - u_{k-1,\mathrm{scaled}}. $$

The current reward defaults are:

| Quantity | Value |
|---|---:|
| `Q_diag` | `[37000, 20000]` |
| `R_diag` | `[2500, 2500]` |
| `k_rel` | `[0.3, 0.01]` |
| `band_floor_phys` | `[0.003, 0.2]` |
| `beta` | `7.0` |
| `gate` | `geom` |
| `reward_scale` | `1.0` |

For output channel `i`, the scaled tracking band is

$$ b_{i,k} = \frac{\max(k_{\mathrm{rel},i} \lvert y_{\mathrm{sp},i,k}^{\mathrm{phys}}\rvert, b_{\mathrm{floor},i}^{\mathrm{phys}})}{y_{\max,i} - y_{\min,i}}. $$

The inside-band gate is

$$ s_{i,k} = \sigma\left(\frac{b_{i,k} - \lvert e_{i,k}\rvert}{\tau_{\mathrm{frac}} b_{i,k}}\right), \quad w_k = \left(\prod_i s_{i,k}\right)^{1/n_y}. $$

The scalar reward is a temperature-sensitive tracking and movement reward with an inside-band bonus:

$$ r_k = -J_{\mathrm{err},k} - J_{\Delta u,k} - J_{\mathrm{lin,out},k} - J_{\mathrm{lin,in},k} + B_{\mathrm{in},k}. $$

The standard DDQN target is the Double-DQN endpoint target:

$$ y_k^{\mathrm{DQN}} = r_k + \gamma Q_{\bar{\theta}}(s_{k+1}, \arg\max_a Q_\theta(s_{k+1},a)). $$

The dueling network changes the Q parameterization:

$$ Q_\theta(s,a) = V_\theta(s) + A_\theta(s,a) - \frac{1}{\lvert\mathcal{A}_H\rvert}\sum_b A_\theta(s,b). $$

## Latest Results

Metrics are rescored with the current reward over the final 20 subepisodes. This avoids mixing older logged rewards with the current reward definition.

| Method | Tail reward | Delta vs OF-MPC | Final reward | First live min | Comp MAE | Temp MAE | Outside band | Unique pairs | Top pair | Top frac | Switch frac | Tail loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| OF-MPC | `6.391` | `0.000` | `6.926` | `9.070` | `0.001545` | `0.1921` | `0.2515` | NA | NA | NA | NA | NA |
| Standard DDQN | `7.336` | `+0.945` | `8.915` | `-1.338` | `0.001152` | `0.2042` | `0.2580` | `86` | `(12, 5)` | `0.0805` | `0.2220` | `53.48` |
| Dueling DDQN | `3.993` | `-2.398` | `1.561` | `0.647` | `0.001252` | `0.2198` | `0.2743` | `79` | `(9, 2)` | `0.1720` | `0.1766` | `22.68` |

The latest standard DDQN improves composition MAE relative to OF-MPC but worsens temperature MAE. The scalar reward gives it a small tail win, but this is not a clean control improvement because the temperature channel and action stability both degraded.

The latest dueling DDQN improves composition relative to OF-MPC but degrades temperature enough to lose clearly under the current reward. The final reward is also much worse than OF-MPC.

Both latest agents have:

- `buffer_capacity = 150000`
- `buffer_size = 79199`
- `tail_reason_accepted_frac = 1.0`
- `tail_projection_frac = 0.0`
- `tail_cooldown_frac = 0.0`
- `reward_probation_enabled = False`

So the latest failure is not due to replay capacity being too small and not due to the horizon safety layer blocking actions.

## Historical Comparison

The table below compares selected older good runs to the latest June 2 runs using the same current reward rescoring.

| Method | Run | Tail reward | Delta vs OF-MPC | Temp MAE | Comp MAE | Unique pairs | Top pair | Top frac | Buffer | Saved Q2 |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Standard DDQN | `20260519_202111` | `10.547` | `+4.156` | `0.1850` | `0.000918` | `31` | `(6, 3)` | `0.5135` | `150000` | `5000` |
| Standard DDQN | `20260520_193757` | `9.177` | `+2.786` | `0.1934` | `0.000939` | `26` | `(6, 3)` | `0.4435` | `150000` | `5000` |
| Dueling DDQN | `20260518_140746` | `8.223` | `+1.832` | `0.2029` | `0.000868` | `51` | `(6, 3)` | `0.5855` | `150000` | `5000` |
| Dueling DDQN | `20260521_154934` | `9.073` | `+2.682` | `0.2007` | `0.000895` | `62` | `(11, 11)` | `0.3135` | `150000` | `1500` |
| Standard DDQN | `20260602_144031` | `7.336` | `+0.945` | `0.2042` | `0.001152` | `86` | `(12, 5)` | `0.0805` | `150000` | `20000` |
| Dueling DDQN | `20260602_144134` | `3.993` | `-2.398` | `0.2198` | `0.001252` | `79` | `(9, 2)` | `0.1720` | `150000` | `20000` |

This supports your memory that distillation horizon runs had good results. The good results are not completely gone after current-reward rescoring. The May 19 and May 20 standard DDQN runs still beat OF-MPC under the current reward. The difference is that those runs were more concentrated and had better temperature behavior.

## Interpretation

### 1. Exploration is not simply off

The latest agents use `exploration_mode = noisy`. In this mode, `epsilon_tail = 0.0` is expected because the exploration is carried by noisy network layers. The saved tail noisy sigma values are:

| Method | Tail noisy sigma |
|---|---:|
| Standard DDQN | `0.0270` |
| Dueling DDQN | `0.0404` |

This does not look like a missing-exploration bug. The more plausible issue is that NoisyNet exploration is still producing action churn late in the run, while the value estimates are not confident enough to settle on a small set of useful horizon recipes.

### 2. Reward is part of the explanation, but not the whole explanation

The current reward is much more sensitive to tray-85 temperature than many older saved runs. The latest runs use `Q2 = 20000`, while several older good runs saved `Q2 = 5000` or `Q2 = 1500`.

However, reward provenance is not the entire story. The May 19 and May 20 standard DDQN runs still score well when rescored with the current reward. They also have better temperature MAE than the latest run. So the current reward did not make horizon adaptation impossible. It made unstable temperature-worse horizon schedules easier to expose.

### 3. The main symptom is horizon-policy churn

The current standard DDQN tail uses `86` of `87` recipes. The top recipe appears only `8.05%` of tail steps. This is almost unconcentrated.

The older strong standard DDQN runs used:

- `31` unique recipes with top pair `(6, 3)` at `51.35%`
- `26` unique recipes with top pair `(6, 3)` at `44.35%`

That contrast is the strongest practical signal. The good horizon supervisor behaves like a stable scheduler with occasional deviations. The latest standard DDQN behaves like a nearly continuous horizon sampler.

### 4. Dueling is not solving the action-ranking problem

Dueling DDQN is more concentrated than standard DDQN in the latest run, but it concentrates on a temperature-worse policy. Its top tail recipe `(9, 2)` appears `17.2%` of the tail. That is more stable than standard DDQN, but still not enough, and the selected recipe distribution does not protect temperature tracking.

The dueling architecture is therefore not the missing piece by itself. It may help represent state value and action advantage, but the current closed-loop action selection still needs a gate, a dwell rule, or a smaller candidate set.

### 5. There is no SG-DQN here yet

I found SG-TD3 result families for weights and residual work, but no SG-DQN horizon implementation or result bundle. The latest horizon defaults have:

- `horizon_safety.enabled = False`
- `release_filter.enabled = False`
- `reward_probation.enabled = False`
- `shadow_default_mpc.enabled = False`

So the current horizon results should be interpreted as ungated DQN and ungated dueling DQN.

## June 2 Exploration And Range Update

The active distillation horizon defaults were changed from NoisyNet exploration to epsilon-greedy exploration for the next horizon reruns:

| Agent | Old exploration | New exploration | New epsilon schedule |
|---|---|---|---|
| Standard DDQN | `noisy` | `epsilon` | linear `0.20 -> 0.02` over `50000` steps |
| Dueling DDQN | `noisy` | `epsilon` | linear `0.20 -> 0.02` over `50000` steps |

This isolates the suspected late-action-churn mechanism more cleanly than NoisyNet. For standard DDQN, the entrypoint was also updated to pass `eps_decay_steps` into `DQNAgent`, because the old exponential schedule would not actually reach the requested low epsilon within one 200-subepisode run.

The horizon range should not be increased for the next run. The saved widened-grid runs used `263` recipes and failed badly. The better question is whether to keep the current `87` recipes for one epsilon-greedy ablation, then reduce to a medium grid if recipe churn persists.

Evidence from the top current-reward 87-recipe runs:

| Method | Run | Tail reward | Unique pairs | Top pair | Top frac | Tail Np range | Tail Nc range | Fraction in `Np 4-12, Nc 2-8` |
|---|---|---:|---:|---|---:|---|---|---:|
| Standard DDQN | `20260519_202111` | `10.547` | `31` | `(6, 3)` | `0.513` | `6-14` | `2-11` | `0.817` |
| Standard DDQN | `20260520_193757` | `9.177` | `26` | `(6, 3)` | `0.444` | `5-14` | `2-13` | `0.810` |
| Dueling DDQN | `20260511_131656` | `9.084` | `73` | `(6, 3)` | `0.678` | `4-14` | `2-13` | `0.911` |
| Dueling DDQN | `20260521_154934` | `9.073` | `62` | `(11, 11)` | `0.314` | `4-14` | `2-13` | `0.640` |
| Standard DDQN | `20260601_160538` | `8.708` | `86` | `(12, 7)` | `0.056` | `4-14` | `2-13` | `0.668` |

Aggregate top-pair frequencies across the top ten 87-recipe runs were dominated by `(6, 3)` at about `34.9%`, followed by `(11, 11)` at about `12.2%`. This argues against a very tight grid such as `Np 4-10, Nc 2-6`, because that would remove `(11, 11)` and `(12, 7)`. A medium grid such as `Np 4-12, Nc 2-8` has `53` actions and keeps `(6, 3)` plus `(12, 7)`, but it removes `(11, 11)`. A slightly larger medium grid such as `Np 4-12, Nc 2-11` has `62` actions and keeps all three historical anchors.

Recommended range decision:

1. Keep the current 87-recipe grid for the first epsilon-greedy rerun so the exploration change is isolated.
2. Do not increase the grid beyond 87 until a frozen evaluation shows the learned policy is stable.
3. If epsilon-greedy still gives high churn, test a medium reduced grid. Prefer `Np 4-12, Nc 2-11` before a tighter grid because it preserves `(6, 3)`, `(11, 11)`, and `(12, 7)`.

## Bugs, Inconsistencies, Or Risks Found

I did not find evidence that the latest poor results are caused by a simple exploration-off bug, replay-capacity regression, or horizon safety override. The main risks are experimental and algorithmic:

- The saved results are training rollouts, not separate frozen greedy evaluations.
- The current report does not have Q-value margin diagnostics, so we cannot tell whether the best action barely beats the alternatives.
- The shadow default MPC diagnostic is disabled, so we cannot measure whether selected horizons improve the MPC objective at decision time.
- There is no dwell penalty or dwell rule, so the agent can change horizon recipes too often.
- The reward can allow composition improvement to coexist with temperature degradation unless the acceptance logic explicitly checks both channels.
- Older logged rewards are not comparable to current logged rewards unless rescored.

## Literature Connections

No new citations are added in this report. The local result pattern is consistent with known DQN-family concerns: off-policy value estimates can rank many near-equivalent discrete actions inconsistently, dueling value-advantage decomposition does not by itself impose action persistence, and scalar reward improvement can hide channel-specific control degradation. These are used here as interpretation, not as formal citation claims.

## Recommended Next Experiments

### 1. Add a frozen greedy evaluation pass

Purpose: determine whether the learned greedy policy is actually poor or whether the training rollout is noisy.

Likely files:

- `distillation_RL_assisted_MPC_horizons_unified.py`
- `distillation_RL_assisted_MPC_horizons_dueling_unified.py`
- `utils/horizon_runner.py`
- `utils/horizon_runner_dueling.py`

Change: after training, rerun the same scenario with `agent.act_eval`, no replay pushes, no training, and noise disabled through the existing eval path.

Metrics to compare: tail reward, temp MAE, comp MAE, outside-band fraction, unique pairs, switch fraction, top-pair fraction.

Result that confirms the idea: frozen evaluation has fewer unique pairs and better temperature without changing the reward.

### 2. Implement SG-DQN as a value-confidence and dwell gate

Purpose: stop low-confidence horizon changes from entering the plant.

Likely files:

- `utils/agent_step_runtime.py`
- `utils/horizon_safety.py`
- `systems/distillation/notebook_params.py`

Change: define an SG-DQN action gate:

$$ a_k^{\mathrm{exec}} = \begin{cases} a_k^{\mathrm{DQN}}, & Q(s_k,a_k^{\mathrm{DQN}}) - Q(s_k,a_k^{\mathrm{hold}}) \ge m_Q \ \mathrm{and}\ d_{\mathrm{hold}} \ge d_{\min}, \\ a_k^{\mathrm{hold}}, & \mathrm{otherwise}. \end{cases} $$

Start with `a_hold = (6, 3)` during release, then hold the last accepted action. Use a minimum dwell of one or two subepisodes before accepting another change.

Metrics to improve: tail unique pairs below `25`, tail switch fraction below `0.05`, temperature MAE at or below OF-MPC, and tail reward above OF-MPC.

Failure mode to watch: the gate may collapse to default `(6, 3)` forever. Track accepted fraction and accepted action counts.

### 3. Turn on shadow default MPC diagnostics

Purpose: determine whether a selected horizon is actually giving a better MPC optimization outcome than `(6, 3)` at the same state.

Likely file:

- `systems/distillation/notebook_params.py`

Change: enable `horizon_safety.shadow_default_mpc.enabled = True` for a diagnostic run, even if action gating remains off.

Metrics to save: selected objective, default objective, first-move difference, selected-minus-default objective, and whether reward improved after accepted changes.

Result that confirms the idea: good horizon changes should have negative selected-minus-default objective or better subsequent reward without temperature penalty.

### 4. Curate a medium action set before retraining

Purpose: reduce the value-ranking burden across nearly equivalent horizon recipes.

Likely file:

- `systems/distillation/notebook_params.py`

Change: run an ablation with a curated action set derived from saved-run frequencies and neighbors. Candidate anchors from saved runs include `(6, 3)`, `(9, 2)`, `(11, 11)`, `(12, 5)`, and `(12, 7)`. Add nearby recipes only if the shadow MPC diagnostics support them.

Metrics to improve: top-pair fraction, unique-pair count, tail loss, temperature MAE.

Result that rejects the idea: the smaller action set still has high switch fraction and worse temperature.

### 5. Test exploration ablations after frozen evaluation exists

Purpose: isolate whether late NoisyNet action churn is harming control.

Likely files:

- `systems/distillation/notebook_params.py`
- `DQN/dqn_agent.py`
- `DuelingDQN/dueling_dqn_agent.py`

Change: compare current NoisyNet to epsilon-greedy and to NoisyNet with late sigma freezing or eval-mode action selection after a fixed training phase.

Metrics to watch: tail noisy sigma, unique recipes, switch fraction, tail loss, and frozen-eval performance.

Result that confirms exploration is the issue: frozen or low-noise evaluation improves temperature and reduces recipe churn without hurting composition.

### 6. Keep the current reward but add channel-aware acceptance

Purpose: avoid a scalar reward win that worsens tray-85 temperature.

Likely files:

- `utils/horizon_safety.py`
- `systems/distillation/notebook_params.py`

Change: do not immediately retune `Q2` downward. Instead, add an acceptance condition that rejects a horizon change if recent temperature band error or predicted shadow default temperature error is worse than default by a tolerance.

Metrics to improve: temperature MAE, outside-band fraction, and scalar reward.

Result that confirms the idea: standard DDQN keeps composition gains while temperature returns to OF-MPC quality.

## Recommended Run Order

1. Run frozen greedy evaluation for the June 2 standard and dueling agents if checkpoints are available.
2. Run one diagnostic standard DDQN with shadow default MPC enabled and no gate.
3. Implement SG-DQN value-margin plus dwell gate.
4. Rerun standard DDQN SG on the 87-action grid.
5. Rerun dueling SG only if standard SG shows the gate is working.
6. Try the curated action set if the 87-action SG run still has high switch fraction.

## Remaining Uncertainty

The largest uncertainty is that the current bundles do not provide a separate post-training evaluation trajectory. Because the saved trajectories are training rollouts, the current diagnosis cannot fully distinguish poor learned value ranking from late exploratory churn. Q-value margins and shadow default MPC diagnostics are also missing. Those two diagnostics should be added before changing the reward again.

## Files Changed

- Created `report/scripts/analyze_distillation_horizon_dqn_dueling_20260602.py`
- Created and updated `report/distillation_horizon_dqn_dueling_diagnosis_2026_06_02.md`
- Updated `systems/distillation/notebook_params.py` so standard and dueling horizon defaults use epsilon-greedy exploration with linear `0.20 -> 0.02` decay over `50000` steps
- Updated `distillation_RL_assisted_MPC_horizons_unified.py` so standard DDQN receives `eps_decay_steps`
- Generated local ignored artifacts under `report/figures/distillation_horizon_dqn_dueling_20260602/`

## How To Verify

Run:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe report\scripts\analyze_distillation_horizon_dqn_dueling_20260602.py
```

Then open:

- `report/distillation_horizon_dqn_dueling_diagnosis_2026_06_02.md`
- `report/figures/distillation_horizon_dqn_dueling_20260602/latest_horizon_pair_summary.csv`
- `report/figures/distillation_horizon_dqn_dueling_20260602/all_horizon_runs_current_reward_summary.csv`
