# Residual and Combined Supervisor Progress Review

Date: 2026-04-29

## Scope

This note reviews the saved residual-policy runs for polymer and distillation, with a separate emphasis on the latest polymer combined runs where the residual agent is active. The analysis uses saved `input_data.pkl` bundles and the physical setpoint reconstruction already used by `report/generate_polymer_change_impact_report.py`.

The key question is whether the residual method is working scientifically, not only whether the episode reward improves.

## Files Inspected

- `RL_assisted_MPC_residual_unified.ipynb`
- `distillation_RL_assisted_MPC_residual_unified.ipynb`
- `RL_assisted_MPC_combined_unified.ipynb`
- `utils/residual_runner.py`
- `utils/combined_runner.py`
- `utils/residual_authority.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`
- `report/04_residual_methods.md`
- `report/05_combined_supervisor.md`
- `report/polymer_change_impact_report.md`
- `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`
- `change-reports/2026-04-25_analyze_distillation_residual_tail_degradation.md`
- `Polymer/Data/mpc_results_dist.pickle`
- `Polymer/Data/mpc_results_nominal.pickle`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- `Distillation/Data/mpc_results_nominal.pickle`
- Polymer residual result bundles under `Polymer/Results/td3_residual_nominal/` and `Polymer/Results/td3_residual_disturb/`
- Polymer combined result bundles under `Polymer/Results/combined_disturb_h_dqn_mismatch__m_td3_mismatch__w_td3_mismatch__r_td3_mismatch_rho/`
- Distillation residual result bundles under `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/` and `Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified/`

## Method Summary

The residual controller keeps the MPC solve as the baseline action and lets an RL actor propose a bounded residual move in scaled input coordinates. In each sample,

$$ u_t^{\mathrm{RL}} = \Pi_{\mathcal{U}}\left(u_t^{\mathrm{MPC}} + \Delta u_t^{\mathrm{res}}\right). $$

The actor first produces a raw normalized residual action,

$$ a_t^{\mathrm{raw}} = \pi_{\theta}(s_t). $$

The raw action is mapped to residual bounds and then projected by the residual-authority layer:

$$ \Delta u_t^{\mathrm{res,exec}} = \rho_t \Delta u_t^{\mathrm{res,raw}}, \qquad \rho_t \in [\rho_{\min}, 1]. $$

The mismatch-state residual runs append innovation and tracking-error features:

$$ e_t^{\mathrm{trk}} = y_t - y_t^{\mathrm{sp}}, \qquad e_t^{\mathrm{inn}} = y_t - \hat{y}_t. $$

The recent refreshed runs use a signed-log mismatch transform rather than hard clipping:

$$ \phi(e) = \operatorname{sign}(e)\log(1 + |e|). $$

The combined polymer supervisor has four agents active around one MPC layer:

- DQN horizon selector
- TD3 matrix multiplier agent
- TD3 weight multiplier agent
- TD3 residual agent

The combined residual component is therefore not isolated. If the reward improves while the offset grows, the cause may be the interaction between the weight/matrix/horizon decisions and the residual projection, not the residual actor alone.

## Metric Definitions

For each saved bundle, the physical setpoint was reconstructed from the scaled setpoint deviation:

$$ y_{\mathrm{sp},t}^{\mathrm{phys}} = y_{\min} + \left(y_{\mathrm{sp},t}^{\mathrm{dev}} + y_{\mathrm{ss}}^{\mathrm{scaled}}\right)(y_{\max} - y_{\min}). $$

The tail tracking metric is the mean physical absolute error over the late part of each setpoint segment:

$$ \mathrm{MAE}_{\mathrm{tail}} = \frac{1}{|\mathcal{T}_{\mathrm{late}}|n_y}\sum_{t \in \mathcal{T}_{\mathrm{late}}}\|y_t - y_{\mathrm{sp},t}\|_1. $$

The offset vector is:

$$ \bar{e}_{\mathrm{tail}} = \frac{1}{|\mathcal{T}_{\mathrm{late}}|}\sum_{t \in \mathcal{T}_{\mathrm{late}}}(y_t - y_{\mathrm{sp},t}). $$

The reward degradation metric is:

$$ \Delta R_{\mathrm{drop}} = \bar{R}_{\mathrm{best10}} - \bar{R}_{\mathrm{last10}}. $$

Larger `Delta R_drop` means the controller found a better reward region earlier and then degraded.

## Polymer Residual Runs

Baseline references:

| Baseline | Tail MAE mean | Tail MAE by output | Tail offset | Final-20 reward |
| --- | ---: | --- | --- | ---: |
| `Polymer/Data/mpc_results_dist.pickle` | 0.00028 | `[0.00008, 0.00049]` | `[-0.00002, 0.00001]` | -4.4173 |
| `Polymer/Data/mpc_results_nominal.pickle` | 0.00397 | `[0.00177, 0.00616]` | `[-0.00097, 0.00026]` | -19.2609 |

The nominal baseline file contains only two reward episodes, so the nominal reward comparison is weaker than the disturbance comparison. The nominal residual tracking conclusion is still useful because the saved physical trajectories show very small final offsets in the later nominal runs.

| Run | Mode | Tail MAE mean | Delta tail MAE vs matching MPC | Tail offset `[eta, T]` | Final-20 reward | Delta reward vs matching MPC | Reward drop | Tail rho |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `td3_residual_nominal/20260402_174941` | nominal | 0.0060 | +0.0021 | `[0.0024, -0.0077]` | -3.5934 | +15.6676 | 0.0272 | n/a |
| `td3_residual_nominal/20260408_121130` | nominal | 0.0003 | -0.0037 | `[0.0000, -0.0001]` | -3.5446 | +15.7163 | 0.0007 | n/a |
| `td3_residual_nominal/20260408_131932` | nominal | 0.0002 | -0.0038 | `[-0.0000, 0.0001]` | -3.5092 | +15.7517 | 0.0163 | n/a |
| `td3_residual_disturb/20260413_004620` | disturb | 0.0026 | +0.0024 | `[-0.0005, 0.0015]` | -3.9313 | +0.4860 | 0.0573 | 0.7259 |
| `td3_residual_disturb/20260420_225631` | disturb | 0.0047 | +0.0044 | `[0.0017, -0.0067]` | -3.6812 | +0.7362 | 0.0257 | 0.6801 |
| `td3_residual_disturb/20260422_181610` | disturb | 0.0045 | +0.0042 | `[0.0003, -0.0001]` | -3.5788 | +0.8386 | 0.0000 | 0.6723 |
| `td3_residual_disturb/20260423_025802` | disturb | 0.0059 | +0.0056 | `[-0.0018, 0.0071]` | -3.7190 | +0.6984 | 0.0438 | 0.6860 |

### Polymer Residual Interpretation

The polymer residual method is partially working, but it is not yet a clean tracking win under disturbance.

What worked:

- The nominal residual runs after April 8 remove the visible nominal offset. Tail MAE drops from the nominal MPC reference of `0.00397` to `0.00020-0.00030`.
- Disturbance residual runs consistently improve reward relative to disturbance MPC. The final-20 reward improves by `+0.4860` to `+0.8386`.
- The refreshed mismatch state and signed-log transform helped policy learning compared with the older hard-clipped mismatch state, as already documented in `report/polymer_change_impact_report.md`.

What did not work yet:

- Under disturbance, all inspected residual runs have worse tail physical MAE than `Polymer/Data/mpc_results_dist.pickle`.
- The best disturbance residual by final-20 reward is `20260422_181610`, but its tail MAE mean is `0.00450`, still above the MPC reference `0.00028`.
- The residual projection is active on essentially every step in the disturbance runs. The raw actor requests are much larger than what is executed. For the latest disturbance residual runs, the mean raw-to-executed residual action gap is about `0.157-0.164` in normalized action units.

Scientific conclusion:

The polymer residual policy has learned reward-improving behavior, but the saved disturbance runs do not prove that residual correction beats the offset-free MPC baseline in physical tracking. The method is promising for nominal offset removal and reward shaping, but the disturbance version is still reward-positive and tracking-negative.

## Polymer Combined Runs With Residual Active

The latest combined runs are the most important current evidence because the reward improvement is much stronger than in residual-only training.

| Run | Tail MAE mean | Delta tail MAE vs disturbance MPC | Tail offset `[eta, T]` | Final-20 reward | Delta reward vs disturbance MPC | Reward drop | Tail rho |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `combined_disturb...rho/20260426_042026` | 0.0090 | +0.0087 | `[0.0006, -0.0015]` | -2.1311 | +2.2863 | 0.0122 | 0.6655 |
| `combined_disturb...rho/20260426_193310` | 0.0077 | +0.0074 | `[0.0008, 0.0036]` | -1.9912 | +2.4261 | 0.0348 | 0.5807 |

### Combined Interpretation

The combined supervisor is clearly learning according to the scalar reward. The latest run improves final-20 reward by `+2.4261` relative to the disturbance MPC reward reference. That is much larger than the residual-only reward gain.

However, the same runs have the largest polymer tail MAE among the residual-active polymer cases:

- latest combined tail MAE mean: `0.00766`
- disturbance MPC tail MAE mean: `0.00028`
- latest residual-only disturbance tail MAE mean: `0.00586`

This matches the observation that the combined run develops a small offset even though nominal behavior does not. The offset is not enormous in raw physical units, but it is systematic enough to matter because the baseline MPC tail error is very small.

The likely mechanism is reward-track mismatch:

- The combined agents can improve the scalar reward by changing transients, input movement, horizon choice, and penalty/model behavior.
- The residual projection prevents unsafe large residual moves, but it does not enforce zero steady-state offset.
- Since all active agents share the same scalar reward, the residual agent may receive credit for a global reward improvement caused by matrix/weight/horizon choices while still leaving a biased late-segment trajectory.

Scientific conclusion:

The combined polymer method is the strongest reward result so far, but it is not yet a clean tracking result. It should be reported as reward improvement with residual-active offset risk, not as a solved residual-control method.

## Distillation Residual Runs

Baseline reference:

| Baseline | Tail MAE mean | Tail MAE by output | Tail offset | Final-20 reward |
| --- | ---: | --- | --- | ---: |
| `Distillation/Data/mpc_results_disturb_fluctuation.pickle` | 0.00509 | `[0.00009, 0.01008]` | `[0.00008, -0.00992]` | -0.3027 |

| Run | Agent | Tail MAE mean | Delta tail MAE vs fluctuation MPC | Tail offset `[x24, T85]` | Final-20 reward | Delta reward vs fluctuation MPC | Reward drop | Tail rho |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `20260414_063405` | TD3 | 0.0196 | +0.0145 | `[0.0002, -0.0162]` | -0.3857 | -0.0830 | 0.1644 | 0.9848 |
| `20260415_191909` | SAC | 0.0109 | +0.0058 | `[-0.0012, -0.0196]` | -0.5291 | -0.2264 | 0.2610 | 1.0000 |
| `20260417_181426` | TD3 | 0.0081 | +0.0030 | `[0.0000, -0.0039]` | 18.0986 | +18.4014 | 0.5753 | 0.5774 |
| `20260420_175629` | TD3 | 0.0113 | +0.0062 | `[0.0004, -0.0098]` | 16.0567 | +16.3595 | 1.2601 | 0.7087 |
| `20260425_082259` | TD3 | 0.0085 | +0.0034 | `[0.0005, -0.0162]` | 14.0966 | +14.3993 | 4.9443 | 0.5790 |
| `20260427_194850` | SAC | 0.0025 | -0.0026 | `[-0.0001, -0.0028]` | 5.5393 | +5.8421 | 14.5848 | 0.5645 |

### Distillation Interpretation

The two distillation problems are real:

1. Reward improves and can surpass the MPC reward scale, but later degrades.
2. A late offset can appear, especially in tray-85 temperature, even when the reward remains above the MPC reference.

The TD3 run on April 25 is the clearest example of the user's observation. It reaches a much better reward region earlier, then drops by `4.9443` reward units from best-10 to last-10. Its tail temperature offset is `-0.0162`, worse than the April 17 run's `-0.0039`.

The April 27 SAC run is more subtle. It has the largest reward drop, `14.5848`, but the final tail physical MAE is actually better than the disturbance MPC baseline. That means the reward degradation and the final tail physical tracking metric are not measuring the same behavior. The degradation likely occurs in earlier late-training windows, reward components, action penalties, or inside-band bonuses rather than only in the final late-setpoint MAE.

The common implementation clue is projection:

- Projection is active on essentially every inspected mismatch residual run.
- The raw action is consistently much larger than the executed residual action.
- The actor is therefore learning in raw action space while the plant sees a heavily projected action.

This can create critic-policy mismatch. The critic sees executed replay actions, but the actor update can still push raw actions toward regions that are mostly collapsed by projection. That explains why reward can improve for a while and then degrade as the actor becomes more aggressive or less aligned with the executable residual envelope.

## Bugs, Inconsistencies, and Risks

- The residual method should not be judged by reward alone. Several runs improve reward while worsening physical tail MAE relative to MPC.
- The projection-active rate is effectively `1.0` in disturbance residual and combined runs. That is not a bug by itself, but it means the raw actor action is not the same control action applied to the plant.
- The combined polymer logs report `state_mode=standard` at the top level even though residual mismatch logs and `rho` authority are present. This is probably a bundle-level naming artifact from the combined runner, but it can confuse report readers.
- Distillation reward families have changed across archived and unified paths, as documented in `report/distillation_reward_audit.md`. Claims such as "surpasses nominal MPC reward" need to say which reward function and baseline file were used.
- The nominal polymer MPC baseline file currently has only two reward episodes. Use it cautiously for reward comparisons.

## Literature Connections

The observed failure mode is consistent with established MPC-RL and safe RL lessons already used in the repo reports:

- TD3 can overvalue out-of-distribution continuous actions, especially after a narrow warm-start phase [Fujimoto2018].
- TD3+BC-style actor anchoring can reduce action drift away from the executable behavior support [FujimotoGu2021].
- Safe policy improvement and baseline fallback are appropriate when a learned policy may degrade a strong baseline [Laroche2019].
- Learning-based MPC needs explicit safety/performance protection rather than relying only on reward shaping [Hewing2020].

These references are already listed in `report/matrix_multiplier_cap_calculation_and_distillation_recovery.md`, so this report does not introduce new citations.

## Recommended Next Experiments

### Experiment 1: Residual Actor Executed-Action Anchor

Purpose: reduce the raw/executed residual mismatch.

Files:

- `TD3Agent/agent.py`
- `SACAgent/sac_agent.py`
- `utils/residual_runner.py`
- `utils/combined_runner.py`

Change:

Add an actor regularization term for the residual agent during the first post-warm-start training window:

$$ \mathcal{L}_{\mathrm{actor}} = -\mathbb{E}[Q(s,\pi(s))] + \lambda_{\mathrm{exec}}\mathbb{E}[\|\pi(s)-a^{\mathrm{exec}}\|_2^2]. $$

Metric that should improve:

- lower raw-to-executed action gap
- lower projection fraction or lower projection magnitude
- no loss in final-20 reward

Failure mode to watch:

- the actor collapses to zero residual and loses the reward gain.

### Experiment 2: Combined Polymer Offset Penalty Gate

Purpose: preserve the combined reward gain while preventing small steady-state bias.

Files:

- `systems/polymer/notebook_params.py`
- `utils/rewards.py`
- `utils/combined_runner.py`

Change:

Add a late-segment or near-setpoint offset penalty that activates when tracking is inside the loose transient band:

$$ r_t^{\mathrm{offset}} = -\lambda_{\mathrm{off}}\|y_t - y_t^{\mathrm{sp}}\|_1\mathbf{1}\{\|y_t - y_t^{\mathrm{sp}}\|_{\infty} < \epsilon_{\mathrm{near}}\}. $$

Metric that should improve:

- combined tail MAE mean below `0.003`
- final-20 reward remains better than residual-only

Failure mode to watch:

- excessive penalty reduces useful transient improvement from the matrix and weight agents.

### Experiment 3: Distillation Degradation Window Replay Audit

Purpose: locate why reward peaks and then degrades.

Files:

- `Distillation/Results/distillation_residual_td3_disturb_fluctuation_mismatch_rho_unified/20260425_082259/input_data.pkl`
- `Distillation/Results/distillation_residual_sac_disturb_fluctuation_mismatch_rho_unified/20260427_194850/input_data.pkl`
- a new analysis script under `report/scripts/`

Change:

Compute per-episode reward, output MAE, input movement, residual raw norm, residual executed norm, projection gap, and inside-band bonus for early, peak, and tail windows.

Metric that should improve:

- diagnosis should identify whether reward degradation is driven by output error, move penalty, projection, or bonus loss.

Failure mode to watch:

- final tail MAE can look good while reward degradation happened earlier, so use windowed episode diagnostics rather than only final episode plots.

### Experiment 4: Residual Authority Schedule Ablation

Purpose: test whether constant projection prevents stable residual learning.

Files:

- `utils/residual_authority.py`
- `systems/polymer/notebook_params.py`
- `systems/distillation/notebook_params.py`

Change:

Compare current `rho_mapping_mode="exp_raw_tracking"` against a stricter near-setpoint shutdown and a smoother authority ramp:

$$ \rho_t = \rho_{\min} + (1-\rho_{\min})(1-\exp[-k(\|e_t\|_{\infty}-\tau)_+]). $$

Metric that should improve:

- lower steady-state offset
- lower raw/executed action gap
- reward drop below `1.0` for distillation TD3 and SAC

Failure mode to watch:

- too strict an authority schedule may remove the only useful residual correction during mismatch.

## Bottom Line

Polymer residual alone is not yet a disturbance-tracking win, but it is a useful learning signal and it removes nominal offset in the later nominal runs. Polymer combined is the highest-reward result and should be prioritized, but the latest combined runs confirm a small offset risk under disturbance. Distillation residual has the same deeper issue: the actor can improve reward and then drift or degrade because the raw policy is not well aligned with the projected action that the plant actually receives.

The next work should focus on executable-action alignment and offset-aware evaluation, not simply more training episodes.
