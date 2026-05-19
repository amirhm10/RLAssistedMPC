# Polymer Markov TD3-Only Run Without Safeguard

Date: 2026-05-16

## Objective

Analyze the latest polymer Markov run from
[RL_assisted_MPC_markov_zbound_008_looser_gate_only_td3_only_unified.ipynb](../RL_assisted_MPC_markov_zbound_008_looser_gate_only_td3_only_unified.ipynb)
and compare it against the safeguarded sibling
[RL_assisted_MPC_markov_zbound_008_looser_gate_only_unified.ipynb](../RL_assisted_MPC_markov_zbound_008_looser_gate_only_unified.ipynb)
and the disturbance MPC baseline.

The main question is:

What actually happens when the polymer Markov controller is forced to execute TD3 actions without LS or nominal fallback?

## Runs compared

- latest TD3-only, no-safeguard run:
  `Polymer/Results/td3_markov_disturb_zbound_008_looser_gate_only_td3_only/20260516_000338/`
- latest safeguarded sibling:
  `Polymer/Results/td3_markov_disturb_zbound_008_looser_gate_only/20260515_230958/`
- disturbance MPC baseline:
  `Polymer/Data/mpc_results_dist.pickle`

## Method reminder

The polymer Markov controller starts from a nominal offset-free MPC plan `U_{0,k}` and applies a learned Markov correction:

$$ z_k = \mathrm{clip}(a_k, -1, 1)\, z_{\max}. $$

The corrected candidate is formed as:

$$ U_{\mathrm{cand},k} = U_{0,k} + M_k z_k, $$

where `M_k` is the Markov correction map built from the local lifted prediction basis.

In the guarded notebook family, the TD3 candidate is executed only if it passes the same three tests used in the unified runner:

$$ S_{\mathrm{pred}}(z_k) > s_{\mathrm{pred,min}}, $$

$$ d_G(z_k) \le d_{\max}, $$

$$ J_{\mathrm{nom}}(U_{\mathrm{cand},k}) \le J_{\mathrm{nom}}(U_{0,k}) + \varepsilon_{\mathrm{abs}} + \tau_{\mathrm{rel}} \lvert J_{\mathrm{nom}}(U_{0,k}) \rvert. $$

If the TD3 candidate fails, the guarded notebook can fall back to the LS candidate and then to nominal MPC.

In the TD3-only notebook, the configuration changes are:

- `rl_fallback_to_ls = False`
- `force_td3_execute = True`

So the same diagnostics are still logged, but they are no longer used to block TD3 execution.

This matters because the saved result bundle confirms:

- `force_td3_execute = True`
- `td3_accepted_fraction = 1.0`
- `ls_fallback_fraction = 0.0`
- `nominal_fallback_fraction = 0.0`

That means the live controller really is a TD3-only controller online, not a fallback-dominated mixture.

## Main result

The no-safeguard TD3-only run is the best of the three in reward terms, and it is also slightly better than the safeguarded sibling on final tracking metrics.

### High-level comparison

| Metric | TD3-only no safeguard | Guarded TD3 sibling | Nominal MPC |
| --- | ---: | ---: | ---: |
| Mean reward | `-3.9313` | `-4.0360` | `-4.4174` |
| Tail-20 mean reward | `-3.6978` | `-3.8800` | `-4.4174` |
| Final episode reward | `-3.7357` | `-3.9214` | `-4.4174` |
| TD3 fraction, tail-20 | `1.0000` | `0.1570` | `0.0000` |
| LS fraction, tail-20 | `0.0000` | `0.8256` | `0.0000` |
| Nominal fraction, tail-20 | `0.0000` | `0.0174` | `1.0000` |

So the experiment succeeds in the narrow sense the user asked for:

1. all live corrective moves come from TD3
2. reward improves relative to the safeguarded variant
3. reward also improves relative to nominal disturbance MPC

![Reward and action-source comparison](figures/polymer_markov_td3_only_without_safeguard_20260516/fig_reward_and_action_source.png)

## What the safeguard was buying

The reward gain does not mean the safeguards were unnecessary. It means the safeguard was trading away authority in exchange for more conservative executed behavior.

The executed diagnostics show that clearly.

### Tail-20 executed diagnostics

| Metric | TD3-only no safeguard | Guarded TD3 sibling |
| --- | ---: | ---: |
| Executed prediction score | `-0.00199` | `0.05103` |
| Executed gain drift | `0.07613` | `0.03480` |
| Executed `\|\|z\|\|` | `0.15325` | `0.09350` |
| Requested raw-action saturation | `0.99919` | `0.99900` |

The important contrast is:

- both runs ask for nearly saturated raw TD3 actions in the tail
- the TD3-only run executes those large requests directly
- the guarded run executes a much smaller correction on average because it falls back mostly to LS

So the positive executed prediction score in the guarded run is not a property of the TD3 actor by itself. It is a property of the TD3-plus-fallback controller that the safeguards assemble online.

The TD3-only run removes that protective mixture. It keeps the reward gain, but it does so with:

- slightly negative executed prediction score
- roughly `2.2x` larger executed drift in the tail
- about `1.64x` larger executed correction norm in the tail

![Executed diagnostics](figures/polymer_markov_td3_only_without_safeguard_20260516/fig_mechanism_diagnostics.png)

![Tail summary](figures/polymer_markov_td3_only_without_safeguard_20260516/fig_tail_summary.png)

## Final closed-loop quality

The final held-out episode still favors the TD3-only controller.

### Final episode metrics

| Metric | TD3-only no safeguard | Guarded TD3 sibling | Nominal MPC |
| --- | ---: | ---: | ---: |
| Reward | `-3.7357` | `-3.9214` | `-4.4174` |
| Viscosity RMSE | `0.1792` | `0.1836` | `0.1917` |
| Temperature RMSE | `0.4371` | `0.4520` | `0.5678` |
| Viscosity IAE | `43.06` | `44.07` | `51.71` |
| Temperature IAE | `102.27` | `105.61` | `212.30` |
| Mean input-move norm | `1.0245` | `1.0132` | `1.1915` |

So the TD3-only run is not winning only on reward shaping. It also gives slightly better final tracking than the safeguarded sibling and clearly better tracking than nominal MPC.

The one small cost visible here is control movement:

- TD3-only has slightly higher mean move size than the safeguarded sibling
- both RL variants still move less than the nominal MPC baseline on average

![Final episode comparison](figures/polymer_markov_td3_only_without_safeguard_20260516/fig_final_episode_vs_guarded_and_mpc.png)

## Interpretation

The cleanest reading of the latest result is:

1. the TD3 actor has learned a useful correction policy in this notebook family
2. removing the safeguard allows that policy to act with full authority
3. full authority improves reward and final tracking on this seed and disturbance realization
4. the improvement comes with weaker model-alignment diagnostics, because the executed policy no longer benefits from LS/nominal filtering

So this is a promising performance result, but not yet a proof that the safeguards are obsolete.

It is still a single latest-run comparison. The current evidence supports:

- better performance for this run
- real TD3-only execution
- a measurable loss of executed safety/alignment margin

It does **not** yet support:

- robustness across seeds
- robustness across disturbance profiles
- a general claim that fallback should be removed from the Markov family

## Risks and caveats

The main scientific risk is over-crediting reward gains to an all-clear TD3 policy when the diagnostics show the opposite direction on alignment:

- the TD3-only controller executes larger corrections
- its executed prediction score is slightly negative in the tail
- its executed drift sits much closer to the guard threshold

That makes the result interesting, not invalid. But the right claim is:

The no-safeguard controller performed better on this run while spending more of its available alignment margin.

## Recommended next experiment

The next experiment should test whether this reward gain is robust or just a favorable single-seed outcome.

Recommended follow-up:

- run the same `z_bound = 0.08` looser-gate family for at least `5` seeds with and without `force_td3_execute`
- keep the disturbance profile fixed to the same disturbance notebook family for a fair A/B comparison
- compare:
  - mean reward
  - tail reward
  - final episode RMSE and IAE
  - executed prediction score
  - executed gain drift
  - executed correction norm
  - any visible instability or saturation episodes

The success criterion should be:

The TD3-only variant keeps its reward/tracking edge across seeds without a large increase in failed episodes or drift-heavy transients.

If that does not hold, then the current run is best interpreted as a high-performing but less conservative special case rather than a general design improvement.

## 2026-05-17 unified fallback follow-up

After the TD3-priority fallback design was added to the shared runner, the latest polymer Markov result folder is:

`Polymer/Results/td3_markov_disturb_zbound_008/20260517_210211/`

This run does **not** represent the intended TD3-priority setup. The saved bundle reports:

- `td3_priority_fallback = {}`
- `summary_metrics["td3_priority_fallback_enabled"] = False`
- `behavioral_cloning_enabled = False`
- `force_td3_execute = False`

So the behavioral-cloning part of the plan was active, but the new less-conservative TD3-priority fallback block was not passed from the unified notebook into the shared runner for this run.

### What happened

The latest run still behaved like the old guarded controller. Across all steps, the executed action source fractions were:

| Source | Fraction |
| --- | ---: |
| LS fallback | `0.7190` |
| TD3 accepted | `0.2214` |
| warm-start LS | `0.0467` |
| nominal fallback | `0.0129` |

In the tail-20 subepisodes, TD3 authority became even smaller:

| Source | Tail-20 fraction |
| --- | ---: |
| LS fallback | `0.7539` |
| TD3 accepted | `0.1657` |
| nominal fallback | `0.0804` |

That is why TD3 looked less effective: it was not actually being given the decision authority we intended. It was mostly being evaluated as a proposal and then replaced by LS.

![Latest priority gap and action-source fractions](figures/polymer_markov_td3_priority_gap_20260517/fig_latest_priority_gap_action_sources.png)

### Why TD3 was blocked

The blocker was not the catastrophic-cap idea. In the tail-20 window:

| Requested TD3 check | Tail-20 pass fraction |
| --- | ---: |
| old `score > 0` test | `0.1658` |
| drift `<= 0.10` | `1.0000` |
| old cost guard | `1.0000` |
| new priority absolute cap `<= 0.10` | `1.0000` |

The maximum requested TD3 gain drift was only `0.0800`, and the maximum requested cost margin was only `0.00465`. Under the intended TD3-priority full phase, these candidates would not have been rejected by drift or cost. They were rejected because the older positive-score veto was still active.

So the conclusion is narrow but important:

The latest polymer unified run did not fail because the new less-conservative fallback design was too conservative. It failed to test that design because the notebook did not pass `td3_priority_fallback` into `markov_cfg`.

### Reward consequence

The reward trace also matches that mechanism:

| Run | Tail-20 mean reward | Final reward |
| --- | ---: | ---: |
| TD3-only no safeguard, 2026-05-16 | `-3.6978` | `-3.7357` |
| latest unified, 2026-05-17 | `-3.8501` | `-3.9026` |
| previous guarded sibling, 2026-05-15 | `-3.8800` | `-3.9214` |

The latest unified run is slightly better than the previous guarded sibling, likely helped by disabled behavioral cloning and updated shared defaults, but it does not recover the TD3-only reward level because it still executes mostly LS.

![Reward comparison for latest unified gap](figures/polymer_markov_td3_priority_gap_20260517/fig_reward_latest_vs_td3_only.png)

### Implementation status after this audit

The notebook pass-through has now been corrected for future runs:

- `RL_assisted_MPC_markov_unified.ipynb` now passes `CTRL.get("td3_priority_fallback", {})` into `markov_cfg`
- `distillation_RL_assisted_MPC_markov_unified.ipynb` has the same pass-through fix
- the shared runner logic remains the intended TD3-priority design:
  - no positive-score hard veto in priority mode
  - phase-aware catastrophic caps
  - LS only as emergency fallback
  - nominal MPC only as last resort
  - executed-action replay remains enabled

The next polymer run from `RL_assisted_MPC_markov_unified.ipynb` should therefore show `summary_metrics["td3_priority_fallback_enabled"] = True` in `input_data.pkl`. If it does not, the run should be treated as a configuration failure rather than an algorithm result.

## 2026-05-17 TD3-priority run before soft handoff

The next polymer Markov result was:

`Polymer/Results/td3_markov_disturb_zbound_008/20260517_235822/`

This run is the clean pre-handoff test: the TD3-priority fallback block was active, but the later authority-ramp/probation soft-handoff logs were not yet present. The saved bundle confirms:

- `summary_metrics["td3_priority_fallback_enabled"] = True`
- `force_td3_execute = False`
- `behavioral_cloning_enabled = False`
- no `td3_authority_scale_log`, so this predates the soft-handoff implementation

That means this is not the TD3-only no-safeguard notebook, and it is not the old fallback-dominated unified run. It is the intended TD3-priority controller before the latest handoff modification.

### Main comparison

| Run | TD3 fraction, all | TD3 fraction, tail-20 | Tail-20 mean reward | Final reward |
| --- | ---: | ---: | ---: | ---: |
| TD3-priority, no soft handoff, 2026-05-17 | `0.9500` | `1.0000` | `-3.6938` | `-3.7263` |
| TD3-only no safeguard, 2026-05-16 | `1.0000` | `1.0000` | `-3.6978` | `-3.7357` |
| no pass-through unified run, 2026-05-17 | `0.2214` | `0.1657` | `-3.8501` | `-3.9026` |

So for polymer, the TD3-priority design did what we wanted: it removed LS domination without requiring `force_td3_execute = True`. In reward terms, it essentially matched the TD3-only no-safeguard result while keeping emergency nominal fallback available.

![Reward and TD3 authority for the pre-handoff priority run](figures/polymer_markov_td3_priority_no_handoff_20260517/fig_reward_and_td3_authority.png)

### Mechanism

The tail-20 action-source comparison is the clearest mechanism check:

| Source | TD3-priority, no soft handoff | no pass-through unified |
| --- | ---: | ---: |
| TD3 accepted | `1.0000` | `0.1657` |
| LS fallback | `0.0000` | `0.7539` |
| nominal fallback | `0.0000` | `0.0804` |

The old positive-score veto would still have blocked most TD3 moves. In the TD3-priority run tail:

| Requested TD3 check | Tail-20 value |
| --- | ---: |
| `score > 0` fraction | `0.1181` |
| drift `<= 0.10` fraction | `1.0000` |
| old cost guard pass fraction | `1.0000` |
| full priority cap pass fraction | `1.0000` |

The requested TD3 gain drift stayed below the cap, with tail maximum `0.0800`, and the tail maximum cost margin was only `0.00464`. The reason TD3 executed is exactly the intended one: priority mode did not use the positive-score veto as a hard acceptance rule.

![Priority mechanism diagnostics](figures/polymer_markov_td3_priority_no_handoff_20260517/fig_priority_mechanism.png)

### Interpretation

This result is stronger than the earlier TD3-only result in one important way. The earlier run proved that forced TD3 could perform well. This run shows that the shared unified runner can give TD3 real authority while still retaining emergency fallback logic.

For polymer, there is no evidence here that the soft-handoff ramp is required for performance. The unsmoothed TD3-priority release already reached the TD3-only reward level:

- tail-20 reward improved by `0.1563` versus the no-pass-through unified run
- final reward improved by `0.1763` versus the no-pass-through unified run
- tail-20 reward was essentially tied with TD3-only no safeguard

The reason we still keep the new soft-handoff design is cross-case robustness. Distillation showed a large post-warm-start reward shock when TD3 was released abruptly, while polymer tolerated the abrupt release well. So the right conclusion is:

The polymer case supports TD3-priority as the correct direction. The soft handoff should be viewed as a cross-case stabilizer, not as something polymer needed to recover performance.
