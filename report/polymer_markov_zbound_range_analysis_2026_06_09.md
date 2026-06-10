# Polymer Markov z-bound Range Analysis

Date: 2026-06-09

## Objective

Assess whether the polymer SG-TD3 Markov standalone runs improve as the Markov correction range expands, and diagnose the newest `z_bound = 0.70` run with `advantage_margin = 1.0`.

## Files Inspected

- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260608_155310/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260608_193914/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260609_013736/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260609_160219/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260609_220217/input_data.pkl`
- Matching compare bundles under `Polymer/Results/disturb_compare_sg_td3_markov_critic_warm3_ls_else_mpc_shadow_mismatch/`
- Analysis script: `report/scripts/analyze_polymer_markov_zbound_20260609.py`
- Summary CSV: `report/figures/polymer_markov_zbound_20260609/polymer_markov_zbound_summary.csv`

## Method

The Markov action perturbs the finite-horizon dynamic matrix through four `io_pair_gain` coordinates. In the active runner, the normalized TD3 action `a_k` is mapped to a bounded Markov correction by

$$ z_k = z_{\max} a_k,\quad a_k \in [-1,1]^4. $$

For four coordinates, the largest possible 2-norm is

$$ \|z_k\|_2 \le 2 z_{\max}. $$

The analysis uses the paired compare bundles for OF-MPC reward rather than the `y_mpc` and `u_mpc` fields inside the RL bundle, because the Markov RL bundle can mirror those fields from the RL trajectory.

## Reward Results

| z_bound | SG margin | Tail-20 RL reward | Tail-20 OF-MPC reward | Tail reward delta | Final reward delta | Worst post-warm RL reward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.0 | -3.7572 | -3.8084 | 0.0512 | 0.0520 | -4.4312 |
| 0.10 | 0.0 | -3.7485 | -3.8084 | 0.0598 | 0.0424 | -4.4621 |
| 0.20 | 0.0 | -3.5404 | -3.8084 | 0.2680 | 0.2429 | -4.6140 |
| 0.50 | 0.0 | -3.0706 | -3.8084 | 0.7378 | 0.7494 | -5.7549 |
| 0.70 | 1.0 | -2.9041 | -3.8084 | 0.9043 | 0.8932 | -7.7115 |

The `0.70` run is the best tail performer so far. It improves tail reward by about `0.904` over OF-MPC, compared with `0.738` for `0.50`.

The tradeoff is the worst post-warm episode. The `0.70` run has the largest release downside, with worst post-warm reward `-7.7115`.

![Reward and z usage](figures/polymer_markov_zbound_20260609/polymer_markov_zbound_reward_and_usage.png)

## Action And Safety Diagnostics

| z_bound | SG margin | Tail z 2-norm q95 | Tail z 2-norm max | Tail policy fraction | Tail near-coordinate-cap fraction | Tail shadow projection fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.0 | 0.1000 | 0.1000 | 0.8396 | 0.8554 | 0.4379 |
| 0.10 | 0.0 | 0.2000 | 0.2000 | 0.5848 | 0.6733 | 0.8363 |
| 0.20 | 0.0 | 0.3923 | 0.4000 | 0.8991 | 0.5829 | 0.9637 |
| 0.50 | 0.0 | 0.9866 | 1.0000 | 0.6931 | 0.1725 | 0.8077 |
| 0.70 | 1.0 | 0.9216 | 1.4000 | 0.1338 | 0.0535 | 0.4366 |

The `0.50` run was not merely better because of noise. It used the larger authority:

- The tail q95 z 2-norm is `0.9866`, close to the four-coordinate vector cap of `1.0`.
- Tail policy selection is still substantial at `69.3%`, so the actor is active.
- Tail shadow projection remains high at `80.8%`, meaning the inactive shadow safety layer would still modify many requested actions.

The `0.70` run changes two variables at once: larger range and stricter SG margin. It has a better tail reward, but the mechanism is different:

- Tail policy selection drops from `69.3%` at `0.50` to `13.4%` at `0.70`.
- Post-warm policy selection drops from `63.5%` to `6.9%`.
- The tail improvement is therefore not mainly caused by more frequent TD3 execution. It is mostly a wider, supervisor-dominated Markov correction regime with occasional TD3 actions.
- The stricter margin does reduce actor passes, but it does not fix the release problem. Worst post-warm reward worsens from `-5.7549` to `-7.7115`.

![Source and safety diagnostics](figures/polymer_markov_zbound_20260609/polymer_markov_zbound_source_and_safety.png)

## Interpretation

The `0.70` result supports the observation that the latest Markov run is better in the tail. It also supports the hypothesis that `advantage_margin = 1.0` reduces the number of actor actions that pass the SG gate.

However, the stricter margin did not solve the post-warm transient. The post-warm crash became worse even though far fewer actor actions passed. That suggests the early downside is not only a TD3-policy release problem. The wider LS/MPC supervisor candidates can also become aggressive when the z range expands.

The evidence now says:

- Larger range improves mature-tail reward up to the tested `0.70` case.
- The SG margin of `1.0` strongly reduces policy passes.
- Tail reward can still improve with fewer TD3 passes, so the supervisor Markov candidate is doing useful work under the wider range.
- Worst post-warm reward worsens as the range expands, so release safety needs a range schedule, not only a stricter critic margin.

## Recommended Next Experiment

Do not expand beyond `0.70` yet. The next clean ablation should separate the range effect from the margin effect:

- File: `RL_assisted_MPC_markov_unified.py`
- Run A: `z_bound = 0.70`, `advantage_margin = 0.0`
- Run B: `z_bound = 0.50`, `advantage_margin = 1.0`
- Compare both against the current `z_bound = 0.70`, `advantage_margin = 1.0` run using:
  - tail-20 reward delta versus OF-MPC,
  - worst first-20 post-warm reward,
  - tail z 2-norm q95 and max,
  - tail near-coordinate-cap fraction,
  - shadow projection fraction,
  - SG policy versus supervisor fractions.

If Run A improves tail reward but keeps the same poor release, the range is the main release-risk source. If Run B reduces the release crash but loses tail reward, the margin is too conservative for the tail. The likely final design is a post-warm z-bound ramp, for example starting near `0.20` and ramping to `0.70`, with a margin that relaxes from `1.0` toward `0.0` after the first 20 post-warm episodes.

## Remaining Uncertainty

This sweep has one run per setting, and the `0.70` point changes both range and SG margin. The tail trend is strong, but the causal mechanism is not isolated. A two- or three-seed repeat of `0.50 margin 0.0`, `0.70 margin 1.0`, and the two proposed ablations is needed before locking in the larger range.
