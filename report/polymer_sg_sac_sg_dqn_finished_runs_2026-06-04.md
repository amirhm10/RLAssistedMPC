# Polymer SG-SAC and SG-DQN Finished-Run Analysis

## Objective

This report analyzes the five completed polymer disturbed runs that finished on
June 3, 2026 before the later SG-SAC deterministic-gate and hidden-7 release
code changes were implemented. The goal is to determine what changed relative to
the nearest previous runs and how much the new SG-SAC or SG-DQN versions
improved.

The five analyzed current runs are:

- SG-SAC residual: `Polymer/Results/sg_sac_residual_critic_warm3_zero_shadow_disturb/20260603_205207/input_data.pkl`
- SG-SAC weights: `Polymer/Results/sg_sac_weights_critic_warm3_identity_shadow_disturb/20260603_205218/input_data.pkl`
- SG-SAC Markov: `Polymer/Results/sg_sac_markov_critic_warm3_ls_else_mpc_shadow_disturb/20260603_212647/input_data.pkl`
- SG-DQN horizon: `Polymer/Results/horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260603_210127/input_data.pkl`
- SG-dueling-DQN horizon: `Polymer/Results/dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260603_210913/input_data.pkl`

These runs are not the new `detgate_hidden7` SG-SAC runs. The saved configs show
3 post-warm action-freeze subepisodes and 3 actor-freeze subepisodes.

## Method Summary

For each current run, I compared against the nearest previous polymer disturbed
ancestor:

- SG-SAC residual versus SG-TD3 residual critic-warm-3 conservative.
- SG-SAC weights versus SG-TD3 weights critic-warm-3 conservative.
- SG-SAC Markov versus SG-TD3 Markov critic-warm-3 LS-or-MPC supervisor.
- SG-DQN horizon versus the previous standard DQN horizon unified run.
- SG-dueling-DQN horizon versus the previous dueling DQN horizon unified run.

The comparison uses post-warm metrics and the final 20 subepisodes, where one
subepisode is 800 simulation steps. Tail-20 therefore means the final 16,000
steps. For reward, higher is better. For tracking error and input movement,
lower is better.

## Mathematical Interpretation

The supervisor-gated controller evaluates a learned policy proposal
`a_pi(s)` against a supervisor action `a_sup(s)`. The continuous SG-TD3 and
pre-detgate SG-SAC runs used a conservative score of the form:

$$ S(s,a) = \min(Q_1(s,a),Q_2(s,a)) - \lambda_Q |Q_1(s,a)-Q_2(s,a)| - \lambda_{\mathrm{sup}} \|a-a_{\mathrm{sup}}\|_2^2 - \lambda_{\mathrm{prev}} \|a-a_{\mathrm{prev}}\|_2^2. $$

The gate admits the policy if:

$$ S(s,a_\pi) > S(s,a_{\mathrm{sup}}) + m. $$

For residual and weights, the saved current SG-SAC configs used margin
`m = 0.5`. For Markov, `m = 0.0`. The completed SG-SAC runs used stochastic
SAC policy samples for live train-mode gate candidates. They did not include
the later deterministic candidate mode, twin-critic dominance veto, or hidden-7
actor/alpha training release.

For SG-DQN horizon selection, the same idea is applied to discrete horizon
actions:

$$ a_{\mathrm{exec}} = \begin{cases} a_\pi, & Q(s,a_\pi) > Q(s,a_{\mathrm{sup}}) + m, \\ a_{\mathrm{sup}}, & \mathrm{otherwise}. \end{cases} $$

The supervisor horizon was the OF-MPC default pair `(Hp,Hc) = (9,3)`, encoded as
action `6`, with zero gate margin.

## Quantitative Results

Positive percentages below mean improvement versus the previous run. Negative
percentages mean degradation. OF-MPC reward improvement is computed from the
saved compare bundles, whose tail OF-MPC reward is `-4.417343`.

| runner | tail reward current | tail reward previous | tail reward vs previous pct | tail reward vs OF-MPC pct | tail scaled MAE current | tail scaled MAE previous | tail scaled MAE vs previous pct | eta MAE current | eta MAE previous | eta MAE vs previous pct | T MAE current | T MAE previous | T MAE vs previous pct | policy post current | policy post previous |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SG-SAC residual | -3.008 | -2.956 | -1.75 | 31.92 | 1.254 | 0.921 | -36.22 | 13.399 | 12.420 | -7.88 | 4.868 | 3.950 | -23.23 | 11.2% | 7.1% |
| SG-SAC weights | -2.841 | -2.703 | -5.11 | 35.68 | 1.223 | 1.210 | -1.10 | 15.316 | 14.354 | -6.71 | 4.595 | 4.205 | -9.27 | 10.6% | 6.9% |
| SG-SAC Markov | -3.792 | -3.804 | 0.33 | 14.17 | NA | NA | NA | NA | NA | NA | NA | NA | NA | 50.0% | 28.4% |
| SG-DQN horizon | -2.631 | -2.680 | 1.83 | 40.43 | 1.097 | 1.101 | 0.35 | 14.031 | 14.374 | 2.39 | 4.333 | 4.375 | 0.95 | 16.0% | NA |
| SG-dueling DQN | -2.555 | -2.672 | 4.39 | 42.17 | 1.108 | 1.157 | 4.28 | 13.990 | 14.810 | 5.54 | 4.413 | 4.531 | 2.61 | 16.9% | NA |

Generated evidence files:

- Metric CSV: `report/figures/2026-06-04_polymer_sg_runs/polymer_sg_run_metrics.csv`
- Reward chart: `report/figures/2026-06-04_polymer_sg_runs/tail_reward_comparison.png`
- Scaled tracking chart: `report/figures/2026-06-04_polymer_sg_runs/tail_scaled_tracking_mae.png`
- Viscosity MAE chart: `report/figures/2026-06-04_polymer_sg_runs/tail_eta_mae.png`
- Temperature MAE chart: `report/figures/2026-06-04_polymer_sg_runs/tail_T_mae.png`
- SG policy authority chart: `report/figures/2026-06-04_polymer_sg_runs/sg_policy_fraction.png`

## Main Interpretation

The SG-DQN family is the clear improvement among these five runs. Standard
SG-DQN improved tail reward by `1.83%`, viscosity tail MAE by `2.39%`, and
temperature tail MAE by `0.95%` versus the previous DQN horizon run. The
dueling SG-DQN was stronger, improving tail reward by `4.39%`, tail scaled
tracking MAE by `4.28%`, viscosity MAE by `5.54%`, and temperature MAE by
`2.61%`.

The SG-SAC residual run did not improve over the earlier SG-TD3 residual run.
It admitted policy actions more often after warm start, increasing from `7.1%`
to `11.2%`, but tail reward worsened by `1.75%`, tail scaled MAE worsened by
`36.22%`, and temperature MAE worsened by `23.23%`. This supports the earlier
diagnosis that residual SG-SAC was over-admitting risky stochastic residual
samples without enough critic agreement.

The SG-SAC weights run is mixed but not a tracking improvement. Post-warm input
movement decreased by `9.32%`, but tail reward worsened by `5.11%`, viscosity
MAE worsened by `6.71%`, and temperature MAE worsened by `9.27%`. This looks
like a quieter controller that did not improve output tracking.

The SG-SAC Markov run is only slightly better than SG-TD3 Markov in reward:
tail reward improved by `0.33%`. It also improved stored Markov diagnostics,
with prediction-score mean decreasing from `0.0369` to `0.0312` and gain-drift
mean decreasing from `0.0339` to `0.0297`. However, the Markov run bundle does
not save `tracking_error_log` or `tracking_error_raw_log`, so I cannot claim a
direct output-tracking improvement for Markov from this bundle alone.

All five current SG runs still beat the saved OF-MPC reward baseline in the
compare bundles. The tail reward improvement over OF-MPC ranges from `14.17%`
for SG-SAC Markov to `42.17%` for SG-dueling DQN. This is useful, but it does
not mean every new SG version beat its previous RL ancestor.

## Implementation Consistency Checks

The saved configs show that the completed SG-SAC runs used the older SG-SAC
configuration:

- action freeze subepisodes: `3`
- actor freeze subepisodes: `3`
- no `candidate_mode = "deterministic"`
- no `critic_dominance_gate_enabled`
- no sampled-action supervisor BC loss

Therefore, these results are the pre-change baseline for the later
`detgate_hidden7` implementation. They should be used as the acceptance
comparison for the new SG-SAC residual, weights, and Markov reruns.

The per-run bundles contain `y_mpc` and `y_line_full`, but in at least the
horizon SG-DQN bundle these fields mirror the live trajectory rather than
providing a separate OF-MPC tracking baseline. I therefore used the compare
bundle rewards for OF-MPC comparison and did not compute OF-MPC tracking MAE
from those per-run fields.

## Risks and Limitations

- This is a single-run comparison for each family. Most current bundles have
  `seed=None`, while the dueling horizon run records `seed=7`. Seed spread is
  needed before claiming robust superiority.
- Markov output-tracking error logs are missing, so Markov improvement is based
  on reward and model-prediction diagnostics only.
- The previous horizon ancestors are non-SG DQN and dueling-DQN runs, so their
  SG policy-fraction fields are not available.
- The SG-SAC residual collapse is consistent with stochastic candidate release,
  but this analysis cannot prove the mechanism without rerunning with the
  deterministic/dominance gate and comparing gate decisions step by step.

## Literature Connections

No new external citations were added in this pass. The local result pattern is
consistent with the standard safe policy improvement interpretation: value
gates help only when the critic ranking is reliable. For residual actions, a
small admitted fraction can still dominate closed-loop behavior because the
action changes the MPC move directly. This is why the later deterministic
candidate and twin-critic dominance changes are scientifically well motivated
without adding a manual residual safety layer.

## Recommended Next Experiment

Run the new SG-SAC `detgate_hidden7` wrappers using the same disturbed polymer
setup:

- `RL_assisted_MPC_residual_supervisor_gated_sac_critic_warm_unified.py`
- `RL_assisted_MPC_weights_supervisor_gated_sac_critic_warm_unified.py`
- `RL_assisted_MPC_markov_supervisor_gated_sac_critic_warm_unified.py`

Acceptance targets:

- Residual SG-SAC should recover at least the previous SG-TD3 residual tail
  benchmark: tail reward better than `-2.956`, tail scaled MAE below `0.921`,
  and temperature MAE below `3.950`.
- Weights SG-SAC should beat the SG-TD3 weights tail reward `-2.703` or, if it
  remains quieter, it should show a clear tracking improvement rather than only
  reduced input movement.
- Markov SG-SAC should keep the small reward/model-diagnostic gain while adding
  explicit saved tracking-error logs so reward and output tracking can be
  checked together.

For SG-DQN, the dueling SG-DQN run is the strongest of the five and should be
kept as the current horizon baseline when comparing future continuous SG-SAC
methods.

## Files Inspected

- `change-reports/2026-06-03_polymer_sg_dqn_horizon_runners.md`
- `change-reports/2026-06-03_polymer_sg_sac_continuous_runners.md`
- `change-reports/2026-06-03_sac_corrections_and_sg_sac_core.md`
- `SACAgent/supervisor_gated_sac_agent.py`
- `DQN/supervisor_gated_dqn_agent.py`
- `TD3Agent/supervisor_replay_buffer.py`
- The five current run bundles listed in the objective.
- The five previous comparison bundles listed in the method summary.
- The five current `disturb_compare_*` bundles used for OF-MPC reward.

## Files Changed

- `report/polymer_sg_sac_sg_dqn_finished_runs_2026-06-04.md`
- `tools/analyze_polymer_sg_finished_runs.py`
- `report/figures/2026-06-04_polymer_sg_runs/polymer_sg_run_metrics.csv`
- `report/figures/2026-06-04_polymer_sg_runs/tail_reward_comparison.png`
- `report/figures/2026-06-04_polymer_sg_runs/tail_scaled_tracking_mae.png`
- `report/figures/2026-06-04_polymer_sg_runs/tail_eta_mae.png`
- `report/figures/2026-06-04_polymer_sg_runs/tail_T_mae.png`
- `report/figures/2026-06-04_polymer_sg_runs/sg_policy_fraction.png`
