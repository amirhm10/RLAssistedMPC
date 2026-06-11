# Final All-Runners Paper Package

Date: 2026-06-11

## Scope

This report includes the finalized standalone and combined runners for both case studies. The main-text set is OF-MPC, horizon SG-DQN, weights SG-TD3, residual SG-TD3, Markov SG-TD3, and the four-agent combined SG runner for polymer and distillation. All metrics below are computed from saved `input_data.pkl` or baseline pickle bundles; no polymer or Aspen simulations are launched.

## Method Frame

The common closed-loop structure is offset-free MPC with an RL supervisor. For a supervisor-gated agent, the executed action is selected by comparing a policy score and a supervisor score:

$$ a_{\mathrm{exec}} = a_{\pi}\ \mathrm{if}\ S(a_{\pi}) - S(a_{\mathrm{sup}}) > m,\ \mathrm{else}\ a_{\mathrm{sup}}. $$

Tracking metrics are computed in physical output units by converting saved scaled-deviation setpoints back to physical coordinates using each bundle's `data_min`, `data_max`, and `steady_states` fields.

## Final Performance Table

| Plant | Runner | Role | Tail reward | Delta vs OF-MPC | Worst post-warm | First live | Tail y1 MAE | Tail y2 MAE | Tail input TV | Tail policy frac |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| polymer | OF-MPC | baseline | -3.808 | 0.000 | -4.384 | -4.338 | 0.052 | 0.129 | 18257.830 | NA |
| polymer | Horizon SG-DQN | standalone | -2.662 | 1.146 | -4.383 | -3.318 | 0.046 | 0.118 | 38485.802 | 0.187 |
| polymer | Weights SG-TD3 | standalone | -2.699 | 1.109 | -4.581 | -3.488 | 0.046 | 0.113 | 29718.303 | 0.101 |
| polymer | Residual SG-TD3 | standalone | -2.935 | 0.874 | -7.106 | -4.212 | 0.040 | 0.119 | 30113.237 | 0.068 |
| polymer | Markov SG-TD3 | standalone | -2.857 | 0.951 | -4.714 | -4.023 | 0.041 | 0.093 | 16917.831 | 0.395 |
| polymer | Combined SG | combined | -1.818 | 1.991 | -6.373 | -2.998 | 0.029 | 0.079 | 37620.266 | NA |
| distillation | OF-MPC | baseline | -0.303 | 0.000 | -0.337 | -0.233 | 0.002 | 0.192 | 1079156.327 | NA |
| distillation | Horizon SG-DQN | standalone | 12.394 | 12.697 | 7.307 | 10.869 | 0.001 | 0.183 | 1189507.162 | 0.130 |
| distillation | Weights SG-TD3 | standalone | 14.416 | 14.719 | 8.307 | 9.572 | 0.001 | 0.160 | 1081027.868 | 0.590 |
| distillation | Residual SG-TD3 | standalone | 30.370 | 30.672 | -12.658 | 8.544 | 0.001 | 0.072 | 1415315.916 | 0.488 |
| distillation | Markov SG-TD3 | standalone | 27.100 | 27.403 | -18.522 | 10.089 | 0.001 | 0.058 | 1282626.863 | 0.407 |
| distillation | Combined SG | combined | 30.132 | 30.435 | 0.934 | 11.455 | 0.001 | 0.070 | 1395862.125 | NA |

## Main Findings

- Polymer: best tail reward among final runners is `Combined SG` with delta `1.991` versus OF-MPC.
- Polymer combined: tail delta is `1.991` and worst post-warm reward is `-6.373`.
- Polymer residual vs Markov: residual tail delta `0.874`, Markov tail delta `0.951`.
- Distillation: best tail reward among final runners is `Residual SG-TD3` with delta `30.672` versus OF-MPC.
- Distillation combined: tail delta is `30.435` and worst post-warm reward is `0.934`.
- Distillation residual vs Markov: residual tail delta `30.672`, Markov tail delta `27.403`.

## Combined-Agent Attribution

The table below is diagnostic attribution, not causal attribution. Causal attribution still requires leave-one-agent-out combined reruns under the same disturbance and seed.

| Plant | Agent | Combined tail delta | Tail policy frac | Tail supervisor frac | Tail fallback frac | Tail gate advantage | Tail authority | Replay size | Critic finite frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| polymer | horizon | 1.991 | 0.211 | 0.039 | 0.750 | 0.675 | 9.604 | 150000 | NA |
| polymer | markov | 1.991 | 0.004 | 0.753 | 0.243 | 0.164 | 0.063 | 150000 | 1.000 |
| polymer | weights | 1.991 | 0.056 | 0.944 | NA | 0.688 | 0.054 | 150000 | NA |
| polymer | residual | 1.991 | 0.048 | 0.952 | NA | 0.398 | 0.015 | 150000 | 1.000 |
| distillation | horizon | 30.435 | 0.210 | 0.040 | 0.750 | 8.049 | 6.944 | 40000 | NA |
| distillation | markov | 30.435 | 0.000 | 0.001 | 0.999 | 379765.306 | 0.002 | 40000 | 1.000 |
| distillation | weights | 30.435 | 0.575 | 0.425 | NA | 16.256 | 0.123 | 40000 | NA |
| distillation | residual | 30.435 | 0.503 | 0.496 | NA | 8.794 | 0.005 | 40000 | 1.000 |

## Diagnostic Caveats

- Standalone success is strong evidence that each agent family is viable by itself under the final disturbance scenario. It is not, by itself, proof that the same agent caused the combined-run improvement.
- The combined attribution table should therefore be read as a mechanism diagnostic: policy execution, fallback, action authority, replay health, and gate advantage. The causal table still needs the leave-one-agent-out reruns.
- Polymer combined: Markov tail policy execution is `0.004` with fallback `0.243`, so the combined gain is real but Markov's direct causal share is still unresolved.
- Distillation combined: Markov tail fallback is `0.999` while weights and residual have much larger policy fractions. Treat the current combined win as a combined-supervisor result, not as a standalone Markov credit claim.

## Next Experiments For The Paper

| Priority | Experiment or log | Scope | Minimum output | Paper use |
| --- | --- | --- | --- | --- |
| P0 | leave-one-agent-out combined ablations | polymer and distillation | no-horizon, no-Markov, no-weights, no-residual combined bundles with the same seed and disturbance | report C_i = J(all agents) - J(all except i) beside the diagnostic source-fraction table |
| P0 | no-SG standalone ablations | horizon, Markov, weights, and residual where practical | keep hard MPC/physical clipping, remove only the SG accept/reject decision, and store identical metrics | main safety claim: SG reduces bad handoff, fallback burden, and worst post-warm episodes |
| P0 | distillation combined provenance rerun | distillation combined | one rerun after the config-snapshot guard with param-noise settings visible in saved JSON/pickle | avoids a provenance footnote in the final combined distillation table |
| P1 | common rescoring freeze | all final bundles | single script that emits final summary, attribution, safety, replay, and logging-gap CSVs | reproducible table-generation method for the journal supplement |
| P1 | seed replication | headline runners if compute allows | two extra seeds for OF-MPC, best standalone, combined, and key ablations | median and interquartile intervals rather than single-run claims |
| P1 | run-native replay and reward audits | every active RL runner | compact replay audit CSV and reward-component episode CSV | appendix tables explaining whether the replay buffer contained informative off-steady-state samples |

## Logging Gap Extract

The full machine-readable gap list is saved as `final_logging_gap_table.csv`. The most paper-relevant gaps are:

| Priority | Plant | Runner or agent | Gap | Recommended fix |
| --- | --- | --- | --- | --- |
| P0 | distillation | Combined SG | combined agent_config_snapshot missing | rerun this combined case once after the config snapshot guard and archive the resulting bundle |
| P0 | distillation | combined Markov | tail Markov branch mostly fallback | inspect solver/source codes and run no-Markov combined ablation under the same disturbance and seed |

## Paper Readiness

What is ready:

- Both plants now have saved standalone and combined bundles with reward histories, trajectories, source logs, losses, and replay snapshots for combined runs.
- Distillation combined now has `input_data.pkl`, closing the earlier attribution gap.
- Polymer Markov and residual parameter-noise standalone runs are now available and should be used as final standalone rows.

What still needs one more paper-safe pass:

- Run leave-one-agent-out combined ablations: no-horizon, no-Markov, no-weights, and no-residual for both plants. This is the clean causal attribution experiment.
- Run no-SG standalone ablations for TD3 families and DQN horizon where practical. Keep hard MPC and physical clipping; remove only the supervisor gate. This supports the safety-gate claim.
- Re-run distillation combined once after the `agent_config_snapshot` guard commit so the final combined bundle explicitly records Markov/residual parameter-noise config provenance.
- Freeze a single common rescoring script for reward, IAE, RMSE, max error, input total variation, source fractions, and safety burden. Then regenerate all main tables from that script.
- Add at least two more seeds for the paper headline rows if compute time allows. If not, report this as single-seed final evidence and avoid statistical claims.

Recommended new logs for every future final run:

- `episode_metrics.csv` with reward, reward components, physical IAE/RMSE/max error, input total variation, saturation/projection/fallback fractions, and source fractions.
- `agent_attribution_summary.csv` for combined runs with one row per agent.
- `config_snapshot.json` and `agent_config_snapshot.json` saved as standalone files in addition to pickle storage.
- Compact replay audit CSV per active agent: replay size, capacity, state dimension, feature ranges, tail-steady feature ranges, and source mix.
- Seed, git commit, baseline bundle path, disturbance profile, setpoint schedule identifier, plant/model identifiers, and exact runner script name.

## Figures And Tables Generated

- `report/figures/final_all_runners_paper_package_20260611/final_all_runners_summary.csv`
- `report/figures/final_all_runners_paper_package_20260611/final_combined_agent_attribution.csv`
- `report/figures/final_all_runners_paper_package_20260611/final_logging_gap_table.csv`
- `report/figures/final_all_runners_paper_package_20260611/final_paper_next_experiments.csv`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_tail_reward_delta_by_family.png`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_reward_histories_all_runners.png`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_metric_panels_all_runners.png`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_tail_tracking_inputs_polymer.png`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_tail_tracking_inputs_distillation.png`
- `report/figures/final_all_runners_paper_package_20260611/fig_final_combined_source_fractions.png`

## Data Files Included

- `Polymer/Results/mpc_offsetfree_disturb_unified/20260608_143324/input_data.pkl`
- `Polymer/Results/horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_mismatch/20260608_151338/input_data.pkl`
- `Polymer/Results/sg_td3_weights_critic_warm3_conservative_disturb_mismatch/20260608_152025/input_data.pkl`
- `Polymer/Results/sg_td3_residual_critic_warm3_conservative_disturb_mismatch/20260610_131706/input_data.pkl`
- `Polymer/Results/sg_td3_markov_critic_warm3_ls_else_mpc_shadow_disturb_mismatch/20260610_170822/input_data.pkl`
- `Polymer/Results/combined_disturb_sg__h_sg_dqn_mismatch__markov_sg_td3_mismatch__w_sg_td3_mismatch__r_sg_td3_mismatch_no_rho/20260611_045747/input_data.pkl`
- `Distillation/Data/mpc_results_disturb_fluctuation.pickle`
- `Distillation/Results/distillation_horizon_sg_disturb_fluctuation/20260609_195213/input_data.pkl`
- `Distillation/Results/distillation_weights_sg_disturb_fluctuation/20260609_203050/input_data.pkl`
- `Distillation/Results/distillation_residual_sg_disturb_fluctuation/20260609_203500/input_data.pkl`
- `Distillation/Results/distillation_markov_sg_disturb_fluctuation/20260611_132048/input_data.pkl`
- `Distillation/Results/distillation_combined_sg_disturb_fluctuation/20260611_095216/input_data.pkl`
