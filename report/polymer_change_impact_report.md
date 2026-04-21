# Polymer Residual And Matrix Change-Impact Report

Date: 2026-04-20

This report compares the latest polymer residual and matrix runs against the closest pre-change reference runs and the disturbance baseline MPC.

Important scope note:

- the latest saved polymer runs analyzed here were generated before the later observer-default rollback
- so the refreshed runs in this report still use `observer_update_alignment="current_measurement_corrector"`
- the report therefore answers the question "did the changes affect the runs you actually executed?" rather than "what would happen under the final defaults after rollback?"

## Runs Compared

| Family | Variant | Saved run |
| --- | --- | --- |
| baseline | baseline | `Polymer\Data\mpc_results_dist.pickle` |
| residual | legacy | `Polymer\Results\td3_residual_disturb\20260413_004620\input_data.pkl` |
| residual | refreshed | `Polymer\Results\td3_residual_disturb\20260420_225631\input_data.pkl` |
| matrix | legacy | `Polymer\Results\td3_multipliers_disturb\20260411_011134\input_data.pkl` |
| matrix | refreshed | `Polymer\Results\td3_multipliers_disturb\20260420_215528\input_data.pkl` |

## Config Comparison

| Family | Variant | base_state_norm_mode | mismatch_feature_transform_mode | observer_update_alignment | rho_mapping_mode | deadband |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |
| residual | legacy | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `False` |
| residual | refreshed | `running_zscore_physical_xhat` | `signed_log` | `current_measurement_corrector` | `exp_raw_tracking` | `True` |
| matrix | legacy | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `False` |
| matrix | refreshed | `running_zscore_physical_xhat` | `signed_log` | `current_measurement_corrector` | `legacy_clipped_linear` | `False` |

## Performance Summary

| Family | Variant | Tail physical MAE mean | Tail scaled MAE mean | Final-20 avg reward | Tail MAE vs baseline |
| --- | --- | --- | --- | --- | --- |
| baseline | baseline | 0.1701 | 0.3156 | -4.4173 | n/a |
| residual | legacy | 0.1700 | 0.3091 | -3.9313 | -0.0001 |
| residual | refreshed | 0.1504 | 0.2747 | -3.6812 | -0.0197 |
| matrix | legacy | 0.1722 | 0.3141 | -4.1899 | 0.0020 |
| matrix | refreshed | 0.1655 | 0.2975 | -4.0549 | -0.0047 |

## State-Conditioning Summary

| Family | View | full-span median | late-span median | late/full ratio |
| --- | --- | --- | --- | --- |
| residual | legacy actual | 0.4223 | 0.0047 | 0.0112 |
| residual | refreshed actual | 2.2758 | 0.0153 | 0.0067 |
| residual | refreshed fixed counterfactual | 0.5049 | 0.0035 | 0.0068 |
| matrix | legacy actual | 0.4787 | 0.0034 | 0.0071 |
| matrix | refreshed actual | 2.4824 | 0.0135 | 0.0054 |
| matrix | refreshed fixed counterfactual | 0.5402 | 0.0033 | 0.0060 |

## Main Findings

1. The refreshed residual run did make a material difference. Tail physical MAE mean improved from `0.1700` to `0.1504` (-11.5%), and final-20 reward improved from `-3.9313` to `-3.6812` (-6.4%). The refreshed run also reduced `rho=1` fraction from `0.5547` to `0.1673` and activated the new deadband on `0.1615` of steps.
2. The refreshed matrix run also changed behavior, but the gain is smaller and more mixed. Tail physical MAE mean improved from `0.1722` to `0.1655` (-3.9%), and final-20 reward improved from `-4.1899` to `-4.0549` (-3.2%). Output 1 improved more clearly than output 2.
3. The new polymer state conditioning definitely changed what the policy saw. On the refreshed residual trajectory, the median full-span of the policy-visible physical `xhat` block is `2.2758` under running normalization versus `0.5049` under a fixed-minmax counterfactual. On the refreshed matrix trajectory, the same comparison is `2.4824` versus `0.5402`.
4. The residual mismatch features no longer pile up at the hard clip. In the legacy residual run, transformed tracking hit exact `|3|` on `0.4084` of samples on average across outputs. In the refreshed residual run, exact-`|3|` mass is essentially zero while the raw tracking p99 is still very large at `203.0935`. That means the new transform is exposing severity information instead of flattening it.
5. The matrix refreshed run also exposes much more mismatch-feature dynamic range than the legacy run. The mean transformed tracking p99 rises from `1.3083` to `5.1685`, and the raw tracking p99 in the refreshed run is `207.6566`. So the changes absolutely affected the matrix policy input, even though the closed-loop improvement is modest rather than dramatic.

## Figures

- [Reward curves](./polymer_change_impact/figures/polymer_change_reward_curves.png)
- [Tail traces](./polymer_change_impact/figures/polymer_change_tail_traces.png)
- [Performance bars](./polymer_change_impact/figures/polymer_change_performance_bars.png)
- [State conditioning](./polymer_change_impact/figures/polymer_change_state_conditioning.png)
- [Feature diagnostics](./polymer_change_impact/figures/polymer_change_feature_diagnostics.png)

## Data Tables

- [Config summary](./polymer_change_impact/data/config_summary.csv)
- [Performance summary](./polymer_change_impact/data/performance_summary.csv)
- [State conditioning summary](./polymer_change_impact/data/state_conditioning_summary.csv)