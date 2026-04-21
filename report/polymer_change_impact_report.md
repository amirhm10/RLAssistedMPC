# Polymer Change-Impact Report

Date: 2026-04-20

This report is now self-contained: the comparison tables, figures, explanations, and source notes are inside the report instead of being listed as external links only.

It covers the polymer runs that are relevant to the recent method-scoped conditioning changes:

- residual legacy vs refreshed
- matrix legacy vs two refreshed runs
- structured-matrix legacy vs refreshed
- reidentification legacy vs refreshed
- the disturbance baseline MPC for reference

Two scope notes matter:

- the first refreshed residual and first refreshed matrix runs were executed before the later observer rollback, so they still used `observer_update_alignment="current_measurement_corrector"`
- the later matrix, structured-matrix, and reidentification reruns used the final default `observer_update_alignment="legacy_previous_measurement"`

## Conditioning Mathematics

The new polymer state-conditioning path changes the policy input in two places:

1. Physical `xhat` block:
   fixed min-max uses `z = 2 (x - x_min) / (x_max - x_min) - 1`
   running normalization uses `z_t = clip((x_t - mu_t) / sqrt(var_t + eps), -c, c)`

2. Mismatch extras (`innovation`, `tracking_error`):
   legacy path used hard clipping near `+/-3`
   new path uses `signed_log(e) = sign(e) log(1 + |e|)`

For polymer, the motivation is that fixed min-max gives a very small local slope when the saved width is huge. Running z-score makes the local slope proportional to `1 / sigma_t`, which is exactly the idea used by Stable Baselines3 `VecNormalize` [SB3].

For reidentification, the numerical issue is different. The identification window is solving a noisy regression problem. When the window is not informative enough, the information matrix becomes ill-conditioned, parameter updates become noise-sensitive, and a guard will reject almost every candidate. That is an inference from our logs, and it is consistent with the identification literature on persistency of excitation and regularized least squares [Mu2022] [Binette2016] [Wang2022] [Hochstenbach2011] [Lim2016].

## Run Set And Configs

| Family | Variant | Saved run | base_state_norm_mode | mismatch_transform | observer_alignment | rho_mapping | candidate_guard | blend_validity |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | `Polymer/Data/mpc_results_dist.pickle` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` | `n/a` |
| residual | legacy | `Polymer/Results/td3_residual_disturb/20260413_004620/input_data.pkl` | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `n/a` | `n/a` |
| residual | refreshed (current-observer run) | `Polymer/Results/td3_residual_disturb/20260420_225631/input_data.pkl` | `running_zscore_physical_xhat` | `signed_log` | `current_measurement_corrector` | `exp_raw_tracking` | `n/a` | `n/a` |
| matrix | legacy | `Polymer/Results/td3_multipliers_disturb/20260411_011134/input_data.pkl` | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `n/a` | `n/a` |
| matrix | refreshed (current-observer run) | `Polymer/Results/td3_multipliers_disturb/20260420_215528/input_data.pkl` | `running_zscore_physical_xhat` | `signed_log` | `current_measurement_corrector` | `legacy_clipped_linear` | `n/a` | `n/a` |
| matrix | refreshed (legacy-observer run) | `Polymer/Results/td3_multipliers_disturb/20260420_234944/input_data.pkl` | `running_zscore_physical_xhat` | `signed_log` | `legacy_previous_measurement` | `legacy_clipped_linear` | `n/a` | `n/a` |
| structured | legacy | `Polymer/Results/td3_structured_matrices_disturb/20260409_193654/input_data.pkl` | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `n/a` | `n/a` |
| structured | refreshed (legacy-observer run) | `Polymer/Results/td3_structured_matrices_disturb/20260420_235100/input_data.pkl` | `running_zscore_physical_xhat` | `signed_log` | `legacy_previous_measurement` | `legacy_clipped_linear` | `n/a` | `n/a` |
| reidentification | legacy | `Polymer/Results/td3_reidentification_disturb/20260415_120803/input_data.pkl` | `legacy_fixed_minmax` | `legacy_hard_clip` | `legacy_previous_measurement` | `legacy_clipped_linear` | `fro_only` | `n/a` |
| reidentification | refreshed (legacy-observer run) | `Polymer/Results/td3_reidentification_disturb/20260420_234346/input_data.pkl` | `running_zscore_physical_xhat` | `signed_log` | `legacy_previous_measurement` | `legacy_clipped_linear` | `fro_only` | `off` |

## Performance Summary

| Family | Variant | Tail phys MAE mean | Tail scaled MAE mean | Final-20 reward | Tail MAE delta vs baseline | Tracking raw p99 | Tracking exact abs3 frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | baseline | 0.1701 | 0.3156 | -4.4173 | n/a | n/a | n/a |
| residual | legacy | 0.1700 | 0.3091 | -3.9313 | -0.0001 | n/a | 0.4084 |
| residual | refreshed (current-observer run) | 0.1504 | 0.2747 | -3.6812 | -0.0197 | 203.0935 | 0.0000 |
| matrix | legacy | 0.1722 | 0.3141 | -4.1899 | 0.0020 | n/a | 0.0000 |
| matrix | refreshed (current-observer run) | 0.1655 | 0.2975 | -4.0549 | -0.0047 | 207.6566 | 0.0000 |
| matrix | refreshed (legacy-observer run) | 0.1728 | 0.3111 | -4.0844 | 0.0026 | 207.5347 | 0.0000 |
| structured | legacy | 0.1671 | 0.3100 | -4.3176 | -0.0030 | n/a | 0.0000 |
| structured | refreshed (legacy-observer run) | 0.1673 | 0.3069 | -4.1990 | -0.0028 | 206.2536 | 0.0000 |
| reidentification | legacy | 0.1672 | 0.3203 | -4.5006 | -0.0029 | n/a | 0.4458 |
| reidentification | refreshed (legacy-observer run) | 0.1767 | 0.3245 | -4.4125 | 0.0066 | 205.8527 | 0.0000 |

The top-level result is mixed rather than uniform:

- Residual improved clearly: tail physical MAE mean moved from `0.1700` to `0.1504` (-11.5%), and final-20 reward moved from `-3.9313` to `-3.6812` (-6.4%).
- Matrix improved in the first refreshed run, but not robustly in the second. The current-observer rerun reached `0.1655` tail physical MAE mean, while the later legacy-observer rerun moved back to `0.1728`. Reward stayed better than legacy in both refreshed runs, but the tail-tracking gain was not stable.
- Structured matrix changed the policy input and improved reward modestly, but not tail MAE. Tail physical MAE mean stayed essentially flat, from `0.1671` to `0.1673`, while final-20 reward improved from `-4.3176` to `-4.1990` (-2.7%).
- Reidentification is the one family that did not benefit. Tail physical MAE mean worsened from `0.1672` to `0.1767`, and the refreshed run ended slightly worse than baseline MPC by `0.0066`.

![Polymer performance overview across residual, matrix, structured matrix, and reidentification runs](./polymer_change_impact/figures/polymer_change_performance_overview.png)

The overview figure shows two separate effects:

- the policy input changed a lot in the refreshed polymer runs, because the normalized physical `xhat` block became much wider than the fixed-minmax counterfactual
- the control benefit is family-dependent: residual benefits the most, matrix and structured matrix benefit partly, and reidentification does not

## Family Tail Traces

![Tail traces for residual, matrix, structured matrix, and reidentification polymer runs](./polymer_change_impact/figures/polymer_change_family_tail_traces.png)

The tail traces make the family-level behavior easier to see:

- residual refreshed stays visibly tighter to the setpoint than residual legacy
- matrix current-observer refresh is the strongest matrix run, while the later legacy-observer refresh gives back part of that tail-tracking gain
- structured matrix refreshed is not a no-op, but its gain is milder than residual: reward improves, while tail MAE stays nearly unchanged
- reidentification refreshed does not settle better than legacy, and it does not beat baseline MPC in the tail

## State Conditioning

| Run | Actual full-span med | Fixed CF full-span med | Full-span gain | Actual late-span med | Fixed CF late-span med | Late-span gain |
| --- | --- | --- | --- | --- | --- | --- |
| Residual refreshed | 2.2758 | 0.5049 | 4.5073 | 0.0153 | 0.0035 | 4.3511 |
| Matrix refreshed current-observer | 2.4824 | 0.5402 | 4.5957 | 0.0135 | 0.0033 | 4.1197 |
| Matrix refreshed legacy-observer | 2.2733 | 0.5031 | 4.5183 | 0.0148 | 0.0037 | 4.0410 |
| Structured matrix refreshed legacy-observer | 2.1689 | 0.4717 | 4.5981 | 0.0137 | 0.0034 | 4.0837 |
| Reidentification refreshed legacy-observer | 2.1264 | 0.4618 | 4.6050 | 0.0115 | 0.0029 | 3.9733 |

![Running-normalization gain on polymer physical xhat policy span](./polymer_change_impact/figures/polymer_change_state_conditioning.png)

This figure confirms that the observation-conditioning change is real, not cosmetic. The refreshed polymer runs all present a much wider physical `xhat` signal to the policy than the fixed-minmax counterfactual on the same saved trajectory. That is why it was reasonable to expect an effect from the new defaults, and the residual family is the clearest case where the better state spread translated into better closed-loop performance.

## Mismatch-Feature Diagnostics

![Polymer mismatch feature diagnostics across legacy and refreshed runs](./polymer_change_impact/figures/polymer_change_feature_diagnostics.png)

The feature-diagnostic figure shows why the transform change matters:

- legacy residual tracking piled up at exact `|3|` on `0.4084` of samples on average across outputs
- refreshed residual tracking exact-`|3|` mass is effectively zero, even though raw tracking p99 stays huge at `203.0935`
- the same pattern appears in the matrix, structured-matrix, and reidentification refreshed runs: the raw mismatch magnitude is still large, but the transform is no longer flattening it into a single clipped bucket

So the transform change clearly improved what the policy can distinguish. The remaining question is whether the downstream family can exploit that richer mismatch information. Residual does. Reidentification currently does not.

## Reidentification: Why It Is Not Working

| Variant | Candidate valid frac | Update event frac | Update success frac | Fallback frac | Cond median | Cond p95 | Residual ratio median | Residual ratio p95 | eta_A req p95 | eta_A app p95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| legacy | 0.0021 | 0.1999 | 0.0004 | 0.1995 | 184829.7 | 2932624.9 | n/a | n/a | n/a | 0.8600 |
| refreshed (legacy-observer run) | 0.0011 | 0.1999 | 0.0002 | 0.1997 | 185509.1 | 2199603.2 | 0.9714 | 5.1943 | 0.9044 | 0.9044 |

![Polymer reidentification health diagnostics](./polymer_change_impact/figures/polymer_reid_identification_health.png)

The reidentification failure diagnosis is strong:

- Updates are attempted often enough. The update-event fraction is `0.1999` in the refreshed run, essentially the same as legacy.
- But almost none of those attempts survive the guard. Candidate-valid fraction is only `0.0011`, and update-success fraction is only `0.0002`.
- The regression windows are numerically bad. The refreshed run has a median condition number of `185509.1` and p95 of `2199603.2`.
- Even when a candidate is formed, it usually does not improve the fit enough. The refreshed residual-ratio median is `0.9714`, which is effectively one, and p95 is `5.1943`, which means many candidate windows are much worse than the incumbent model.
- The RL agent is still requesting aggressive identification authority. In the refreshed run, `eta_A` requested p95 is `0.9044` and applied p95 is `0.9044`; for `eta_B`, those are `0.9459` and `0.9459`. Because `blend_validity_mode` is off, there is almost no moderation from the validity layer.

So the main problem is not that the RL state is blind anymore. The main problem is that the online identification layer almost never produces a trustworthy candidate model. The policy is asking for identification action, but the identification engine mostly rejects or falls back, and when it does evaluate a candidate the window is poorly conditioned and often not actually better.

That interpretation is consistent with the literature:

- persistency of excitation is the condition that makes the regression informative enough to uniquely determine parameters [Mu2022]
- in process-control settings, online re-identification may be impossible without enough excitation [Binette2016]
- adaptive MPC papers therefore use information-matrix tests to decide whether a data window is informative enough to trigger a model update [Wang2022]
- when the least-squares problem is ill-conditioned, regularization is a standard fix because it reduces sensitivity to noise and numerical error [Hochstenbach2011] [Lim2016]

The result in this repository matches that story closely. The refreshed reidentification run still sees the mismatch better than before, but the identification subproblem is not healthy enough to convert that information into good model updates.

## Conclusions

From the saved polymer runs, the recent changes did matter, but not in one uniform way:

- yes, the observation-conditioning and mismatch-transform changes are clearly changing the policy-visible state in polymer
- yes, those changes helped the residual family materially
- yes, they affected matrix and structured matrix, but matrix is sensitive to the observer choice and structured matrix is mostly reward-level improvement rather than a clear tail-MAE win
- no, the same changes did not fix polymer reidentification, because that family is currently bottlenecked by candidate-model quality, conditioning, and validity, not only by RL-state scaling

The immediate project implication is that polymer residual remains the strongest beneficiary of the new conditioning path, matrix and structured matrix are secondary candidates for further tuning, and polymer reidentification needs identification-layer changes next: informative-window gating, stronger candidate validation, and likely some regularization in the online estimator.

## Sources

- [SB3] Stable Baselines3 `VecNormalize` implementation and running-mean/running-variance observation normalization.
- [Mu2022] Mu et al., *Persistence of excitation for identifying switched linear systems*, Automatica 2022.
- [Binette2016] Binette and Srinivasan, *On the Use of Nonlinear Model Predictive Control without Parameter Adaptation for Batch Processes*, Processes 2016.
- [Wang2022] *Offset-free ARX-based adaptive model predictive control applied to a nonlinear process*, ISA Transactions 2022.
- [Hochstenbach2011] Hochstenbach and Reichel, *Fractional Tikhonov regularization for linear discrete ill-posed problems*, BIT Numerical Mathematics 2011.
- [Lim2016] Lim and Pang, *l1-regularized recursive total least squares based sparse system identification for the error-in-variables*, SpringerPlus 2016.

[SB3]: https://stable-baselines3.readthedocs.io/en/v2.7.0/_modules/stable_baselines3/common/vec_env/vec_normalize.html
[Mu2022]: https://doi.org/10.1016/j.automatica.2021.110142
[Binette2016]: https://www.mdpi.com/2227-9717/4/3/27
[Wang2022]: https://www.sciencedirect.com/science/article/abs/pii/S0019057821002937
[Hochstenbach2011]: https://doi.org/10.1007/s10543-011-0313-9
[Lim2016]: https://doi.org/10.1186/s40064-016-3120-6