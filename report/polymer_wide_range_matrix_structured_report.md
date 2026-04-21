# Polymer Wide-Range Matrix and Structured Report

Date: 2026-04-21

This report focuses on the latest widened-range polymer matrix and structured-matrix runs and compares them against the previous legacy/narrow runs, the baseline MPC, and the latest polymer reidentification run.

The goal is to answer four questions with data from the saved runs:

1. Why did the widened matrix/structured methods improve, and why did reidentification still fail?
2. Why does structured wide achieve better reward while not improving the held-out evaluation episode?
3. Why do the wide runs first degrade and then recover, and what is a practical fix?
4. Can the range be widened further, and what should limit that when the controller uses a state-space model?

## Run Set

| Run | Saved bundle | run_mode | state_mode | observer | base_state_norm | mismatch_transform | range_profile | update_family | disturbance consistent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline MPC | `Polymer/Data/mpc_results_dist.pickle` | disturb | n/a | n/a | n/a | n/a | n/a | n/a | yes |
| Matrix legacy | `Polymer/Results/td3_multipliers_disturb/20260411_011134/input_data.pkl` | disturb | mismatch | n/a | n/a | n/a | n/a | n/a | yes |
| Matrix narrow refresh | `Polymer/Results/td3_multipliers_disturb/20260420_234944/input_data.pkl` | disturb | mismatch | legacy_previous_measurement | running_zscore_physical_xhat | signed_log | n/a | n/a | yes |
| Matrix wide | `Polymer/Results/td3_multipliers_disturb/20260421_011145/input_data.pkl` | disturb | mismatch | legacy_previous_measurement | running_zscore_physical_xhat | signed_log | n/a | n/a | yes |
| Structured legacy | `Polymer/Results/td3_structured_matrices_disturb/20260409_193654/input_data.pkl` | disturb | mismatch | n/a | n/a | n/a | tight | block | yes |
| Structured narrow refresh | `Polymer/Results/td3_structured_matrices_disturb/20260420_235100/input_data.pkl` | disturb | mismatch | legacy_previous_measurement | running_zscore_physical_xhat | signed_log | tight | block | yes |
| Structured wide | `Polymer/Results/td3_structured_matrices_disturb/20260421_013208/input_data.pkl` | disturb | mismatch | legacy_previous_measurement | running_zscore_physical_xhat | signed_log | wide | block | yes |
| Reidentification refresh | `Polymer/Results/td3_reidentification_disturb/20260420_234346/input_data.pkl` | disturb | mismatch | legacy_previous_measurement | running_zscore_physical_xhat | signed_log | n/a | n/a | yes |

All compared runs use the same polymer disturbance schedule. The saved `qi`, `qs`, and `ha` arrays in the result bundles are numerically identical to the baseline schedule, so the widened-range comparison is not confounded by different disturbances.

## Main Comparison

| Run | Tail phys MAE | Final test phys MAE | Final test out1 MAE | Final test out2 MAE | Final test reward | Final-10 reward | Final test input move |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline MPC | 0.1701 | 0.1701 | 0.0674 | 0.2729 | -4.4174 | -4.4173 | 0.0305 |
| Matrix legacy | 0.1722 | 0.1717 | 0.0645 | 0.2789 | -4.1667 | -4.1908 | 0.0310 |
| Matrix narrow refresh | 0.1728 | 0.1730 | 0.0627 | 0.2833 | -4.0860 | -4.0888 | 0.0306 |
| Matrix wide | 0.1592 | 0.1534 | 0.0494 | 0.2574 | -3.1837 | -3.2069 | 0.0338 |
| Structured legacy | 0.1671 | 0.1637 | 0.0651 | 0.2623 | -4.2805 | -4.3062 | 0.0317 |
| Structured narrow refresh | 0.1673 | 0.1642 | 0.0637 | 0.2647 | -4.1771 | -4.1885 | 0.0294 |
| Structured wide | 0.1537 | 0.1925 | 0.0521 | 0.3330 | -3.2593 | -3.2519 | 0.0702 |
| Reidentification refresh | 0.1767 | 0.1770 | 0.0678 | 0.2862 | -4.4135 | -4.4127 | 0.0325 |

- Matrix wide is the strongest run in this study. Final test physical MAE drops from `0.1730` in the previous narrow refresh to `0.1534` (-11.3%), beating both the legacy matrix run `0.1717` and baseline MPC `0.1701`.
- Structured wide is mixed. Its overall late-run tail MAE improves from `0.1673` to `0.1537`, but the held-out final test gets worse: `0.1642` to `0.1925` (+17.2%).
- Reidentification still fails to convert the richer RL state into better control. Its final test MAE remains `0.1770`, worse than matrix wide and worse than baseline MPC.

![Reward curves for matrix and structured narrow versus wide runs](./polymer_wide_range_matrix_structured/figures/wide_range_reward_curves.png)

The reward curves show the same pattern in both wide runs: a severe early deterioration followed by recovery to a better late-stage reward than the narrow runs. The matrix wide trough is deeper (`-34.5039` at episode `19`) than the structured wide trough (`-24.2145` at episode `12`), but both recover only much later, around episodes `139` and `141` respectively.

## Final Test Episode

![Final held-out test episode outputs for matrix and structured families](./polymer_wide_range_matrix_structured/figures/wide_range_final_test_outputs.png)

Matrix wide improves both outputs in the held-out test episode: output 1 MAE falls from `0.0627` to `0.0494`, and output 2 MAE falls from `0.2833` to `0.2574`.
Structured wide does not. Output 1 improves from `0.0637` to `0.0521`, but output 2 gets substantially worse: `0.2647` to `0.3330`. This is why the final test mean degrades even though the overall late-run tail and reward look better.

## Why Matrix Wide Worked And Reidentification Did Not

The latest matrix result is successful because the controller is allowed to choose from a small, direct, always-realized model family. There is no identification gate between the action and the prediction model used by MPC. The action changes three smooth multipliers and the model correction is immediately available to the optimizer on every step.

In the final test episode, matrix wide mainly uses a damped `A` correction and a stronger first-input `B` correction: `alpha` mean is `0.9536`, while `delta1` mean is `1.1400` and its p95 reaches `1.2000`. That is a coherent low-dimensional correction, not a noisy online identification problem.

Reidentification is fundamentally different. The policy may request strong blend authority, but the online identification engine almost never produces an admissible new model. In the latest polymer reidentification run, candidate-valid fraction is only `0.0011` and update-success fraction is only `0.0002`. Median condition number is `185509.1` and p95 is `2199603.2`. So reidentification is bottlenecked by data informativity and numerical conditioning, not by the policy alone.

This difference matches the literature. The direct multiplier methods behave like bounded parametric adaptation inside a fixed low-dimensional family. Reidentification, by contrast, needs persistently informative data and a numerically healthy regression problem. The adaptive MPC and identification literature repeatedly stresses persistence of excitation, targeted excitation, and dual control/experiment design as prerequisites for successful online model maintenance [Berberich2022] [Heirung2015] [Heirung2017] [Parsi2020] [Oshima2024].

## Why Structured Wide Gets Better Reward But Worse Evaluation

The answer is in the reward itself. The reward is built from scaled output errors and input moves, with output weights `Q = [5, 1]`. That means output 1 is five times more expensive than output 2 in the quadratic term. So a policy can improve reward by helping output 1 a lot even if output 2 gets worse.

| Run | Final test scaled MAE out1 | Final test scaled MAE out2 | Weighted quad out1 | Weighted quad out2 | Final test input move | Final test reward |
| --- | --- | --- | --- | --- | --- | --- |
| Matrix narrow refresh | 0.2706 | 0.3530 | 3.3931 | 0.5451 | 0.0306 | -4.0860 |
| Matrix wide | 0.2105 | 0.3198 | 2.5968 | 0.4680 | 0.0338 | -3.1837 |
| Structured narrow refresh | 0.2751 | 0.3292 | 3.5527 | 0.4875 | 0.0294 | -4.1771 |
| Structured wide | 0.2227 | 0.4169 | 2.5662 | 0.5310 | 0.0702 | -3.2593 |
![Reward versus evaluation tradeoff for the wide matrix and structured runs](./polymer_wide_range_matrix_structured/figures/wide_range_reward_tradeoff.png)

Structured wide improves output 1 sharply: final test scaled MAE drops from `0.2751` to `0.2227`. But output 2 worsens from `0.3292` to `0.4169`. Because output 1 has the larger weight, the weighted output-1 term falls from `3.5527` to `2.5662`, while the output-2 term rises only modestly from `0.4875` to `0.5310`. The result is a better reward even though the held-out evaluation episode is worse on mean physical tracking.

This strongly suggests that if the evaluation objective values both outputs more evenly, then yes, the reward parameters should be revisited. The run is not benefiting from a changed disturbance profile; the disturbance schedules are identical. It is benefiting from an objective mismatch between the training reward and the evaluation metric.

## Why The Wide Runs First Degrade And Then Recover

The early deterioration is consistent with a larger action/model-search space. Widening the multiplier ranges expands the set of prediction models the agent can induce. Early in training, the replay buffer is dominated by low-quality exploratory transitions from this larger space, so the policy gets worse before it learns which stronger corrections are actually useful. Once enough informative transitions accumulate, the policy recovers and starts exploiting the wider authority.

The control literature and RL literature both suggest practical fixes:

- Progressive widening / continuation: instead of jumping directly from narrow to wide bounds, increase the bounds in stages once reward, solver success, and held-out tracking have stabilized. This is conceptually similar to coarse-to-fine action selection and continuation-style training [Seo2025].
- Smoother exploration: use temporally coherent exploration rather than step-to-step jagged action noise. Autoregressive exploration is a direct literature-backed way to reduce violent exploratory swings in continuous control [Korenkevych2019].
- Data-aware excitation only when needed: if model learning is part of the method, dual/adaptive MPC papers recommend adding excitation only when uncertainty is high or data are not informative enough, rather than exciting continuously [Heirung2015] [Heirung2017] [Parsi2020].

In this repository, the most practical fix is staged widening. Start from the narrow run, continue training with an intermediate range, and widen again only after the held-out test episode improves and the model-usage statistics are not saturating.

## How Far Can The Range Be Widened Safely?

The nominal polymer physical `A` has spectral radius `0.9464`. In the final test episode, matrix wide already reaches a derived p95 spectral radius of `1.0934` and max `1.1357`. Structured wide is more aggressive still: mean spectral radius is `1.1054`, p95 is `1.1357`, max is `1.1357`, and several structured multipliers hit the hard upper bound `1.2` at p95.

So the answer is different for the two methods:

- Matrix: a slightly wider range may still be worth testing, but not symmetrically. The successful wide matrix policy mainly uses stronger `B` correction and slightly smaller `A`, so the safer next ablation is to widen `B` more than `A`, not to raise every bound uniformly.
- Structured: the current wide run already looks too aggressive for held-out evaluation. It pushes multiple grouped multipliers to `1.2`, keeps the test spectral radius above `1.0` on average, and doubles the final-test input movement from `0.0294` to `0.0702`. Widening further before adding guards is not justified by these results.

With a state-space model in the loop, the practical limit is not a single scalar bound. It is the largest uncertainty set for which the prediction model remains numerically admissible for MPC: stabilizable/detectable enough for the observer-controller pair, solver-feasible, and not so aggressive that held-out performance collapses. Robust MPC literature frames this as keeping the model family inside an uncertainty set where recursive feasibility and robust performance can still be guaranteed or approximated [Chen2024] [Limon2013] [Kothare2010].

For this project, a practical safe-widening recipe is:

1. Keep the current solve fallback for structured wide.
2. Add an explicit spectral-radius cap or smooth fade-back to nominal when the prediction model gets too aggressive.
3. Widen bounds asymmetrically, favoring `B` before `A`.
4. Use a staged widening schedule tied to held-out test MAE and solver/fallback statistics.
5. Treat the empirical limit as reached when p95 multiplier use is already on the hard bound and held-out test performance no longer improves.

## Model-Usage Diagnostics

![Wide-range multiplier usage and spectral-radius diagnostics](./polymer_wide_range_matrix_structured/figures/wide_range_model_usage.png)

Matrix wide uses the expanded range in a targeted way. It does not simply saturate everything upward. Instead, it tends to push the first input gain higher while keeping `A` on average below nominal. Structured wide behaves differently: several grouped multipliers have p95 equal to the hard upper bound `1.2`, and the test spectral radius stays above `1.1` on average. That explains why structured wide can still improve reward while giving an over-aggressive held-out trajectory.

## Conclusions

- The latest matrix wide run is genuinely more successful than the previous matrix runs. Its final test MAE `0.1534` beats the previous narrow refresh `0.1730`, the legacy matrix run `0.1717`, and baseline MPC `0.1701`.
- Structured wide is not a clean success. It improves late-run reward and aggregate tail MAE, but its held-out final test MAE degrades from `0.1642` to `0.1925` because it over-optimizes the heavily weighted first output and becomes much more aggressive on the prediction model and control effort.
- Direct multiplier methods work better than reidentification here because they operate in a small always-realized model family. Reidentification still fails due to poor candidate validity (`0.0011`) and poor conditioning, not because the new state conditioning failed.
- The degradation-then-recovery pattern is expected when the authority range is widened abruptly. The most practical fix is staged widening with smoother exploration and explicit held-out performance checks.
- Matrix may be widened a bit further, but only asymmetrically and with monitoring. Structured should not be widened further until a stability/admissibility guard is added.

## Sources

- [Berberich2022] Forward-looking persistent excitation in model predictive control.
- [Heirung2015] MPC-based dual control with online experiment design.
- [Heirung2017] Dual adaptive model predictive control.
- [Parsi2020] Active exploration in adaptive model predictive control.
- [Oshima2024] Targeted excitation and re-identification methods for multivariate process and model predictive control.
- [Korenkevych2019] Autoregressive Policies for Continuous Control Deep Reinforcement Learning.
- [Seo2025] Continuous Control with Coarse-to-fine Reinforcement Learning.
- [Chen2024] Robust model predictive control with polytopic model uncertainty through System Level Synthesis.
- [Limon2013] Robust feedback model predictive control of constrained uncertain systems.
- [Kothare2010] Robust model predictive control design with input constraints.

[Berberich2022]: https://doi.org/10.1016/j.automatica.2021.110033
[Heirung2015]: https://doi.org/10.1016/j.jprocont.2015.04.012
[Heirung2017]: https://doi.org/10.1016/j.automatica.2017.01.030
[Parsi2020]: https://www.research-collection.ethz.ch/handle/20.500.11850/461407
[Oshima2024]: https://doi.org/10.1016/j.jprocont.2024.103190
[Korenkevych2019]: https://doi.org/10.24963/ijcai.2019/382
[Seo2025]: https://proceedings.mlr.press/v270/seo25a.html
[Chen2024]: https://doi.org/10.1016/j.automatica.2023.111431
[Limon2013]: https://doi.org/10.1016/j.jprocont.2012.08.003
[Kothare2010]: https://doi.org/10.1016/j.isatra.2009.10.003