# Polymer Wide-Range Matrix and Structured Report With Distillation Counterpart

Date: 2026-04-22

This report focuses on the latest widened-range polymer matrix and structured-matrix runs, then extends the same admissibility and design logic to the distillation column as a cross-system counterpart.

The goal is to answer five questions with data from the saved runs and the shared polymer model:

1. Why did the widened matrix/structured methods improve, and why did reidentification still fail?
2. What reward is actually used in polymer and distillation, and how should it be changed if both outputs should matter more evenly?
3. Is there a mathematical way to set the multiplier range, instead of widening blindly?
4. Why do the wide runs first degrade and then recover, and how can residual-style ideas help?
5. Is polymer reidentification still worth pursuing, or is it currently dominated by direct multiplier methods?

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

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_reward_curves.png" alt="Reward curves for matrix and structured narrow versus wide runs" width="1200" style="max-width: 100%; height: auto;" />

The reward curves show the same pattern in both wide runs: a severe early deterioration followed by recovery to a better late-stage reward than the narrow runs. The matrix wide trough is deeper (`-34.5039` at episode `19`) than the structured wide trough (`-24.2145` at episode `12`), but both recover only much later, around episodes `139` and `141` respectively.

## Final Test Episode

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_final_test_outputs.png" alt="Final held-out test episode outputs for matrix and structured families" width="1200" style="max-width: 100%; height: auto;" />

Matrix wide improves both outputs in the held-out test episode: output 1 MAE falls from `0.0627` to `0.0494`, and output 2 MAE falls from `0.2833` to `0.2574`.
Structured wide does not. Output 1 improves from `0.0637` to `0.0521`, but output 2 gets substantially worse: `0.2647` to `0.3330`. This is why the final test mean degrades even though the overall late-run tail and reward look better.

## Latest High-Cap Reruns

After the earlier report, new polymer matrix and structured runs were generated under the widened default bounds with the intended physical `A` cap logic. Those reruns matter because they test whether the new cap policy helps in practice.

| Run | Saved bundle | Tail phys MAE | Final test phys MAE | Out1 MAE | Out2 MAE | Final-10 reward | Final test input move |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix first capped rerun | `Polymer/Results/td3_multipliers_disturb/20260421_174016/input_data.pkl` | 0.1491 | 0.1487 | 0.0530 | 0.2443 | -3.5948 | 0.0302 |
| Matrix latest capped rerun | `Polymer/Results/td3_multipliers_disturb/20260421_201555/input_data.pkl` | 0.1569 | 0.1519 | 0.0536 | 0.2502 | -3.5982 | 0.0303 |
| Structured first capped rerun | `Polymer/Results/td3_structured_matrices_disturb/20260421_174057/input_data.pkl` | 0.3146 | 0.2454 | 0.0825 | 0.4082 | -4.6748 | 0.0679 |
| Structured latest capped rerun | `Polymer/Results/td3_structured_matrices_disturb/20260421_201436/input_data.pkl` | 0.1640 | 0.1551 | 0.0488 | 0.2614 | -3.6136 | 0.0377 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_latest_cap_reruns.png" alt="Latest matrix and structured reruns after the cap update" width="1200" style="max-width: 100%; height: auto;" />

The current matrix capped rerun remains strong, but it is now a tradeoff rather than a strict improvement. Relative to the first capped rerun, final test MAE moves from `0.1487` to `0.1519`, while reward improves slightly from `-3.5844` to `-3.5773`. The policy is still using the capped `A` side correctly and keeping `B` wide.
The structured family changed much more. The first capped-structured attempt was still effectively uncapped on the `A` side and performed poorly, but the latest rerun with the actual `A` cap applied improves final test MAE from `0.2454` to `0.1551`, cuts output-2 MAE from `0.4082` to `0.2614`, and improves reward from `-4.2549` to `-3.5554`.

The reason is now clear in the saved bundles. Both matrix reruns use `high_coef = [1.0566, 1.25, 1.25]`, so the scalar `A` multiplier was capped as intended in both cases. The first structured capped attempt still saved `structured_high_a = [1.25, 1.25, 1.25, 1.25]`, so the `A` cap was effectively missing there. The latest structured rerun saves `structured_high_a = [1.0566, 1.0566, 1.0566, 1.0566]` with `structured_high_b = [1.25, 1.25]`, which means the intended tight-`A`, wide-`B` design is finally the one being trained and evaluated.

The structural effect is visible in the spectral diagnostics. The failed first capped-structured attempt had mean/p95 spectral radius `0.9459` / `1.1598` and `3` fallback events. The latest capped-structured rerun brings that down to `0.9000` / `0.9999` with zero fallback events. So the main structured problem was not that the cap idea was wrong; it was that the cap had not actually been applied in the earlier rerun.

## Does Structured Need A Different Cap Than Matrix?

Probably yes in the long run, but the new reruns sharpen the conclusion. There are two separate questions:

1. If the intended structured `A` cap is truly enforced, is it mathematically reasonable?
2. Even if it is mathematically reasonable, should structured still use a stricter cap than matrix because it perturbs several coupled `A` groups at once?

The first question can be answered directly from the model family. I sampled the polymer block-structured family with `B \in [0.75, 1.25]` fixed and varied only the `A`-side upper bound:

| Structured A high | Unstable frac | Near-unit frac | Spectral p95 | Spectral max |
| --- | --- | --- | --- | --- |
| 1.0000 | 0.000 | 0.000 | 0.9337 | 0.9464 |
| 1.0200 | 0.000 | 0.000 | 0.9506 | 0.9652 |
| 1.0300 | 0.000 | 0.000 | 0.9613 | 0.9747 |
| 1.0400 | 0.000 | 0.016 | 0.9710 | 0.9842 |
| 1.0500 | 0.000 | 0.042 | 0.9781 | 0.9936 |
| 1.0566 | 0.000 | 0.077 | 0.9870 | 0.9998 |
| 1.0800 | 0.066 | 0.187 | 1.0043 | 1.0221 |
| 1.1000 | 0.175 | 0.274 | 1.0236 | 1.0410 |
| 1.1500 | 0.370 | 0.455 | 1.0698 | 1.0881 |
| 1.2500 | 0.595 | 0.652 | 1.1598 | 1.1830 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_structured_asymmetric_cap_frontier.png" alt="Structured asymmetric A/B cap frontier" width="1200" style="max-width: 100%; height: auto;" />

That frontier shows that the intended structured `A` cap of `1.0566` is actually a reasonable first bound when `B` stays wide. In random sampling, `A_high = 1.0566` gives unstable fraction `0.000`, spectral p95 `0.9870`, and spectral max `0.9998`. So the matrix-derived cap is not obviously too loose for the structured family in a pure admissibility sense.

But structured is still more delicate than matrix because its action changes several `A`-side groups and the off-diagonal coupling together. A single scalar `alpha` cap from the plain matrix family does not fully capture that geometry. The frontier suggests a practical structured hierarchy:

- `A_high ~= 1.0566` is a reasonable first structured cap and now has one encouraging rerun behind it.
- If you want more margin, `A_high ~= 1.04-1.05` is cleaner: unstable fraction stays zero and the near-unit fraction drops sharply.
- The latest structured success means the next structured question is not "does the cap work at all?" but "should the structured `A` cap be slightly tighter than the matrix cap for better consistency?"

## Why Matrix Wide Worked And Reidentification Did Not

The latest matrix result is successful because the controller is allowed to choose from a small, direct, always-realized model family. There is no identification gate between the action and the prediction model used by MPC. The action changes three smooth multipliers and the model correction is immediately available to the optimizer on every step.

In the current latest capped matrix test episode, the policy still mainly uses a damped `A` correction and a stronger first-input `B` correction: `alpha` mean is `0.9319`, while `delta1` mean is `1.1264` and its p95 reaches `1.2496`. That is a coherent low-dimensional correction, not a noisy online identification problem.

Reidentification is fundamentally different. The policy may request strong blend authority, but the online identification engine almost never produces an admissible new model. In the latest polymer reidentification run, candidate-valid fraction is only `0.0011` and update-success fraction is only `0.0002`. Median condition number is `185509.1` and p95 is `2199603.2`. So reidentification is bottlenecked by data informativity and numerical conditioning, not by the policy alone.

This difference matches the literature. The direct multiplier methods behave like bounded parametric adaptation inside a fixed low-dimensional family. Reidentification, by contrast, needs persistently informative data and a numerically healthy regression problem. The adaptive MPC and identification literature repeatedly stresses persistence of excitation, targeted excitation, and dual control/experiment design as prerequisites for successful online model maintenance [Berberich2022] [Heirung2015] [Heirung2017] [Oshima2024].

## The Polymer Reward Actually Used

The RL notebooks do not use a plain `Q = [5, 1]` quadratic reward. They use the shared relative-band reward from `utils/rewards.py` with setpoint-dependent bands, inside-band gating, linear edge penalties, and an inside-band bonus.

$$
r_t = \Big(-\sum_i e^{\mathrm{eff}}_{i,t} - \sum_i \ell_{i,t} - \sum_j m_{j,t} + \sum_i b_{i,t}\Big) \cdot \texttt{reward\_scale}
$$

where the output balance depends on the scaled band

$$
\text{band}_{i,t} = \frac{\max(k_{\mathrm{rel},i}|y^{sp}_{i,t}|,\; \text{band\_floor}_{i})}{y^{max}_i - y^{min}_i},
\qquad
\text{slope at edge}_i = 2 Q_i \cdot \text{band}_{i,t}.
$$

So the relevant output balance is not just the raw `Q_diag`. It is `Q_diag` filtered through the setpoint-dependent band width.

| Setpoint | Output | Physical setpoint | Band (phys) | Band (scaled) | Edge slope | Bonus prefactor |
| --- | --- | --- | --- | --- | --- | --- |
| SP1 | out1 | 4.500 | 0.0135 | 0.06091 | 0.6310 | 0.1345 |
| SP1 | out2 | 324.000 | 0.0972 | 0.12437 | 0.2239 | 0.0974 |
| SP2 | out1 | 3.400 | 0.0102 | 0.04602 | 0.4767 | 0.0768 |
| SP2 | out2 | 321.000 | 0.0963 | 0.12322 | 0.2218 | 0.0957 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_reward_balance.png" alt="Reward geometry and implied equalization targets" width="1200" style="max-width: 100%; height: auto;" />

If `Q2` stays at `90`, then an edge-equalized `Q1` would be about `212.4`, while a bonus-equalized `Q1` would be about `510.3`. The current `Q1 = 518.0` is therefore almost exactly the bonus-equalized value, not the edge-equalized value.
That is the rigorous explanation for the "even output" issue. Inside the reward band, the outputs are treated much more evenly than the old report implied. But outside the band, output 1 still carries a much steeper correction slope, so the policy can gain reward by helping output 1 more aggressively even when output 2 gets worse.

A practical reward fix depends on what you want to equalize:

- Equal band-edge urgency: move `Q1` toward about `212` while keeping `Q2 = 90`.
- Keep the current inside-band bonus balance: leave `Q1` near about `510` and increase output-2 edge penalties separately.
- Clean implementation: separate inside-band bonus weights from outside-band penalty weights instead of forcing one `Q_diag` to do both jobs.

The saved runs already show this reward tradeoff in the final test episode:

| Run | Final test scaled MAE out1 | Final test scaled MAE out2 | Reward quad out1 | Reward quad out2 | Reward linear out1 | Reward linear out2 | Reward bonus out1 | Reward bonus out2 | Final test input move | Final test reward |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Matrix narrow refresh | 0.2706 | 0.3530 | 3.5152 | 0.4906 | 0.0677 | 0.0335 | 0.0099 | 0.0210 | 0.0306 | -4.0860 |
| Matrix wide | 0.2105 | 0.3198 | 2.6903 | 0.4212 | 0.0507 | 0.0305 | 0.0119 | 0.0188 | 0.0338 | -3.1837 |
| Structured narrow refresh | 0.2751 | 0.3292 | 3.6805 | 0.4387 | 0.0692 | 0.0310 | 0.0278 | 0.0234 | 0.0294 | -4.1771 |
| Structured wide | 0.2227 | 0.4169 | 2.6586 | 0.4779 | 0.0523 | 0.0381 | 0.0051 | 0.0075 | 0.0702 | -3.2593 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_reward_tradeoff.png" alt="Reward versus evaluation tradeoff for the wide matrix and structured runs" width="1200" style="max-width: 100%; height: auto;" />

Structured wide improves output 1 sharply: final test scaled MAE drops from `0.2751` to `0.2227`. But output 2 worsens from `0.3292` to `0.4169`. Under the actual polymer reward, the output-1 quadratic and linear penalties both fall materially, while the output-2 penalties rise by less. The reward therefore improves even though the held-out mean physical error gets worse.

## Distillation Reward Geometry

The distillation reward uses the same relative-band structure as polymer, but the current parameter balance is much more asymmetric toward output 1. The same formulas apply,

$$
\text{edge slope}_i = 2 Q_i\,\text{band}_i,
\qquad
\text{bonus prefactor}_i = \beta Q_i\,\text{band}_i^2,
$$

so the relevant balance question is again how `Q_diag` interacts with the scaled reward band, not just the raw `Q_diag` entries themselves.

| Setpoint | Output | Physical setpoint | Band (phys) | Band (scaled) | Edge slope | Bonus prefactor |
| --- | --- | --- | --- | --- | --- | --- |
| SP1 | out1 | 0.0130 | 0.0039 | 0.00731 | 540.8755 | 13.8367 |
| SP1 | out2 | -23.0000 | 0.4600 | 0.02582 | 77.4739 | 7.0026 |
| SP2 | out1 | 0.0280 | 0.0084 | 0.01574 | 1164.9627 | 64.1890 |
| SP2 | out2 | -21.0000 | 0.4200 | 0.02358 | 70.7370 | 5.8377 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_cross_system_reward_balance.png" alt="Cross-system reward balance sweep" width="1200" style="max-width: 100%; height: auto;" />

For polymer, the current `Q1 = 518.0` is close to the bonus-equalized value `510.3` but well above the edge-equalized value `212.4`. That is why polymer still favors output 1 outside the reward band even though the inside-band bonus is relatively balanced.
For distillation, the asymmetry is much larger. The current `Q1 = 37000` is far above both the edge-equalized target `3773` and the bonus-equalized target `11045` with `Q2` fixed at `1500`. So the current distillation reward strongly prioritizes output 1 both near and outside the reward band.

That suggests different reward moves for the two systems:

- Polymer: if the evaluation objective should become more even, reduce `Q1` from `518` toward roughly `212` or separate the inside-band bonus weights from the outside-band penalty weights.
- Distillation: the current `Q1` is so dominant that a meaningful even-output ablation should test `Q1` in the rough range `3773-11045` instead of `37000`.

## Mathematical Limits For The Multiplier Range

For the plain matrix family, observability and controllability rank are not what limit the range. If the physical model is changed only through `A -> alpha A`, then

$$
\mathcal{O}(\alpha A, C) = \begin{bmatrix} C \\ \alpha C A \\ \alpha^2 C A^2 \\ \vdots \end{bmatrix},
\qquad
\mathcal{C}(\alpha A, B) = \begin{bmatrix} B & \alpha A B & \alpha^2 A^2 B & \cdots \end{bmatrix}.
$$

For any `alpha > 0`, these are just row-wise or column-wise scalings of the nominal observability and controllability matrices, so the rank stays the same. That is exactly what the polymer model shows numerically:

| alpha | rho(alpha A) | obs rank | ctrb rank | obs cond | ctrb cond |
| --- | --- | --- | --- | --- | --- |
| 0.850 | 0.8044 | 7 | 7 | 171.2 | 1136.9 |
| 1.000 | 0.9464 | 7 | 7 | 108.4 | 716.3 |
| 1.200 | 1.1357 | 7 | 7 | 97.1 | 550.0 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_admissibility_frontier.png" alt="Matrix and structured admissibility frontier" width="1200" style="max-width: 100%; height: auto;" />

The useful matrix bound comes instead from open-loop spectral admissibility. Since `rho(A_nom) = 0.9464`, requiring `rho(alpha A) < 1` gives `alpha < 1 / rho(A_nom) = 1.0566`. That is the clean analytical upper bound if every candidate prediction `A` must remain open-loop stable.
The lower bound is different. For positive `alpha`, neither observability nor open-loop stability imposes a nontrivial lower bound. So there is no comparable analytical `alpha_min > 0` from these criteria alone. The practical lower bound comes from how much model-speed distortion you are willing to tolerate, not from rank loss.

There are two practical lower-bound rules that do make sense:

1. Trust-region rule: require the relative model change to stay small,

$$
\frac{\|\alpha A - A\|_F}{\|A\|_F} = |\alpha - 1| \le \varepsilon_A
\quad \Rightarrow \quad
\alpha \in [1-\varepsilon_A,\; 1+\varepsilon_A].
$$

2. Time-scale rule: require the dominant spectral radius to stay above a chosen fraction of nominal,

$$
\rho(\alpha A) \ge \kappa \rho(A) \quad \Rightarrow \quad \alpha \ge \kappa,
$$

where `kappa` is a modeling-choice floor such as `0.85` or `0.90`. This is not a stability necessity. It is a way to stop the learned prediction model from becoming unrealistically fast.

For the structured block family, entrywise multipliers can in principle change the rank properties, so a sampled frontier is useful. On the polymer model, however, the sampled positive ranges still keep full observability rank. The frontier is again instability, not rank loss:

| Upper bound | Unstable frac | Near-unit frac | Spectral p95 | Min obs rank | Bad obs frac |
| --- | --- | --- | --- | --- | --- |
| 1.02 | 0.000 | 0.000 | 0.9634 | 7 | 0.000 |
| 1.05 | 0.000 | 0.143 | 0.9892 | 7 | 0.000 |
| 1.08 | 0.158 | 0.381 | 1.0146 | 7 | 0.000 |
| 1.10 | 0.311 | 0.474 | 1.0308 | 7 | 0.000 |
| 1.12 | 0.391 | 0.518 | 1.0490 | 7 | 0.000 |
| 1.15 | 0.460 | 0.566 | 1.0723 | 7 | 0.000 |
| 1.20 | 0.556 | 0.617 | 1.1145 | 7 | 0.000 |
This gives a practical polymer rule. If you want a mostly stable structured family without frequent unstable prediction models, the common upper bound should stay near about `1.05`. By `1.08`, instability already appears. By `1.20`, roughly half the sampled block models are unstable.

## What About The B Multipliers?

The `B` side is different from `A`. In the matrix family the model update is `B(\delta) = B \operatorname{diag}(\delta_1, \delta_2)`. For any strictly positive `delta_j`, controllability rank is preserved for exactly the same reason as above: the controllability matrix is just column-scaled block by block. So, again, rank does not give a useful finite bound.

What `delta_j` changes is the perceived input authority. If `G_ss = C(I-A)^{-1}B` is the steady-state gain, then

$$
G_{ss}(\delta) = G_{ss}\,\operatorname{diag}(\delta_1, \delta_2).
$$

So the predicted output gain of input `j` scales linearly with `delta_j`, while the input move required to get the same correction scales approximately like `1 / delta_j`.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_b_multiplier_design.png" alt="B-multiplier gain and move-ratio explanation" width="1200" style="max-width: 100%; height: auto;" />

That means `B` selection should be done with actuator and trust considerations, not with pole placement logic. A practical lower-bound calculation is

$$
\delta_{j,\min} \ge \frac{\Delta u^{nom}_{j,\;p}}{h^{avail}_{j,\;min}},
$$

where `Delta u^{nom}_{j,p}` is a representative nominal move level such as the 95th percentile under baseline MPC, and `h^{avail}_{j,min}` is the minimum available headroom for that input over the scenarios you care about. If the learned model is allowed to believe the input is too weak, the MPC will ask for moves that may not fit inside the real actuator margin.

A practical upper-bound calculation is a gain-trust region in log space,

$$
|\log \delta_j| \le \varepsilon_B
\quad \Longleftrightarrow \quad
\delta_j \in [e^{-\varepsilon_B}, e^{\varepsilon_B}],
$$

which is often a better way to think about input-gain uncertainty than a raw linear bound because equal multiplicative uncertainty is treated symmetrically above and below `1.0`.

That gives a practical design rule for `B` multipliers:

- Lower bound: choose `delta_min` so the required move inflation `1 / delta_min` still fits inside actuator headroom on representative disturbances.
- Upper bound: choose `delta_max` so the predicted input gain increase stays inside the uncertainty set you are willing to trust, or inside a symmetric trust region such as `|log(delta_j)| <= eps_B`.

| delta | Predicted gain ratio | Required move ratio | Interpretation |
| --- | --- | --- | --- |
| 0.75 | 0.75x | 1.333x | aggressive low-gain assumption; large move inflation |
| 0.85 | 0.85x | 1.176x | moderate low-gain assumption |
| 0.95 | 0.95x | 1.053x | conservative low-gain assumption |
| 1.05 | 1.05x | 0.952x | conservative high-gain assumption |
| 1.25 | 1.25x | 0.800x | aggressive high-gain assumption |

With the widened defaults requested in this update, both polymer and distillation now use `delta \in [0.75, 1.25]` on the `B` side. That means up to about `1.333x` required-move inflation on the low side and up to `1.25x` predicted input gain on the high side. The systems differ only on the `A` side, where polymer is capped at `alpha <= 1.0566` and distillation at `alpha <= 1.1929`.

That is why the safest widening order is still `B` before `A`: `B` does not move the open-loop poles, but it does change how hard the MPC will push the actuators. So `B` should be bounded by input-headroom and validation logic, not by spectral radius.

## Distillation Counterpart

The same mathematics does not transfer one-for-one across systems. Distillation is much less fragile than polymer at the model level.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_cross_system_admissibility.png" alt="Cross-system admissibility comparison" width="1200" style="max-width: 100%; height: auto;" />

| System | rho(A_nom) | alpha_max stable | Structured unstable frac at 1.20 | Structured p95 spectral at 1.20 |
| --- | --- | --- | --- | --- |
| Polymer | 0.9464 | 1.0566 | 0.556 | 1.1145 |
| Distillation | 0.8383 | 1.1929 | 0.013 | 0.9872 |
Polymer has `alpha_max_stable = 1.0566`, while distillation has `alpha_max_stable = 1.1929`. At the same structured upper bound `1.20`, polymer's sampled unstable fraction is `0.556`, but distillation's is only `0.013`. So distillation is mathematically much more tolerant of multiplier widening than polymer.

However, the currently saved disturbance runs do not yet show that widening alone solves distillation performance. The latest disturbance bundles available in this tree are baseline fluctuation, matrix SAC disturbance, and structured SAC disturbance:

| Run | Final test phys MAE | Final test out1 MAE | Final test out2 MAE | Final test reward | Alpha / spectral usage |
| --- | --- | --- | --- | --- | --- |
| Distillation baseline fluctuation | 0.0954 | 0.0016 | 0.1892 | -0.2651 | n/a |
| Distillation matrix SAC disturbance | 0.1569 | 0.0047 | 0.3091 | -0.5055 | alpha mean 0.9849, p95 0.9943 |
| Distillation structured SAC disturbance | 0.2544 | 0.0042 | 0.5045 | -0.7832 | spectral mean 0.7721, p95 0.7758 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_distillation_counterpart.png" alt="Distillation final-test counterpart" width="1200" style="max-width: 100%; height: auto;" />

The current saved disturbance matrix run does not beat baseline (`0.1569` vs `0.0954`), and the current saved structured disturbance run is worse still (`0.2544`). So the distillation section changes the conclusion in an important way: the model-level admissibility landscape is wider, but the currently saved RL policies are not exploiting it well.

That is exactly why the cross-system figure matters. Polymer needs guards because the model family is fragile. Distillation does not need those guards for the same mathematical reason, but it still needs better policy learning and reward alignment before wider ranges will automatically help.

## How To Prevent Catastrophic Distillation Episodes While Keeping Wide Search

The current distillation failure mode is not a pure stability failure. It is a performance failure: the RL policy can move the prediction model into a region where the solve still succeeds, but the closed-loop reward and tracking collapse. That means a structured solve fallback alone is not enough for the plain matrix family.

A practical wide-search design is to keep the raw policy search space wide, but gate the effective `A` deviation by mismatch magnitude and by a nominal-cost acceptance test. One useful formulation is

$$
\theta_t^{eff} = 1 + \lambda_t\big(\theta_t^{rl} - 1\big),
\qquad
\lambda_t = \min\big(g_{\tau}(\tau_t),\; g_J(J_t^{cand}, J_t^{nom})\big),
$$

where `theta` is the multiplier vector, `tau_t` is a raw mismatch magnitude such as `max |tracking_error_raw|`, `g_tau` is a deadband-plus-exponential gate, and `g_J` is a backtracking acceptance filter that shrinks the proposal whenever the candidate one-step predicted cost is much worse than nominal.

For distillation, the safest version is asymmetric:

1. keep the raw search space wide for both `A` and `B`,
2. keep the effective `A` tube narrow near setpoint,
3. allow `B` to stay wider because it changes authority more than pole location,
4. add the same nominal fallback/backtracking logic to plain matrix that structured already uses on solve failure.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_distillation_safe_search.png" alt="Distillation safe wide-search explainer" width="1200" style="max-width: 100%; height: auto;" />

The figure makes the idea concrete. Near setpoint, the effective `A` range can stay close to the current `[0.99, 1.01]` tube even though the raw action still searches over `[0.75, 1.25]`. As the mismatch grows, the gate opens and the same raw policy can use more of the wide space. The second panel shows the other half of the safeguard: if the candidate MPC objective is much worse than nominal, backtrack the blend factor before applying the model.

This is the cleanest way to keep wide search without repeating the catastrophic early distillation episodes. It uses the same residual-style authority idea already present elsewhere in the repo, but applies it to model deviation instead of residual control. The design direction is consistent with adaptive/robust MPC approaches that keep uncertainty updates inside recursively feasible sets and with output-feedback MPC schemes that refresh estimator dynamics only under protected model changes [AdaptiveMPC2020] [RAMPC2023] [OutputFeedbackRMPC2019].

## Will Shifted MPC Warm Start Help Across Methods?

Here `shifted MPC warm start` means initializing the solver with the previous optimal move sequence shifted one step forward, not the notebook's episode-schedule warm-start segment. The optimizer-level idea is

$$
U_t^\star = \arg\min_U J_t(U),
\qquad
U_t^{(0)} = \mathcal{S}(U_{t-1}^\star),
$$

where `\mathcal{S}` drops the first move of the previous solution and appends a terminal guess. MPC warm-start literature uses this because consecutive subproblems are usually close in parameter space, so a shifted initialization reduces solver work and can reduce abnormal solves when the optimization problem size stays fixed [Diehl2005] [Jerez2014].

The important distinction for this repo is that warm start is primarily a solver-conditioning tool, not a learning fix. It can help matrix, structured, weights, residual, reidentification, combined, and even baseline MPC solves converge more smoothly. It does not by itself fix the early wide-range deterioration if the underlying issue is that exploration is driving the policy through a much worse model family.

| System | Family | Warm start implemented | Default on | Fixed-size MPC subproblem | Observer refresh hook | Exploration mode |
| --- | --- | --- | --- | --- | --- | --- |
| polymer | baseline | 1 | 0 | 1 | 0 | none |
| polymer | horizon | 1 | 0 | 0 | 0 | epsilon |
| polymer | dueling_horizon | 1 | 0 | 0 | 0 | noisy |
| polymer | matrix | 1 | 0 | 1 | 0 | param_noise |
| polymer | structured | 1 | 0 | 1 | 0 | param_noise |
| polymer | weights | 1 | 0 | 1 | 0 | param_noise |
| polymer | residual | 1 | 0 | 1 | 0 | param_noise |
| polymer | reidentification | 1 | 0 | 1 | 1 | param_noise |
| polymer | combined | 1 | 0 | 0 | 0 | mixed: horizon=epsilon, cont=param_noise |
| distillation | baseline | 1 | 0 | 1 | 0 | none |
| distillation | horizon | 1 | 0 | 0 | 0 | noisy |
| distillation | dueling_horizon | 1 | 0 | 0 | 0 | noisy |
| distillation | matrix | 1 | 0 | 1 | 0 | param_noise |
| distillation | structured | 1 | 0 | 1 | 0 | param_noise |
| distillation | weights | 1 | 0 | 1 | 0 | param_noise |
| distillation | residual | 1 | 0 | 1 | 0 | param_noise |
| distillation | reidentification | 1 | 0 | 1 | 1 | param_noise |
| distillation | combined | 1 | 0 | 0 | 0 | mixed: horizon=noisy, cont=param_noise |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_warmstart_exploration.png" alt="Warm-start and exploration diagnostics across methods" width="1200" style="max-width: 100%; height: auto;" />

| Representative run | Saved bundle | Mean normalized step change | P95 normalized step change | Decision change fraction | Reward trough gap |
| --- | --- | --- | --- | --- | --- |
| Polymer horizon | `Polymer/Results/horizon_disturb_unified/20260421_133651/input_data.pkl` | 1.3004 | 2.3333 | 0.9786 | 1.7475 |
| Polymer matrix wide | `Polymer/Results/td3_multipliers_disturb/20260421_132620/input_data.pkl` | 0.1158 | 0.5774 | 0.9159 | 26.9340 |
| Polymer structured wide | `Polymer/Results/td3_structured_matrices_disturb/20260421_013208/input_data.pkl` | 0.1360 | 0.5834 | 0.9330 | 20.9626 |
| Polymer weights | `Polymer/Results/td3_weights_disturb/20260421_004946/input_data.pkl` | 0.1014 | 0.5241 | 0.8375 | 2.2421 |
| Polymer residual | `Polymer/Results/td3_residual_disturb/20260420_225631/input_data.pkl` | 0.0068 | 0.0248 | 1.0000 | 1.7507 |
| Polymer reidentification | `Polymer/Results/td3_reidentification_disturb/20260420_234346/input_data.pkl` | 0.0236 | 0.0547 | 0.9500 | 0.1236 |
| Distillation horizon | `Distillation/Results/distillation_horizon_disturb_fluctuation_standard_unified/20260420_175006/input_data.pkl` | 1.2508 | 3.3333 | 0.9135 | 6.5285 |
| Distillation matrix SAC | `Distillation/Results/distillation_matrix_sac_disturb_fluctuation_standard_unified/20260415_104840/input_data.pkl` | 0.0692 | 0.2371 | 0.9750 | 4.9696 |
| Distillation structured SAC | `Distillation/Results/distillation_structured_matrix_sac_disturb_fluctuation_standard_unified/20260415_120923/input_data.pkl` | 0.0645 | 0.1805 | 0.9750 | 6.5381 |

Across the codebase, shifted warm start is already implemented in every MPC-solving runner, but the current defaults leave it off everywhere. The figure and representative-run table show why the likely payoff is method-dependent.

- Horizon methods are the weakest warm-start candidates. Their normalized decision change is about `1.3004` in polymer and `1.2508` in distillation, and the decision changes on more than `0.9786` and `0.9135` of steps. Because the chosen horizon changes frequently, the underlying MPC dimension changes too, so shifted warm start has limited structural value here.
- Matrix and structured methods are stronger warm-start candidates. Their normalized decision change is much lower: about `0.1158` for polymer matrix wide, `0.1360` for polymer structured wide, and even lower for the current distillation disturbance runs. Those methods keep a fixed-size subproblem and only change the prediction model or weights, which is the regime where warm-start logic is most defensible.
- Residual is the cleanest candidate. The current polymer residual representative run has normalized residual-action step change only `0.0068`. That means a shifted move sequence is likely to help solver continuity, especially near setpoint, without asking the optimizer to absorb large structural changes.
- Weights and reidentification sit between those extremes. Weights has moderate normalized step change `0.1014`, so warm start is likely worth trying. Reidentification has very smooth blend coefficients `0.0236`, but warm start alone will not solve its model-quality bottleneck.

So the practical conclusion is: yes, warm start is worth testing beyond matrices, but mainly as a solve-quality and solve-time improvement. The most promising first targets are residual, matrix, structured, weights, and reidentification. Horizon should be lower priority because its problem size changes too often.

## Would Periodic Observer Updating Help?

Every mismatch-capable method already updates the observer state every step. The real question is whether the observer model or observer gain should be redesigned occasionally when the effective prediction model drifts. The short answer is: redesigning observer poles every episode is probably too aggressive, especially for distillation. A thresholded or slower refresh is more plausible than an unconditional episode-by-episode redesign. With a fixed observer gain `L`, the estimation error behaves qualitatively like

$$
e_{k+1} \approx (\hat A - L C)e_k + (\Delta A\,x_k + \Delta B\,u_k),
$$

so a fixed nominal observer becomes less appropriate when the controller keeps the prediction model far from nominal for long intervals. Adaptive/output-feedback MPC literature supports observer redesign when the model is time-varying or adaptively updated, but only when the redesigned observer is built from a sufficiently trustworthy model [Mayne2009] [Souza2021] [Heirung2017].

| Case | Mean A drift | Mean B drift | Mean spectral radius |
| --- | --- | --- | --- |
| Polymer matrix latest capped | 0.0952 | 0.1563 | 0.8819 |
| Polymer structured latest capped | 0.1788 | 0.1489 | 0.9000 |
| Distillation matrix SAC | 0.0161 | 0.0189 | n/a |
| Distillation structured SAC | 0.0120 | 0.0157 | 0.7721 |
<img src="./polymer_wide_range_matrix_structured/figures/wide_range_observer_refresh_support.png" alt="Observer-refresh support diagnostics" width="1200" style="max-width: 100%; height: auto;" />

| Reidentification run | Candidate valid frac | Update success frac | Update event frac | Condition median | Condition p95 |
| --- | --- | --- | --- | --- | --- |
| Polymer reidentification | 0.0011 | 0.0002 | 0.1999 | 185536.0 | 2199679.8 |
| Distillation reidentification | 0.2177 | 0.0109 | 0.0475 | 20706.9 | 33159.2 |

The observer-refresh question splits cleanly by system. In the polymer wide runs, mean model drift is substantial: around `0.0952` to `0.1788` on the `A` side and `0.1563` to `0.1489` on the `B` side. In the current distillation matrix/structured runs, the same drift is only about `0.0161` to `0.0120`. That means periodic observer redesign is far more plausible as a polymer-wide-multiplier stabilization tool than as a fix for the current distillation matrix/structured runs.

But periodic observer refresh only helps if the refreshed model is trustworthy. That is exactly why it is not a good first fix for current polymer reidentification: candidate-valid fraction is only `0.0011` and update-success fraction is only `0.0002`. There is almost no accepted model stream to refresh from. Distillation reidentification is materially healthier on that axis: candidate-valid fraction `0.2177`, update-success fraction `0.0109`, and condition numbers an order of magnitude lower.

So the practical recommendation is narrow, not blanket: periodic observer redesign is worth exploring only if it is based on a smoothed accepted effective model and protected by the same admissibility checks as the controller. The best first candidates are polymer matrix/structured with slow refresh from interval-mean effective models, and distillation reidentification where candidate quality is already much better than polymer. For current distillation matrix/structured, per-episode pole redesign is unlikely to be the main fix because the model drift is too small; gating and acceptance filters are more relevant there. Polymer reidentification should not use observer refresh as a primary rescue mechanism until its identification layer becomes trustworthy.

## Is The Current Exploration Noise Making The Wide-Range Problem Worse?

Yes, especially in the continuous multiplier families. The current defaults use the same continuous exploration schedule across matrix, structured, weights, residual, reidentification, and the continuous branches of combined supervision: parameter-noise scale starts at `0.2`, decays toward `0.02` with an exponential factor `0.99995`, and the perturbed actor is resampled every `4` environment steps. Horizon uses epsilon/noisy exploration instead. That means the continuous methods see the same exploration magnitude even when the feasible model family has been widened dramatically.

The representative diagnostics make the distinction clear. Residual and reidentification are smooth under their current policy outputs, with normalized step change about `0.0068` and `0.0236`. Polymer matrix and structured wide are much more aggressive, around `0.1158` and `0.1360`, and weights is similar at `0.1014`. Those are exactly the families that show the deepest early reward troughs.

This is why warm start and exploration should not be conflated. Warm start can reduce solver friction, but it does not remove the fact that the current actor-noise process is injecting the same nominal exploration scale into a much larger admissible model set. Literature on parameter-noise exploration and temporally coherent policies supports making exploration smoother or staged when the action-to-dynamics map is very sensitive [Plappert2018] [Korenkevych2019] [Seo2025].

In practical repo terms, the most sensible order is:

1. keep the current wide bounds, but enable shifted warm start first in residual, matrix, structured, weights, and reidentification,
2. reduce early continuous exploration amplitude or slow the param-noise resample cadence when the range is widened,
3. only then consider periodic observer redesign where sustained model drift justifies it.

## Why The Wide Runs First Degrade And Then Recover

The early deterioration is consistent with a larger action/model-search space. Widening the multiplier ranges expands the set of prediction models the agent can induce. Early in training, the replay buffer is dominated by low-quality exploratory transitions from this larger space, so the policy gets worse before it learns which stronger corrections are actually useful. Once enough informative transitions accumulate, the policy recovers and starts exploiting the wider authority.

The control literature and RL literature both suggest practical fixes:

- Progressive widening / continuation: instead of jumping directly from narrow to wide bounds, increase the bounds in stages once reward, solver success, and held-out tracking have stabilized [Seo2025].
- Smoother exploration: use temporally coherent exploration rather than step-to-step jagged action noise [Korenkevych2019].
- Data-aware excitation only when needed: if model learning is part of the method, add excitation only when uncertainty is high or the data are not informative enough [Heirung2015] [Heirung2017] [Oshima2024].
- Robust uncertainty-set shaping: treat the multiplier family as an uncertainty set and grow that set only while feasibility and worst-case behavior remain acceptable [Chen2024] [Limon2013] [Kothare1996].

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_practical_fixes_explainer.png" alt="Literature-backed practical fixes explained" width="1200" style="max-width: 100%; height: auto;" />

The four panels in the figure are not copied from the papers. They are explanatory plots built from the mechanisms those papers discuss, translated into this project's setting.

What each paper-backed idea means here in concrete terms:

- Progressive widening / continuation: widen the allowed multiplier range in phases. In this repo, that means promoting `high_coef` and the structured range profile only after held-out MAE, fallback rate, and p95 multiplier saturation are acceptable. This directly addresses the early degradation seen in the wide polymer reward curves.
- Smoother exploration: replace jagged per-step multiplier noise with temporally coherent exploration, so the replay buffer contains locally consistent trajectories rather than violent model jumps. In practical terms, use an autoregressive action-noise process or a slower parameter-noise refresh for matrix and structured agents.
- Data-aware excitation only when needed: make extra exploration or model-learning authority a function of uncertainty or poor informativeness. In this repo, that can be the same mismatch-based gate used for residual-style authority, but applied to exploration amplitude or reidentification authority instead of directly to `u_res`.
- Robust uncertainty-set shaping: use the admissibility frontier to decide whether a candidate range should even be trainable. The polymer structured frontier shows why a uniform `1.20` upper bound is too wide as a default uncertainty set, while the distillation frontier shows that the same number is not equally dangerous there.

A practical implementation in this repository would be:

1. Start from the current narrow run.
2. Continue training with an intermediate range first, not with the full wide range.
3. Use a smoothed exploration process for the multiplier action.
4. Gate extra exploration or multiplier authority by mismatch magnitude when the trajectory is already near setpoint.
5. Only promote to the next wider range if held-out MAE improves and p95 multiplier use is not already saturating.
6. Stop widening when the held-out episode stops improving or the sampled spectral statistics cross the chosen admissibility limit.

## Residual-Style Gating For Matrix And Structured Multipliers

The saved wide runs show another practical issue. The policy keeps the model far from nominal even when the band-normalized raw tracking error is already small. That is exactly the situation where the residual method's deadband idea can help.

Use the same raw mismatch feature already logged in mismatch mode:

$$
\tau_t = \max_i |\mathrm{tracking\_error\_raw}_{i,t}|.
$$

Because `tracking_error_raw` is already normalized by the tracking scale, `tau_t \le 1` means the outputs are inside the reward band. Then apply a residual-style gate to the multiplier deviation:

$$
g_t = \begin{cases}
0, & \tau_t \le 1, \\
1 - \exp(-k(\tau_t - 1)), & \tau_t > 1,
\end{cases}
\qquad
m^{eff}_t = 1 + g_t (m^{rl}_t - 1).
$$

The same equation applies to structured multipliers `theta_t`. This keeps the policy free to make large corrections when tracking is bad, but collapses back toward the nominal model near setpoint.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_authority_gate.png" alt="Residual-style gate effect on multiplier deviation" width="1200" style="max-width: 100%; height: auto;" />

| Run | Tracking bin | Count | Current mean deviation | Gated mean deviation |
| --- | --- | --- | --- | --- |
| Matrix wide | <=1 | 346 | 0.1047 | 0.0000 |
| Matrix wide | 1-3 | 113 | 0.0864 | 0.0198 |
| Matrix wide | 3-10 | 91 | 0.0849 | 0.0647 |
| Matrix wide | 10-30 | 132 | 0.1139 | 0.1124 |
| Matrix wide | >30 | 118 | 0.1393 | 0.1393 |
| Structured wide | <=1 | 101 | 0.1555 | 0.0000 |
| Structured wide | 1-3 | 199 | 0.1557 | 0.0421 |
| Structured wide | 3-10 | 219 | 0.1378 | 0.1090 |
| Structured wide | 10-30 | 146 | 0.1511 | 0.1496 |
| Structured wide | >30 | 135 | 0.1664 | 0.1664 |
In the current wide matrix run, the mean deviation in the low-tracking bin is still about `0.1047`, and `52.9%` of low-tracking test steps still deviate from nominal by more than `0.1`. The structured-wide run is even more aggressive. The gate figure shows that a deadband plus exponential gate would cut that low-tracking authority sharply without changing the high-tracking regime nearly as much.

This is the cleanest way to borrow the residual idea here. The same mismatch signal and the same authority logic can be reused, but the action being gated is the model deviation instead of a residual control move.

## How Far Can The Range Be Widened Safely?

The nominal polymer physical `A` has spectral radius `0.9464`. In the final test episode, matrix wide already reaches a derived p95 spectral radius of `1.0934` and max `1.1357`. Structured wide is more aggressive still: mean spectral radius is `1.1054`, p95 is `1.1357`, max is `1.1357`, and several structured multipliers hit the hard upper bound `1.2` at p95.

So the answer is different for the two methods:

- Matrix: a slightly wider range may still be worth testing, but not symmetrically. The successful wide matrix policy mainly uses stronger `B` correction and slightly smaller `A`, so the safer next ablation is to widen `B` more than `A`, not to raise every bound uniformly.
- Structured: the current wide run already looks too aggressive for held-out evaluation. It pushes multiple grouped multipliers to `1.2`, keeps the test spectral radius above `1.0` on average, and doubles the final-test input movement from `0.0294` to `0.0702`. Widening further before adding guards is not justified by these results.

With a state-space model in the loop, the practical limit is not a single scalar bound. It is the largest uncertainty set for which the prediction model remains numerically admissible for MPC: stabilizable/detectable enough for the observer-controller pair, solver-feasible, and not so aggressive that held-out performance collapses. Robust MPC literature frames this as keeping the model family inside an uncertainty set where feasibility and robust performance can still be guaranteed or approximated [Chen2024] [Limon2013] [Kothare1996].

For this project, a practical safe-widening recipe is:

1. Keep the current solve fallback for structured wide.
2. Add an explicit spectral-radius cap or smooth fade-back to nominal when the prediction model gets too aggressive.
3. Widen bounds asymmetrically, favoring `B` before `A`.
4. Use a staged widening schedule tied to held-out test MAE and solver/fallback statistics.
5. Treat the empirical limit as reached when p95 multiplier use is already on the hard bound and held-out test performance no longer improves.

## Model-Usage Diagnostics

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_model_usage.png" alt="Wide-range multiplier usage and spectral-radius diagnostics" width="1200" style="max-width: 100%; height: auto;" />

Matrix wide uses the expanded range in a targeted way. It does not simply saturate everything upward. Instead, it tends to push the first input gain higher while keeping `A` on average below nominal. Structured wide behaves differently: several grouped multipliers have p95 equal to the hard upper bound `1.2`, and the test spectral radius stays above `1.1` on average. That explains why structured wide can still improve reward while giving an over-aggressive held-out trajectory.

## Is Reidentification Useless Now?

For the current polymer implementation, the honest answer is: reidentification is currently dominated, not theoretically useless. The widened direct-multiplier methods are clearly more effective today because they avoid the identification bottleneck and exploit a low-dimensional model family that the MPC can use immediately.

On the evidence in these runs, the research priority should be matrix/structured continuation before polymer reidentification. Matrix wide reaches final-test MAE `0.1534` while reidentification stays at `0.1770` with candidate-valid fraction `0.0011`. Until the identification windowing, excitation, and candidate validation are redesigned, polymer reidentification is not competitive with direct multiplier supervision.

That said, the literature does not support calling reidentification useless in general. It says the opposite: reidentification can work when the controller deliberately creates informative data and validates updates carefully [Berberich2022] [Heirung2015] [Heirung2017] [Oshima2024]. In this project, that means reidentification should be treated as a separate adaptive-control problem, not as a drop-in replacement for multiplier tuning.

## Conclusions

- The latest matrix wide run is genuinely more successful than the previous matrix runs. Its final test MAE `0.1534` beats the previous narrow refresh `0.1730`, the legacy matrix run `0.1717`, and baseline MPC `0.1701`.
- The latest capped matrix rerun stays in that successful regime. Relative to the first capped rerun, it trades a small MAE regression (`0.1487` to `0.1519`) for slightly better reward, while still using the capped `A` side and wide `B` side as intended.
- The latest structured rerun is the real correction in this update. Once the `A` cap was actually applied, final test MAE improved from `0.2454` to `0.1551`, spectral p95 dropped below `1`, and fallback events disappeared.
- Structured is therefore no longer a simple failure story. The right conclusion is narrower: structured can work with tight `A` / wide `B`, but it is more cap-sensitive than plain matrix and still likely needs a slightly tighter `A` limit or stronger acceptance logic for consistency.
- The actual polymer reward is bonus-balanced more than edge-balanced. If both outputs should matter more evenly in evaluation, the most direct reward fix is to reduce `Q1` toward about `212` or to separate inside-band bonus weights from outside-band penalty weights.
- Distillation reward balance is currently much more asymmetric than polymer. With `Q2 = 1500`, an even-output ablation should test `Q1` in the rough range `3773-11045`, not the current `37000`.
- For the matrix family, observability and controllability do not bound positive `alpha`; the meaningful analytical upper bound is `alpha < 1.0566` if all candidate `A` matrices must stay open-loop stable. There is no comparable analytical lower bound, so the lower side should be chosen by a trust-region or time-scale rule. For `B`, the useful bounds are about gain-trust and actuator headroom, not spectral stability.
- For structured updates, the matrix-derived cap `1.0566` is still a sensible first `A`-side bound. Monte Carlo sampling with `B_high = 1.25` gives zero unstable fraction at that cap. If more margin is wanted, the report's asymmetric frontier suggests `A_high` around `1.04-1.05` before tightening `B`.
- Distillation is mathematically much less fragile than polymer: `alpha_max_stable` is `1.1929` instead of `1.0566`, and the structured unstable fraction at `1.20` is only `0.013` instead of `0.556`. But the currently saved disturbance RL runs still do not beat the distillation baseline, so wider admissibility alone is not enough. A safer next design is wide raw search with mismatch-gated effective `A` authority plus nominal-cost backtracking/fallback.
- Shifted MPC warm start is worth testing beyond the matrix family, but primarily as a solver-quality improvement. The highest-payoff candidates in the current codebase are residual, matrix, structured, weights, and reidentification. Horizon is a weaker target because the subproblem dimension changes too often.
- Periodic observer redesign should not be treated as a blanket fix, and every-episode pole redesign is probably too aggressive. It is most defensible when sustained model drift is present and the refreshed model is trustworthy. That makes polymer matrix/structured and distillation reidentification more plausible targets than current distillation matrix/structured or polymer reidentification.
- The degradation-then-recovery pattern is expected after abrupt widening. The most practical fix is staged widening, smoother exploration, mismatch-gated authority, and uncertainty-set shaping before wider training ranges are accepted.
- Polymer reidentification is currently dominated by the widened direct-multiplier methods and should be deprioritized until the identification layer itself is redesigned around informative-window generation and candidate validation.

## Sources

- [Diehl2005] A Real-Time Iteration Scheme for Nonlinear Optimization in Optimal Feedback Control.
- [Jerez2014] Embedded online optimization for model predictive control at megahertz rates.
- [Berberich2022] Forward-looking persistent excitation in model predictive control.
- [Heirung2015] MPC-based dual control with online experiment design.
- [Heirung2017] Dual adaptive model predictive control.
- [Mayne2009] Robust output feedback model predictive control of constrained linear systems: Time varying case.
- [Oshima2024] Targeted excitation and re-identification methods for multivariate process and model predictive control.
- [Plappert2018] Parameter Space Noise for Exploration.
- [Souza2021] Stochastic MPC with adaptive nonlinear Kalman filtering for output-feedback control of systems subject to multiplicative uncertainty.
- [Korenkevych2019] Autoregressive Policies for Continuous Control Deep Reinforcement Learning.
- [Seo2025] Continuous Control with Coarse-to-fine Reinforcement Learning.
- [Chen2024] Robust model predictive control with polytopic model uncertainty through System Level Synthesis.
- [AdaptiveMPC2020] Adaptive model predictive control for a class of constrained linear systems with parametric uncertainties.
- [RAMPC2023] Robust adaptive model predictive control with persistent excitation conditions.
- [Limon2013] Robust feedback model predictive control of constrained uncertain systems.
- [Kothare1996] Robust constrained model predictive control using linear matrix inequalities.
- [OutputFeedbackRMPC2019] Output feedback robust MPC for linear systems with norm-bounded model uncertainty and disturbance.

[Diehl2005]: https://doi.org/10.1137/S0363012902400713
[Jerez2014]: https://doi.org/10.1109/TAC.2014.2351991
[Berberich2022]: https://doi.org/10.1016/j.automatica.2021.110033
[Heirung2015]: https://doi.org/10.1016/j.jprocont.2015.04.012
[Heirung2017]: https://doi.org/10.1016/j.automatica.2017.01.030
[Mayne2009]: https://doi.org/10.1016/j.automatica.2009.05.009
[Oshima2024]: https://doi.org/10.1016/j.jprocont.2024.103190
[Plappert2018]: https://openreview.net/forum?id=ByBAl2eAZ
[Souza2021]: https://doi.org/10.1016/j.automatica.2021.109951
[Korenkevych2019]: https://www.ijcai.org/proceedings/2019/0382.pdf
[Seo2025]: https://proceedings.mlr.press/v270/seo25a.html
[Chen2024]: https://doi.org/10.1016/j.automatica.2023.111431
[AdaptiveMPC2020]: https://doi.org/10.1016/j.automatica.2020.108974
[RAMPC2023]: https://doi.org/10.1016/j.automatica.2023.110959
[Limon2013]: https://doi.org/10.1016/j.jprocont.2012.08.003
[Kothare1996]: https://doi.org/10.1016/0005-1098(96)00063-5
[OutputFeedbackRMPC2019]: https://doi.org/10.1016/j.automatica.2019.07.002