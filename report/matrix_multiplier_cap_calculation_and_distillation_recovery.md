# Matrix Multiplier Cap Calculation And Distillation Recovery

Date: 2026-04-24

This report rewrites the cap-selection logic for the matrix and structured-matrix supervisors. It also explains why the polymer cap now works while the distillation cap still does not give better-than-MPC behavior, even after tightening the `A` multiplier to `[0.99, 1.01]`, keeping `B` wide, and reducing TD3 exploration and target policy smoothing noise to `0.01`.

The latest distillation result described by the user is not available as a saved result bundle in this tree. I treat it as a user-provided observation from 2026-04-24:

- TD3 exploration noise and target policy smoothing noise were reduced to `0.01`.
- `A` multiplier authority was capped tightly, approximately `[0.99, 1.01]`.
- `B` multiplier authority was kept wide.
- Performance still degraded heavily after release, then recovered after roughly 100 episodes.
- The recovered policy still did not beat baseline MPC.

That conclusion is consistent with the current notebook and runner code. The distillation cap issue is now less about open-loop `A` stability and more about policy release, `B` authority, reward alignment, and lack of a safe-improvement acceptance layer.

## Executive Summary

The cap should not be calculated from one scalar rule. Use a two-layer rule:

1. **Admissibility cap**: remove model candidates that are numerically unsafe, unstable, infeasible, or structurally invalid.
2. **Performance cap**: among admissible candidates, keep only the region that does not degrade baseline MPC on short protected rollouts or nominal-cost checks.

For scalar matrix multipliers:

$$
\begin{aligned}
A_{\theta} &= \begin{bmatrix}
\alpha A_0^{\mathrm{phys}} & 0 \\
0 & I
\end{bmatrix}, \\
B_{\theta} &= \begin{bmatrix}
B_0^{\mathrm{phys}}\operatorname{diag}(\delta_1,\delta_2) \\
0
\end{bmatrix}.
\end{aligned}
$$

The `A` high cap has a clean analytical first pass:

$$
\alpha_{\max,\mathrm{stable}} = \frac{\rho_{\mathrm{target}}}{\rho(A_0^{\mathrm{phys}})}.
$$

Using `rho_target = 1` gives the currently documented values:

| System | `rho(A0_phys)` | `1 / rho(A0_phys)` | Interpretation |
| --- | ---: | ---: | --- |
| Polymer | 0.9464 | 1.0566 | Tight `A` cap is necessary. |
| Distillation | 0.8383 | 1.1929 | Stability permits much wider `A`, but performance does not. |

For `B`, there is no equivalent spectral-stability cap. `B` caps must come from finite-horizon gain trust, actuator headroom, prediction-error checks, and closed-loop baseline non-regression.

The polymer result now makes sense: tight `A` plus wide `B` gave useful gain correction without letting the prediction poles become too aggressive. The latest capped polymer matrix and structured reruns both work:

| Polymer run | Final test phys MAE | Final test reward | `A` drift | `B` drift | Spectral p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matrix first capped | 0.1487 | -3.5844 | 0.1132 | 0.2146 | 1.0000 |
| Matrix latest capped | 0.1519 | -3.5773 | 0.0952 | 0.1563 | 0.9990 |
| Structured first capped | 0.2454 | -4.2549 | 0.1748 | 0.1893 | 1.1598 |
| Structured latest capped | 0.1551 | -3.5554 | 0.1788 | 0.1489 | 0.9999 |

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_latest_cap_reruns.png" alt="Latest polymer capped matrix and structured reruns" width="1200" style="max-width: 100%; height: auto;" />

The distillation result changes the lesson. Because distillation is already using tight `A`, wide `B`, and smaller noise, the remaining degradation is not solved by another static `A` cap. The next fix should be:

- nominal-cost backtracking/fallback for both matrix and structured matrix,
- mismatch-gated effective model authority,
- staged `B` authority release,
- TD3+BC-style actor anchoring during the post-warm-start handoff,
- reward rebalance for the distillation outputs.

## Current Code Facts

The current scalar matrix path uses `utils/multiplier_mapping.py` to map normalized actor actions in `[-1, 1]` into asymmetric physical multiplier bounds. The nominal action maps to multiplier `1.0`.

For scalar matrix updates, `utils/matrix_runner.py` applies:

$$
\begin{aligned}
A^{\mathrm{phys}}_{\mathrm{cand}} &= \alpha A^{\mathrm{phys}}_0, \\
B^{\mathrm{phys}}_{\mathrm{cand}} &= B^{\mathrm{phys}}_0 \operatorname{diag}(\delta).
\end{aligned}
$$

For structured updates, `utils/structured_model_update.py` builds either a block or band multiplier family. The active distillation default is block mode:

$$
\begin{aligned}
\theta_A &= \left[
\theta_{A,\mathrm{block1}},
\theta_{A,\mathrm{block2}},
\theta_{A,\mathrm{block3}},
\theta_{A,\mathrm{off}}
\right], \\
\theta_B &= \left[
\theta_{B,1},
\theta_{B,2}
\right].
\end{aligned}
$$

In block mode, diagonal state groups are scaled by their own `theta_A` values, off-block couplings are scaled by `theta_A_off`, and each input column of `B` is scaled by its corresponding `theta_B`.

The current distillation config in `systems/distillation/config.py` sets:

| Quantity | Current value |
| --- | --- |
| `DISTILLATION_MATRIX_ALPHA_UPPER_CAP` | `1.1929` |
| `DISTILLATION_MATRIX_ALPHA_DEFAULT_LOW` | `0.99` |
| `DISTILLATION_MATRIX_ALPHA_DEFAULT_HIGH` | `1.01` |
| `DISTILLATION_DEFAULT_MULTIPLIER_LOW` | `0.75` |
| `DISTILLATION_DEFAULT_MULTIPLIER_HIGH` | `1.25` |

The current distillation TD3 matrix defaults in `systems/distillation/notebook_params.py` now also use:

| TD3 field | Current value |
| --- | ---: |
| `target_policy_smoothing_noise_std` | 0.01 |
| `std_start` | 0.01 |
| `std_end` | 0.01 |
| `param_noise_std_start` | 0.01 |
| `param_noise_std_end` | 0.01 |
| `post_warm_start_action_freeze_subepisodes` | 5 |
| `post_warm_start_actor_freeze_subepisodes` | 5 |

That means the newest degradation is happening even after the first round of conservative release changes. In the active distillation matrix notebook output, the first several subepisodes stay at nominal multipliers, then the first live learned region degrades sharply. The visible output shows:

| Subepisode | Avg reward | Mean `alpha` | Mean `delta` |
| ---: | ---: | ---: | --- |
| 1 | 11.6800 | 1.0000 | `[1.0000, 1.0000]` |
| 2 | 17.6939 | 1.0000 | `[1.0000, 1.0000]` |
| 3 | 17.6866 | 1.0000 | `[1.0000, 1.0000]` |
| 4 | 17.6821 | 1.0000 | `[1.0000, 1.0000]` |
| 5 | 17.6792 | 1.0000 | `[1.0000, 1.0000]` |
| 6 | -1137.8245 | 0.9846 | `[0.9851, 1.0105]` |
| 7 | -915.3582 | 1.0695 | `[0.9968, 0.9943]` |

The active structured notebook output shows a similar pattern in a less extreme form: `theta_A` stays very close to `1.0`, while `theta_B` moves more, and late visible rewards remain far below the nominal warm-start reward. This supports the diagnosis that the problem is not only `A` high instability. The learned policy is still finding model changes that are admissible but bad for closed-loop performance.

## Why Stability Caps Are Not Enough

A stability cap answers this question:

> Will the candidate prediction model have acceptable open-loop physical-model poles?

The distillation failure asks a different question:

> Will the candidate prediction model improve the closed-loop MPC decision under the real nonlinear plant, disturbance profile, reward scaling, and input constraints?

Those are not equivalent.

For scalar `A`, open-loop stability is easy:

$$
\rho(\alpha A_0^{\mathrm{phys}}) = |\alpha| \rho(A_0^{\mathrm{phys}}),
$$

so a positive upper bound can be computed directly. But even with `alpha` restricted to `[0.99, 1.01]`, the finite-horizon input-output prediction can still change through `B`:

$$
G_N(A,B) = \begin{bmatrix}
C B \\
C A B \\
C A^2 B \\
\vdots \\
C A^{N-1}B
\end{bmatrix}.
$$

The MPC optimizer sees `G_N`, not only `rho(A)`. A wide `B` range can change predicted move authority and input allocation while all poles remain stable. That is useful in polymer, but in distillation it can make the optimizer trust a gain correction that the Aspen plant does not reward.

The cap design therefore needs three levels:

1. **Pole admissibility** for `A`.
2. **Finite-horizon gain trust** for `A` and `B` together.
3. **Closed-loop non-regression** against baseline MPC.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_cross_system_admissibility.png" alt="Cross-system admissibility comparison" width="1200" style="max-width: 100%; height: auto;" />

The existing cross-system admissibility figure shows why polymer and distillation split. Polymer needs the tight `A` cap because the nominal `A` is already close to the unit circle. Distillation has more open-loop spectral margin, but its saved and user-reported RL runs still degrade because admissible models can still be poor control models.

## Cap Calculation Procedure

Use this procedure for both matrix and structured matrix. The difference is only the candidate generator.

### Step 1. Define The Candidate Family

For matrix:

$$
\theta = [\alpha, \delta_1, \delta_2].
$$

For structured block mode:

$$
\theta = \left[
\theta_{A,\mathrm{block1}},
\theta_{A,\mathrm{block2}},
\theta_{A,\mathrm{block3}},
\theta_{A,\mathrm{off}},
\theta_{B,1},
\theta_{B,2}
\right].
$$

The raw actor space can remain wide, but the effective model space should be filtered or gated before MPC receives it.

### Step 2. Compute The Analytical `A` Ceiling

Choose a target spectral margin:

$$
\rho_{\mathrm{target}} \in [0.98, 0.995]
$$

for routine closed-loop training, or `rho_target = 1.0` only for a hard mathematical upper bound.

Then:

$$
\begin{aligned}
\alpha_{\mathrm{high}}^{\mathrm{stab}} &= \min\left(
\alpha_{\mathrm{user}}^{\mathrm{high}},
\frac{\rho_{\mathrm{target}}}{\rho(A_0^{\mathrm{phys}})}
\right).
\end{aligned}
$$

For polymer, `rho(A0_phys) ~= 0.9464`, so:

$$
\frac{1.0}{0.9464} \approx 1.0566.
$$

That is why the polymer `A` cap around `1.0566` is meaningful.

For distillation, `rho(A0_phys) ~= 0.8383`, so:

$$
\frac{1.0}{0.8383} \approx 1.1929.
$$

But the active distillation default is already much tighter than that:

$$
\alpha \in [0.99, 1.01].
$$

So for distillation, the next improvement should not be "make `A` even tighter" unless diagnostics show `A` is still causing a specific issue. It should be "keep `A` tight near nominal, but add performance filters and `B` authority control."

### Step 3. Calculate A Finite-Horizon Gain Cap

For each candidate model, compute the finite-horizon Markov matrix:

$$
\begin{aligned}
G_N(A_{\theta}, B_{\theta}) &= \begin{bmatrix}
C B_{\theta} \\
C A_{\theta}B_{\theta} \\
\cdots \\
C A_{\theta}^{N-1}B_{\theta}
\end{bmatrix}.
\end{aligned}
$$

Then define a gain-change ratio:

$$
\begin{aligned}
r_G(\theta) &= \frac{\left\|G_N(A_{\theta}, B_{\theta}) - G_N(A_0, B_0)\right\|_F}{\left\|G_N(A_0, B_0)\right\|_F + \epsilon}.
\end{aligned}
$$

Use the MPC prediction horizon for `N`: polymer matrix currently uses `predict_h = 9`; distillation matrix uses `predict_h = 6`.

A practical initial rule:

$$
\begin{aligned}
r_G(\theta) &\le \epsilon_G, \\
\epsilon_G &\in [0.10, 0.25].
\end{aligned}
$$

Use `0.10` for distillation until the policy reliably beats MPC, because the latest distillation run shows that wide `B` authority still creates a bad release.

### Step 4. Add Input-Headroom And Move-Size Caps

The `B` multiplier affects how much the MPC thinks each manipulated input can move the outputs. It should be capped by headroom:

$$
\begin{aligned}
h_j(t) &= \min\left(
\frac{u_{\max,j} - u_{t,j}}{u_{\max,j} - u_{\min,j}},
\frac{u_{t,j} - u_{\min,j}}{u_{\max,j} - u_{\min,j}}
\right).
\end{aligned}
$$

If the candidate model repeatedly drives low headroom or large input moves, shrink the effective `B` range:

$$
\begin{aligned}
\delta_{j}^{\mathrm{eff}} &= 1 + \lambda_{B,j}(\delta_j^{\mathrm{raw}} - 1), \\
\lambda_{B,j} &\in [0,1].
\end{aligned}
$$

A first rule for distillation:

$$
\begin{aligned}
\lambda_{B,j} &= \min\left(
1,
\frac{\operatorname{p50}(h_j)}{h_{\mathrm{safe}}}
\right), \\
h_{\mathrm{safe}} \in [0.15,0.25].
\end{aligned}
$$

This is better than narrowing `B` permanently, because it keeps `B` available when the column has room and shrinks it when the optimizer is already near an actuator edge.

### Step 5. Add A Nominal-Cost Acceptance Test

For each step, compare the candidate model against the nominal model before applying the candidate prediction model:

$$
\begin{aligned}
J_t^{\mathrm{cand}} &= J\!\left(
U_t^{\mathrm{cand}};
A_{\theta},
B_{\theta},
x_t,
y_t^{\mathrm{sp}}
\right), \\
J_t^{\mathrm{nom}} &= J\!\left(
U_t^{\mathrm{nom}};
A_0,
B_0,
x_t,
y_t^{\mathrm{sp}}
\right).
\end{aligned}
$$

Accept the candidate only if:

$$
J_t^{\mathrm{cand}} \le J_t^{\mathrm{nom}} + \epsilon_J.
$$

If it fails, backtrack toward nominal:

$$
\begin{aligned}
\theta^{\mathrm{eff}} &= 1 + \lambda(\theta^{\mathrm{raw}} - 1), \\
\lambda &\leftarrow \beta \lambda, \\
\beta &\in [0.25,0.5].
\end{aligned}
$$

until the candidate passes or `lambda = 0`, which recovers nominal MPC.

This is the missing piece in distillation. The current structured runner has a solve-failure fallback, but a solve can succeed while the closed-loop decision is still bad. Distillation needs a **performance fallback**, not only a numerical fallback.

## Structured Cap Calculation

For structured matrix, do not derive the cap only from the scalar matrix `alpha`. Sample the structured family.

Let:

$$
\Theta = \{\theta^{(1)}, \ldots, \theta^{(M)}\}.
$$

be random or grid-sampled candidates from the raw range. For each candidate, compute:

$$
\begin{aligned}
\rho_i &= \rho(A_{\theta^{(i)}}^{\mathrm{phys}}), \\
r_{G,i} &= r_G(\theta^{(i)}), \\
f_i &= \mathbf{1}\{\text{MPC solve and cost acceptance pass}\}.
\end{aligned}
$$

Build the accepted set:

$$
\begin{aligned}
\Theta_{\mathrm{acc}} &= \left\{
\theta^{(i)}
:
\rho_i \le \rho_{\mathrm{target}},
\quad
r_{G,i} \le \epsilon_G,
\quad
f_i = 1
\right\}.
\end{aligned}
$$

Then choose per-coordinate bounds from robust quantiles:

$$
\begin{aligned}
\theta_{\mathrm{low},j} &= Q_{0.05}(\Theta_{\mathrm{acc},j}), \\
\theta_{\mathrm{high},j} &= Q_{0.95}(\Theta_{\mathrm{acc},j}).
\end{aligned}
$$

Finally, test the worst corners. If the joint high corner fails, shrink the high quantile from `0.95` to `0.90`, or shrink only the coordinate with the largest contribution to `r_G`.

The existing polymer structured frontier supports this approach:

| Structured `A_high` | `B_high` | Unstable frac | Near-unit frac | Spectral p95 | Spectral max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.0000 | 1.25 | 0.0000 | 0.0000 | 0.9337 | 0.9464 |
| 1.0400 | 1.25 | 0.0000 | 0.0158 | 0.9710 | 0.9842 |
| 1.0500 | 1.25 | 0.0000 | 0.0418 | 0.9781 | 0.9936 |
| 1.0566 | 1.25 | 0.0000 | 0.0770 | 0.9870 | 0.9998 |
| 1.0800 | 1.25 | 0.0655 | 0.1870 | 1.0043 | 1.0221 |
| 1.2500 | 1.25 | 0.5945 | 0.6520 | 1.1598 | 1.1830 |

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_structured_asymmetric_cap_frontier.png" alt="Structured asymmetric A/B cap frontier" width="1200" style="max-width: 100%; height: auto;" />

For polymer, `A_high = 1.0566` was acceptable because it kept the structured spectral p95 below `1`. For distillation, `A_high = 1.01` is already conservative. Therefore, the structured distillation cap should be computed mostly through `G_N`, cost acceptance, and `B` authority, not through a tighter stability cap.

## Why Distillation Still Degrades

The current evidence points to four causes.

### 1. The Released Actor Can Still Leave The Warm-Start Support

Warm start fills replay with nominal MPC actions. The current Phase 1 hidden release then keeps executing the nominal action for several subepisodes while the critic catches up. This is good, and it is now implemented through `utils/phase1_hidden_release.py`.

But after release, the actor objective is still the usual TD3 deterministic policy improvement:

$$
\max_{\theta_{\pi}}
\mathbb{E}_{s \sim \mathcal{D}}
\left[
Q_{\phi}(s,\pi_{\theta_{\pi}}(s))
\right].
$$

There is no term that says:

$$
\pi_{\theta_{\pi}}(s) \approx a_{\mathrm{nom}}
$$

during the first live deployment. So the critic can still overvalue actions outside the narrow baseline-action support. This is exactly the failure mode discussed in TD3 and offline RL literature: value approximation error and out-of-distribution actions can lead the actor toward bad actions [Fujimoto2018] [FujimotoGu2021] [Kumar2020].

### 2. Reducing Noise Reduces Variance, Not Bias

Changing exploration and target policy smoothing noise to `0.01` makes the run less random, but it does not fix a biased actor objective or a wrong cost landscape. If the critic ranks a bad `B` correction too highly, small noise can still converge to that correction. It can also slow discovery of useful alternatives, which explains the user-observed pattern:

- heavy degradation after release,
- recovery only after enough online data accumulates,
- final performance still below MPC.

### 3. Distillation `B` Is A Performance-Sensitive Direction

The active structured notebook output shows `theta_A` near `1.0` while `theta_B` moves noticeably. That is exactly what the configured bounds allow: tight `A`, wide `B`.

The problem is that `B` changes input-output authority directly. In distillation, reflux and reboiler duty are strongly coupled and constrained. A wrong gain correction can make MPC believe an input move will be more useful than it actually is, or make it allocate effort to the wrong manipulated input. This can worsen tray-24 composition or tray-85 temperature even when `A` is perfectly stable.

### 4. The Distillation Reward Is Still Unbalanced

The distillation reward defaults are:

$$
\begin{aligned}
Q &= \operatorname{diag}(37000, 1500), \\
R &= \operatorname{diag}(2500, 2500).
\end{aligned}
$$

The reward also uses relative bands and inside-band bonus logic from `utils/rewards.py`, so the effective importance is not just `Q1/Q2`. Still, the earlier wide-range report already found that distillation is much more reward-asymmetric than polymer. If the first output dominates policy learning, the policy can recover reward while still not beating MPC in mean physical MAE or output-2 behavior.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_cross_system_reward_balance.png" alt="Cross-system reward balance" width="1200" style="max-width: 100%; height: auto;" />

## Recommended Distillation Fix

Do not keep narrowing `A`. Keep the current `A` default and add a protected effective-authority layer:

$$
\theta_t^{\mathrm{eff}} = 1 + \lambda_t(\theta_t^{\mathrm{raw}} - 1).
$$

Use separate gates for `A` and `B`:

$$
\begin{aligned}
\lambda_{A,t} &= \min(g_{\tau,t}, g_{J,t}, g_{\mathrm{ramp},t}), \\
\lambda_{B,t} &= \min(g_{\tau,B,t}, g_{J,t}, g_{\mathrm{headroom},t}, g_{\mathrm{ramp},B,t}).
\end{aligned}
$$

The mismatch gate should open only when tracking is bad:

$$
\tau_t = \max_i |e_{\mathrm{raw},i}(t)|.
$$

$$
\begin{aligned}
g_{\tau,t} &= \begin{cases}
0, & \tau_t \le 1, \\
1 - \exp[-k(\tau_t - 1)], & \tau_t > 1.
\end{cases}
\end{aligned}
$$

This makes the multiplier method behave like residual authority:

- near setpoint, use nominal MPC,
- when mismatch is large, allow model correction,
- if candidate cost is worse than nominal, backtrack to nominal,
- as training progresses, widen authority gradually.

<img src="./polymer_wide_range_matrix_structured/figures/wide_range_distillation_safe_search.png" alt="Distillation safe wide-search concept" width="1200" style="max-width: 100%; height: auto;" />

## TD3+BC Handoff Anchor

The next actor-side fix is a TD3+BC-style handoff regularizer. During the first `K` post-release subepisodes, change the actor loss from pure deterministic policy improvement to:

$$
\begin{aligned}
\mathcal{L}_{\mathrm{actor}} &= -\mathbb{E}_{s \sim \mathcal{D}}
\left[
Q_{\phi}(s,\pi_{\theta}(s))
\right] \\
&\quad
+ \lambda_{\mathrm{BC}}(e)
\mathbb{E}_{(s,a_0)\sim\mathcal{D}_{\mathrm{ws}}}
\left[
\|\pi_{\theta}(s)-a_0\|_2^2
\right].
\end{aligned}
$$

For matrix and structured matrix:

$$
a_0 = a_{\mathrm{nom}},
$$

the normalized action that maps to all multipliers equal to `1.0`.

Use a decaying weight:

$$
\lambda_{\mathrm{BC}}(e) = \lambda_{\mathrm{BC},0}\exp\left(-\frac{e-e_0}{T_{\mathrm{BC}}}\right).
$$

First distillation values to test:

| Parameter | First value |
| --- | ---: |
| `lambda_BC_0` | 0.1 |
| `T_BC` | 10 subepisodes |
| BC dataset | warm-start plus recent nominal/fallback transitions |
| apply to | TD3 matrix and structured matrix |

This is directly supported by TD3+BC and offline-to-online RL literature: when training starts from narrow logged behavior, actor regularization reduces out-of-distribution action selection [FujimotoGu2021] [Kumar2020] [Nair2021].

## Algorithm 1: Offline Cap Identification

```text
Input:
  nominal model A0, B0, C
  MPC horizons Np, Nc
  plant/result validation scenarios
  raw low/high bounds for A and B
  spectral target rho_target
  gain-change threshold eps_G
  cost-regression threshold eps_J

Build candidate set:
  if matrix:
      sample theta = [alpha, delta_1, delta_2]
  if structured:
      sample theta_A groups and theta_B columns

For each theta:
  construct A_theta, B_theta
  reject if structure is invalid or non-finite
  reject if rho(A_theta_phys) > rho_target
  compute G_N(A_theta, B_theta)
  reject if ||G_N(theta) - G_N(0)||_F / ||G_N(0)||_F > eps_G
  run short nominal-cost or rollout check
  reject if candidate cost exceeds nominal cost by eps_J

Return:
  accepted candidate set
  low/high caps as robust accepted quantiles
  corner-test report
```

## Algorithm 2: Online Protected Multiplier Execution

```text
Input:
  actor raw action a_raw
  multiplier bounds low/high
  nominal multiplier vector theta_nom = 1
  current tracking mismatch tau_t
  actuator headroom h_t
  nominal MPC cost J_nom

Map:
  theta_raw = map_action_to_bounds(a_raw, low, high)

Gate:
  lambda_A = min(mismatch_gate(tau_t), release_ramp(t), cost_gate)
  lambda_B = min(mismatch_gate_B(tau_t), release_ramp_B(t), headroom_gate(h_t), cost_gate)

Build:
  theta_eff_A = 1 + lambda_A * (theta_raw_A - 1)
  theta_eff_B = 1 + lambda_B * (theta_raw_B - 1)

Accept:
  while candidate predicted cost is worse than nominal:
      lambda_A *= beta
      lambda_B *= beta
      rebuild candidate

Fallback:
  if no accepted candidate:
      theta_eff = ones

Run MPC with theta_eff.
```

## Algorithm 3: Distillation Experiment Sequence

Run these in order. Do not open a wider search until the previous line stops degrading relative to MPC.

| Experiment | `A` range | `B` range | New protection | Success criterion |
| --- | --- | --- | --- | --- |
| D0 current TD3 | `[0.99, 1.01]` | `[0.75, 1.25]` | none beyond current Phase 1 | reproduce degradation/recovery |
| D1 cost fallback | `[0.99, 1.01]` | `[0.75, 1.25]` | nominal-cost backtracking | no release collapse |
| D2 `A` gate | raw wide or current | `[0.75, 1.25]` | mismatch-gated `A` authority | no worse than D1 |
| D3 `B` gate | current | `[0.75, 1.25]` raw | headroom and mismatch-gated `B` authority | lower final MAE than D1 |
| D4 TD3+BC | current | raw wide with gates | actor anchor to nominal action | faster recovery and no release trough |
| D5 reward rebalance | current | gated wide | lower `Q1` ablation | improve output-2 without losing output-1 |

The most important metric is not final reward alone. Track all of these:

$$
\Delta R_{\mathrm{release}} = \bar{R}_{\mathrm{live},1{:}10} - \bar{R}_{\mathrm{warm},\mathrm{tail}}.
$$

$$
\Delta \mathrm{MAE}_{\mathrm{test}} = \mathrm{MAE}_{\mathrm{RL},\mathrm{test}} - \mathrm{MAE}_{\mathrm{MPC},\mathrm{test}}.
$$

$$
\operatorname{fallback\_frac},
\quad
\operatorname{near\_bound\_frac},
\quad
\operatorname{p95}(|\theta^{\mathrm{eff}}-1|),
\quad
\operatorname{p95}(\Delta u).
$$

Declare success only if:

- `Delta R_release` is not a large negative drop,
- final test MAE beats or matches MPC,
- output-2 MAE does not silently degrade,
- fallback is not doing all the work,
- the learned policy uses nonzero authority only when mismatch warrants it.

## Literature Mapping

| Topic | Paper | Useful lesson for this project |
| --- | --- | --- |
| TD3 overestimation and target smoothing | [Fujimoto2018] | Delayed policy updates and clipped target noise reduce overestimation, but do not guarantee safe release from narrow data support. |
| TD3+BC | [FujimotoGu2021] | A simple behavior-cloning term can keep a TD3 actor close to dataset actions under offline or narrow-support training. |
| Conservative critics | [Kumar2020] | Distribution shift can make standard off-policy critics overvalue actions outside the dataset. |
| Offline-to-online handoff | [Nair2021] | Prior data can accelerate learning, but the handoff needs behavior-aware updates. |
| Safe policy improvement | [Laroche2019] | Fall back to the baseline policy where uncertainty is high. This maps directly to nominal-MPC fallback for multiplier policies. |
| Trust-region policy updates | [Schulman2015] | Large policy changes are risky; staged authority and action-distance penalties are practical trust regions. |
| Constrained policy optimization | [Achiam2017] | Safety constraints should be explicit, not only encoded in reward. |
| Learning-based MPC safety | [Hewing2020] | MPC is a good safety layer for learning, but learned model or policy changes need uncertainty-aware protection. |
| Robust MPC uncertainty sets | [Kothare1996], [Limon2013], [Mayne2005] | Treat the multiplier family as an uncertainty set and grow it only while feasibility/performance remain acceptable. |
| Adaptive MPC and excitation | [Berberich2022], [Heirung2015] | Model adaptation requires informative data and guarded model maintenance, not blind widening. |

## Final Recommendations

1. For polymer, keep the current tight-`A`, wide-`B` matrix cap. The latest capped matrix and structured runs show that this design is now working.
2. For polymer structured matrix, keep `A_high` around `1.0566` for now, but test `1.04-1.05` as a lower-risk consistency ablation.
3. For distillation, do not spend the next run only tightening `A`. The latest observation already used `[0.99, 1.01]` and still degraded.
4. Add nominal-cost backtracking/fallback to matrix and structured matrix. This is the highest-priority distillation fix.
5. Add mismatch-gated effective authority for both `A` and `B`, with a stricter gate for `A` and a headroom-aware gate for `B`.
6. Add TD3+BC actor anchoring during the post-warm-start handoff. The current hidden release helps, but the first live actor can still leave safe support.
7. Run a distillation reward-balance ablation. The current reward can recover while still not beating MPC in physical MAE, especially if output 2 is underprotected.
8. Only after D1-D5 work should `A` be widened toward the larger distillation analytical stability cap. The current bottleneck is policy/performance safety, not spectral admissibility.

## Sources

- [Fujimoto2018]: Addressing Function Approximation Error in Actor-Critic Methods. https://proceedings.mlr.press/v80/fujimoto18a.html
- [FujimotoGu2021]: A Minimalist Approach to Offline Reinforcement Learning. https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html
- [Kumar2020]: Conservative Q-Learning for Offline Reinforcement Learning. https://neurips.cc/virtual/2020/poster/17621
- [Nair2021]: AWAC: Accelerating Online Reinforcement Learning with Offline Datasets. https://openreview.net/forum?id=OJiM1R3jAtZ
- [Laroche2019]: Safe Policy Improvement with Baseline Bootstrapping. https://proceedings.mlr.press/v97/laroche19a
- [Schulman2015]: Trust Region Policy Optimization. https://proceedings.mlr.press/v37/schulman15
- [Achiam2017]: Constrained Policy Optimization. https://proceedings.mlr.press/v70/achiam17a
- [Hewing2020]: Learning-Based Model Predictive Control: Toward Safe Learning in Control. https://www.annualreviews.org/content/journals/10.1146/annurev-control-090419-075625
- [Kothare1996]: Robust constrained model predictive control using linear matrix inequalities. https://doi.org/10.1016/0005-1098(96)00063-5
- [Limon2013]: Robust feedback model predictive control of constrained uncertain systems. https://doi.org/10.1016/j.jprocont.2012.08.003
- [Mayne2005]: Robust model predictive control of constrained linear systems with bounded disturbances. https://doi.org/10.1016/j.automatica.2004.08.019
- [Berberich2022]: Forward-looking persistent excitation in model predictive control. https://doi.org/10.1016/j.automatica.2021.110033
- [Heirung2015]: MPC-based dual control with online experiment design. https://doi.org/10.1016/j.jprocont.2015.04.012
