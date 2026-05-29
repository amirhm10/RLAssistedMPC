# Distillation Markov Safety-Layer Audit And Restoration Plan

Date: 2026-05-29  
Case study: Aspen C2 splitter distillation column  
Controller family: TD3-assisted Markov-corrected MPC  
Scenario: `run_mode = "disturb"`, `disturbance_profile = "fluctuation"`

## Executive Summary

The Markov runner is the most sensitive of the current distillation RL-assisted MPC families because the TD3 action does not directly trim the first input move. It changes the lifted input-output response used by MPC. A good Markov correction can make MPC more predictive and more aggressive in the right direction. A bad Markov correction can make MPC confidently optimize the wrong local model.

The historical evidence says both things are true:

| Run family | Tail-20 reward | Final reward | TD3 tail fraction | Nominal fallback tail fraction | Main lesson |
|---|---:|---:|---:|---:|---|
| May 16 guarded | 17.529 | 16.242 | 0.000 | 0.933 | Safe but TD3 nearly absent |
| May 16 TD3-only | 18.050 | 19.219 | 1.000 | 0.000 | TD3 can help, but first-block temperature tradeoff was poor |
| May 18 TD3-only retuned | 21.978 | 22.223 | 1.000 | 0.000 | Reward retuning fixed much of the late tracking tradeoff, but early release shock remained severe |
| May 18 soft handoff | 21.247 | 20.386 | 1.000 | 0.000 in tail, 0.046 overall | Best historical safety-performance compromise |
| May 22 protected gate | 14.130 | 13.267 | 0.000 | 0.932 | This is the "copy-paste nominal" failure mode |
| May 23 controlled authority | -25.638 | -22.910 | 1.000 | 0.000 | Small TD3 authority alone did not make Markov safe |
| May 28 forced TD3 with z-safety | -27.582 | -27.171 | 1.000 | 0.000 | Magnitude clipping alone is not a quality filter |

The main conclusion is:

**Bring back Markov safety, but do not bring back the old hard gate in the same form. The old hard gate could make the controller look safe by making Markov nearly indistinguishable from nominal MPC. The better restoration is TD3-first with catastrophic fallback, z trust-region safety, reward probation, and source-aware diagnostics.**

The strongest reference for the next run is the May 18 soft-handoff behavior. It preserved TD3 authority while reducing the worst early crash:

| Metric | May 18 soft handoff | May 18 TD3-only no safeguard |
|---|---:|---:|
| Tail-20 reward | 21.247 | 21.978 |
| Final reward | 20.386 | 22.223 |
| Worst first-20 reward | -7.857 | -37.100 |
| Overall nominal fallback fraction | 0.0466 | 0.0000 |
| Tail TD3 fraction | 1.000 | 1.000 |

That is the behavior to recover: not nominal copy-paste, not unfiltered TD3.

## Files Inspected

Implementation files:

- `distillation_RL_assisted_MPC_markov_unified.py`
- `systems/distillation/notebook_params.py`
- `utils/markov_runner.py`
- `utils/behavioral_cloning.py`
- `utils/markov_diagnostics.py`

Saved raw bundles inspected directly:

- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260518_184548/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_td3_only_no_safeguard_unified/20260518_091937/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260522_190448/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260523_224108/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260528_230949/input_data.pkl`
- `Distillation/Results/distillation_markov_td3_disturb_fluctuation_unified/20260528_230949/markov_stage_diagnostics.csv`

Prior analysis reports and generated summaries:

- `report/distillation_markov_td3_only_family_2026_05_16.md`
- `report/distillation_markov_td3_decision_authority_2026_05_17.md`
- `report/distillation_markov_z_bound_safety_2026_05_20.md`
- `report/distillation_latest_5runner_safety_analysis_2026_05_29.md`
- `report/figures/distillation_markov_td3_only_family_20260516/summary_metrics.csv`
- `report/figures/distillation_markov_td3_decision_authority_20260517/summary_metrics.csv`
- `report/figures/distillation_markov_z_safety_20260520/markov_z_safety_summary.csv`
- `report/figures/distillation_latest_settings_20260529/latest_settings_summary_metrics.csv`

## What The Markov Method Is Doing

The baseline MPC uses the identified offset-free linear model around the Aspen steady state. The controlled outputs are:

$$ y_k = [x_{24,\mathrm{C_2H_6},k}, T_{85,k}]^\top. $$

The manipulated inputs are:

$$ u_k = [F_{\mathrm{reflux},k}, Q_{\mathrm{reb},k}]^\top. $$

The nominal augmented model is:

$$ x^+_{\mathrm{aug},k}=A_{\mathrm{aug}}x_{\mathrm{aug},k}+B_{\mathrm{aug}}\Delta u_k,\qquad y_k=C_{\mathrm{aug}}x_{\mathrm{aug},k}. $$

The nominal lifted MPC solves:

$$ U_0^\star=\arg\min_U \sum_{j=1}^{N_p} e_{k+j}^\top Qe_{k+j}+\sum_{j=0}^{N_c-1}\Delta u_{k+j}^\top R\Delta u_{k+j}. $$

The Markov controller builds a nominal lifted response matrix `G0`, then lets TD3 or LS propose a low-dimensional correction:

$$ G(z)=G_0+\sum_{i=1}^{4} z_iG_i. $$

For the active distillation basis:

- `z_1`: output 1 response to input 1
- `z_2`: output 1 response to input 2
- `z_3`: output 2 response to input 1
- `z_4`: output 2 response to input 2

The TD3 actor produces a raw action:

$$ a_{\theta,k}\in[-1,1]^4. $$

The runner maps it into Markov-correction coordinates:

$$ z_{\theta,k}=z_{\max}\,\mathrm{clip}(a_{\theta,k},-1,1). $$

After optional BC handoff and z-safety projection, the corrected MPC solves:

$$ U_z^\star=\arg\min_U J(U;G(z)). $$

The plant receives the first move of the chosen MPC sequence. This is why Markov is not a direct-action residual policy. The TD3 action changes the prediction geometry inside MPC.

## Why Markov Needs Different Safety Than Weights Or Residual

The current result should not be interpreted as "Markov is bad." It should be interpreted as "Markov is high leverage and needs candidate-quality safety."

Weights changes the penalty matrix. MPC still solves with the same model and constraints. Residual applies a small first-move trim after nominal MPC. Markov changes the lifted response model. If that response model is wrong, MPC can generate a physically plausible but directionally poor input sequence.

The May 28 failure is a good example:

| Diagnostic | May 28 Markov value |
|---|---:|
| Tail-20 reward | -27.582 |
| TD3 source fraction in tail | 1.000 |
| LS fallback fraction in tail | 0.000 |
| Nominal fallback fraction in tail | 0.000 |
| z requested projection fraction in tail | 1.000 |
| z vector projection fraction in tail | 1.000 |
| z coordinate clip fraction in tail | 0.000 |
| Tail mean z 2-norm | 0.060 |
| Tail q95 abs z coordinate | 0.030 |
| Requested cost-guard pass fraction in tail | 0.077 |
| Requested prediction-score mean in tail | -0.184 |
| Requested gain-drift mean in tail | 0.030 |

The actor was bounded, but the bounded action was still bad. z-safety kept the correction small in a geometric sense, but it did not check whether the corrected MPC candidate was useful.

## Historical Mechanism Timeline

### May 16: Guarded Markov Was Too Conservative

The May 16 guarded and relaxed runs had acceptable reward, but they were mostly not TD3 controllers.

| Variant | Tail-20 reward | Tail TD3 | Tail LS | Tail nominal |
|---|---:|---:|---:|---:|
| Guarded | 17.529 | 0.0000 | 0.0673 | 0.9328 |
| Relaxed | 17.376 | 0.0003 | 0.1425 | 0.8573 |
| LS only | 17.529 | 0.0000 | 0.0673 | 0.9328 |
| TD3 without LS | 17.287 | 0.0000 | 0.0000 | 1.0000 |
| TD3 only | 18.050 | 1.0000 | 0.0000 | 0.0000 |

This is the important copy-paste warning. The guarded and LS-only branches were nearly identical. The gate protected the plant, but it also suppressed TD3 almost completely.

The post-warm requested-step gate statistics explain why:

| Variant | TD3 gate pass | Score pass | Drift pass | Cost pass |
|---|---:|---:|---:|---:|
| Guarded | 0.0066 percent | 5.70 percent | 100 percent | 0.80 percent |
| Relaxed | 0.0066 percent | 2.54 percent | 100 percent | 1.52 percent |
| TD3 without LS | 0.0842 percent | 1.20 percent | 100 percent | 1.68 percent |
| TD3-only trajectory under old gate | 1.8816 percent | 12.99 percent | 100 percent | 9.50 percent |

The old gate would reject almost all of the same TD3 behavior that produced the best late reward in the May 16 family. That means the old hard prediction-score and cost gates were not just safety filters. They were also TD3-erasing filters.

### May 18: TD3 Had Real Upside, But Needed Release Safety

The May 18 TD3-only no-safeguard run had the strongest late reward and much better tracking after reward retuning:

| Metric | May 18 TD3-only | OF-MPC reference |
|---|---:|---:|
| Tail-20 reward | 21.978 | 13.898 |
| Final reward | 22.223 | 13.136 |
| Tail SP1 temperature MAE | 0.086 K | 0.172 K |
| Tail SP2 temperature MAE | 0.118 K | 0.212 K |
| Tail SP1 composition MAE | 0.000561 | 0.001508 |
| Tail SP2 composition MAE | 0.000873 | 0.001582 |

But it had unsafe release behavior:

| Metric | Value |
|---|---:|
| Worst first-20 reward | -37.100 |
| Negative first-20 episodes | 5 |
| Overall negative average episode fraction | 0.030 |

The May 18 soft-handoff run solved much of that early-release problem while keeping TD3 relevant:

| Metric | Soft handoff | TD3-only no safeguard |
|---|---:|---:|
| Tail-20 reward | 21.247 | 21.978 |
| Final reward | 20.386 | 22.223 |
| Worst first-20 reward | -7.857 | -37.100 |
| Overall TD3 accepted fraction | 0.950 | 1.000 |
| Overall nominal fallback fraction | 0.0466 | 0.000 |
| Probation trigger count | 8 | not active |

This is the key design target.

### May 22: Protected Gate Became Near-Nominal

The May 22 protected-gate Markov run was safe but too conservative:

| Metric | May 22 protected Markov | OF-MPC |
|---|---:|---:|
| Tail-20 reward | 14.130 | 13.898 |
| Final reward | 13.267 | 13.136 |
| Overall TD3 accepted fraction | 0.000 |
| Overall nominal fallback fraction | 0.931 |
| Tail nominal fallback fraction | 0.932 |

This matches your memory. The safety layer made performance almost a copy-paste of nominal MPC because TD3 effectively never entered the plant.

### May 23: Small Authority Alone Was Not Enough

The May 23 controlled-authority Markov run shows the opposite failure:

| Metric | May 23 controlled authority |
|---|---:|
| Tail-20 reward | -25.638 |
| Final reward | -22.910 |
| Overall TD3 accepted fraction | 0.950 |
| Overall nominal fallback fraction | 0.0457 |
| Mean TD3 authority scale | 0.288 |
| Probation active fraction | 0.945 |
| Probation trigger count | 190 |
| z bound | 0.04 |
| z 2-norm in tail | 0.020 |

This run is important because it proves that small Markov corrections can still be harmful if the accepted correction points in the wrong direction. Authority scaling and probation are useful, but they are not replacements for candidate-quality checks.

### May 28: z-Safety Alone Failed Under Forced TD3

The May 28 run kept z-safety active but disabled the layers that decide whether a candidate should reach the plant:

- `force_td3_execute = True`
- `td3_priority_fallback.enabled = False`
- reward probation inactive
- LS fallback inactive
- nominal fallback inactive
- protected release gate disabled

The z vector projection was active on every tail step and forced the vector norm to 0.06. Because four coordinates were involved, the projection made the coordinates sit near plus or minus 0.03. That is why the visible band looked like `[-0.03, 0.03]` even though the coordinate bound was `[-0.04, 0.04]`.

The failure mechanism is not insufficient z authority. It is poor candidate quality being forced through.

## Safety Layer Audit

Each layer below has a direct verdict: restore, restore in softened form, keep as diagnostic, or do not restore as a hard gate.

### 1. Hard z Coordinate Bound

What it does:

The TD3 raw action is clipped to `[-1, 1]` and mapped into the configured Markov range. The active default is:

$$ z_i\in[-0.04,0.04]. $$

Historical audit:

The old `z_bound = 0.05` was usable in the May 18 TD3-only and soft-handoff runs, but the z-safety audit showed that no-safeguard TD3 frequently pushed coordinates toward the hard bound. The May 20 audit found that the TD3-only no-safeguard run had q90 abs z coordinate about 0.048 and q95 z 2-norm about 0.092. Large z 2-norm was strongly associated with negative step rewards.

The May 28 run used `z_bound = 0.04`, but still failed badly. This proves that the bound is necessary but not sufficient.

Verdict:

**Bring back and keep. Do not increase to 0.05 yet.**

How it helps:

It prevents extreme Markov-model perturbations and keeps the action space more learnable. It also makes the safety envelope easier to reason about.

Copy-paste risk:

Low by itself. The hard z bound limits action magnitude, but it does not force nominal fallback. The copy-paste risk comes from the acceptance gate and fallback hierarchy, not from the outer z range.

### 2. z Vector Norm Cap

What it does:

After coordinate clipping, the runner can project the full z vector into a trust region:

$$ \lVert z\rVert_2\le z_{\mathrm{norm,max}}. $$

The active default is:

$$ z_{\mathrm{norm,max}}=0.06. $$

Historical audit:

This layer was active in the May 28 run. It explains why the four coordinates settled near 0.03 instead of 0.04. If four coordinates are all near 0.04, the vector norm would be about 0.08, so the 0.06 norm cap scales the vector down by about 0.75.

But May 28 still failed. The cap restricted magnitude, not quality. The requested candidates had tail prediction-score mean -0.184 and only 7.7 percent of tail requests passed the old cost guard.

Verdict:

**Bring back and keep. Keep 0.06 for the next recovery run.**

How it helps:

It prevents all four Markov directions from becoming large at the same time. It is especially useful because the per-coordinate cap does not prevent high combined model perturbation.

Copy-paste risk:

Low to medium. A very small norm cap could make Markov too close to nominal. The current 0.06 cap is not the reason May 22 became nominal-like, and it did not prevent May 28 from executing TD3. It is a useful trust region.

### 3. Dynamic z-Safety Cap By Phase Or Probation

What it does:

The runner can use a smaller effective coordinate cap during protected, ramp, full, or probation phases. The older safety schedule used values like:

| Phase | Example effective cap |
|---|---:|
| protected | 0.02 |
| ramp start | 0.03 |
| ramp end | 0.04 |
| full | 0.04 |
| probation | 0.02 |

The May 28 active default made all phase caps equal to 0.04, so it no longer tightened early or under probation.

Historical audit:

The May 23 controlled-authority run had dynamic z-safety and probation, but it still collapsed. This tells us dynamic z caps alone do not solve candidate quality. However, the May 18 soft handoff reduced release shock, and authority/probation were part of that behavior.

Verdict:

**Bring back, but only as a support layer.**

How it helps:

It makes early release and reward-collapse periods less aggressive. It should reduce the chance that a fresh TD3 policy uses full Markov authority immediately after warm start.

Copy-paste risk:

Medium if the cap stays too small for too long. Use it as a phase schedule, not a permanent lock. A good target is full-phase cap 0.04, not full-phase cap 0.02.

### 4. Adaptive LS Markov Correction

What it does:

The runner fits a least-squares Markov correction from recent prediction errors:

$$ z_{\mathrm{LS}}=\arg\min_z \sum_{\tau\in\mathcal{W}}\lVert y_\tau-\hat y_\tau(z)\rVert_W^2+\lambda_z\lVert z\rVert_2^2. $$

It can be used as:

- a warm-start source
- a fallback action
- a behavioral-cloning target
- a diagnostic shadow candidate

Historical audit:

LS was useful as a safe reference, but LS-dominated Markov was not the final goal. In the May 16 family, guarded and LS-only were nearly identical. In May 22, the protected-gate Markov result had TD3 accepted fraction 0 and nominal fallback about 93 percent. The result was safe, but not really a TD3 Markov method.

Verdict:

**Bring back as shadow, warm reference, and emergency fallback. Do not let it dominate by default.**

How it helps:

It gives the controller a model-consistent fallback and gives TD3 a reasonable early teacher. It also provides a good diagnostic for whether TD3 is moving in a direction that recent prediction errors support.

Copy-paste risk:

High if LS becomes the primary executed source. The report should always show TD3, LS, and nominal source fractions. If tail TD3 fraction is near zero, the run should not be claimed as learned Markov improvement.

### 5. LS Fallback

What it does:

If TD3 is rejected and the LS candidate is acceptable, the runner executes LS:

$$ z_{\mathrm{exec}}=z_{\mathrm{LS}}. $$

Historical audit:

LS fallback is safer than forced bad TD3. But old guarded runs frequently used LS or nominal instead of TD3. In May 16 guarded, tail LS fraction was 0.067 and nominal fraction was 0.933. In May 16 relaxed, LS increased to 0.143 while TD3 remained almost zero. That is not enough TD3 authority.

Verdict:

**Bring back, but only after TD3-first catastrophic screening.**

How it helps:

It prevents a bad Markov correction from reaching the plant when TD3 is clearly dangerous.

Copy-paste risk:

Medium to high if LS fallback triggers too often. The target should be rare LS fallback, not LS as the main live controller. A practical warning threshold is tail LS fraction above 0.10 unless the run is explicitly an LS-only ablation.

### 6. Nominal MPC Fallback

What it does:

If TD3 and LS are not acceptable, the runner executes the nominal lifted MPC solution:

$$ z_{\mathrm{exec}}=0,\qquad U_{\mathrm{exec}}=U_0^\star. $$

Historical audit:

Nominal fallback is the final safety net. It also caused the strongest copy-paste behavior when used too often. May 22 Markov had tail nominal fallback fraction 0.932 and tail reward 14.130, very close to OF-MPC 13.898.

Verdict:

**Bring back as last-resort fallback. Do not let it become the main policy.**

How it helps:

It prevents unstable or clearly harmful Markov candidates from being applied.

Copy-paste risk:

Very high if the acceptance gate is too strict. Any report should flag Markov runs with tail nominal fallback fraction above 0.20 as fallback-dominated. Above 0.80, the run is essentially nominal.

### 7. Prediction-Improvement Score

What it does:

The runner scores whether the Markov correction improves recent output prediction:

$$ s_{\mathrm{pred}}=\mathrm{SSE}_{\mathrm{nominal}}-\mathrm{SSE}_{\mathrm{corrected}}-\lambda_z\lVert z\rVert_2^2. $$

The old guarded logic required positive prediction score:

$$ s_{\mathrm{pred}}>s_{\min}. $$

Historical audit:

This was one of the strongest TD3-erasing gates. The May 16 successful TD3-only behavior would have mostly failed the old score gate. In the May 16 family, post-warm score pass rates were only 1.20 to 12.99 percent depending on variant, yet the TD3-only run had the best late reward in that family.

The May 18 retuned TD3-only run had much better performance and a less negative tail score than May 16. The May 28 failed run had a very negative tail score, around -0.184. So the score is useful as a risk signal, but zero is not the correct universal threshold.

Verdict:

**Do not bring back as a hard positive veto. Bring back as diagnostic and as a catastrophic floor.**

How it helps:

It can detect when TD3 is making the model prediction much worse, especially in protected and ramp phases.

Copy-paste risk:

Very high if the rule is `score > 0` on every step. That was one of the reasons old guarded Markov became nominal or LS-like.

Recommended use:

- log it every step
- block extremely negative candidates during protected and ramp phases
- use softer thresholds in full authority
- combine it with cost, drift, z norm, and predicted tracking risk instead of using it alone

### 8. Gain-Drift Gate

What it does:

The runner computes the drift between the corrected and nominal lifted matrices:

$$ d_{\mathrm{gain}}=\frac{\lVert G(z)-G_0\rVert}{\lVert G_0\rVert}. $$

The default guard is:

$$ d_{\mathrm{gain}}\le 0.10. $$

Historical audit:

Gain drift was not the main bottleneck. In the May 16 family, requested drift passed essentially 100 percent of the time. In May 28, tail requested gain drift was about 0.03, well below 0.10, but the run failed. In May 18 soft handoff, drift was smaller and performance was good.

Verdict:

**Bring back as a catastrophic sanity check, but do not treat it as the main safety mechanism.**

How it helps:

It prevents structurally extreme lifted-response changes. It is useful for catching numerical or boundary-seeking policies.

Copy-paste risk:

Low at the current 0.10 limit. It would become copy-paste only if tightened far below the observed useful TD3 envelope.

### 9. Nominal-Cost Guard

What it does:

For a candidate Markov correction, the runner solves the corrected MPC, then evaluates that candidate sequence under the nominal lifted model. It computes:

$$ \Delta J_{\mathrm{nom}}=J_{\mathrm{nom}}(U_z^\star)-J_{\mathrm{nom}}(U_0^\star). $$

The old direct guard accepted if:

$$ \Delta J_{\mathrm{nom}}\le \epsilon_{\mathrm{abs}}+\epsilon_{\mathrm{rel}}\lvert J_{\mathrm{nom}}(U_0^\star)\rvert. $$

Historical audit:

The old nominal-cost guard was too strict when used as a hard approval test. In the May 16 family, cost pass fractions were very low. The successful TD3-only trajectory would mostly have failed the old gate. The latest May 28 run had tail cost-pass fraction 0.077 and failed badly, so the cost signal is still meaningful.

The important distinction is:

- strict cost guard means TD3 almost never executes
- no cost guard means bad TD3 can be forced through
- broad catastrophic cost cap can preserve TD3 while blocking clear disasters

Verdict:

**Bring back as a phase-aware catastrophic cap, not as the old tiny hard gate.**

How it helps:

It blocks corrected MPC plans that are clearly worse than the nominal plan under the nominal model.

Copy-paste risk:

Very high if the tolerance is near the old direct gate. The May 16 evidence shows that strict cost gating can reject about 98 to 99 percent of useful TD3 behavior.

Recommended use:

Use the TD3-priority caps as the starting point:

| Phase | Absolute cap | Relative cap |
|---|---:|---:|
| protected | 0.02 | 5.0 |
| ramp | 0.05 | 20.0 |
| full | 0.10 | 50.0 |

These are not "reward improvement" gates. They are catastrophic fallback caps.

### 10. TD3 Priority Fallback

What it does:

This layer changes the decision hierarchy. Instead of making TD3 prove it is better every step, TD3 is the primary candidate after warm start and fallback is used only when the candidate is too risky.

Historical audit:

The May 18 soft-handoff result is the best evidence in favor of this layer:

- tail-20 reward 21.247
- overall TD3 accepted fraction 0.950
- nominal fallback fraction 0.0466
- worst first-20 reward -7.857 instead of -37.100

May 22 shows the danger if the gate becomes too restrictive. May 23 shows the danger if priority fallback accepts TD3 without enough quality discrimination.

Verdict:

**Bring back. This should be the main Markov safety layer, but it must be TD3-first and catastrophic, not TD3-erasing.**

How it helps:

It keeps TD3 live enough to learn and improve, while retaining LS and nominal MPC as recovery tools.

Copy-paste risk:

Medium. It depends on thresholds. The report should require source-fraction evidence. A good recovery run should have tail TD3 fraction above 0.70 and nominal fallback below 0.10 unless there is a documented disturbance or release event.

### 11. TD3 Authority Scaling

What it does:

The priority fallback has an authority ramp that can reduce TD3 action magnitude during protected or probation phases:

$$ a_{\mathrm{scaled}}=\alpha_k a_{\theta,k},\qquad 0\le\alpha_k\le1. $$

Developed defaults include:

- protected scale 0.25
- ramp start scale 0.25
- ramp end scale 1.0
- full scale 1.0
- cooldown scale 0.25 during probation

Historical audit:

Authority scaling helped the May 18 soft-handoff result reduce release shock. But May 23 proves authority scaling alone is not enough. That run had mean authority scale about 0.288 and still collapsed.

Verdict:

**Bring back as a release-safety layer, not as a substitute for candidate quality.**

How it helps:

It makes early TD3 actions smaller while the actor and critic are still adapting.

Copy-paste risk:

Medium if authority remains low for the full run. The cap should ramp to 1.0 after the protected phase unless reward probation is active.

### 12. Reward Probation

What it does:

The runner compares new subepisode reward against a warm reference. If reward collapses by more than a threshold, it triggers cooldown:

$$ \bar r_{\mathrm{episode}} < \bar r_{\mathrm{warm,ref}}-\Delta r_{\mathrm{collapse}}. $$

During cooldown, TD3 authority can be reduced and z caps can be tightened.

Historical audit:

May 18 soft handoff had 8 probation triggers and reduced the worst first-20 crash from -37.100 to -7.857. That is strong evidence that probation is useful. May 23 had 190 probation triggers and still collapsed, which means probation cannot be the only safety mechanism.

Verdict:

**Bring back. It is one of the most useful release-shock layers.**

How it helps:

It reacts to actual closed-loop harm rather than only predicted model metrics.

Copy-paste risk:

Medium if probation stays active too long. The report should log probation trigger count, active fraction, and tail authority scale. If tail probation is still active, the run is not fully released.

### 13. Protected Behavioral Cloning To LS Action

What it does:

The TD3 actor can be trained with an auxiliary BC loss toward a safe target. For Markov, the target is LS action:

$$ \mathcal{L}_{\mathrm{BC}}=\lVert a_\theta(s_k)-a_{\mathrm{LS},k}\rVert^2. $$

Historical audit:

LS-target BC is helpful for initializing TD3 near a model-consistent correction. But if it lasts too long or is combined with strict fallback, it teaches the actor to imitate the fallback policy. That can make the Markov method converge toward LS or nominal behavior rather than learning a useful TD3 correction.

The May 16 analysis already showed that guarded and LS-only were almost identical. That is the warning sign.

Verdict:

**Bring back only as short early support or diagnostic. Do not use long LS imitation as the main training signal after release.**

How it helps:

It can reduce chaotic early exploration and provide a sensible starting policy.

Copy-paste risk:

High if LS-target BC remains strong after warm start while fallback is also frequent. The replay buffer and actor loss both become dominated by fallback-like behavior.

### 14. Protected BC Release Gate

What it does:

The release gate compares TD3 raw action to the safe BC target and blocks live release until TD3 is close enough:

- mean action gap threshold
- max coordinate gap threshold
- required window fraction

Historical audit:

This is exactly the type of layer that can create the nominal-copy behavior the user remembered. A hard release gate can protect the plant, but it can also prevent TD3 from entering the loop. May 22 is the warning case: TD3 accepted fraction 0 and nominal fallback about 93 percent.

Verdict:

**Do not bring back as a permanent hard gate. Keep it as diagnostic or use it only to trigger soft authority reduction.**

How it helps if softened:

It tells us whether TD3 is far from LS. That is useful information. Instead of blocking TD3 completely, a large gap should shrink authority, trigger backtracking, or tighten z caps.

Copy-paste risk:

Very high if used as a hard live-release condition.

### 15. BC Handoff Raw-Action Blending

What it does:

BC handoff blends the TD3 raw action with the safe action:

$$ a_{\mathrm{blend},k}=(1-\alpha_k)a_{\mathrm{safe},k}+\alpha_ka_{\theta,k}. $$

The active protected BC defaults use:

- start authority 0.1
- end authority 1.0
- active subepisodes 10

Historical audit:

BC handoff helped in the May 18 soft-handoff family, but it did not save the May 28 run. In May 28, handoff was active but `force_td3_execute=True` and priority fallback was disabled. Once the handoff reached full authority, bad candidates were still forced through.

Verdict:

**Bring back and keep, but never treat it as the only Markov safety layer.**

How it helps:

It smooths the transition from safe behavior to TD3 behavior and reduces abrupt release shock.

Copy-paste risk:

Medium if the safe action dominates for too long. The handoff authority should reach 1.0, and the report should verify that tail TD3 remains active.

### 16. Old TD3 Authority Ramp Diagnostic

What it does:

There is an older `td3_authority_ramp` path with a Markov mode named `z_safety_live_release`. It currently behaves mostly as diagnostic release-gate machinery and is disabled in the active defaults.

Historical audit:

The more useful authority path is now inside `td3_priority_fallback.authority_ramp` and the BC handoff. The older ramp should not be stacked blindly with those layers.

Verdict:

**Keep disabled or diagnostic unless a specific ablation needs it.**

How it helps:

It can log release behavior, but it is not the main mechanism to restore.

Copy-paste risk:

Medium if stacked with release gate, BC handoff, and priority ramp at the same time. Too many authority limiters can make TD3 irrelevant.

### 17. Executed-Action Replay

What it does:

The runner can store the executed action in replay:

$$ (s_k,a_{\mathrm{exec},k},r_k,s_{k+1}). $$

This is the active default:

$$ \texttt{rl_store_executed_action_in_replay=True}. $$

Historical audit:

This is correct when fallback changes the action. If the plant executes nominal fallback but replay stores the rejected TD3 action, the critic is trained on a transition that did not come from the stored action.

The downside is that if fallback is frequent, replay becomes nominal or LS-heavy. That is what makes copy-paste behavior self-reinforcing.

Verdict:

**Keep. Also add or continue source-aware replay diagnostics.**

How it helps:

It preserves physical consistency in replay.

Copy-paste risk:

Medium indirectly. The replay rule itself is correct. The copy-paste risk comes from fallback being too frequent. The solution is not to store the wrong action. The solution is to reduce fallback dominance and log requested-versus-executed gaps.

### 18. Action-Source Accounting

What it does:

The runner logs the execution source:

| Code | Meaning |
|---:|---|
| 0 | nominal no Markov |
| 1 | warm-start LS |
| 2 | TD3 accepted |
| 3 | LS fallback |
| 4 | nominal fallback |
| 5 | LS no RL |

Historical audit:

This was essential for discovering the copy-paste issue. Reward alone made some guarded runs look okay. Source fractions revealed that TD3 was absent.

Verdict:

**Keep and make it a required report metric.**

How it helps:

It separates real learned Markov behavior from fallback-dominated safety.

Copy-paste risk:

None. This is a diagnostic layer.

### 19. Candidate Diagnostics

What it does:

The runner logs:

- requested prediction score
- requested gain drift
- requested nominal-cost margin
- requested cost-guard pass
- LS candidate diagnostics
- executed candidate diagnostics
- z projection activity
- first-move difference from nominal

Historical audit:

The May 28 run would be impossible to diagnose correctly without these logs. They showed that TD3 was forced through even though candidate quality was poor.

Verdict:

**Keep. Promote some diagnostics into safety decisions, but carefully.**

How it helps:

It tells us why the controller accepted, softened, or rejected a candidate.

Copy-paste risk:

None from logging. Risk appears only when diagnostics become too-strict hard vetoes.

## Layer Restoration Verdict Table

| Layer | Restore? | Recommended role | Main help | Copy-paste risk |
|---|---|---|---|---|
| Hard z coordinate bound | Yes | Always-on outer bound | Limits extreme model correction | Low |
| z vector norm cap | Yes | Always-on trust region | Prevents all coordinates being large together | Low to medium |
| Dynamic z cap | Yes, softened | Release and probation support | Reduces early authority | Medium |
| Adaptive LS | Yes, limited | Shadow, warm reference, emergency fallback | Provides model-consistent safe action | High if dominant |
| LS fallback | Yes, limited | Emergency fallback after TD3 check | Prevents bad TD3 application | Medium to high |
| Nominal fallback | Yes, last resort | Final safety net | Prevents catastrophic candidates | Very high if frequent |
| Prediction score | Partly | Diagnostic and catastrophic floor | Flags model-worsening corrections | Very high if positive hard veto |
| Gain drift | Yes | Catastrophic sanity check | Blocks structurally extreme corrections | Low |
| Nominal-cost guard | Yes, changed | Phase-aware catastrophic cap | Blocks clearly bad candidates | Very high if strict |
| TD3 priority fallback | Yes | Main decision safety layer | Keeps TD3 primary but recoverable | Medium |
| Authority scaling | Yes | Release and cooldown | Reduces shock | Medium |
| Reward probation | Yes | Closed-loop harm response | Reduces release crashes | Medium |
| LS-target BC | Partly | Short early support | Initializes actor near safe correction | High if prolonged |
| BC release gate | Diagnostic only | Soft trigger, not hard block | Detects TD3-LS disagreement | Very high if hard |
| BC handoff | Yes | Smooth transition | Reduces sudden authority jump | Medium |
| Old authority ramp | Usually no | Diagnostic only | Extra logging | Medium |
| Executed-action replay | Yes | Replay correctness | Trains critic on real transition | Medium indirectly |
| Source accounting | Yes | Required diagnostic | Detects fallback domination | None |
| Candidate diagnostics | Yes | Required diagnostic and selected filters | Explains safety decisions | None as logging |

## Recommended Markov Recovery Configuration

For the next Markov run, I would not return to the May 16 or May 22 hard gate. I would restore the May 18 soft-handoff philosophy with the newer z-safety envelope.

Recommended next-run settings:

```python
force_td3_execute = False
rl_fallback_to_ls = True
td3_priority_fallback["enabled"] = True
td3_priority_fallback["authority_ramp"]["enabled"] = True
td3_priority_fallback["reward_probation"]["enabled"] = True
rl_store_executed_action_in_replay = True
z_bound = 0.04
z_safety["enabled"] = True
z_safety["vector_norm_cap"]["enabled"] = True
z_safety["vector_norm_cap"]["max_norm"] = 0.06
```

For phase caps, use a modest release schedule rather than all caps equal:

```python
z_safety["protected_cap"] = 0.025  # or 0.03 if 0.025 is too restrictive
z_safety["ramp_start_cap"] = 0.03
z_safety["ramp_end_cap"] = 0.04
z_safety["full_cap"] = 0.04
z_safety["probation_cap"] = 0.025
```

For candidate quality:

- keep gain drift limit at 0.10
- use TD3-priority cost caps, not the old strict direct cost gate
- do not require positive prediction score in the full phase
- use prediction score as a protected/ramp warning and catastrophic floor
- record requested, LS, and executed diagnostics every step

## How To Avoid The Nominal Copy-Paste Trap

Every future Markov run should be judged using both reward and authority.

A Markov run is not a successful TD3 Markov run if:

- tail TD3 fraction is near zero
- nominal fallback fraction is above 0.80
- LS fallback fraction dominates the tail
- reward is close to OF-MPC and source logs show mostly nominal

Suggested labels:

| Tail source behavior | Interpretation |
|---|---|
| TD3 above 0.70, nominal below 0.10 | TD3-primary Markov |
| TD3 between 0.20 and 0.70 | Hybrid Markov |
| TD3 below 0.20, nominal or LS high | Fallback-dominated safety run |
| TD3 near zero, reward near OF-MPC | Nominal copy-paste |

The May 22 run should be labeled fallback-dominated, not Markov-improved.

## What Still Needs A New Layer

The implemented safety stack can recover much of the May 18 behavior, but one layer is still missing for a final-quality Markov method: **candidate softening or backtracking**.

Instead of choosing between full TD3 and full fallback, evaluate scaled candidates:

$$ z_\alpha=z_{\mathrm{safe}}+\alpha(z_{\mathrm{TD3}}-z_{\mathrm{safe}}),\qquad \alpha\in\{1.0,0.75,0.5,0.25,0.0\}. $$

Choose the largest alpha that passes catastrophic checks. This would preserve TD3 direction while reducing the chance of forced bad Markov corrections.

This is not currently the main implemented layer. It should be the next development step after the immediate recovery run.

## Recommended Next Experiments

### Experiment 1: Restore TD3-Priority Soft Handoff With Current z-Safety

Purpose:
recover the May 18 soft-handoff behavior without returning to nominal copy-paste.

Files:

- `systems/distillation/notebook_params.py`
- `distillation_RL_assisted_MPC_markov_unified.py`
- `utils/markov_runner.py`

Change:

- set `force_td3_execute=False`
- set `rl_fallback_to_ls=True`
- enable `td3_priority_fallback`
- enable reward probation
- keep `z_bound=0.04`
- keep vector norm cap 0.06

Success criteria:

- tail-20 reward above 18
- worst first-20 reward above -10
- tail TD3 fraction above 0.70
- tail nominal fallback below 0.10
- no persistent vector projection saturation at every tail step

Failure criteria:

- TD3 fraction near zero means the gate is too conservative
- tail reward negative means candidate quality is still unsafe
- probation active in tail means release never fully recovered

### Experiment 2: Prediction-Score Threshold Ablation

Purpose:
find a score rule that prevents May 28-style bad corrections without rejecting May 18-style useful TD3.

Variants:

- no hard score veto, diagnostic only
- protected and ramp floor only
- full-phase catastrophic floor only
- rolling score trend rather than stepwise veto

Metrics:

- tail TD3 fraction
- tail reward
- first-20 minimum reward
- requested score quantiles
- score-triggered fallback fraction

Expected result:

A positive hard veto will likely recreate nominal copy-paste. A negative catastrophic floor may help.

### Experiment 3: Cost-Cap Envelope Ablation

Purpose:
separate useful Markov deviation from catastrophic wrong-model behavior.

Variants:

- current TD3-priority caps
- tighter protected cap only
- tighter protected and ramp caps
- full-phase cap unchanged

Metrics:

- cost-triggered fallback fraction
- tail TD3 fraction
- nominal fallback fraction
- reward and tracking by setpoint block

Expected result:

Tight full-phase caps are likely to suppress TD3. Tighter protected caps may reduce early release shock without harming final authority.

### Experiment 4: Backtracking Candidate Shield

Purpose:
avoid binary accept-or-fallback behavior.

Implementation:

- add candidate list over alpha values
- use LS accepted candidate or zero as `z_safe`
- evaluate the corrected MPC sequence for each alpha
- execute the largest acceptable alpha

Success criteria:

- lower early crash than no-safeguard TD3
- higher tail TD3 authority than hard guarded Markov
- fewer nominal fallback events than May 22

### Experiment 5: Source-Aware Replay Audit

Purpose:
verify whether the replay buffer is learning from TD3 behavior or fallback behavior.

Metrics:

- replay pushed steps by action source
- executed-versus-requested raw-action gap
- TD3 fraction in training windows
- BC loss versus actor loss

Expected result:

If fallback is rare, executed-action replay is correct and not suppressive. If fallback dominates, the run is likely training on nominal or LS behavior.

## Final Recommendation

For Markov, do not simply "bring everything back." Bring back the layers that made the May 18 soft-handoff result safe, but avoid the exact hard-gate behavior that made May 16 and May 22 look like nominal MPC.

Immediate restoration:

- keep hard z bound at 0.04
- keep z vector norm cap at 0.06
- turn off `force_td3_execute`
- turn on TD3-priority fallback
- turn on reward probation
- turn on LS and nominal fallback as emergency recovery
- keep executed-action replay
- keep BC handoff
- keep release gate diagnostic, not hard-blocking
- use prediction score and nominal cost as risk signals, not strict improvement proof

Research direction:

The final Markov method should be **TD3-primary corrected MPC with safety softening**, not forced TD3 and not fallback-dominated nominal MPC.

## Remaining Uncertainty

The conclusions above are based on saved single-run trajectories and dated analysis bundles, not a multi-seed Markov study. The mechanism evidence is strong because source logs, candidate diagnostics, and reward histories all tell the same story, but the exact thresholds still need controlled reruns.

The largest open question is not whether safety should return. It should. The open question is how much of the safety should be hard fallback versus soft backtracking. The historical evidence favors soft backtracking as the next development step.

## Figures And Data Evidence Used

- `report/figures/distillation_markov_td3_only_family_20260516/fig_action_source_and_reward_summary.png`
- `report/figures/distillation_markov_td3_only_family_20260516/fig_td3_only_reward_tradeoff.png`
- `report/figures/distillation_markov_td3_decision_authority_20260517/fig_gate_breakdown.png`
- `report/figures/distillation_markov_td3_decision_authority_20260517/fig_td3_only_gate_misalignment.png`
- `report/figures/distillation_markov_z_safety_20260520/fig_z_norm_reward_risk.png`
- `report/figures/distillation_latest_settings_20260529/fig_markov_source_and_guard.png`
- `report/figures/distillation_latest_settings_20260529/fig_markov_safety_history.png`
