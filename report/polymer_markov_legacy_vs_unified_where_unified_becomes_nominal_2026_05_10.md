# Polymer Markov: Where Does Unified Become Nominal?

Date: 2026-05-10

Latest runs compared:

- Legacy: `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/`
- Unified: `Polymer/Results/td3_markov_disturb/20260510_204834/`

Generated figures:

- `report/figures/polymer_markov_legacy_vs_unified_20260510/legacy_vs_unified_metric_grid.png`
- `report/figures/polymer_markov_legacy_vs_unified_20260510/legacy_vs_unified_prediction_diagnostics.png`

## Files inspected

- `polymer_markov_corrected_mpc_legacy.ipynb`
- `polymer_markov_corrected_mpc_unified.ipynb`
- `report/scripts/generate_polymer_markov_correction_assets_legacy.py`
- `report/scripts/analyze_polymer_markov_latest_legacy_vs_unified.py`
- `utils/markov_runner.py`
- `systems/polymer/notebook_params.py`
- `report/polymer_markov_latest_run_analysis_2026_05_09.md`
- `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/markov_stage_diagnostics.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/summary_metrics.json`
- `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/acceptance_summary.csv`
- `Polymer/Results/polymer_markov_corrected_mpc/20260510_204243/prediction_score_summary.csv`
- `Polymer/Results/td3_markov_disturb/20260510_204834/markov_stage_diagnostics.csv`

## What the current method is doing

Both notebooks implement the same Markov-corrected lifted MPC idea:

$$ G(z) = G_0 + \sum_{j=1}^{r} z_j \Delta G_j, \qquad z_j \in [-0.05, 0.05]. $$

The LS teacher and TD3 policy are both screened by the same prediction-improvement score:

$$ S_{\mathrm{pred}}(z) = \sum_{\tau \in \mathcal{T}_k} \left(\|W_y e_\tau^{\mathrm{nom}}\|_2^2 - \|W_y e_\tau^{z}\|_2^2 \right) - \lambda_z \|z\|_2^2, \qquad \lambda_z = 10^{-3}. $$

The accepted live action is still chosen by the same guard logic:

1. accept TD3 if solve success, positive enough score, bounded gain drift, and nominal-cost guard pass
2. otherwise execute the accepted LS correction
3. otherwise fall back to nominal MPC

The important point is that the Markov math is now essentially aligned. The latest legacy run also uses the shared unified reward, so reward mismatch is no longer the main explanation.

## Mathematical interpretation

The two runs now share the same key mathematical ingredients:

- same `basis_family = "io_pair_gain"`
- same `z_bound = 0.05`
- same `prediction_window = 20`
- same `s_pred_min = 1e-6`
- same `gain_drift_max = 0.10`
- same nominal-cost guard tolerances
- same observer alignment: `legacy_previous_measurement`
- same reward family: shared relative-QR reward
- same nominal reference path in the unified defaults: `lifted_g0_prototype`

So the current gap is not a different control objective or a different Markov correction formula. It is mainly a different closed-loop data regime and action-release schedule.

## Main result interpretation

### Run-level comparison

| Quantity | Legacy | Unified | Unified minus legacy |
| --- | ---: | ---: | ---: |
| First TD3 step | 8005 | 9 | earlier by 7996 steps |
| Mean executed `z`-norm | 0.0855 | 0.0782 | -0.0074 |
| Mean executed gain drift | 0.03595 | 0.03411 | -0.00184 |
| Mean first-move diff norm | 0.001315 | 0.001532 | +0.000217 |
| Mean full-sequence diff norm | 0.004273 | 0.004951 | +0.000678 |
| Mean executed nominal-cost margin | 3.56e-5 | 3.82e-5 | +2.56e-6 |
| Mean executed prediction score | 0.03280 | 0.02401 | -0.00878 |
| Mean LS prediction score | 0.04119 | 0.03065 | -0.01054 |
| Mean requested TD3 score | 0.00362 | 0.00059 | -0.00302 |
| Overall TD3 accepted fraction | 0.3235 | 0.3950 | +0.0715 |
| Overall LS fallback fraction | 0.4668 | 0.5685 | +0.1018 |
| Overall nominal fallback fraction | 0.0111 | 0.0363 | +0.0252 |
| Warm-start LS fraction | 0.1984 | 0.0000 | -0.1984 |

### What the plots say

The requested six signals show three clear facts.

1. Unified is not more nominal by raw move-difference norm.
   Its executed first-move and full-sequence differences are actually a bit larger than legacy on average.

2. Unified is weaker in prediction value.
   Both its executed score and its LS teacher score are materially lower than legacy.

3. Legacy works because the teacher dominates for a long time.
   The first 10 episodes are essentially LS warm-start in the legacy run, and even the tail of the run is still heavily LS-driven.

## Where does unified become nominal?

Short answer:

It does not undergo a late dramatic collapse into source-4 nominal fallback. Instead, it becomes nominal-proximal almost immediately.

That conclusion comes from three pieces of evidence.

### 1. There is no late source-4 takeover

Unified nominal fallback is only `3.63%` overall and `3.34%` over the last 5000 steps. The last 10 unified episodes stay around `2.9%` to `4.4%` nominal fallback, not `50%` to `100%`.

So the unified controller is not "failing because it is mostly executing nominal MPC" in the literal action-source sense.

### 2. Unified is nominal-proximal in cost from the beginning

Its executed nominal-cost margin is already on the order of `4e-5` in the first few episodes and stays on the same order in the last few episodes. That means the accepted corrected sequence is being kept extremely close to the nominal solution in nominal-cost space for the entire run.

This is why I would say unified becomes nominal immediately, not late:

- not because it chooses source-4 often
- but because the guard keeps every accepted correction extremely close to the nominal solution in value

### 3. Unified corrections are lower-value corrections from episode 1

Its mean requested TD3 score is only `0.00059`, compared with `0.00362` for legacy. Its mean executed score is `0.02401`, compared with `0.03280` for legacy. Its mean LS score is `0.03065`, compared with `0.04119` for legacy.

So the unified run is not collapsing late. It is operating in a lower-score regime from the start.

## Why legacy works and unified does not

The cleanest answer is:

Legacy is not mainly winning because TD3 learned a better Markov policy. It is winning because it protects the run with a long teacher-only warm start and continues to let the stronger LS teacher dominate execution.

### Exact codewise differences that matter

#### 1. Warm-start length is not the same

Legacy builds its config from the matrix episode defaults, which still carry:

- `warm_start = 10` in `systems/polymer/notebook_params.py`
- then the legacy notebook only overrides `n_tests`, not `warm_start`

Unified Markov defaults explicitly set:

- `warm_start = 0` in `systems/polymer/notebook_params.py`

This is the biggest behavioral difference in the latest pair.

Observed consequence:

- legacy first TD3 action source appears at step `8005`
- unified first TD3 action source appears at step `9`

So unified begins live TD3 almost immediately, while legacy lets LS teacher execution populate the history and replay stream for 10 full episodes first.

#### 2. Replay data are therefore not the same

Both runners store the executed action in replay, not just the requested action. That means the early replay distribution depends heavily on what was actually executed.

Because legacy executes warm-start LS for about `19.84%` of the whole run, its replay buffer is seeded by teacher-quality trajectories before TD3 is asked to carry the loop.

Unified has no such teacher-only block. It starts mixing TD3 and LS fallback immediately, so its replay and closed-loop state distribution are different from step 9 onward.

#### 3. Legacy remains teacher-dominated even late

Last 5000 steps:

- legacy TD3 fraction: `0.2066`
- legacy LS fallback fraction: `0.7158`
- legacy nominal fallback fraction: `0.0776`

Unified last 5000 steps:

- unified TD3 fraction: `0.3826`
- unified LS fallback fraction: `0.5840`
- unified nominal fallback fraction: `0.0334`

So the legacy run that "works" is still mostly being carried by LS, not by a clearly superior TD3 policy.

That is the most important interpretive correction in this comparison.

#### 4. Unified’s LS teacher is also weaker

This is the part that proves the issue is not only TD3.

Source-conditioned executed prediction scores:

| Executed source | Legacy executed score | Unified executed score |
| --- | ---: | ---: |
| TD3 accepted | 0.03346 | 0.02353 |
| LS fallback | 0.03323 | 0.02589 |
| Nominal fallback | 0.00000 | 0.00000 |

The unified LS fallback itself is worse than legacy LS fallback. That means the problem is not "same teacher, worse actor." The entire closed-loop data regime feeding the teacher fit is weaker.

### What is not the root cause anymore

The latest pair does **not** support blaming:

- reward mismatch
- different Markov basis
- different `z` bound
- different prediction window
- different observer alignment
- different nominal lifted reference path

Those are now aligned enough that the remaining gap is dominated by execution schedule and trajectory distribution.

## Bugs, inconsistencies, or risks found

1. The legacy notebook display cells still contain stale printed values from an older run (`n_tests: 200`, `warm_start: 10`) even though the notebook source now overrides `n_tests` to 50. The saved run artifacts are more trustworthy than those stale rendered outputs.

2. The latest legacy run still reflects the older warm-start regime. So saying "legacy and unified are now the same" is still not correct at runtime, even though the core Markov math is now mostly aligned.

3. If the goal is a fair apples-to-apples TD3 comparison, comparing a `warm_start=10` legacy run to a `warm_start=0` unified run is not fair. It is comparing two different release protocols.

## Figure/report updates made

- Created `report/figures/polymer_markov_legacy_vs_unified_20260510/legacy_vs_unified_metric_grid.png`
- Created `report/figures/polymer_markov_legacy_vs_unified_20260510/legacy_vs_unified_prediction_diagnostics.png`
- Created `report/figures/polymer_markov_legacy_vs_unified_20260510/comparison_summary.json`
- Created this report:
  `report/polymer_markov_legacy_vs_unified_where_unified_becomes_nominal_2026_05_10.md`

## Literature connections

No external literature was needed for this comparison. The key issue was implementation and closed-loop data distribution, not a theoretical gap in the Markov correction idea itself.

## Recommended next experiment

Run one controlled A/B test:

- keep reward, basis, `z_bound`, prediction window, gain-drift guard, cost guard, observer alignment, and nominal lifted solver identical
- change only `warm_start`
- compare `warm_start = 10` versus `warm_start = 0` in the unified runner

Expected result:

- if warm start is the main cause, unified LS score and executed prediction score should move much closer to legacy when `warm_start = 10`
- if the gap remains large, then the next suspect is a more subtle trajectory or state-update mismatch

Metrics to watch:

- LS prediction score
- executed prediction score
- TD3 accepted fraction
- LS fallback fraction
- nominal fallback fraction
- episode-average reward

## Remaining uncertainty

The one remaining uncertainty is that the latest unified run directory did not save a lightweight summary JSON like the legacy run did, so this comparison relied on the full stage-diagnostics CSV rather than a richer bundle parse. That is enough for the six requested signals and the action-source analysis, but not enough to claim more detailed reward-component decomposition yet.

## Files changed

- `report/scripts/analyze_polymer_markov_latest_legacy_vs_unified.py`
- `report/polymer_markov_legacy_vs_unified_where_unified_becomes_nominal_2026_05_10.md`

## How to verify the changes

1. Re-run:
   `C:\Users\HAMEDI\miniconda3\python.exe report/scripts/analyze_polymer_markov_latest_legacy_vs_unified.py`
2. Confirm the figure files are regenerated under:
   `report/figures/polymer_markov_legacy_vs_unified_20260510/`
3. Open this report and cross-check the cited metrics against:
   `report/figures/polymer_markov_legacy_vs_unified_20260510/comparison_summary.json`
