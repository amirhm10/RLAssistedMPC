# Practical handoff, recovery and steady-state assessment

October 24, 2026. Internal analysis of saved simulation data. The column E1 snapshot is frozen at 22:36 UTC: 533/640 pairs, including combined 213/320. The live combined run continues independently.

**The strongest supported argument is reduced disruption and faster recovery when introducing learned assistance, particularly for residual and combined control.** The results do not show uniform superiority across every channel, operating phase, comparator or metric. No manuscript or reviewer response has been edited.

## Evidence and validation

The analysis includes all 100 completed Scenario 2 training runs: two plants, five assistance families, SG/plain, five prescribed seeds (7, 11, 23, 37, 53), and 300 episodes per run. There are two matched saved nominal-MPC references, 12 E2 controller/reference trajectories including the supplemental recovered polymer threshold, 3,200 completed polymer E1 pairs and the frozen 533-pair column prefix. E3 is independently recomputed from saved forecast probes and reproduces all 576 existing summary rows.

All primary trajectory arrays are finite and aligned after the control action. All stored RL rewards agree with the same fixed plant-specific reward; the vectorized calculation is also checked against the unchanged scalar reward implementation. References match. Raw files have SHA-256 provenance. All 3,733 E1 paired returns and output RMSEs are recomputed from raw NPZ signals. The state analysis joins 960 standalone horizon/weight decisions to their original training histories; every captured initial tracking error matches exactly.

Some original exports omit reward-parameter metadata. An explicit saved parameter record is accepted only after the entire original reward array agrees with that score. Native numerical runtimes differ across recoveries; identical seeds do not imply bitwise identical training. Optional unset diagnostics and missing solver-failure counters are not converted to zeros.

No simulator is opened or reset. Original scientific sources, trajectories, results, checkpoints, archives and the unrelated dirty polymer runner remain untouched. Derived analysis runs use the existing private numerical runtime, one numerical thread and low process priority.

## Handoff: combined controller, episodes 14-20

These are means of the five seed-level results. Reward is higher-is-better. Output RMSE and input activity are lower-is-better. Input activity is mean absolute input movement normalized by each input's identification range.

| Plant | Mode | Reward | Output 1 RMSE | Output 2 RMSE | Input activity |
| --- | --- | ---: | ---: | ---: | ---: |
| Polymer | Nominal MPC | -4.361 | 0.19031 | 0.55628 | 0.03558 |
| Polymer | Ungated RL | -12.726 | 0.30869 | 1.13013 | 0.55146 |
| Polymer | Critic SG | -4.358 | 0.18645 | 0.59529 | 0.13941 |
| Column | Nominal MPC | 10.149 | 0.002755 | 0.41007 | 0.00746 |
| Column | Ungated RL | -178.806 | 0.009585 | 1.48059 | 0.06019 |
| Column | Critic SG | -15.150 | 0.006316 | 0.56468 | 0.01582 |

Relative to ungated combined RL, SG reduces the two handoff output RMSEs by about **39.6% and 47.3% for polymer**, and **34.1% and 61.9% for the column**. Normalized input activity falls about **74.7% and 73.7%**. Percentages are ratios of the displayed seed means, not significance probabilities.

Both plants have 5/5 positive paired combined handoff reward gains. Standalone residual also wins 5/5 in both plants. The gains are much smaller and less consistent for horizon and weights.

| Plant | Family | Positive paired handoff reward gains | Mean SG minus plain reward |
| --- | --- | ---: | ---: |
| Polymer | Horizon | 2/5 | -0.00448 |
| Polymer | Weights | 2/5 | -0.10110 |
| Polymer | Matrix | 4/5 | 0.18560 |
| Polymer | Residual | 5/5 | 8.50136 |
| Polymer | Combined | 5/5 | 8.36754 |
| Column | Horizon | 1/5 | -0.43915 |
| Column | Weights | 3/5 | 0.33744 |
| Column | Matrix | 5/5 | 59.53856 |
| Column | Residual | 5/5 | 120.34568 |
| Column | Combined | 5/5 | 163.65577 |

Retaining nominal MPC still performs better during column release; SG does not eliminate every release dip. The practical benefit is reducing the damage of introducing RL while retaining subsequent adaptation. It does not mean every baseline metric is preserved.

![Handoff rewards](figures/01_handoff_rewards.png)

## Recovery speed

Two complementary criteria are reported: episode mean reward at least the matched nominal-MPC episode, and both output RMSEs no worse than the matched nominal-MPC episode. Recovery requires three consecutive qualifying episodes. The reported time is the first qualifying episode minus release episode 14; two later episodes confirm it. This measures benchmark-relative performance recovery, not a physical settling-time constant.

The search ends at episode 200 before the next operating change. Non-recovery is censored and is never dropped or replaced by an invented recovery time. All seeds contribute to the median; a censored median is reported as not reached. One- and five-episode sensitivities, operating-change recovery and the accumulated positive reward deficit over the entire fixed window are retained in the data. Later relapses remain visible in the unsmoothed curves and deficit measure.

| Study | Plant | Recovery criterion | Critic SG | Comparator |
| --- | --- | --- | ---: | ---: |
| Five-seed Scenario 2 | Polymer | Reward | median 3 episodes | ungated median 15 |
| Five-seed Scenario 2 | Polymer | Both output RMSEs | median 5 | ungated median 19 |
| Five-seed Scenario 2 | Column | Reward | median 6; 5/5 recover | ungated median not reached; 2/5 recover by 200 |
| Five-seed Scenario 2 | Column | Both output RMSEs | median 24; 5/5 recover | ungated median 78; 4/5 recover |
| Single-seed E2 | Polymer | Reward | 1 | threshold 25 |
| Single-seed E2 | Polymer | Both output RMSEs | 1 | threshold 25 |
| Single-seed E2 | Column | Reward | 5 | threshold 37 |
| Single-seed E2 | Column | Both output RMSEs | 37 | threshold 22 |

The replicated comparison strongly favors SG recovery against ungated RL. The threshold comparison is more nuanced: SG recovers reward faster in both plants, but the single column threshold realization reaches the both-output MPC-RMSE criterion earlier. That counterexample is retained. A claim of faster recovery must name its metric.

![Reward recovery](figures/12_recovery_reward.png)

![Physical-error recovery](figures/13_recovery_both_output_rmse.png)

## Is the simple threshold better?

The raw recheck confirms that SG is better over the handoff window in both plants and over the full post-release window. Threshold uses the same four-channel learner structure with a tracking-error threshold acceptance rule; it is not an MPC-only controller.

| Plant | Window | Critic SG reward | Threshold reward |
| --- | --- | ---: | ---: |
| Polymer | Release 14-20 | -4.134 | -10.983 |
| Polymer | Postrelease 14-200 | -1.973 | -4.096 |
| Polymer | Evaluation 200 | -1.680 | -3.466 |
| Column | Release 14-20 | -28.217 | -96.982 |
| Column | Postrelease 14-200 | 25.017 | 11.754 |
| Column | Evaluation 200 | 35.394 | 36.248 |

Column pre-release SG/threshold histories match exactly. Polymer has small pre-release differences: maximum output difference 0.0005544 and input difference 0.06008 in saved physical units, with identical references. These are separately trained end-to-end comparisons; later replay and policies can differ after their acceptance decisions diverge. E2 has one realization per method, not a five-seed threshold replication.

The initial concern came from the column final episode alone. SG has lower final RMSE for both outputs, but threshold has slightly higher reward. Independent reward decomposition explains this: threshold's near-target bonus is 39.5881 versus SG's 38.4068, outweighing larger aggregate penalties. The reward is nonlinear, so the ordering of its mean need not match output RMSE ordering. No label swap, scoring mismatch or smoothing error was found.

The recovered polymer threshold is also included as supplementary same-seed data and remains worse than SG at handoff. It is not counted as an independent additional seed.

## Horizon and weights: steady-state and value-equivalence hypothesis

The state classification uses only the ten steps BEFORE the decision: constant reference, output increments no more than 5% of a current tracking band per step, and input increments no more than 0.5% of the input identification range per step. Being near the target is recorded separately from being dynamically quiet. Stricter and looser thresholds are retained.

The input comparisons below condition on DIFFERENT proposed configurations, excluding the trivial case where policy and supervisor propose the same action. Nearly unchanged means at most 0.01% of the input identification range.

| Plant | Channel | Nearly unchanged first input, quasi steady | Changing/transition | Median abs return difference, steady | Changing/transition |
| --- | --- | ---: | ---: | ---: | ---: |
| Polymer | Horizon | 55/167 | 9/160 | 0.06250 | 0.39874 |
| Polymer | Weights | 48/208 | 1/192 | 0.01178 | 0.20226 |
| Column | Horizon | 15/23 | 3/42 | 4.76066 | 28.18497 |
| Column | Weights | 19/29 | 3/51 | 0.36200 | 4.43259 |

The data support smaller consequences and more first-input equivalence near steady operation. In the column, about 65% of differing horizon/weight proposals leave the first input nearly unchanged in quasi-steady conditions, versus about 6-7% during changing conditions.

The reward-equivalence and supervisor-retention parts of the hypothesis must be distinguished. The next table uses a 200-step near-tie sensitivity of 0.5% of max(1, absolute supervisor return), not an assertion of exactly identical values.

| Plant | Channel | Near-tied returns among differing quasi-steady proposals | Supervisor selected in those ties |
| --- | --- | ---: | ---: |
| Polymer | Horizon | 156/167 | 27/156 |
| Polymer | Weights | 197/208 | 169/197 |
| Column | Horizon | 10/23 | 0/10 |
| Column | Weights | 22/29 | 4/22 |

Value ambiguity is supported, particularly in polymer. Supervisor retention in ambiguous cases is strong for polymer weights but not universal; column horizons use the RL proposal in all ten of their quasi-steady near-ties. The gate compares estimated advantages and margins; it does not directly detect steady state or observe the future realized return.

Restricting the same diagnostic to quasi-steady decisions already inside both tracking bands gives polymer horizon 81/88 near-tied returns and weights 158/164; the supervisor is retained in 19/81 and 132/158 of those ties. Both column subsets are unchanged. The conclusion therefore also holds near the target, rather than depending only on slow operation away from it.

Equivalent outcomes can make fine value ranking difficult and can motivate retaining a default action when estimated advantage is insufficient. **The data do not directly establish that this ambiguity destabilized training, or that the supervisory layer consistently takes over in every steady-state case.** Those remain interpretations to test, rather than conclusions upgraded from equal rewards.

Some negative sign counts are practically small. They do not explain all unfavorable acceptances. A state that is steady now can also face a reference switch inside the 200-step rollout; short-term input equivalence and long-term return equivalence are therefore shown separately. The analysis does not claim that the optimization objective, Hessian, optimizer or complete future input sequence is mathematically identical.

![Steady-state effects](figures/15_steady_state_effects.png)

![Reward-equivalent settings and supervisor retention](figures/17_reward_equivalence_and_supervisor.png)

## Other findings and limits

- E1 includes genuinely lower-return acceptances and beneficial rejections, including material losses. Its alternatives change the FIRST action and then use the same frozen gated continuation; they are not independently trained always-policy or always-supervisor controllers. One training seed and dependent, prospectively selected states do not support seed-level confidence intervals or plant-wide failure-frequency estimates.
- E3 executed predictors improve lead-eight RMSE in 14/24 groups per plant, worsen it in 8/24 and tie in 2/24. Polymer ratios to nominal are approximately 0.984-1.009; column ratios span 0.434-1.295. Known future inputs are supplied. Better control does not imply uniformly better prediction.
- Nominal checkpoint forecasts exist only in mature/adapted windows. Missing release/change cells are explicitly shown as not sampled.
- Column combined SG improves both final output RMSEs over ungated RL in every paired seed. Polymer ungated combined RL has slightly better mean final reward/output RMSEs than SG, while SG uses less input movement. The strong polymer argument is handoff mitigation and recovery, not uniform final-performance dominance.
- The saved handoff effects are smaller and less consistent for horizon/weights. The standalone paired-decision consequences are also generally smaller for those channels than for residual/dynamic matrix. These conditional diagnostics do not identify causal individual-agent contributions without ablations.
- SG can use more input movement than nominal MPC. Input activity and physical-output trade-offs remain visible in every relevant comparison.

## Readiness and remaining run

Combined column E1 is the last run in the CURRENT execution queue. Four standalone E1 configurations, both 50-run repetition campaigns, selected E2 comparisons and E3 stages have finished. Frozen-policy unseen-reference evaluations and separately scoped E4 interaction/E5 sensitivity work have not been supplied by this pool; completing this queue does not automatically close every reviewer dependency.

Amir has selected a practical handoff/recovery emphasis and deferred formal stability/convergence analysis to future work. No reviewer comment is marked resolved by this assessment. No response has been drafted and no manuscript/PDF in the paper repository has been edited. All 46 frozen original files remain intact.

The case-15/17 parallel adapter already executes the two comparison branches concurrently. Extra cases cannot accelerate that synchronous interface without a new scheduler and checkpoint/recovery validation. No further migration or live scientific change is attempted here.

## Files and reproduction

This folder contains the independent analysis scripts, all per-seed/window/decision tables, metric definitions, SHA-256 provenance and the complete figure set. The offline gallery and PDF booklet show every figure; SVG counterparts support vector export. Raw research results and native recovery data remain in their original verified locations and retained lossless archives.

Files inspected are enumerated by the provenance tables and include original reward/alignment/E1/E3 source, current accepted recovery audits, and the paper roadmap/backlog/response placeholders. Files changed are confined to this new assessment folder and concise operational notes. Original scientific runs, results, sources, queue plans and unrelated dirty user files remain intact.

Reproduction order: analyze.py controls; analyze.py gates; verify_comparators.py; plot_results.py; reward_check.py; recovery_analysis.py; steady_state_analysis.py; verify_analysis.py. Use the documented base interpreter with -B. E3 verification writes fresh derived destinations; use verify_comparators.py --e2-only to refresh E2 without repeating the closed E3 derivation.
