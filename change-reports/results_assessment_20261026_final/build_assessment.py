"""Assemble the final evidence assessment; never write manuscript responses."""
import re
import analyze as a


def readable(text):
    # Preserve literal paths and code while spacing prose copied from run notes.
    chunks=re.split(r'(`[^`]*`)',text)
    for i in range(0,len(chunks),2):
        for word in ('October','episodes','episode','seed','all','All','combined','column','provides','among','the'):
            chunks[i]=re.sub(r'\b'+word+r'(?=\d)',word+' ',chunks[i])
        chunks[i]=re.sub(r'(?<=[a-z])(?=\d)',' ',chunks[i])
        chunks[i]=chunks[i].replace(',2026',', 2026')
    return ''.join(chunks)


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


readiness=[
('R1.1','Ready to prepare','Final comparative abstract numbers and abbreviation cleanup','Final values verified; no abstract written.'),
('R1.2','Ready to prepare','Contributions and practical scope','Keep claims within the tested linear-MPC assistance setting.'),
('R1.3','Ready to prepare','Positioning against MPC tuning','E2 evidence available; verify literature attribution during writing.'),
('R1.4','Open analytical dependency','Requested substantive analytical result','Formal stability/convergence/unbiasedness is not established. Author prefers future work; this does not itself resolve the request.'),
('R1.5','Literature check needed','Two named RL papers','Full-text comparison and citation check remain separate writing work.'),
('R1.6','Ready to prepare','Approved simple/BO/LS/ARX numerical comparisons','Adapted BO and known ARX limitations must remain explicit; no extra E7 reproduction campaign is required by the accepted scope.'),
('R1.7','Ready to prepare','Short conclusion and measured future-work limits','Use final evidence; avoid general superiority.'),
('R2.1','Ready to prepare','Completed critic-versus-realized-decision audit','Evidence is mixed/unfavorable in important cells; supports honest diagnostic discussion, not dependable critic ranking or calibrated confidence.'),
('R2.2','Partial evidence','Critic semantics, scalar reward and interactions','Output-wise metrics and reward decomposition exist; E4 joint-action evidence is not supplied by this campaign.'),
('R2.3','Partial evidence','Paired seeds and unseen multistep references','Five seeds per family/mode/plant are complete; frozen-policy unseen-reference evaluation is not supplied.'),
('R2.4','Ready to prepare','Linear MPC scope and guarantee terminology','Describe empirical mitigation; no formal safety, recursive-feasibility or stability guarantee.'),
('R2.5','Ready to prepare','Four separate gates using a common principle','Preserve channel-specific critics, supervisors, margins and buffers.'),
('R2.6','Ready to prepare','Abrupt release protocol','State release episode14 and distinguish learning from action availability.'),
('R2.7','Ready to prepare','Readable figures','New PNG/SVG figures and booklet are ready for selection; manuscript figures are not replaced.'),
('R3.1','Ready to prepare','Nomenclature and first use','Editorial pass remains to be performed.'),
('R3.2','Ready to prepare','Predictor accuracy and identification detail','576 E3 rows reproduced; known-future-input, window and single-seed limitations remain.'),
('R3.3','Literature check needed','Nonlinear surrogate trade-offs','Verified literature discussion remains; no nonlinear benchmark claim is supported.'),
('R3.4','Ready to prepare','Safe-RL keyword and guarantee language','Observed handoff mitigation is distinct from certified constraint satisfaction.'),
('R3.5','Ready to prepare','Supervisor origins in the diagram','Implementation can support the diagram; no conceptual/manuscript figure has been edited.'),
('R3.6','Partial evidence','Sequential decisions and joint limitations','Conditional first-action E1 data do not establish joint causal effects; E4 remains separate.'),
('R3.7','Additional empirical evidence needed','Adverse calibration, model order and disturbances','No verified completed E5 stress campaign was identified in the audited pool.'),
('R3.8','Ready to prepare','Architecture, activations and smaller networks','Follow accepted E6 justification/limitations route; no new architecture training is authorized.'),
('R3.9','Additional empirical evidence needed','Margin/bound/reward-scale sensitivity','E5 controller-parameter sweeps remain separate. Analysis-window and dwell sensitivities do not replace them.'),
('R3.10','Ready to prepare','NMPC omission and modeling premise','Follow accepted E7 scope/limitations route; no numerical superiority over NMPC is established.')]


def main():
    a.save_csv('reviewer_readiness.csv',[dict(comment=x,status=y,evidence=z,remaining=w) for x,y,z,w in readiness])
    counts={s:sum(r[1]==s for r in readiness) for s in dict.fromkeys(r[1] for r in readiness)}
    source=a.ROOT.parent/'Rl_assisted_paper/RL_assisted_paper/manuscript_revised/REVISION_ROADMAP.md'
    scope=a.ROOT.parent/'Rl_assisted_paper/RL_assisted_paper/manuscript_revised/REVIEW_EXPERIMENT_PLAN_2026_10_12.md'
    ready='# Reviewer readiness check - final campaign evidence\n\n'
    ready+='**Yes, evidence-based revision work can begin. Completing this campaign does not make every reviewer request complete or the submission ready. No reviewer response is drafted, no comment is marked resolved, and no manuscript is edited.**\n\n'
    ready+='The final campaign provides 100 replicated Scenario2 runs, all3840 E1 pairs,12 E2 comparator/reference trajectories and576 independently reproduced E3 rows. The threshold comparison has one realization per method and plant; the five-seed evidence compares SG with ungated RL, not with a five-seed threshold arm. The completed E1 main run is an instrumented seed7 trajectory, not a sixth independent replicate.\n\n'
    ready+=table(['Comment','Readiness','Evidence/work available','Remaining limit'],readiness)+'\n\n'
    ready+='## Decision and order of work\n\n'
    ready+='There are16 comments ready for evidence-based preparation,2 needing targeted literature checks, and6 with partial or missing analytical/empirical dependencies. These are readiness labels, not resolved-comment counts. Begin later with scope/terminology and release definitions, then the completed E1/E2/E3 evidence and readable figures. Preserve the adverse gate-ranking cells and the threshold counterexamples. Keep unseen-reference evaluation, E4 joint consequences, and E5 stress/supervisory sensitivity as distinct evidence decisions. The author has chosen a practical emphasis and formal theory as future work; R1.4 must still be handled explicitly rather than relabeled resolved by that choice. E6/E7 retain their accepted justification-and-limitations scope, with no new architecture or NMPC runs. No further run is started by this assessment.\n\n'
    ready+='The most defensible practical result is reduced disruption and faster benchmark-relative recovery against ungated assistance, and better seven-episode handoff behavior and reward recovery than the tested threshold realization. Reliable critic ordering, universal superiority, formal safety and physical plant deployment have not been established.\n\n'
    ready+='## Sources and preservation\n\n'
    ready+=f'Comment mapping: `{source}`. Accepted E5/E6/E7 scope: `{scope}` and the paper experiment backlog. These paper records still contain historical running states; current completion is established by the October26 code-repository receipts, not those older status paragraphs. No paper-memory or response file is silently updated.\n\n'
    ready+='Read-only guards cover47 accessible manuscript/PDF/readiness files. Two existing ancillary PDFs (cover letter and highlights) could not be read locally and remain untouched; their typeset appearance is not audited here. This does not block numerical analysis, but final submission validation is a later task. The October24 analysis is preserved byte for byte.\n'
    (a.HERE/'REVIEWER_READINESS.md').write_text(readable(ready),encoding='utf-8')
    a.save_json('reviewer_readiness_summary.json',dict(total_comments=24,status_counts=counts,
        can_begin_revision_preparation=True,all_comments_resolved=False,submission_ready=False,
        responses_drafted=0,manuscript_edits=0,source=str(source),source_sha256=a.digest(source),scope_source=str(scope),scope_source_sha256=a.digest(scope)))

    old=(a.ROOT/'change-reports/results_assessment_20261024/ASSESSMENT.md').read_text(encoding='utf-8')
    start=old.index('**The strongest supported argument')
    text='# Final practical handoff, recovery and critic-decision assessment\n\n'
    text+='October26,2026. All scheduled campaign runs are complete. This new version includes all640/640 column E1 pairs, combined320/320, the complete accepted300-cycle combined training trajectory, and the original3200 polymer pairs. It preserves the October24 assessment rather than overwriting it.\n\n'
    text+=old[start:]
    text=text.replace('the frozen 533-pair column prefix','all640 completed column pairs').replace('All 3,733 E1','All3,840 E1')
    text=text.replace('The analysis does not show uniform','The analysis does not show uniform')
    pair=a.csv_rows(a.DATA/'threshold_pairwise.csv')
    recovery=a.csv_rows(a.DATA/'threshold_recovery_summary.csv')
    summary='## Direct threshold comparison: handoff and recovery\n\n'
    summary+='E2 is an end-to-end training comparison at seed7. The threshold accepts an eligible learned proposal when the maximum output tracking error divided by its tracking band is no larger than its warm-up-calibrated cutoff. Its cutoff is max(1,1.25 times the95th percentile warm-up normalized error). Critic scores remain shadow diagnostics in that arm and do not determine acceptance. Learners, bounds, readiness, release and replay machinery are retained. Later policies/replay can diverge after the choices differ; this is not a selector-only fixed-policy experiment.\n\n'
    summary+=table(['Plant','Handoff output1 RMSE reduction','Output2 RMSE reduction','Input activity reduction','Fewer band excursions (percentage points)'],[[r['plant'].title(),f"{float(r['rmse1_reduction_percent']):.1f}%",f"{float(r['rmse2_reduction_percent']):.1f}%",f"{float(r['normalized_input_activity_reduction_percent']):.1f}%",f"{float(r['band_excursion_reduction_percentage_points']):.1f}"] for r in pair if r['window']=='release_14_20'])+'\n\n'
    summary+='Primary handoff window: episodes14-20. Lower input activity and fewer excursions support reduced empirical handoff disruption; tracking bands are performance targets, not certified safe operating limits. Nominal MPC is still better on several handoff metrics, and SG uses more input activity than nominal. No formal safety, feasibility, operating-envelope or deployment guarantee follows.\n\n'
    summary+=table(['Plant','Recovery criterion','Critic gate','Simple threshold'],[[plant.title(),criterion.replace('_',' '),next(r['episodes_after_release'] for r in recovery if (r['plant'],r['method'],r['criterion'])==(plant,'sg',criterion)),next(r['episodes_after_release'] for r in recovery if (r['plant'],r['method'],r['criterion'])==(plant,'threshold',criterion))] for plant in ('polymer','column') for criterion in ('reward','both_output_rmse')])+'\n\n'
    summary+='Recovery numbers are episodes after release14, using the first of three consecutive benchmark-qualifying episodes. They measure training-performance recovery against matched nominal MPC, not physical settling times. Critic gating recovers reward earlier in both plants; the column threshold recovers both output RMSEs earlier (22 versus37). The accumulated positive reward deficit over episodes14-200 is2.57 versus56.07 for polymer and321.60 versus1564.02 for the column (critic versus threshold; sums of episode-mean deficits). The raw curves and one-/five-episode sensitivities remain available.\n\n'
    summary+='Window sensitivity also matters: over the shorter column episodes14-16 window, critic composition RMSE is64.6% higher than threshold, while temperature RMSE, reward and input activity are better. Over the specified14-20 and the14-25 windows both output RMSEs favor the critic. These are disclosed analysis sensitivities, not a reason to change the primary window. Threshold also has higher column episode200 reward despite worse RMSE, as independently explained by the near-target bonus. A universal handoff or recovery superiority claim is therefore unsupported.\n\n'
    summary+='![Direct threshold handoff](figures/18_threshold_handoff.png)\n\n![Direct threshold recovery](figures/19_threshold_recovery.png)\n\n'
    where=text.index('## Horizon and weights:')
    text=text[:where]+summary+text[where:]

    gates=a.csv_rows(a.DATA/'combined_final_gate_summary.csv')
    section='## What the complete combined decision audit actually shows\n\n'
    section+='All320 column combined decisions are available:80 per channel and20 per channel/window cell. Among200 policy acceptances,100 have lower200-step realized return than the matched supervisor alternative;96 are below minus1 reward unit under the retained magnitude sensitivity. There are57 beneficial rejections. Counts describe prospectively sampled, dependent states from one training seed, not population failure probabilities or independent experiments.\n\n'
    section+=table(['Channel','Lower-return accepts / accepts','Mean selected gain vs supervisor','Median selected gain','Critic regret','Supervisor-first-action regret'],[[r['channel'].title(),r['harmful_accepted']+'/'+r['accepted'],f"{float(r['mean_selected_gain_vs_supervisor']):.2f}",f"{float(r['median_selected_gain_vs_supervisor']):.2f}",f"{float(r['critic_regret']):.2f}",f"{float(r['always_supervisor_regret']):.2f}"] for r in gates])+'\n\n'
    section+='The mean selected first-action gain is negative for each channel when the four sampled phases are equally weighted. This is unfavorable evidence for consistent critic ranking at those states. At the adapted window, Markov, weights and residual have positive mean selected gains (20.17,7.52 and11.71), while horizon remains negative (-22.56). Both favorable and unfavorable cells are retained. Means can be influenced by large losses; medians and per-state returns are provided. Of32 extended800-step comparisons, one changes the sign of the primary200-step return difference.\n\n'
    section+='Only the candidate first action changes and the later controller is the same frozen gated continuation. The supervisor-first-action comparator is therefore neither a trained threshold controller nor an always-MPC trajectory. These diagnostics do not contradict a favorable end-to-end handoff comparison, but they prevent using that comparison as proof that the critic reliably identifies the better action. Neither finite-horizon returns nor twin-critic penalties establish calibrated uncertainty or a stability theorem.\n\n'
    section+='![Final combined decisions](figures/20_final_combined_decisions.png)\n\nThe accepted full main trajectory is shown separately; it is not an extra replicate in the100-run study. All four learners, full recovery state and original technical replay alternatives are retained. Native integration after the accepted cold restore is not claimed bitwise identical to the original failed suffix. The final accepted generation is selected by the pre-existing recovery protocol, not by favorable performance.\n\n![Final combined training](figures/21_final_combined_training.png)\n\n'
    where=text.index('## Other findings and limits')
    groups=a.csv_rows(a.DATA/'combined_steady_groups.csv')
    section+='### Steady state in the final combined run\n\n'
    section+='The same pre-decision classification and numerical sensitivities were applied to160 final combined horizon/weight states; captured errors match the saved main history. Different configurations more often produce nearly unchanged first inputs at quasi-steady states: horizon2/8 versus2/65 changing states; weights7/15 versus4/65. Near-tied200-step returns occur in3/8 quasi-steady horizon proposals and5/15 quasi-steady weight proposals. The supervisor is retained in0/3 and4/5 of those ties, respectively. Thus value ambiguity and conservative retention are supported in some subsets, especially weights, rather than as a universal rule.\n\n'
    section+='Five quasi-steady horizon states and three quasi-steady weight states still have accepted losses below minus1; all eight begin inside both tracking bands. Steady operation therefore does not explain away every unfavorable decision. A quiet present does not imply an unchanged future reference or an identical MPC solution. The data still do not directly establish that reward-equivalent choices caused training confusion or that the gate detects steady state explicitly.\n\n![Combined steady-state check](figures/22_combined_steady_state.png)\n\n'
    text=text[:where]+section+text[where:]
    first=text.index('## Readiness and remaining run');last=text.index('## Files and reproduction',first)
    text=text[:first]+'## Reviewer readiness\n\nAll scheduled runs in this campaign have completed; the hourly monitor is disabled after verified final preservation. We can begin evidence-based revision work, but cannot mark all reviewer requests resolved. The24-comment readiness table is in [REVIEWER_READINESS.md](REVIEWER_READINESS.md). Remaining distinct dependencies include frozen-policy unseen references, E4 joint-action consequences, E5 stress/supervisory sensitivity, literature checks and R1.4\'s analytical request. The author\'s practical emphasis and theory-as-future-work decision is preserved; it is not presented as a mathematical result. No response has been drafted or manuscript edited.\n\n'+text[last:]
    text=text.replace('Reproduction order: analyze.py controls; analyze.py gates; verify_comparators.py; plot_results.py; reward_check.py; recovery_analysis.py; steady_state_analysis.py; verify_analysis.py.','Reproduction order: analyze.py controls; analyze.py gates; verify_comparators.py; plot_results.py; reward_check.py; recovery_analysis.py; steady_state_analysis.py; final_comparison.py; build_assessment.py; verify_analysis.py; final_checks.py; package_figures.py.')
    text+='\nThe new package has24 PNG/SVG figure pairs, all3840 raw E1 comparisons and complete source/metric provenance. Inspected files are enumerated in the provenance tables, final completion receipts and read-only paper snapshot. Changed files are confined to this new versioned assessment. The85 prior assessment files, scientific sources/results/recovery data, manuscript/response files and unrelated dirty polymer runner are preserved.\n'
    marker='## Evidence and validation'
    text=text.replace(marker,'**Important qualification from the complete E1 audit:** 100 of200 column-combined policy acceptances have lower200-step return than their matched supervisor alternative. The observed handoff benefit does not establish reliable critic action ranking; the negative results are reported below.\n\n'+marker)
    (a.HERE/'ASSESSMENT.md').write_text(readable(text),encoding='utf-8')
    ledger=[
        dict(claim='Lower primary seven-episode handoff disruption than threshold',classification='empirical observation',evidence='threshold_pairwise.csv; E2 raw verified trajectories',limit='One realization per method/plant; shorter column composition window is a counterexample.'),
        dict(claim='Faster recovery than threshold',classification='metric-dependent empirical observation',evidence='threshold_recovery_summary.csv',limit='Reward favors critic in both plants; both-output RMSE favors threshold in the column.'),
        dict(claim='SG improves handoff and recovery versus ungated combined RL',classification='replicated empirical observation',evidence='paired_seed_differences.csv; recovery_times.csv',limit='Five paired training seeds; tested scenario, no held-out-reference generalization claim.'),
        dict(claim='Critic reliably ranks the better action',classification='unsupported generalization',evidence='combined_final_gate_summary.csv; gate_groups.csv',limit='Many lower-return acceptances and beneficial rejections; negative phase-weighted mean selected gain in each column combined channel.'),
        dict(claim='Steady operation can produce similar outcomes from different settings',classification='partly supported interpretation',evidence='steady_state_groups.csv; combined_steady_groups.csv',limit='Not all near-steady choices are equivalent; no direct evidence of training confusion.'),
        dict(claim='Formal closed-loop safety/stability',classification='not established',evidence='No theorem or certified operating constraints supplied',limit='Empirical tracking mitigation only; R1.4 remains an explicit dependency.'),
        dict(claim='All reviewer requirements complete',classification='not established',evidence='reviewer_readiness.csv',limit='Campaign complete; unseen references, E4/E5, analytical and literature work remain distinct.')]
    a.save_csv('claim_evidence_ledger.csv',ledger)
    print('Final assessment and24-comment readiness check written')


if __name__=='__main__':main()
