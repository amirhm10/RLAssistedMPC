"""Independent final-table identities and preservation checks; no simulator."""
from collections import Counter
import subprocess
import analyze as a

np=a.np
rows=a.csv_rows(a.DATA/'gate_decisions.csv')
assert len(rows)==3840
assert Counter(r['plant'] for r in rows)==dict(polymer=3200,column=640)
combined=[r for r in rows if (r['plant'],r['category'])==('column','combined')]
assert len(combined)==320
assert len({(r['channel'],r['window'],r['step']) for r in combined})==320
assert len(Counter((r['channel'],r['window']) for r in combined))==16
assert all(n==20 for n in Counter((r['channel'],r['window']) for r in combined).values())
for summary in a.csv_rows(a.DATA/'combined_final_gate_summary.csv'):
    group=[r for r in combined if r['channel']==summary['channel']]
    delta=np.array([float(r['delta_return']) for r in group])
    take=np.array([r['take_policy']=='True' for r in group])
    gain=np.where(take,delta,0)
    assert len(group)==int(summary['pairs'])==80
    assert take.sum()==int(summary['accepted'])
    assert np.sum(take&(delta<0))==int(summary['harmful_accepted'])
    np.testing.assert_allclose(float(summary['mean_selected_gain_vs_supervisor']),gain.mean())
    # Best-candidate regret minus supervisor regret is exactly minus selected gain.
    np.testing.assert_allclose(float(summary['critic_regret'])-float(summary['always_supervisor_regret']),-gain.mean())
assert sum(r['take_policy']=='True' for r in combined)==200
assert sum(r['harmful_acceptance']=='True' for r in combined)==100
assert sum(r['material_harmful_acceptance']=='True' for r in combined)==96

metrics=a.csv_rows(a.DATA/'threshold_handoff_metrics.csv')
e2=a.csv_rows(a.DATA/'e2_metrics.csv')
for row in metrics:
    if row['window'] not in ('release_14_20','postrelease_14_200','evaluation_200'):continue
    method='offsetfree' if (row['plant'],row['method'])==('polymer','nominal') else row['method']
    old=next(r for r in e2 if (r['plant'],r['method'],r['window'])==(row['plant'],method,row['window']))
    for new,previous in (('rmse1','rmse_output1'),('rmse2','rmse_output2'),('mean_reward','mean_reward'),('normalized_input_activity','normalized_input_movement')):
        np.testing.assert_allclose(float(row[new]),float(old[previous]),rtol=1e-10,atol=1e-9)
    np.testing.assert_allclose(float(row['outside_either_tracking_band_fraction']),1-float(old['joint_band_fraction']))
for row in a.csv_rows(a.DATA/'threshold_pairwise.csv'):
    sg=next(r for r in metrics if (r['plant'],r['method'],r['window'])==(row['plant'],'sg',row['window']))
    th=next(r for r in metrics if (r['plant'],r['method'],r['window'])==(row['plant'],'threshold',row['window']))
    for field in ('rmse1','rmse2','normalized_input_activity'):
        np.testing.assert_allclose(float(row[field+'_reduction_percent']),100*(1-float(sg[field])/float(th[field])))
recovery=a.csv_rows(a.DATA/'threshold_recovery_summary.csv')
assert len(recovery)==8
expected={('polymer','sg','reward'):1,('polymer','sg','both_output_rmse'):1,
          ('polymer','threshold','reward'):25,('polymer','threshold','both_output_rmse'):25,
          ('column','sg','reward'):5,('column','sg','both_output_rmse'):37,
          ('column','threshold','reward'):37,('column','threshold','both_output_rmse'):22}
for r in recovery:assert int(r['episodes_after_release'])==expected[r['plant'],r['method'],r['criterion']]

states=a.csv_rows(a.DATA/'combined_steady_decisions.csv')
assert len(states)==160
for r in a.csv_rows(a.DATA/'combined_steady_groups.csv'):
    group=[s for s in states if (s['channel'],s['quasi_steady'])==(r['channel'],r['quasi_steady'])]
    different=[s for s in group if s['different_proposal']=='True']
    assert len(group)==int(r['pairs']) and len(different)==int(r['different_proposals'])
    assert sum(s['return_near_tie']=='True' for s in different)==int(r['near_tied_returns'])
    assert sum(s['material_harmful_accepted']=='True' for s in group)==int(r['material_harmful_accepted'])
extended=a.csv_rows(a.DATA/'combined_long_horizon_sensitivity.csv')
assert len(extended)==32 and sum(r['sign_reversal']=='True' for r in extended)==1
current=a.read(a.DATA/'combined_final_training_curve.json')
assert len(current['reward'])==len(current['rmse1'])==len(current['rmse2'])==300
assert len(a.read(a.DATA/'repeated_provenance.json'))==102
assert len(a.read(a.DATA/'e2_provenance.json'))==12
assert len(a.csv_rows(a.DATA/'e3_verified_metrics.csv'))==576

original=a.read(a.DATA/'original_assessment_frozen.json')
for row in original['unchanged_files']:
    assert a.digest(row['path'])==row['sha256'],row['path']
paper=a.read(a.DATA/'paper_readonly_snapshot.json')
for row in paper['files']:assert a.digest(row['path'])==row['sha256'],row['path']
ready=a.read(a.DATA/'reviewer_readiness_summary.json')
assert ready['total_comments']==sum(ready['status_counts'].values())==24
assert ready['status_counts']['Ready to prepare']==16
assert ready['responses_drafted']==ready['manuscript_edits']==0
assert not ready['all_comments_resolved'] and not ready['submission_ready']
job=a.read(a.ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1/budget20_job.json')
scientific=0
for path,checksum in job['sources'].items():
    source=a.ROOT/path
    if source.is_file():assert a.digest(source)==checksum,path;scientific+=1
assert scientific==139
campaign=a.read(a.ROOT/'change-reports/hourly_run_checks_20261024/reconciliation_inputs.json')['metadata']['horizon']['source_sha256']
assert len(campaign)==143
for path,checksum in campaign.items():assert a.digest(a.ROOT/path)==checksum,path
assert len(list(a.FIG.glob('*.png')))==len(list(a.FIG.glob('*.svg')))==24
a.save_json('final_verification.json',dict(passed=True,e1_pairs=3840,combined_final_pairs=320,
    four_balanced_channels_and_windows=True,threshold_metric_rows_checked=len(metrics),
    threshold_recovery_rows_checked=8,combined_steady_states_checked=160,
    extended800_step_cases_checked=32,earlier_assessment_files_unchanged=len(original['unchanged_files']),
    accessible_paper_files_unchanged=len(paper['files']),unreadable_ancillary_pdfs_left_untouched=len(paper['unavailable_files']),
    job_source_files_unchanged=scientific,campaign_seals_unchanged=len(campaign),manuscript_edits=0,reviewer_comments_answered=0,
    total_figures=24,readiness_comments_checked=24,
    analysis_script_hashes={p.name:a.digest(p) for p in a.HERE.glob('*.py')}))
print('Final verification passed:3840 paired decisions,24 figures,24-comment readiness and preservation guards')
