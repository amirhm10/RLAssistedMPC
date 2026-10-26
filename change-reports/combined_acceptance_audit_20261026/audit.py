"""Offline follow-up: severity and operating regime of final combined acceptances.

Reads completed scientific records only. Writes this new supplement only; the
final assessment, scientific sources, recovery state and manuscript stay frozen.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FINAL = ROOT / 'change-reports/results_assessment_20261026_final'
sys.path.insert(0, str(FINAL))
import analyze as a

np, plt = a.np, a.plt
CHANNELS = ('horizon', 'weights', 'markov', 'residual')
LABELS = ('Horizon', 'Weights', 'Markov', 'Residual')
DATA = HERE / 'data'
FIG = HERE / 'figures'
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

CONTRACT = dict(
    question='Are the 100 lower-return acceptances mainly steady-state near-ties, and how can end-to-end handoff/reward recovery still be better?',
    mode='focused-audit', scope='Accepted completed column combined E1, seed 7, all 320 pairs; previously verified E2 context.',
    allowed_actions='Read closed data; create a separate derived supplement. No simulation, training, manuscript or reviewer-response edits.',
    primary_metric='Policy minus supervisor discounted return over 200 controls, gamma from each record.',
    independent_unit='One instrumented training realization; sampled, phase-stratified decision states are not independent seeds.',
    legacy_absolute_sensitivity=1.0,
    relative_return_sensitivities=[.001, .005, .01, .05],
    relative_denominator='max(1, abs(supervisor discounted return)); cancellation-robust denominator sensitivity also reported.',
    steady_definition='Previous 10 controls: unchanged reference, max output increment <=5% tracking band, max input increment <=0.5% ID range.',
    near_target_definition='Both pre-decision errors inside their reference-relative performance bands.',
    negligible_first_move='Max branch first-input difference <=0.01% of its ID range.',
    small_physical_change='Absolute difference between branch normalized-error RMSEs <=1% of a tracking band on BOTH outputs.',
    joint_near_equivalence='Return gap <=0.5%, negligible first move, AND small physical RMSE change.',
    thresholds_status='Analysis sensitivities fixed before this recalculation; not validated safety bounds or a practical-equivalence test.',
    prior_package='Preserved byte-for-byte; previous material_harmful_acceptance label refers only to the absolute -1 sensitivity.',
)


def write_json(name, value):
    (DATA / name).write_text(json.dumps(a.plain(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (DATA / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(a.plain(rows))


def frozen_hashes():
    return {str(p.relative_to(FINAL)): a.digest(p)
            for p in FINAL.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def metrics(group):
    accepted = [r for r in group if r['accepted']]
    neg = [r for r in accepted if r['negative']]
    positive = [r for r in accepted if r['delta200'] > 0]
    rejected_bad = [r for r in group if not r['accepted'] and r['negative']]
    rejected_good = [r for r in group if not r['accepted'] and r['delta200'] > 0]
    loss = np.asarray([-r['delta200'] for r in neg])
    relative = np.asarray([100*r['relative_gap'] for r in neg])
    return dict(pairs=len(group), accepted=len(accepted), negative_accepted=len(neg),
        positive_accepted=len(positive), tied_accepted=sum(r['delta200'] == 0 for r in accepted),
        absolute_loss_over_1=sum(-r['delta200'] > 1 for r in neg),
        negative_quasi_steady=sum(r['quasi_steady'] for r in neg),
        negative_near_target=sum(r['near_target'] for r in neg),
        negative_relative_le_0p1percent=sum(r['relative_gap'] <= .001 for r in neg),
        negative_relative_le_0p5percent=sum(r['relative_gap'] <= .005 for r in neg),
        negative_relative_le_1percent=sum(r['relative_gap'] <= .01 for r in neg),
        negative_relative_le_5percent=sum(r['relative_gap'] <= .05 for r in neg),
        negative_steady_and_near_tie=sum(r['quasi_steady'] and r['relative_gap'] <= .005 for r in neg),
        negative_negligible_first_move=sum(r['first_move_negligible'] for r in neg),
        negative_small_physical_change=sum(r['small_physical_change'] for r in neg),
        negative_joint_near_equivalence=sum(r['joint_near_equivalence'] for r in neg),
        negative_both_output_rmse_no_worse=sum(r['rmse1_delta'] <= 0 and r['rmse2_delta'] <= 0 for r in neg),
        negative_either_rmse_worse_by_gt_1percent_band=sum(r['max_worsening_band_rmse'] > .01 for r in neg),
        negative_both_rmse_worse_by_gt_1percent_band=sum(r['min_worsening_band_rmse'] > .01 for r in neg),
        negative_worse_band_occupancy=sum(r['inside_joint_band_delta'] < -1e-12 for r in neg),
        negative_upcoming_reference_change=sum(r['next_reference_change'] < 200 for r in neg),
        negative_positive_first20=sum(r['delta20'] > 0 for r in neg),
        negative_positive_first_step=sum(r['delta1'] > 0 for r in neg),
        median_loss=float(np.median(loss)) if len(loss) else None,
        q25_loss=float(np.quantile(loss, .25)) if len(loss) else None,
        q75_loss=float(np.quantile(loss, .75)) if len(loss) else None,
        p90_loss=float(np.quantile(loss, .9)) if len(loss) else None,
        max_loss=float(loss.max()) if len(loss) else None,
        median_relative_loss_percent=float(np.median(relative)) if len(loss) else None,
        p90_relative_loss_percent=float(np.quantile(relative, .9)) if len(loss) else None,
        negative_loss_sum=float(loss.sum()),
        positive_accepted_gain_sum=sum(r['delta200'] for r in positive),
        gate_gain_vs_supervisor_sum=sum(r['delta200'] for r in accepted),
        rejected_bad_policy_count=len(rejected_bad),
        rejected_bad_policy_loss_avoided_sum=sum(-r['delta200'] for r in rejected_bad),
        rejected_good_policy_count=len(rejected_good),
        rejected_good_policy_gain_missed_sum=sum(r['delta200'] for r in rejected_good),
        gate_gain_vs_policy_sum=-sum(r['delta200'] for r in group if not r['accepted']))


def main():
    before = frozen_hashes()
    write_json('research_contract.json', CONTRACT)
    completion_path = ROOT/'change-reports/combined_inference_recovery_20261026/completion.json'
    completion = a.read(completion_path)
    training, training_proof = a.load(completion['trajectory']['path'], completion['trajectory']['sha256'])
    y, u, _, ref = a.trajectories(training)
    ranges = np.asarray(training['data_max'])-training['data_min']
    parameter_source = next(r for r in a.read(FINAL/'data/repeated_provenance.json')
        if (r['plant'], r['category'], r['mode']) == ('column', 'horizon', 'sg'))
    template, _ = a.load(parameter_source['path'], parameter_source['sha256'])
    params = template['reward_params']
    del template
    du = (u-np.vstack([training['steady_states']['ss_inputs'], u[:-1]]))/ranges[:2]
    gate_proof = next(r for r in a.read(FINAL/'data/gate_provenance.json')['sources']
        if r['plant'] == 'column' and r['completed_prefix_records'] == 320)
    decision_path = Path(gate_proof['path'])
    records = a.safe_lines(decision_path)
    canonical = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    assert hashlib.sha256(canonical).hexdigest() == gate_proof['canonical_prefix_sha256']
    prior = {(r['channel'], int(r['step'])): r for r in a.csv_rows(FINAL/'data/gate_decisions.csv')
        if (r['plant'], r['category']) == ('column', 'combined')}
    assert len(records) == len(prior) == 320
    rows, inspected = [], []
    alignment_max = reward_difference_max = 0.
    seen = set()
    for i, r in enumerate(records):
        key = (r['channel'], r['step'])
        assert key not in seen
        seen.add(key)
        assert r['status'] == 'completed' and not r['smoke'] and r['continuation'] == 'gated'
        p = decision_path.parent/r['trace']
        sha = a.digest(p)
        assert sha == prior[key]['trace_sha256']
        step = int(r['step'])
        captured = np.asarray(r['initial_tracking_error_physical'])
        alignment = float(np.max(np.abs((y[step]-ref[step])-captured)/ranges[2:]))
        alignment_max = max(alignment_max, alignment)
        assert alignment <= 1e-4
        band0 = np.maximum(np.abs(ref[step])*params['k_rel'], params['band_floor_phys'])
        steady = bool(np.max(np.abs(ref[step-10:step+1]-ref[step])) < 1e-9
            and np.max(np.abs(np.diff(y[step-10:step+1], axis=0))/band0) <= .05
            and np.max(np.abs(du[step-10:step])) <= .005)
        take = int(r['gate']['selected_source']) == 2
        np.testing.assert_allclose(r['gate']['action'],
            r['gate']['policy_action' if take else 'supervisor_action'], rtol=0, atol=0)
        assert r['metrics']['200']['selected'] == ('policy' if take else 'supervisor')
        branch_data = {}
        with np.load(p, allow_pickle=False) as z:
            for branch in ('policy', 'supervisor'):
                signals = {name: z[branch+'__'+name][:200] for name in ('reward', 'y', 'u', 'reference')}
                assert all(np.isfinite(v).all() for v in signals.values())
                assert signals['y'].shape == signals['u'].shape == signals['reference'].shape == (200, 2)
                assert signals['reward'].shape == (200,)
                assert len(z[branch+'__reward']) == r['rollout_steps']
                previous_input = z[branch+'__previous_input']
                temp = dict(data_min=training['data_min'], data_max=training['data_max'],
                    steady_states=dict(ss_inputs=previous_input))
                rescored, moves, components = a.rescored(np.vstack([y[step], signals['y']]),
                    signals['u'], signals['reference'], temp, params, details=True)
                reward_difference_max = max(reward_difference_max, float(np.max(np.abs(rescored-signals['reward']))))
                np.testing.assert_allclose(rescored, signals['reward'], rtol=1e-7, atol=1e-5)
                discount = r['gamma_per_control_step']**np.arange(200)
                error = signals['y']-signals['reference']
                bands = np.maximum(np.abs(signals['reference'])*params['k_rel'], params['band_floor_phys'])
                ret = float(discount@signals['reward'])
                rmse = np.sqrt(np.mean(error**2, axis=0))
                np.testing.assert_allclose(ret, r['metrics']['200'][branch]['return'], rtol=1e-10, atol=1e-7)
                np.testing.assert_allclose(rmse, r['metrics']['200'][branch]['rmse_physical'], rtol=1e-8, atol=1e-8)
                branch_data[branch] = dict(**signals, ret=ret, rmse=rmse,
                    norm_rmse=np.sqrt(np.mean((error/bands)**2, axis=0)),
                    inside=float(np.mean(np.all(np.abs(error) <= bands, axis=1))),
                    movement=float(np.abs(moves).sum(axis=1).mean()),
                    components={k:float(discount@v) for k,v in components.items()},
                    solver_failures=int(np.sum(~np.asarray(z[branch+'__sol_success'][:200], dtype=bool))))
            np.testing.assert_array_equal(branch_data['policy']['reference'], branch_data['supervisor']['reference'])
            np.testing.assert_array_equal(z['policy__previous_input'], z['supervisor__previous_input'])
        policy, sup = branch_data['policy'], branch_data['supervisor']
        delta = policy['ret']-sup['ret']
        np.testing.assert_allclose(delta, r['metrics']['200']['delta_return'], rtol=1e-9, atol=1e-7)
        np.testing.assert_allclose(delta, float(prior[key]['delta_return']), rtol=1e-9, atol=1e-7)
        assert take == (prior[key]['take_policy'] == 'True')
        relative = abs(delta)/max(1., abs(sup['ret']))
        denom_abs = max(1., float(discount@np.abs(sup['reward'])))
        symmetric = abs(delta)/max(1., abs(sup['ret']), abs(policy['ret']))
        first_move = float(np.max(np.abs(policy['u'][0]-sup['u'][0])/ranges[:2]))
        normalized_rmse_delta = policy['norm_rmse']-sup['norm_rmse']
        small_physical = bool(np.max(np.abs(normalized_rmse_delta)) <= .01)
        changes = np.flatnonzero(np.any(np.abs(sup['reference']-sup['reference'][0]) > 1e-8, axis=1))
        next_change = int(changes[0]) if len(changes) else 200
        prefix = max(1, min(10, next_change))
        reward_delta = policy['reward']-sup['reward']
        row = dict(channel=r['channel'], window=r['window'], step=step, episode=r['episode'],
            accepted=take, negative=delta < 0, delta200=delta,
            policy_return=policy['ret'], supervisor_return=sup['ret'],
            discounted_per_step_delta=delta/float(discount.sum()),
            relative_gap=relative, symmetric_relative_gap=symmetric,
            abs_reward_normalized_gap=abs(delta)/denom_abs,
            delta1=float(reward_delta[0]), delta10=float(discount[:10]@reward_delta[:10]),
            delta20=float(discount[:20]@reward_delta[:20]), delta80=float(discount[:80]@reward_delta[:80]),
            prefix_before_reference_change=prefix,
            delta_prefix=float(discount[:prefix]@reward_delta[:prefix]),
            relative_prefix_gap=abs(float(discount[:prefix]@reward_delta[:prefix]))/max(1.,abs(float(discount[:prefix]@sup['reward'][:prefix]))),
            next_reference_change=next_change, quasi_steady=steady,
            near_target=bool(np.all(np.abs(captured) <= band0)),
            proposal_different=bool(np.max(np.abs(np.asarray(r['gate']['policy_action'])-r['gate']['supervisor_action'])) > 1e-8),
            first_move_range_fraction=first_move, first_move_negligible=first_move <= 1e-4,
            small_physical_change=small_physical,
            joint_near_equivalence=bool(relative <= .005 and first_move <= 1e-4 and small_physical),
            rmse1_delta=float(policy['rmse'][0]-sup['rmse'][0]),
            rmse2_delta=float(policy['rmse'][1]-sup['rmse'][1]),
            norm_rmse1_delta=float(normalized_rmse_delta[0]), norm_rmse2_delta=float(normalized_rmse_delta[1]),
            max_worsening_band_rmse=float(normalized_rmse_delta.max()),
            min_worsening_band_rmse=float(normalized_rmse_delta.min()),
            inside_joint_band_delta=policy['inside']-sup['inside'],
            input_activity_delta=policy['movement']-sup['movement'],
            policy_solver_failures=policy['solver_failures'], supervisor_solver_failures=sup['solver_failures'],
            score_advantage=r['gate'].get('advantage_policy_supervisor'),
            trace=str(p), trace_sha256=sha)
        for name in policy['components']:
            row['delta_'+name] = policy['components'][name]-sup['components'][name]
        np.testing.assert_allclose(sum(row['delta_'+name] for name in policy['components']), delta, rtol=1e-7, atol=1e-5)
        rows.append(row)
        inspected.append(dict(path=str(p), sha256=sha))
        if (i+1) % 80 == 0:
            print('Verified paired traces', i+1, '/ 320', flush=True)
    assert set(Counter((r['channel'], r['window']) for r in rows).values()) == {20}
    assert len(set((r['channel'], r['window']) for r in rows)) == 16
    all_summary = metrics(rows)
    assert all_summary['accepted'] == 200 and all_summary['negative_accepted'] == 100
    summary = [dict(channel=ch, **metrics([r for r in rows if r['channel'] == ch])) for ch in CHANNELS]
    windows = ('release', 'prechange_mature', 'plant_change', 'adapted')
    phase_summary = [dict(window=w, **metrics([r for r in rows if r['window'] == w])) for w in windows]
    neg = [r for r in rows if r['accepted'] and r['negative']]
    sensitivity = []
    for threshold in (.001, .005, .01, .05):
        sensitivity.append(dict(relative_threshold=threshold,
            supervisor_return_denominator=sum(r['relative_gap'] <= threshold for r in neg),
            symmetric_return_denominator=sum(r['symmetric_relative_gap'] <= threshold for r in neg),
            discounted_absolute_reward_denominator=sum(r['abs_reward_normalized_gap'] <= threshold for r in neg)))
    steady_sensitivity = []
    for history in (5, 10, 20):
        for rate_scale in (.5, 1., 2.):
            steady_rows=[]
            for r in rows:
                step=r['step']; b=np.maximum(np.abs(ref[step])*params['k_rel'], params['band_floor_phys'])
                yes=bool(np.max(np.abs(ref[step-history:step+1]-ref[step])) < 1e-9
                    and np.max(np.abs(np.diff(y[step-history:step+1],axis=0))/b) <= .05*rate_scale
                    and np.max(np.abs(du[step-history:step])) <= .005*rate_scale)
                if r['accepted'] and r['negative'] and yes: steady_rows.append(r)
            steady_sensitivity.append(dict(history=history, rate_scale=rate_scale,
                negative_accepted_quasi_steady=len(steady_rows),
                negative_accepted_steady_and_near_tie=sum(r['relative_gap'] <= .005 for r in steady_rows)))
    write_csv('all_320_decisions.csv', rows)
    write_csv('negative_accepted_100.csv', neg)
    write_csv('channel_summary.csv', summary)
    write_csv('phase_summary.csv', phase_summary)
    write_csv('return_threshold_sensitivity.csv', sensitivity)
    write_csv('steady_threshold_sensitivity.csv', steady_sensitivity)
    component_groups = [('all_negative_accepts', neg)]
    component_groups += [(w, [r for r in neg if r['window'] == w]) for w in windows]
    component_groups += [('both_physical_rmse_no_worse', [r for r in neg
        if r['rmse1_delta'] <= 0 and r['rmse2_delta'] <= 0])]
    component_rows = []
    for name, group in component_groups:
        component_rows.append(dict(group=name, count=len(group), delta_return_sum=sum(r['delta200'] for r in group),
            **{k:sum(r[k] for r in group) for k in ('delta_quadratic_tracking','delta_input_movement',
                'delta_outside_band','delta_inside_band','delta_tracking_bonus')}))
    write_csv('negative_reward_components.csv', component_rows)
    write_json('overall_summary.json', all_summary)
    make_figures(rows, summary, phase_summary)
    make_examples(rows, decision_path, params)
    guard = frozen_hashes()
    assert guard == before, 'Prior final package changed during this audit'
    verification = dict(at_utc=datetime.now(timezone.utc).isoformat(),
        all_320_unique_complete_balanced_pairs_verified=True,
        selected_source_matches_executed_candidate_and_selected_branch=True,
        independently_recomputed_branch_rewards_returns_and_rmse=True,
        max_recomputed_step_reward_difference=reward_difference_max,
        predecision_alignment_range_fraction_max=alignment_max,
        canonical_decision_sha256=hashlib.sha256(canonical).hexdigest(),
        main_trajectory=training_proof, reward_parameter_source=parameter_source,
        final_package_files_unchanged=len(before), final_package_sha256_before=before,
        traces=inspected, no_sources_or_recovery_files_changed=True,
        final_e2_context_sources=[dict(path=str(FINAL/'data'/name),sha256=a.digest(FINAL/'data'/name)) for name in
            ('threshold_handoff_metrics.csv','threshold_pairwise.csv','threshold_recovery_summary.csv')],
        inspected_analysis_and_instruction_files=[dict(path=str(p),sha256=a.digest(p)) for p in
            [ROOT/'AGENTS.md', *[ROOT/('.agents/skills/'+s+'/SKILL.md') for s in
                ('research-result-loop','research-orchestrator','experiment-and-statistics','research-reporting-reproducibility')],
             FINAL/'analyze.py', FINAL/'final_comparison.py', FINAL/'data/gate_provenance.json',
             FINAL/'data/gate_decisions.csv', FINAL/'data/research_contract.json', FINAL/'data/reviewer_readiness_summary.json',
             FINAL/'package_manifest.json', completion_path, ROOT/'review_numeric_recovery.py', ROOT/'utils/rewards.py']])
    write_json('verification.json', verification)
    print('OVERALL',json.dumps(a.plain(all_summary)), flush=True)
    print('CHANNELS',json.dumps(a.plain(summary)), flush=True)
    print('PHASES',json.dumps(a.plain(phase_summary)), flush=True)
    print('RETURN SENSITIVITY',json.dumps(sensitivity), flush=True)
    print('STEADY SENSITIVITY',json.dumps(steady_sensitivity), flush=True)


def make_figures(rows, summary, phases):
    neg=[r for r in rows if r['accepted'] and r['negative']]
    colors=['#126c91','#e4972f','#72994f','#9c6b99']
    fig,ax=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ch,label,color in zip(CHANNELS,LABELS,colors):
        values=np.sort([100*r['relative_gap'] for r in neg if r['channel']==ch])
        ax[0,0].step(values,np.arange(1,len(values)+1)/len(values)*100,where='post',label=label,color=color)
    ax[0,0].axvline(.5,color='#777777',ls=':',lw=1,label='0.5% sensitivity')
    ax[0,0].set(xscale='log',xlabel='Loss / abs supervisor return (%)',ylabel='Cumulative share of negative accepts (%)',title='How small are the return losses?')
    ax[0,0].legend(fontsize=8)
    x=np.arange(4)
    ax[0,1].bar(x,[r['negative_accepted'] for r in summary],color='#e4b0a9',label='All negative accepts')
    ax[0,1].bar(x-.18,[r['negative_relative_le_0p5percent'] for r in summary],width=.32,color='#126c91',label='Return loss <=0.5%')
    ax[0,1].bar(x+.18,[r['negative_quasi_steady'] for r in summary],width=.32,color='#72994f',label='Quasi-steady before decision')
    ax[0,1].set(xticks=x,xticklabels=LABELS,ylabel='Decision count',title='Near-ties and steady operation differ',ylim=(0,49))
    ax[0,1].legend(fontsize=8)
    for i,r in enumerate(summary):
        ax[0,1].text(i,r['negative_accepted']+.6,str(r['negative_accepted']),ha='center',fontsize=9)
    for i,r in enumerate(summary):
        ax[1,0].bar(i,r['gate_gain_vs_supervisor_sum']/80,color='#d45c38',width=.4)
        ax[1,0].bar(i+.4,r['gate_gain_vs_policy_sum']/80,color='#126c91',width=.4)
    ax[1,0].set(xticks=x+.2,xticklabels=LABELS,ylabel='Mean 200-step return difference',title='Local value of gating varies by channel')
    ax[1,0].axhline(0,color='#777777',lw=.8)
    from matplotlib.patches import Patch
    ax[1,0].legend(handles=[Patch(color='#d45c38',label='Gate minus supervisor first action'),Patch(color='#126c91',label='Gate minus policy first action')],fontsize=8)
    recovery=a.csv_rows(FINAL/'data/threshold_recovery_summary.csv')
    values={mode:[int(next(r for r in recovery if (r['plant'],r['method'],r['criterion']) ==
        ('column',mode,criterion))['episodes_after_release']) for criterion in ('reward','both_output_rmse')]
        for mode in ('sg','threshold')}
    ax[1,1].bar(np.array([0,1])-.18,values['sg'],width=.32,color='#126c91',label='Critic gate')
    ax[1,1].bar(np.array([0,1])+.18,values['threshold'],width=.32,color='#e4972f',label='Simple threshold')
    for j,vals in enumerate((values['sg'],values['threshold'])):
        for i,v in enumerate(vals): ax[1,1].text(i+(-.18 if j==0 else .18),v+.5,str(v),ha='center')
    ax[1,1].set(xticks=[0,1],xticklabels=['Reward','Both output RMSEs'],ylabel='Episodes after release (lower is faster)',ylim=(0,44),title='Separate E2 column recovery: criterion matters')
    ax[1,1].legend(fontsize=8)
    for axis in ax.flat:axis.grid(axis='y',alpha=.15)
    fig.suptitle('Final combined acceptance audit: 100 negative-return choices among 200 accepts\nE1 = matched first action, then frozen gating; E2 = separately trained threshold comparison. Each uses seed 7.',fontsize=12)
    fig.savefig(FIG/'acceptance_severity.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'acceptance_severity.svg',bbox_inches='tight')
    plt.close(fig)


def make_examples(rows, decision_path, params):
    """Show a worst loss and a median late loss under explicit selection rules."""
    neg=[r for r in rows if r['accepted'] and r['negative']]
    worst=min(neg,key=lambda r:r['delta200'])
    adapted=sorted([r for r in neg if r['window']=='adapted'],key=lambda r:-r['delta200'])
    median_late=adapted[(len(adapted)-1)//2]
    examples=[dict(selection='Largest absolute accepted loss', **worst),
              dict(selection='Lower median absolute loss among adapted negative accepts', **median_late)]
    write_json('example_selection.json', examples)
    fig,axes=plt.subplots(2,3,figsize=(12.5,7),constrained_layout=True)
    for i,r in enumerate(examples):
        with np.load(r['trace'],allow_pickle=False) as z:
            for branch,label,color in (('policy','Accepted policy first action','#d45c38'),
                                       ('supervisor','Supervisor first action','#126c91')):
                reference=z[branch+'__reference'][:200]
                bands=np.maximum(np.abs(reference)*params['k_rel'],params['band_floor_phys'])
                normalized=(z[branch+'__y'][:200]-reference)/bands
                for j in range(2):
                    axes[i,j].plot(np.arange(1,201),normalized[:,j],color=color,lw=1.2,label=label)
            rd=z['policy__reward'][:200]-z['supervisor__reward'][:200]
            axes[i,2].plot(np.arange(1,201),np.cumsum(.99**np.arange(200)*rd),color='#62527d',lw=1.5)
        for j in range(2):
            axes[i,j].axhline(1,color='#999999',ls=':',lw=.8)
            axes[i,j].axhline(-1,color='#999999',ls=':',lw=.8)
            axes[i,j].set(ylabel='Signed error / current tracking band',
                title=('Composition' if j==0 else 'Temperature')+' | '+r['channel'].title()+f", step {r['step']}")
        axes[i,2].axhline(0,color='#999999',ls=':',lw=.8)
        axes[i,2].set(ylabel='Cumulative discounted reward difference',
            title=f"{r['window']}: return loss {100*r['relative_gap']:.2f}%")
        for ax in axes[i]:
            ax.set(xlabel='Controls after the sampled decision');ax.grid(alpha=.15)
            if r['next_reference_change']<200:ax.axvline(r['next_reference_change']+1,color='#777777',ls='--',lw=.7)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Why negative return is not a uniform failure severity\nTop: largest absolute loss. Bottom: median adapted loss. Dashed vertical line = next reference change; dotted levels = tracking bands.',fontsize=12)
    fig.savefig(FIG/'illustrative_traces.png',dpi=180,bbox_inches='tight')
    fig.savefig(FIG/'illustrative_traces.svg',bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
