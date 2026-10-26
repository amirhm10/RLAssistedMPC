"""Final offline handoff/recovery audit and completed combined diagnostics."""
from collections import Counter
import gc
import analyze as a

np, plt = a.np, a.plt
COLORS = dict(sg='#126c91', threshold='#e4972f', nominal='#777777')
NAMES = dict(sg='Critic gate', threshold='Simple threshold', nominal='Nominal MPC')


def threshold_metrics():
    sources = a.read(a.DATA/'e2_provenance.json')
    rows, provenance = [], []
    windows = dict(release_14_20=(14,20), first_episode_14=(14,14),
                   release_14_16=(14,16), release_14_25=(14,25),
                   postrelease_14_200=(14,200), evaluation_200=(200,200))
    for plant in ('polymer','column'):
        parameter_source = next(r for r in a.read(a.DATA/'repeated_provenance.json')
            if (r['plant'],r['category'],r['mode']) == (plant,'horizon','sg'))
        template, _ = a.load(parameter_source['path'],parameter_source['sha256'])
        params = template['reward_params']; del template; gc.collect()
        reference = None
        for mode in ('sg','threshold','nominal'):
            source = next(r for r in sources if r['plant']==plant and
                r['method']==(mode if mode!='nominal' or plant=='column' else 'offsetfree'))
            data, proof = a.load(source['path'],source['sha256'])
            y,u,stored,ref = a.trajectories(data)
            if reference is None: reference=ref.copy()
            else: np.testing.assert_allclose(ref,reference,rtol=0,atol=1e-7)
            reward,du = a.rescored(y,u,ref,data,params)
            if mode!='nominal': np.testing.assert_allclose(reward,stored,rtol=1e-7,atol=1e-5)
            bands=np.maximum(np.abs(ref)*params['k_rel'],params['band_floor_phys'])
            error=y[1:]-ref; norm=np.abs(error)/bands; joint=norm.max(axis=1)
            n=int(data['time_in_sub_episodes'])
            for name,(first,last) in windows.items():
                lo,hi=(first-1)*n,last*n; e=error[lo:hi]
                rows.append(dict(plant=plant,method=mode,window=name,first_episode=first,last_episode=last,
                    steps=hi-lo,mean_reward=float(reward[lo:hi].mean()),
                    rmse1=float(np.sqrt(np.mean(e[:,0]**2))),rmse2=float(np.sqrt(np.mean(e[:,1]**2))),
                    peak_error1=float(np.abs(e[:,0]).max()),peak_error2=float(np.abs(e[:,1]).max()),
                    outside_either_tracking_band_fraction=float(np.mean(joint[lo:hi]>1)),
                    peak_tracking_band_ratio=float(joint[lo:hi].max()),
                    q95_tracking_band_ratio=float(np.quantile(joint[lo:hi],.95)),
                    normalized_input_activity=float(np.abs(du[lo:hi]).sum(axis=1).mean()),
                    largest_normalized_input_jump=float(np.abs(du[lo:hi]).max())))
            proof.update(plant=plant,method=mode,parameter_source=parameter_source['path'],
                threshold=data.get('comparison_threshold'),core_arrays_finite=True)
            provenance.append(proof)
            del data,y,u,stored,ref,reward,du,error,norm,joint; gc.collect()
    a.save_csv('threshold_handoff_metrics.csv',rows)
    a.save_json('threshold_handoff_provenance.json',dict(sources=provenance,
        primary_window='episodes 14-20',window_sensitivities=['14','14-16','14-25'],
        handoff_safety_meaning='Empirical disruption, tracking-band excursions and input movement. Tracking bands are not certified safe operating limits.',
        independent_threshold_realizations_per_plant=1,no_threshold_seed_confidence_intervals=True,
        end_to_end_training_comparison_not_fixed_policy_selector_ablation=True))
    comparisons=[]
    for plant in ('polymer','column'):
        for window in windows:
            sg=next(r for r in rows if (r['plant'],r['method'],r['window'])==(plant,'sg',window))
            th=next(r for r in rows if (r['plant'],r['method'],r['window'])==(plant,'threshold',window))
            comparisons.append(dict(plant=plant,window=window,reward_gain=sg['mean_reward']-th['mean_reward'],
                **{k+'_reduction_percent':100*(1-sg[k]/th[k]) for k in
                   ('rmse1','rmse2','normalized_input_activity')},
                band_excursion_reduction_percentage_points=100*(th['outside_either_tracking_band_fraction']-sg['outside_either_tracking_band_fraction'])))
    a.save_csv('threshold_pairwise.csv',comparisons)

    fig,axes=plt.subplots(2,3,figsize=(12.5,7),constrained_layout=True)
    for i,plant in enumerate(('polymer','column')):
        rr={m:next(r for r in rows if (r['plant'],r['method'],r['window'])==(plant,m,'release_14_20')) for m in NAMES}
        for j,m in enumerate(NAMES):
            axes[i,0].bar(np.arange(2)+(j-1)*.24,[rr[m]['rmse1']/rr['threshold']['rmse1'],rr[m]['rmse2']/rr['threshold']['rmse2']],width=.22,color=COLORS[m],label=NAMES[m])
        axes[i,0].axhline(1,color='#999999',ls=':',lw=.7)
        axes[i,0].set(xticks=[0,1],xticklabels=['Output 1','Output 2'],ylabel='RMSE / threshold RMSE',title=plant.title()+': tracking error')
        for j,m in enumerate(NAMES):
            value=100*rr[m]['outside_either_tracking_band_fraction']
            axes[i,1].bar(j,value,color=COLORS[m]);axes[i,1].text(j,value+1,f'{value:.1f}%',ha='center',fontsize=9)
            axes[i,2].bar(j,rr[m]['normalized_input_activity'],color=COLORS[m])
        axes[i,1].set(xticks=range(3),xticklabels=['Critic','Threshold','MPC'],ylim=(0,110),ylabel='Steps outside either tracking band (%)',title='Tracking-band excursions')
        axes[i,2].set(xticks=range(3),xticklabels=['Critic','Threshold','MPC'],ylabel='Mean sum abs input change / ID ranges',title='Input activity (lower is better)')
        for ax in axes[i]: ax.grid(axis='y',alpha=.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,0),ncol=3,fontsize=9)
    fig.suptitle('Handoff disruption: critic gate versus simple tracking-error threshold\nEpisodes 14-20; one E2 realization per method and plant. Tracking bands are not hard safety constraints.',fontsize=12)
    a.image(fig,'18_threshold_handoff')


def recovery_figure():
    curves=a.read(a.DATA/'e2_reward_curves.json');recovery=a.csv_rows(a.DATA/'recovery_times.csv')
    summary=[]
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for i,plant in enumerate(('polymer','column')):
        baseline=next(r for r in curves if r['plant']==plant and r['method'] in ('nominal','offsetfree'))
        for m in ('sg','threshold'):
            curve=next(r for r in curves if (r['plant'],r['method'])==(plant,m))
            difference=np.asarray(curve['reward'])-baseline['reward']
            axes[i,0].plot(np.arange(14,81),difference[13:80],color=COLORS[m],label=NAMES[m],lw=1.5)
            for j,criterion in enumerate(('reward','both_output_rmse')):
                r=next(r for r in recovery if (r['plant'],r['study'],r['mode'],r['event'],r['criterion'],r['consecutive_episodes'])==(plant,'E2',m,'handoff',criterion,'3'))
                assert r['recovered']=='True'
                value=int(r['episodes_after_event']);x=j+(-.19 if m=='sg' else .19)
                axes[i,1].bar(x,value,width=.34,color=COLORS[m],label=NAMES[m] if j==0 else None)
                axes[i,1].text(x,value+.6,str(value),ha='center',fontsize=11)
                summary.append(dict(plant=plant,method=m,criterion=criterion,episodes_after_release=value,
                    confirmed_episode=int(r['confirmed_episode']),reward_deficit_episode_sum=float(r['reward_deficit_episode_sum'])))
        axes[i,0].axhline(0,color='#777777',lw=.7,ls='--')
        axes[i,0].set(title=plant.title()+': unsmoothed reward recovery',xlabel='Episode',ylabel='Mean reward minus matched nominal MPC')
        axes[i,1].set(title=plant.title()+': first sustained recovery',xticks=[0,1],xticklabels=['Reward','Both output RMSEs'],ylabel='Episodes after release 14 (lower is faster)',ylim=(0,44))
        for ax in axes[i]:ax.grid(axis='y',alpha=.18)
    axes[0,0].legend(fontsize=9);axes[0,1].legend(fontsize=9)
    fig.suptitle('Recovery speed depends on the criterion\nFirst of three qualifying episodes; column threshold recovers both output RMSEs earlier. Single-seed E2 evidence.',fontsize=12)
    a.image(fig,'19_threshold_recovery')
    a.save_csv('threshold_recovery_summary.csv',summary)


def final_combined():
    rows=a.csv_rows(a.DATA/'gate_decisions.csv')
    rows=[r for r in rows if (r['plant'],r['category'])==('column','combined')]
    assert len(rows)==320
    summary=[];extended=[]
    for channel in a.CATS[:-1]:
        group=[r for r in rows if r['channel']==channel];assert len(group)==80
        delta=np.array([float(r['delta_return']) for r in group]);take=np.array([r['take_policy']=='True' for r in group])
        summary.append(dict(channel=channel,pairs=80,accepted=int(take.sum()),
            harmful_accepted=int(np.sum(take&(delta<0))),material_harmful_accepted=int(np.sum(take&(delta < -1))),
            beneficial_rejected=int(np.sum(~take&(delta>0))),
            critic_regret=float(np.maximum(0,np.where(take,-delta,delta)).mean()),
            always_supervisor_regret=float(np.maximum(0,delta).mean()),
            always_policy_regret=float(np.maximum(0,-delta).mean()),
            median_selected_gain_vs_supervisor=float(np.median(np.where(take,delta,0))),
            median_accepted_delta=float(np.median(delta[take])) if take.any() else None,
            mean_selected_gain_vs_supervisor=float(np.where(take,delta,0).mean())))
    source=next(r for r in a.read(a.DATA/'gate_provenance.json')['sources'] if r['plant']=='column' and r['completed_prefix_records']==320)
    for r in a.safe_lines(a.Path(source['path'])):
        if r['rollout_steps']<800:continue
        with np.load(a.Path(source['path']).parent/r['trace'],allow_pickle=False) as arrays:
            delta={}
            for h in (200,800):
                delta[h]=float(np.dot(r['gamma_per_control_step']**np.arange(h),arrays['policy__reward'][:h]-arrays['supervisor__reward'][:h]))
                np.testing.assert_allclose(delta[h],r['metrics'][str(h)]['delta_return'],rtol=1e-8,atol=1e-7)
        extended.append(dict(channel=r['channel'],window=r['window'],step=r['step'],delta200=delta[200],delta800=delta[800],sign_reversal=bool(delta[200]*delta[800]<0)))
    a.save_csv('combined_final_gate_summary.csv',summary);a.save_csv('combined_long_horizon_sensitivity.csv',extended)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for j,(key,label,color) in enumerate((('critic_regret','Critic gate','#126c91'),('always_supervisor_regret','Supervisor first action','#777777'),('always_policy_regret','Policy first action','#d45c38'))):
        axes[0].bar(np.arange(4)+(j-1)*.25,[r[key] for r in summary],width=.23,label=label,color=color)
    axes[0].set(xticks=range(4),xticklabels=a.LABELS[:-1],ylabel='Mean 200-step realized decision regret',title='Compared at the same sampled states')
    axes[0].set_yscale('symlog',linthresh=1);axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=3)
    for j,r in enumerate(summary):
        axes[1].bar(j,r['accepted'],color='#bed8df',label='Accepted policy' if j==0 else None)
        axes[1].bar(j,r['harmful_accepted'],color='#c65d52',label='Lower-return acceptance' if j==0 else None)
        axes[1].text(j,r['accepted']+1,f"{r['harmful_accepted']}/{r['accepted']}",ha='center',fontsize=10)
    axes[1].set(xticks=range(4),xticklabels=a.LABELS[:-1],ylim=(0,84),ylabel='Count out of 80 decisions per channel',title='Critic decisions are not uniformly favorable')
    axes[1].legend(fontsize=8)
    fig.suptitle('Completed column combined E1: 320 matched decisions, seed 7\nOnly the first candidate changes; both use frozen gated continuation. These are not whole threshold-controller runs.',fontsize=12)
    a.image(fig,'20_final_combined_decisions')

    completion=a.read(a.ROOT/'change-reports/combined_inference_recovery_20261026/completion.json')
    data,proof=a.load(completion['trajectory']['path'],completion['trajectory']['sha256'])
    params=data.get('reward_params')
    if params is None:
        param_source=next(r for r in a.read(a.DATA/'repeated_provenance.json') if (r['plant'],r['category'],r['mode'])==('column','horizon','sg'))
        template,_=a.load(param_source['path'],param_source['sha256']);params=template['reward_params'];del template
    baseline=next(r for r in a.read(a.DATA/'reward_curves.json') if r['plant']=='column' and r['mode']=='baseline')
    base_source=next(r for r in a.read(a.DATA/'repeated_provenance.json') if r['plant']=='column' and r['mode']=='baseline')
    base_data,_=a.load(base_source['path'],base_source['sha256'])
    ref=a.reference_physical(base_data).copy();del base_data
    metrics,curve,_=a.control_rows(data,'column','combined','E1 accepted',7,params,ref,proof)
    a.save_csv('combined_final_training_metrics.csv',metrics);a.save_json('combined_final_training_curve.json',curve)
    proof.update(source='Accepted final E1 instrumented main trajectory, not an additional independent replicate.',
        accepted_recovery='combined_inference_recovery_20261026/completion.json',
        technical_suffix_replays_have_preserved_originals=True)
    a.save_json('combined_final_training_provenance.json',proof)
    # Test the steady-state interpretation in the final combined context too.
    y,u,_,_=a.trajectories(data);ranges=np.asarray(data['data_max'])-data['data_min']
    du=(u-np.vstack([data['steady_states']['ss_inputs'],u[:-1]]))/ranges[:2]
    state_rows=[];max_alignment=0.
    for r in a.safe_lines(a.Path(source['path'])):
        if r['channel'] not in ('horizon','weights'):continue
        step=int(r['step']);band=np.maximum(np.abs(ref[step])*params['k_rel'],params['band_floor_phys'])
        captured=np.asarray(r['initial_tracking_error_physical'])
        alignment=float(np.max(np.abs((y[step]-ref[step])-captured)/ranges[2:]))
        max_alignment=max(max_alignment,alignment);assert alignment<=1e-4
        quasi=bool(np.max(np.abs(ref[step-10:step+1]-ref[step]))<1e-9 and
            np.max(np.abs(np.diff(y[step-10:step+1],axis=0))/band)<=.05 and
            np.max(np.abs(du[step-10:step]))<=.005)
        different=bool(np.max(np.abs(np.asarray(r['gate']['policy_action'])-r['gate']['supervisor_action']))>1e-8)
        with np.load(a.Path(source['path']).parent/r['trace'],allow_pickle=False) as z:
            move=float(np.max(np.abs(z['policy__u'][0]-z['supervisor__u'][0])/ranges[:2]))
        delta=float(r['metrics']['200']['delta_return'])
        tie=bool(abs(delta)/max(1.,abs(r['metrics']['200']['supervisor']['return']))<=.005)
        take=int(r['gate']['selected_source'])==2
        state_rows.append(dict(channel=r['channel'],window=r['window'],step=step,
            quasi_steady=quasi,near_target=bool(np.all(np.abs(captured)<=band)),different_proposal=different,
            first_move_negligible=move<=1e-4,first_move_input_range_fraction=move,
            return_near_tie=tie,selected_policy=take,harmful_accepted=bool(take and delta<0),
            material_harmful_accepted=bool(take and delta < -1),delta_return=delta))
    assert len(state_rows)==160
    a.save_csv('combined_steady_decisions.csv',state_rows)
    state_groups=[]
    for channel in ('horizon','weights'):
        for quasi in (False,True):
            group=[r for r in state_rows if r['channel']==channel and r['quasi_steady']==quasi]
            different=[r for r in group if r['different_proposal']]
            ties=[r for r in different if r['return_near_tie']]
            state_groups.append(dict(channel=channel,quasi_steady=quasi,pairs=len(group),
                different_proposals=len(different),negligible_first_moves=sum(r['first_move_negligible'] for r in different),
                near_tied_returns=len(ties),supervisor_among_ties=sum(not r['selected_policy'] for r in ties),
                material_harmful_accepted=sum(r['material_harmful_accepted'] for r in group),
                near_target_material_harmful=sum(r['material_harmful_accepted'] and r['near_target'] for r in group)))
    a.save_csv('combined_steady_groups.csv',state_groups)
    a.save_json('combined_steady_provenance.json',dict(classified_decisions=160,
        max_training_trace_alignment_range_fraction=max_alignment,
        thresholds_match_previous_standalone_analysis=True,
        engineering_sensitivities_not_controller_settings_or_safety_bounds=True))
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),constrained_layout=True)
    labels=[]
    for i,r in enumerate(state_groups):
        labels.append(r['channel'].title()+'\n'+('Steady' if r['quasi_steady'] else 'Changing'))
        denominator=r['different_proposals']
        if denominator:
            axes[0].bar(i-.18,100*r['negligible_first_moves']/denominator,width=.34,color='#126c91')
            axes[0].bar(i+.18,100*r['near_tied_returns']/denominator,width=.34,color='#e4972f')
        axes[1].bar(i,r['material_harmful_accepted'],color='#c65d52')
        axes[1].text(i,r['material_harmful_accepted']+.3,f"{r['material_harmful_accepted']}/{r['pairs']}",ha='center',fontsize=9)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='#126c91',label='Nearly unchanged first input'),Patch(color='#e4972f',label='Near-tied 200-step returns')],fontsize=8)
    axes[0].set(ylabel='Percent of DIFFERENT proposed configurations',ylim=(0,110))
    axes[1].set(ylabel='Accepted policy with delta return below -1',ylim=(0,max(r['material_harmful_accepted'] for r in state_groups)+5))
    for ax in axes:ax.set(xticks=range(4),xticklabels=labels);ax.grid(axis='y',alpha=.18)
    fig.suptitle('Final combined horizon/weight decisions: does steady operation make choices equivalent?\nPre-decision history only; same engineering sensitivities as the standalone analysis. One seed, 160 decisions.',fontsize=12)
    a.image(fig,'22_combined_steady_state')
    fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    for ax,key,label in zip(axes.flat,('reward','rmse1','rmse2','movement'),('Episode mean reward','Ethane-composition RMSE (mole fraction)','Tray85 temperature RMSE (K)','Normalized input activity')):
        ax.plot(np.arange(1,301),curve[key],color='#126c91',lw=1.2,label='Accepted combined E1 main run')
        ax.plot(np.arange(1,301),baseline[key],color='#777777',lw=.9,label='Matched nominal MPC')
        for event in (14,201):ax.axvline(event,color='#aaaaaa',ls=':',lw=.8)
        ax.set(xlabel='Episode',ylabel=label);ax.grid(alpha=.18)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Final combined main-training trajectory: 300/300 episodes\nInstrumented seed 7 run with accepted full-state recovery; shown separately from the five-seed study and E2 threshold test.',fontsize=12)
    a.image(fig,'21_final_combined_training')
    print('Final combined gate summaries:',a.json.dumps(summary),flush=True)
    print('Extended800-step checks:',len(extended),'sign reversals:',sum(r['sign_reversal'] for r in extended),flush=True)
    print('Combined steady-state groups:',a.json.dumps(state_groups),flush=True)


if __name__=='__main__':
    threshold_metrics();recovery_figure();final_combined()
