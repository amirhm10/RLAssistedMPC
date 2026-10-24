"""Test near-steady/low-consequence explanations from PRE-decision history."""
import gc
from collections import defaultdict
import analyze as a
np,plt=a.np,a.plt
manifest=a.read(a.DATA/"gate_provenance.json")
rows,sources=[],[]
for plant in ("polymer","column"):
    chosen=[]
    for source in manifest["sources"]:
        if source["plant"]!=plant:
            continue
        records=a.safe_lines(a.Path(source["path"]))[:source["completed_prefix_records"]]
        if records[0]["category"] in ("horizon","weights"):
            chosen.append((records[0]["category"],a.Path(source["path"]),records))
    template_path=next(path.parent.parent/"training.pkl" for cat,path,_ in chosen if cat=="horizon")
    template,tsource=a.load(template_path)
    if "reward_params" in template:
        params=template["reward_params"]
    else:
        reference=next(r for r in a.read(a.DATA/"repeated_provenance.json") if
            (r["plant"],r["category"],r["mode"])==(plant,"horizon","sg"))
        parameter_template,_=a.load(reference["path"],reference["sha256"])
        params=parameter_template["reward_params"]
        del parameter_template
    del template;gc.collect()
    for category,path,records in chosen:
        training,src=a.load(path.parent.parent/"training.pkl")
        y,u,_,ref=a.trajectories(training)
        score,_=a.rescored(y,u,ref,training,params)
        np.testing.assert_allclose(score,training["rewards_step"],rtol=1e-7,atol=1e-5)
        assert len(y)-1==int(training["nFE"])
        ranges=np.asarray(training["data_max"])-np.asarray(training["data_min"])
        previous=np.vstack([training["steady_states"]["ss_inputs"],u[:-1]])
        du=(u-previous)/ranges[:2]
        max_alignment=0
        for record in records:
            step=int(record["step"])
            measured_error=y[step]-ref[step]
            captured_error=np.asarray(record["initial_tracking_error_physical"])
            discrepancy=float(np.max(np.abs(measured_error-captured_error)/ranges[2:]))
            max_alignment=max(max_alignment,discrepancy)
            if discrepancy>1e-4:
                raise ValueError("Pre-decision training alignment requires investigation: "+str((plant,category,step,discrepancy)))
            band=np.maximum(np.abs(ref[step])*params["k_rel"],params["band_floor_phys"])
            lookback=10
            history=y[step-lookback:step+1]
            constant_ref=bool(np.max(np.abs(ref[step-lookback:step+1]-ref[step]))<1e-9)
            output_rate=float(np.max(np.abs(np.diff(history,axis=0))/band))
            input_rate=float(np.max(np.abs(du[step-lookback:step])))
            quasi=bool(constant_ref and output_rate<=.05 and input_rate<=.005)
            near=bool(np.all(np.abs(captured_error)<=band))
            trace=path.parent/record["trace"]
            with np.load(trace,allow_pickle=False) as z:
                pu,su=z["policy__u"][:200],z["supervisor__u"][:200]
                py,sy=z["policy__y"][:200],z["supervisor__y"][:200]
                pr,sr=z["policy__reward"][:200],z["supervisor__reward"][:200]
                references=z["policy__reference"][:200]
                np.testing.assert_allclose(references,z["supervisor__reference"][:200],rtol=0,atol=1e-8)
                changes=np.flatnonzero(np.any(np.abs(references-references[0])>1e-8,axis=1))
                next_change=int(changes[0]) if len(changes) else 200
                first_move=float(np.max(np.abs(pu[0]-su[0])/ranges[:2]))
                short=max(1,min(10,next_change))
                short_difference=float(np.dot(record["gamma_per_control_step"]**np.arange(short),(pr-sr)[:short]))
                first_reward_difference=float(abs(pr[0]-sr[0])/max(1.,abs(sr[0])))
                prefix_supervisor_return=float(np.dot(record["gamma_per_control_step"]**np.arange(short),sr[:short]))
                prefix_relative=abs(short_difference)/max(1.,abs(prefix_supervisor_return))
            metrics=record["metrics"]["200"]
            delta=float(metrics["delta_return"])
            relative=abs(delta)/max(1.,abs(metrics["supervisor"]["return"]))
            rmse_change=float(np.max(np.abs(np.asarray(metrics["policy"]["rmse_physical"])-
                                  np.asarray(metrics["supervisor"]["rmse_physical"]))/band))
            take=int(record["gate"]["selected_source"])==2
            different=bool(np.max(np.abs(np.asarray(record["gate"]["policy_action"])-
                              np.asarray(record["gate"]["supervisor_action"])))>1e-8)
            # These are explicit post-hoc numerical/engineering sensitivities,
            # not new controller tolerances or certified error bounds.
            negligible=bool(first_move<=1e-4 and rmse_change<=.01 and relative<=.005)
            rows.append(dict(plant=plant,category=category,window=record["window"],step=step,
                quasi_steady=quasi,near_target=near,steady_near_target=quasi and near,
                reference_constant_past_10_steps=constant_ref,
                max_output_change_per_step_band_fraction=output_rate,
                max_input_change_per_step_range_fraction=input_rate,
                initial_error_max_band_fraction=float(np.max(np.abs(captured_error)/band)),
                different_proposed_action=different,first_move_max_input_range_fraction=first_move,
                first_move_negligible=first_move<=1e-4,
                first_move_numerically_equal=first_move<=1e-8,
                rmse_change_max_band_fraction=rmse_change,
                absolute_return_difference=abs(delta),delta_return=delta,
                relative_return_difference=relative,selected_policy=take,
                selected_source=int(record["gate"]["selected_source"]),
                predicted_score_advantage=record["gate"].get("advantage_policy_supervisor"),
                configured_advantage_margin=record["gate_config"]["advantage_margin"],
                first_step_reward_relative_difference=first_reward_difference,
                first_step_reward_tie=first_reward_difference<=.001,
                first_step_reward_numerically_equal=first_reward_difference<=1e-8,
                short_prefix_relative_return_difference=prefix_relative,
                return_200_tie=relative<=.005,
                harmful_accepted=bool(take and delta<0),negligible_outcome=negligible,
                harmful_but_negligible=bool(take and delta<0 and negligible),
                short_prefix_steps=short,short_prefix_delta_return=short_difference,
                future_reference_change_within_200=next_change<200,
                steps_to_next_reference_change=next_change))
        src.update(plant=plant,category=category,predecision_state_max_alignment_range_fraction=max_alignment)
        sources.append(src)
        del training,y,u,du;gc.collect()
        print("Pre-decision history verified",plant,category,len(records),max_alignment,flush=True)
a.save_csv("steady_state_decisions.csv",rows)
a.save_json("steady_state_provenance.json",dict(sources=sources,classified_decisions=len(rows),
    scope="Standalone horizon/weights only; both plants, one training seed, complete four-window coverage.",
    quasi_steady="Reference constant over preceding 10 steps, max output increment <=5% of current tracking band per step, max input increment <=0.5% of identification input range per step.",
    near_target="Both PRE-decision output errors inside the existing physical reward bands; distinct from low-rate steady operation.",
    negligible_first_move="Maximum normalized first-input difference <=0.0001 of identification input range (0.01%).",
    negligible_outcome="Negligible first move AND each 200-step output RMSE change <=1% of current tracking band AND abs return difference <=0.5% of max(1,abs supervisor return).",
    limits="Post-hoc diagnostic thresholds; no optimized-objective/Hessian equivalence, causal channel ablation or state-independent no-effect claim.",
    future_reference_changes="Reported separately: a presently steady state can still have different 200-step outcomes across a forthcoming setpoint change."))
grouped=defaultdict(list)
for row in rows:
    grouped[row["plant"],row["category"],row["quasi_steady"]].append(row)
summary=[]
for (plant,category,steady),group in grouped.items():
    harmful=[r for r in group if r["harmful_accepted"]]
    different=[r for r in group if r["different_proposed_action"]]
    tied=[r for r in different if r["return_200_tie"]]
    summary.append(dict(plant=plant,category=category,quasi_steady=steady,pairs=len(group),
        different_proposals=sum(r["different_proposed_action"] for r in group),
        different_and_negligible_first_move=sum(r["first_move_negligible"] for r in different),
        differing_proposals_immediate_reward_ties=sum(r["first_step_reward_tie"] for r in different),
        differing_proposals_200_return_ties=len(tied),
        supervisor_among_200_return_ties=sum(r["selected_source"]==1 for r in tied),
        policy_among_200_return_ties=sum(r["selected_source"]==2 for r in tied),
        supervisor_among_different_proposals=sum(r["selected_source"]==1 for r in different),
        observed_source_codes=sorted({r["selected_source"] for r in different}),
        differing_proposals_median_abs_return_difference=float(np.median([r["absolute_return_difference"] for r in different])) if different else None,
        negligible_first_moves=sum(r["first_move_negligible"] for r in group),
        median_first_move_input_range_fraction=float(np.median([r["first_move_max_input_range_fraction"] for r in group])),
        median_abs_return_difference=float(np.median([r["absolute_return_difference"] for r in group])),
        harmful_acceptances=len(harmful),harmful_but_negligible=sum(r["negligible_outcome"] for r in harmful),
        harmful_at_steady_near_target=sum(r["steady_near_target"] for r in harmful)))
a.save_csv("steady_state_groups.csv",summary)
sensitivity=[]
for plant in ("polymer","column"):
    for category in ("horizon","weights"):
        group=[r for r in rows if (r["plant"],r["category"])==(plant,category)]
        for output_tol,input_tol in ((.02,.002),(.05,.005),(.10,.01)):
            settled=[r for r in group if r["reference_constant_past_10_steps"] and
                r["max_output_change_per_step_band_fraction"]<=output_tol and
                r["max_input_change_per_step_range_fraction"]<=input_tol]
            sensitivity.append(dict(plant=plant,category=category,output_increment_band_fraction=output_tol,
                input_increment_range_fraction=input_tol,quasi_steady_pairs=len(settled),
                negligible_first_moves=sum(r["first_move_negligible"] for r in settled)))
a.save_csv("steady_state_sensitivity.csv",sensitivity)
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
for i,plant in enumerate(("polymer","column")):
    for j,(field,label) in enumerate((("different_and_negligible_first_move","Nearly unchanged first MPC input (%)"),
                                      ("differing_proposals_median_abs_return_difference","Median absolute 200-step return difference"))):
        for x,category in enumerate(("horizon","weights")):
            for steady,offset,color in ((False,-.18,"#e4972f"),(True,.18,"#126c91")):
                r=next((r for r in summary if (r["plant"],r["category"],r["quasi_steady"])==(plant,category,steady)),None)
                if not r:
                    continue
                value=100*r[field]/r["different_proposals"] if j==0 else r[field]
                axes[i,j].bar(x+offset,value,width=.34,color=color,label=("Changing / transition" if not steady else "Quasi steady") if x==0 else None)
                axes[i,j].text(x+offset,value,f"n={r['different_proposals']}",ha="center",va="bottom",fontsize=8)
        axes[i,j].set(xticks=[0,1],xticklabels=["Horizon","Weights"],ylabel=label,title=plant.title())
        if j==0:axes[i,j].set_ylim(0,115)
        else:axes[i,j].set_yscale("symlog",linthresh=.01)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=2,fontsize=9)
fig.suptitle("Test of the steady-state explanation: compare only DIFFERENT proposed configurations\nPRE-decision history defines steady; first-input tolerance = 0.01% of input identification range",fontsize=13)
a.image(fig,"15_steady_state_effects")
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
for i,plant in enumerate(("polymer","column")):
    for j,category in enumerate(("horizon","weights")):
        group=[r for r in rows if (r["plant"],r["category"])==(plant,category) and r["harmful_accepted"]]
        ax=axes[i,j]
        for steady,color,label in ((False,"#e4972f","Changing / transition"),(True,"#126c91","Quasi steady")):
            rr=[r for r in group if r["quasi_steady"]==steady]
            ax.scatter([r["initial_error_max_band_fraction"] for r in rr],
                       [100*r["relative_return_difference"] for r in rr],color=color,s=22,label=label,alpha=.75)
        ax.axhline(.5,color="#777777",ls="--",lw=.8,label="0.5% return sensitivity")
        ax.axvline(1,color="#777777",ls=":",lw=.8)
        ax.set_xscale("symlog",linthresh=1);ax.set_yscale("symlog",linthresh=.5)
        ax.set(title=f"{plant.title()}: {category}, {len(group)} lower-return acceptances",
            xlabel="PRE-decision max tracking error / band",ylabel="Abs return difference / max(1, abs Gsup) (%)")
        ax.legend(fontsize=7)
fig.suptitle("Are the unfavorable horizon/weight decisions simply tiny differences near steady operation?\nAll lower-return acceptances shown; normalized return differences are sensitivities, not safety probabilities",fontsize=13)
a.image(fig,"16_unfavorable_choices_context")
fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
for i,plant in enumerate(("polymer","column")):
    for j in range(2):
        ax=axes[i,j]
        for x,category in enumerate(("horizon","weights")):
            for settled,offset,color in ((False,-.18,"#e4972f"),(True,.18,"#126c91")):
                r=next(r for r in summary if (r["plant"],r["category"],r["quasi_steady"])==(plant,category,settled))
                denominator=r["different_proposals"] if j==0 else r["differing_proposals_200_return_ties"]
                numerator=r["differing_proposals_immediate_reward_ties"] if j==0 else r["supervisor_among_200_return_ties"]
                value=100*numerator/denominator if denominator else 0
                ax.bar(x+offset,value,width=.34,color=color,label=("Changing / transition" if not settled else "Quasi steady") if x==0 else None)
                ax.text(x+offset,value+1,f"{numerator}/{denominator}" if denominator else "No ties",ha="center",fontsize=8)
        ax.set(xticks=[0,1],xticklabels=["Horizon","Weights"],ylim=(0,115),
            ylabel="Different proposals with\nimmediate reward tie (%)" if j==0 else "Supervisor selected among\n200-step return ties (%)",
            title=plant.title())
fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncol=2,fontsize=9)
fig.suptitle("Different settings can be reward-equivalent: does the supervisor retain authority?\nImmediate reward tie: 0.1% sensitivity; 200-step return tie: 0.5% sensitivity. Same proposals excluded.",fontsize=13)
a.image(fig,"17_reward_equivalence_and_supervisor")
print(a.json.dumps(summary,indent=2))
