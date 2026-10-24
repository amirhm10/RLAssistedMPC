"""Check aggregation alignment and recovery timing against the saved curves."""
import analyze as a
np=a.np

rows=a.csv_rows(a.DATA/"repeated_metrics.csv")
curves=a.read(a.DATA/"reward_curves.json")
e2=a.csv_rows(a.DATA/"e2_metrics.csv")
e2curves=a.read(a.DATA/"e2_reward_curves.json")
checked=0
for r in rows:
    curve=next(c for c in curves if c["plant"]==r["plant"] and c["category"]==r["category"] and
               c["mode"]==r["mode"] and str(c["seed"] or "")==r["seed"])
    first,last=a.WINDOWS[r["window"]]
    sl=slice(first-1,last)
    np.testing.assert_allclose(np.mean(curve["reward"][sl]),float(r["mean_reward"]),rtol=1e-10,atol=1e-9)
    for out in (1,2):
        # RMSE across equally long episodes is RMS of episode RMSEs,
        # not the arithmetic mean of those RMSEs.
        np.testing.assert_allclose(np.sqrt(np.mean(np.asarray(curve[f"rmse{out}"][sl])**2)),
                                   float(r[f"rmse_output{out}"]),rtol=1e-10,atol=1e-10)
    checked+=1
windows={"release_14_20":(14,20),"postrelease_14_200":(14,200),
         "learning_tail_191_199":(191,199),"last_five_196_200":(196,200),"evaluation_200":(200,200)}
for r in e2:
    c=next(c for c in e2curves if (c["plant"],c["method"])==(r["plant"],r["method"]))
    first,last=windows[r["window"]]
    np.testing.assert_allclose(np.mean(c["reward"][first-1:last]),float(r["mean_reward"]),rtol=1e-10,atol=1e-9)
    for i in (1,2):
        np.testing.assert_allclose(np.sqrt(np.mean(np.asarray(c[f"rmse{i}"][first-1:last])**2)),
                                   float(r[f"rmse_output{i}"]),rtol=1e-10,atol=1e-10)
    checked+=1
recovery=a.csv_rows(a.DATA/"recovery_times.csv")
for r in recovery:
    if r["study"]=="Scenario 2":
        c=next(c for c in curves if (c["plant"],c["category"],c["mode"],str(c["seed"]))==
            (r["plant"],r["category"],r["mode"],r["seed"]))
        base=next(c for c in curves if c["plant"]==r["plant"] and c["mode"]=="baseline")
    else:
        c=next(c for c in e2curves if (c["plant"],c["method"])==(r["plant"],r["mode"]))
        base=next(c for c in e2curves if c["plant"]==r["plant"] and c["method"] in ("nominal","offsetfree"))
    if r["criterion"]=="reward":
        mask=np.asarray(c["reward"])>=np.asarray(base["reward"])-1e-10
    else:
        mask=(np.asarray(c["rmse1"])<=np.asarray(base["rmse1"])+1e-12)&(
               np.asarray(c["rmse2"])<=np.asarray(base["rmse2"])+1e-12)
    start,stop,dwell=int(r["search_first_episode"])-1,int(r["search_last_episode"]),int(r["consecutive_episodes"])
    candidates=[i for i in range(start,stop-dwell+1) if np.count_nonzero(mask[i:i+dwell])==dwell]
    if r["recovered"]=="True":
        assert candidates and int(r["first_recovered_episode"])==min(candidates)+1
        assert int(r["confirmed_episode"])==min(candidates)+dwell<=stop
        assert int(r["episodes_after_event"])==min(candidates)-start
    else:
        assert not candidates and r["episodes_after_event"]==""
    checked+=1
steady=a.csv_rows(a.DATA/"steady_state_decisions.csv")
groups=a.csv_rows(a.DATA/"steady_state_groups.csv")
provenance=a.read(a.DATA/"steady_state_provenance.json")
assert len(steady)==provenance["classified_decisions"]==960
assert len(groups)==8
assert all(s["predecision_state_max_alignment_range_fraction"]==0 for s in provenance["sources"])
assert len({(r["plant"],r["category"],r["step"]) for r in steady})==len(steady)
for r in steady:
    assert r["selected_source"] in ("1","1.0","2","2.0")
    assert (r["selected_policy"]=="True")== (float(r["selected_source"])==2)
    assert (r["first_move_negligible"]=="True")== (float(r["first_move_max_input_range_fraction"])<=1e-4)
    assert (r["return_200_tie"]=="True")== (float(r["relative_return_difference"])<=.005)
for g in groups:
    selected=[r for r in steady if (r["plant"],r["category"],r["quasi_steady"])==
              (g["plant"],g["category"],g["quasi_steady"])]
    differing=[r for r in selected if r["different_proposed_action"]=="True"]
    ties=[r for r in differing if r["return_200_tie"]=="True"]
    assert len(selected)==int(g["pairs"])
    assert len(differing)==int(g["different_proposals"])
    assert len(ties)==int(g["differing_proposals_200_return_ties"])
    assert sum(r["selected_policy"]=="False" for r in ties)==int(g["supervisor_among_200_return_ties"])
    assert sum(r["selected_policy"]=="True" for r in ties)==int(g["policy_among_200_return_ties"])
assert len(list((a.HERE/"figures").glob("*.png")))==19
assert len(list((a.HERE/"figures").glob("*.svg")))==19
a.save_json("verification.json",dict(passed=True,aggregation_and_recovery_rows_checked=checked,
    steady_state_decisions_checked=len(steady),steady_state_group_summaries_checked=len(groups),
    repeated_runs=100,reference_runs=2,e2_controller_trajectories=12,
    e1_raw_pairs=3733,e3_reproduced_summary_rows=576,
    all_episode_metric_windows_and_censoring_checked=True,
    analysis_source_sha256={p.name:a.digest(p) for p in a.HERE.glob("*.py")}))
print("Verified aggregation and recovery rows:",checked)
