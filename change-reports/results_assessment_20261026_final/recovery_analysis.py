"""Sustained performance recovery; censored observations are never dropped."""
import analyze as a
np,plt=a.np,a.plt
curves=a.read(a.DATA/"reward_curves.json")
comparators=a.read(a.DATA/"e2_reward_curves.json")
results=[]


def first_sustained(mask,start,stop,dwell):
    for i in range(start,stop-dwell+1):
        if all(mask[i:i+dwell]):
            return i
    return None


def assess(curve,baseline,study,category,mode,seed,first,last):
    reward=np.asarray(curve["reward"]);base=np.asarray(baseline["reward"])
    error1=np.asarray(curve["rmse1"]);error2=np.asarray(curve["rmse2"])
    within_reward=reward>=base-1e-10
    within_outputs=(error1<=np.asarray(baseline["rmse1"])+1e-12)&(error2<=np.asarray(baseline["rmse2"])+1e-12)
    for definition,mask in (("reward",within_reward),("both_output_rmse",within_outputs)):
        for dwell in (1,3,5):
            start=first_sustained(mask,first-1,last,dwell)
            results.append(dict(study=study,plant=curve["plant"],category=category,mode=mode,seed=seed,
                event="handoff" if first==14 else "operating_change",criterion=definition,
                consecutive_episodes=dwell,search_first_episode=first,search_last_episode=last,
                recovered=start is not None,first_recovered_episode=None if start is None else start+1,
                confirmed_episode=None if start is None else start+dwell,
                episodes_after_event=None if start is None else start-(first-1),
                reward_deficit_episode_sum=float(np.maximum(base[first-1:last]-reward[first-1:last],0).sum()),
                deficit_window=f"{first}--{last}",worst_episode_reward=float(reward[first-1:last].min())))


for curve in curves:
    if curve["mode"]=="baseline":
        continue
    base=next(r for r in curves if r["plant"]==curve["plant"] and r["mode"]=="baseline")
    for first,last in ((14,200),(201,299)):
        assess(curve,base,"Scenario 2",curve["category"],curve["mode"],curve["seed"],first,last)
for curve in comparators:
    if curve["method"] in ("nominal","offsetfree"):
        continue
    base=next(r for r in comparators if r["plant"]==curve["plant"] and r["method"] in ("nominal","offsetfree"))
    assess(curve,base,"E2","combined" if curve["method"] in ("sg","threshold","threshold_recovered") else curve["method"],
           curve["method"],7,14,200)
a.save_csv("recovery_times.csv",results)
a.save_json("recovery_definitions.json",dict(primary_consecutive_episodes=3,sensitivity_consecutive_episodes=[1,5],
    reward="Mean step reward at least matched nominal MPC in each qualifying episode.",
    both_output_rmse="Each of the two output RMSEs no worse than matched nominal MPC in each qualifying episode.",
    timing="First qualifying episode minus event episode; recovery is confirmed only after the next dwell-1 episodes.",
    handoff_window="14--200; censor at the next operating change, never substitute an observed recovery time.",
    operating_change_window="201--299; exclude deterministic evaluation episode 300.",
    deficit="Sum of positive episode-mean reward deficits against matched MPC across the entire fixed window; not area under physical error.",
    qualification="Performance recovery relative to a benchmark, not a mathematical stability or physical settling-time test."))


def points(ax,plant,study,criterion):
    modes=("plain","sg") if study=="Scenario 2" else ("threshold","sg")
    for x,mode in enumerate(modes):
        rr=[r for r in results if (r["plant"],r["study"],r["category"],r["mode"],r["event"],
            r["criterion"],r["consecutive_episodes"])==(plant,study,"combined",mode,"handoff",criterion,3)]
        for offset,r in zip(np.linspace(-.12,.12,len(rr)),rr):
            ax.scatter(x+offset,r["episodes_after_event"] if r["recovered"] else 187,
                marker="o" if r["recovered"] else "^",s=45,
                color="#d45c38" if mode=="plain" else "#e4972f" if mode=="threshold" else "#126c91")
        values=["NR" if not r["recovered"] else str(r["episodes_after_event"]) for r in rr]
        ax.text(x,-.16,", ".join(values),transform=ax.get_xaxis_transform(),ha="center",fontsize=9)
    ax.axhline(187,color="#999999",ls=":",lw=.7)
    ax.set(xticks=[0,1],xticklabels=["Ungated RL" if study=="Scenario 2" else "Threshold RL","Critic-gated RL"],
           ylabel="Episodes after release",title=plant.title()+": "+study)
    ax.set_yscale("symlog",linthresh=10)
    ax.set_ylim(-.4,210);ax.set_yticks([0,1,3,10,30,100,187],["0","1","3","10","30","100","NR by 200"])
    ax.grid(axis="y",alpha=.2)


for criterion,label,number in (("reward","Reward returns to nominal-MPC level","12"),
                              ("both_output_rmse","Both output RMSEs return to nominal-MPC level","13")):
    fig,axes=plt.subplots(2,2,figsize=(11,8.5),constrained_layout=True)
    for i,plant in enumerate(("polymer","column")):
        for j,study in enumerate(("Scenario 2","E2")):
            points(axes[i,j],plant,study,criterion)
    fig.suptitle(label+"\nFirst of three qualifying episodes; circles recover, triangles are not recovered by episode 200\nFive seeds in Scenario 2; one seed in E2. NR is censored, not an observed recovery at the upper limit.",fontsize=12)
    a.image(fig,number+"_recovery_"+criterion)


paired=a.csv_rows(a.DATA/"paired_seed_differences.csv")
gates=a.csv_rows(a.DATA/"gate_decisions.csv")
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for i,plant in enumerate(("polymer","column")):
    for x,cat in enumerate(a.CATS):
        v=[float(r["reward_gain"]) for r in paired if (r["plant"],r["category"],r["window"])==(plant,cat,"release")]
        axes[i,0].bar(x,np.mean(v),color="#126c91",alpha=.55)
        axes[i,0].scatter(x+np.linspace(-.15,.15,5),v,color="#222222",s=20)
    axes[i,0].axhline(0,color="k",lw=.6)
    axes[i,0].set(xticks=range(5),xticklabels=a.LABELS,ylabel="SG minus ungated mean reward",
                   title=plant.title()+": handoff benefit, five paired seeds")
    axes[i,0].set_yscale("symlog",linthresh=.5)
    for context,offset,color in (("standalone",-.18,"#126c91"),("combined",.18,"#e4972f")):
        values=[]
        for cat in a.CATS[:-1]:
            r=[float(v["delta_return"]) for v in gates if (v["plant"],v["category"],v["channel"],v["window"])==
               (plant,cat if context=="standalone" else "combined",cat,"release")]
            values.append(np.mean(np.abs(r)))
        axes[i,1].bar(np.arange(4)+offset,values,width=.34,color=color,label=context)
    axes[i,1].set(xticks=range(4),xticklabels=a.LABELS[:-1],ylabel="Mean absolute paired 200-step return difference",
        title=plant.title()+": observed action-choice consequences, seed 7")
    axes[i,1].set_yscale("log");axes[i,1].legend(fontsize=8)
fig.suptitle("Channel scope matters: show both the gating benefit and the opportunity to change the outcome\nPaired-decision effect sizes are conditional diagnostics across sampled states, not causal channel ablations",fontsize=13)
a.image(fig,"14_channel_effect_sizes")
print("Combined, three-episode recovery:")
for r in results:
    if r["category"]=="combined" and r["event"]=="handoff" and r["consecutive_episodes"]==3:
        print({k:r[k] for k in ("plant","study","mode","seed","criterion","episodes_after_event","recovered")})
