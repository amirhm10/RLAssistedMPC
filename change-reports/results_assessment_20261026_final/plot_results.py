"""Plots from the newly verified, frozen assessment tables; no simulator calls."""
import analyze as a
np,plt=a.np,a.plt
from matplotlib.colors import TwoSlopeNorm

curves=a.read(a.DATA/"reward_curves.json")
e2curves=a.read(a.DATA/"e2_reward_curves.json")
metrics=a.csv_rows(a.DATA/"repeated_metrics.csv")
e2=a.csv_rows(a.DATA/"e2_metrics.csv")
gate=a.csv_rows(a.DATA/"gate_groups.csv")
e3=a.csv_rows(a.DATA/"e3_verified_metrics.csv")
assert len(metrics)==612
names={"polymer":"Polymer CSTR","column":"Distillation column"}
methods={"sg":"Critic-gated RL","plain":"Ungated RL","baseline":"Nominal MPC",
         "nominal":"Nominal MPC","offsetfree":"Nominal MPC","threshold":"Threshold-gated RL",
         "threshold_recovered":"Threshold retry*","ls":"LS-only","bo":"BO tuning",
         "arx_regularized_v1":"ARX-RLS"}
colors={**a.COLORS,"threshold":"#e4972f","threshold_recovered":"#bc8d49",
        "nominal":"#777777","offsetfree":"#777777","ls":"#9278a5",
        "bo":"#3c9272","arx_regularized_v1":"#7e6050"}


def reward(ax,rows,mode,label,limit=None):
    if not rows:
        return
    values=np.asarray([r["reward"] for r in rows])
    if limit:
        values=values[:,:limit]
    x=np.arange(1,values.shape[1]+1)
    ax.plot(x,values.mean(axis=0),label=label,color=colors[mode],lw=1.6)
    if len(values)>1:
        sd=values.std(axis=0,ddof=1)
        ax.fill_between(x,values.mean(axis=0)-sd,values.mean(axis=0)+sd,color=colors[mode],alpha=.16,lw=0)


def handoff():
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),constrained_layout=True)
    for i,plant in enumerate(names):
        for mode in ("baseline","plain","sg"):
            rr=[r for r in curves if r["plant"]==plant and r["mode"]==mode and
                r["category"]==("baseline" if mode=="baseline" else "combined")]
            reward(axes[i,0],rr,mode,methods[mode],30)
        for mode in ("nominal" if plant=="column" else "offsetfree","threshold","sg"):
            reward(axes[i,1],[r for r in e2curves if r["plant"]==plant and r["method"]==mode],
                   mode,methods[mode],30)
        for j in range(2):
            axes[i,j].axvline(13.5,color="#555555",ls="--",lw=.8,label="Release at episode 14" if i==0 else None)
            axes[i,j].axvspan(13.5,20.5,color="#d9e4ec",alpha=.35)
            axes[i,j].set(xlabel="Episode",ylabel="Mean step reward (higher is better)",
                title=f"{names[plant]}: "+("Scenario 2, five paired seeds" if j==0 else "E2 Scenario 1, seed 7"))
            axes[i,j].grid(alpha=.18)
        axes[i,0].legend(fontsize=8)
        axes[i,1].legend(fontsize=8)
    fig.suptitle("Handoff is the primary test: show the release transient, not only the final episode\nUnsmoothed rewards; five-seed shading is sample SD; E2 comparisons are separate single-seed training",fontsize=13)
    a.image(fig,"01_handoff_rewards")


def learning():
    for plant in names:
        fig,axes=plt.subplots(2,3,figsize=(13,7.2),constrained_layout=True)
        for index,cat in enumerate(a.CATS):
            ax=axes.flat[index]
            for mode in ("baseline","plain","sg"):
                rr=[r for r in curves if r["plant"]==plant and r["mode"]==mode and
                    r["category"]==("baseline" if mode=="baseline" else cat)]
                reward(ax,rr,mode,methods[mode])
            ax.axvline(200.5,color="k",ls="--",lw=.8)
            ax.set(title=a.LABELS[index],xlabel="Episode",ylabel="Mean step reward")
            ax.grid(alpha=.15)
        axes.flat[5].axis("off")
        handles,labels=axes.flat[0].get_legend_handles_labels()
        axes.flat[5].legend(handles,labels,loc="center",frameon=False)
        axes.flat[5].text(.5,.15,"Five seeds per mode.\nBands: sample SD, not confidence intervals.\nDashed line: operating change at episode 201.\nNo smoothing; numerical failures were recovered.",ha="center",va="center",fontsize=10)
        fig.suptitle(f"{names[plant]}: all five assistance families, all 300 episodes",fontsize=14)
        a.image(fig,"02_learning_"+plant)


def paired_handoff():
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    fields=["rmse_output1","rmse_output2","normalized_input_movement"]
    for i,plant in enumerate(names):
        labels=(["Viscosity RMSE (L/g)","Temperature RMSE (K)","Range-normalized input movement / step"]
                if plant=="polymer" else ["Composition RMSE (mole fraction)","Temperature RMSE (native units)",
                                          "Range-normalized input movement / step"])
        for j,field in enumerate(fields):
            ax=axes[i,j]
            for seed in (7,11,23,37,53):
                rr=[next(r for r in metrics if (r["plant"],r["category"],r["mode"],r["seed"],r["window"])==
                    (plant,"combined",mode,str(seed),"release")) for mode in ("plain","sg")]
                values=[float(r[field]) for r in rr]
                ax.plot([0,1],values,"o-",color="#bbbbbb",lw=.9,ms=4)
                ax.scatter([0,1],values,c=[colors["plain"],colors["sg"]],s=25,zorder=3)
            base=next(r for r in metrics if r["plant"]==plant and r["mode"]=="baseline" and r["window"]=="release")
            ax.axhline(float(base[field]),color=colors["baseline"],ls="--",label="Nominal MPC")
            ax.set(xticks=[0,1],xticklabels=["Ungated RL","Critic-gated RL"],ylabel=labels[j],
                   title=names[plant]+": episodes 14–20")
            ax.grid(axis="y",alpha=.2)
            if j==0:ax.legend(fontsize=8)
    fig.suptitle("Combined controller at handoff: each connector is the same prescribed seed\nLower is better on every axis; nominal MPC is one common saved reference",fontsize=13)
    a.image(fig,"03_paired_handoff")


def evaluation():
    fig,axes=plt.subplots(2,4,figsize=(14,7.3),constrained_layout=True)
    fields=["mean_reward","rmse_output1","rmse_output2","normalized_input_movement"]
    for i,plant in enumerate(names):
        labels=["Reward ↑","Viscosity RMSE ↓" if plant=="polymer" else "Composition RMSE ↓",
                "Temperature RMSE ↓","Normalized movement ↓"]
        for j,field in enumerate(fields):
            ax=axes[i,j]
            for x,mode in enumerate(("baseline","plain","sg")):
                rr=[r for r in metrics if r["plant"]==plant and r["window"]=="evaluation" and
                    r["mode"]==mode and r["category"]==("baseline" if mode=="baseline" else "combined")]
                v=np.array([float(r[field]) for r in rr])
                ax.bar(x,v.mean(),color=colors[mode],alpha=.7,width=.65)
                ax.scatter(x+np.linspace(-.13,.13,len(v)),v,color="#222222",s=16,zorder=4)
            ax.set(xticks=[0,1,2],xticklabels=["MPC","RL","SG-RL"],title=names[plant]+"\n"+labels[j])
            ax.grid(axis="y",alpha=.15)
    fig.suptitle("Final deterministic evaluation (episode 300): report late performance separately\nDots are five seeds; MPC has one saved reference. Input activity is a cost, not a tracking metric.",fontsize=13)
    a.image(fig,"04_final_evaluation")


def physical():
    final=a.read(a.DATA/"combined_final_trajectories.json")
    for plant in names:
        fig,axes=plt.subplots(2,2,figsize=(12,7.5),constrained_layout=True)
        for mode in ("baseline","plain","sg"):
            rr=[r for r in final if r["plant"]==plant and r["mode"]==mode]
            for j,(key,out) in enumerate((("y",0),("y",1),("u",0),("u",1))):
                values=np.asarray([np.asarray(r[key])[:,out] for r in rr])
                x=np.arange(1,values.shape[1]+1)
                ax=axes.flat[j]
                ax.plot(x,values.mean(axis=0),color=colors[mode],label=methods[mode],lw=1.5)
                if len(values)>1:
                    sd=values.std(axis=0,ddof=1)
                    ax.fill_between(x,values.mean(axis=0)-sd,values.mean(axis=0)+sd,color=colors[mode],alpha=.15,lw=0)
        ref=np.asarray(next(r for r in final if r["plant"]==plant)["ref"])
        for j in range(2):
            axes.flat[j].plot(np.arange(1,len(ref)+1),ref[:,j],"k--",lw=1,label="Setpoint")
        ylabels=(["Viscosity (L/g)","Reactor temperature (K)","Coolant flow (saved physical units)","Monomer flow (saved physical units)"]
                 if plant=="polymer" else ["Ethane mole fraction","Tray temperature (native units)",
                 "Reflux flow (saved native units)","Reboiler duty (saved native units)"])
        for ax,label in zip(axes.flat,ylabels):
            ax.set(xlabel="Control step within episode 300",ylabel=label)
            ax.grid(alpha=.15)
        axes[0,0].legend(fontsize=8)
        fig.suptitle(f"{names[plant]}: combined-controller outputs and inputs, final episode\nMean ± sample SD across five seeds; baseline is one saved trajectory",fontsize=13)
        a.image(fig,"05_physical_"+plant)


def comparator_metrics():
    fig,axes=plt.subplots(2,4,figsize=(15,8.2),constrained_layout=True)
    for i,plant in enumerate(names):
        order=(["offsetfree","ls","arx_regularized_v1","bo","threshold","threshold_recovered","sg"] if plant=="polymer"
               else ["nominal","ls","bo","threshold","sg"])
        for j,(field,label) in enumerate((("mean_reward","Reward ↑"),("rmse_output1","Output 1 RMSE ↓"),
                                          ("rmse_output2","Output 2 RMSE ↓"),("normalized_input_movement","Input movement ↓"))):
            ax=axes[i,j]
            for x,method in enumerate(order):
                row=next(r for r in e2 if (r["plant"],r["method"],r["window"])==(plant,method,"evaluation_200"))
                ax.bar(x,float(row[field]),color=colors[method],alpha=.8)
            ax.set(xticks=range(len(order)),xticklabels=[methods[m].replace(" RL","") for m in order],
                   title=names[plant]+"\n"+label)
            ax.tick_params(axis="x",labelrotation=55,labelsize=8)
            ax.grid(axis="y",alpha=.15)
    fig.suptitle("E2 comparator study: final episode 200, one realization per controller\n*Threshold retry is supplementary same-seed recovery data, not another independent seed",fontsize=13)
    a.image(fig,"06_e2_comparators")


def gate_diagnostics():
    fig,axes=plt.subplots(2,1,figsize=(12,8),constrained_layout=True)
    keys=[(cat,cat) for cat in a.CATS[:-1]]+[("combined",cat) for cat in a.CATS[:-1]]
    for ax,plant in zip(axes,names):
        for j,(field,color,label) in enumerate((("policy_regret","#d45c38","Policy first"),
                                               ("supervisor_regret","#999999","Supervisor first"),
                                               ("gate_regret","#126c91","Actual gate"))):
            values=[]
            for cat,ch in keys:
                rr=[r for r in gate if (r["plant"],r["category"],r["channel"],r["window"])==(plant,cat,ch,"release")]
                assert len(rr)==1
                values.append(float(rr[0][field]))
            ax.bar(np.arange(8)+(j-1)*.24,values,width=.23,color=color,label=label)
        ax.set(xticks=range(8),xticklabels=[f"Solo {l}" for l in a.LABELS[:-1]]+
            [f"Combined {l}" for l in a.LABELS[:-1]],ylabel="Mean decision regret\n(original discounted reward units)",
            title=names[plant]+": release window only")
        ax.set_yscale("symlog",linthresh=.01)
        ax.grid(axis="y",alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("Matched E1 diagnostics: lower regret is better; zero means the better tested first action was selected\n200-step common frozen continuation, one training seed; these are initial-choice alternatives, not three trained controllers",fontsize=12)
    a.image(fig,"07_gate_decision_regret")
    fig,axes=plt.subplots(2,1,figsize=(12,7.5),constrained_layout=True)
    for ax,plant in zip(axes,names):
        for j,(field,color,label) in enumerate((("harmful_accepted","#d45c38","Any lower-return acceptance"),
                                               ("material_harmful_accepted","#6a3540","Return loss > 1 unit"))):
            vals=[];counts=[]
            for cat,ch in keys:
                r=next(r for r in gate if (r["plant"],r["category"],r["channel"],r["window"])==(plant,cat,ch,"release"))
                n=int(r["accepted"]);k=int(r[field])
                vals.append(100*k/n if n else np.nan);counts.append((k,n))
            bars=ax.bar(np.arange(8)+(j-.5)*.35,vals,width=.33,color=color,label=label)
            for bar,(k,n) in zip(bars,counts):
                ax.text(bar.get_x()+bar.get_width()/2,(bar.get_height() if n else 0)+1,
                        f"{k}/{n}" if n else "NA",ha="center",fontsize=8)
        ax.set(xticks=range(8),xticklabels=[f"Solo {l}" for l in a.LABELS[:-1]]+
            [f"Combined {l}" for l in a.LABELS[:-1]],ylim=(0,115),ylabel="% of policy acceptances",
            title=names[plant]+": release window")
        ax.legend(fontsize=8)
        ax.grid(axis="y",alpha=.15)
    fig.suptitle("The gate is imperfect: show unfavorable accepted proposals, including a magnitude sensitivity\nCounts use 200-step return differences. A one-unit threshold is a labeled sensitivity check, not a safety limit.",fontsize=12)
    a.image(fig,"08_gate_acceptance_errors")


def predictions():
    windows=["release","prechange_mature","plant_change","adapted"]
    fig,axes=plt.subplots(2,2,figsize=(12,8.5),constrained_layout=True)
    all_matrices=[]
    for i,plant in enumerate(names):
        sourceplant="distillation" if plant=="column" else "polymer"
        datasets=sorted({r["dataset"] for r in e3 if r["plant"]==sourceplant})
        for j,dataset in enumerate(datasets):
            outputnames=list(dict.fromkeys(r["output"] for r in e3 if r["plant"]==sourceplant))
            matrix=[]
            for cat in ("markov","combined"):
                for output in outputnames:
                    values=[]
                    for window in windows:
                        rr=[r for r in e3 if (r["plant"],r["category"],r["dataset"],r["window"],r["horizon"],r["output"])==
                            (sourceplant,cat,dataset,window,"8",output)]
                        if not rr:
                            values.append(np.nan)
                            continue
                        nom=next(float(r["rmse_physical"]) for r in rr if r["predictor"]=="nominal")
                        val=next(float(r["rmse_physical"]) for r in rr if r["predictor"]=="executed")
                        values.append(val/nom)
                    matrix.append(values)
            all_matrices.append((i,j,dataset,np.asarray(matrix)))
    vmax=max(np.nanmax(abs(np.log2(mat))) for _,_,_,mat in all_matrices)
    for i,j,dataset,mat in all_matrices:
        ax=axes[i,j]
        im=ax.imshow(np.log2(mat),cmap="RdBu_r",vmin=-vmax,vmax=vmax,aspect="auto")
        for y in range(4):
            for x in range(4):
                ax.text(x,y,f"{mat[y,x]:.2f}×" if np.isfinite(mat[y,x]) else "Not sampled",
                        ha="center",va="center",fontsize=10,
                        color="white" if np.isfinite(mat[y,x]) and abs(np.log2(mat[y,x]))>.6*vmax else "#101010")
        ax.set(xticks=range(4),xticklabels=["Release","Mature","Change","Adapted"],
               yticks=range(4),yticklabels=["Markov: output 1","Markov: output 2","Combined: output 1","Combined: output 2"],
               title=list(names.values())[i]+": "+dataset.replace("_"," "))
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label="log2(executed predictor RMSE / nominal predictor RMSE)")
    fig.suptitle("E3 held-out forecasts at lead 8: ratios below 1 improve, above 1 worsen\nKnown actual future inputs, one seed; prediction quality and closed-loop tracking are different questions",fontsize=13)
    a.image(fig,"09_prediction_accuracy")


def coverage():
    rows=a.csv_rows(a.DATA/"gate_decisions.csv")
    windows=["release","prechange_mature","plant_change","adapted"]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),constrained_layout=True)
    for ax,plant in zip(axes,names):
        mat=np.zeros((5,4))
        for i,cat in enumerate(a.CATS):
            target=(100 if plant=="polymer" else 20)*(4 if cat=="combined" else 1)
            for j,w in enumerate(windows):
                n=sum((r["plant"],r["category"],r["window"])==(plant,cat,w) for r in rows)
                mat[i,j]=n/target
                ax.text(j,i,f"{n}/{target}",ha="center",va="center",fontsize=11,zorder=5)
        ax.imshow(mat,cmap="Blues",vmin=0,vmax=1.4,aspect="auto")
        ax.set(xticks=range(4),xticklabels=["Release","Mature","Change","Adapted"],
               yticks=range(5),yticklabels=a.LABELS,title=names[plant])
    fig.suptitle("E1 coverage in this frozen analysis: 3,200 polymer + 640/640 column pairs\nFinal accepted column generations: all five configurations and all four windows complete",fontsize=13)
    a.image(fig,"10_capture_coverage")


if __name__=="__main__":
    handoff();learning();paired_handoff();evaluation();physical()
    comparator_metrics();gate_diagnostics();predictions();coverage()
    print("All figures generated",flush=True)
