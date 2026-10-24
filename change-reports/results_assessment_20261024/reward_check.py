"""Explain the column final-reward/RMSE ranking using the actual fixed reward."""
import analyze as a
np,plt=a.np,a.plt
rows=[]
source=a.ROOT/"Distillation/Results/column_review/20261018_recovered"
for method in ("sg","threshold"):
    folder=source/method/"a1";status=a.read(folder/"status.json")
    d,provenance=a.load(folder/"training.pkl",status["files"]["training.pkl"])
    y,u,stored,ref=a.trajectories(d)
    values,du,components=a.rescored(y,u,ref,d,d["reward_params"],details=True)
    np.testing.assert_allclose(values,stored,rtol=1e-7,atol=1e-5)
    n=int(d["time_in_sub_episodes"])
    for window,(first,last) in {"release":(14,20),"evaluation":(200,200)}.items():
        lo,hi=(first-1)*n,last*n
        err=y[lo+1:hi+1]-ref[lo:hi]
        row=dict(method=method,window=window,mean_reward=float(values[lo:hi].mean()),
            rmse1=float(np.sqrt(np.mean(err[:,0]**2))),rmse2=float(np.sqrt(np.mean(err[:,1]**2))),
            mae1=float(np.mean(np.abs(err[:,0]))),mae2=float(np.mean(np.abs(err[:,1]))),
            input_movement_cost=float(np.mean(np.sum(du[lo:hi]**2*d["reward_params"]["R_diag"],axis=1))),
            **{key:float(v[lo:hi].mean()) for key,v in components.items()})
        rows.append(row)
a.save_csv("column_reward_decomposition.csv",rows)
fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
components=["quadratic_tracking","input_movement","outside_band","inside_band","tracking_bonus"]
colors=["#ca5c5c","#9c6c96","#e09a69","#d1b07a","#4e9b85"]
for ax,window in zip(axes,("release","evaluation")):
    selected=[next(r for r in rows if r["method"]==method and r["window"]==window) for method in ("sg","threshold")]
    negative=np.zeros(2);positive=np.zeros(2)
    for key,color in zip(components,colors):
        v=np.array([r[key] for r in selected])
        ax.bar([0,1],v,bottom=np.where(v>=0,positive,negative),color=color,label=key.replace("_"," "))
        positive+=np.maximum(v,0);negative+=np.minimum(v,0)
    ax.scatter([0,1],[r["mean_reward"] for r in selected],color="k",marker="D",label="Total reward",zorder=5)
    ax.axhline(0,color="#555555",lw=.6)
    ax.set(xticks=[0,1],xticklabels=["Critic-gated RL","Threshold-gated RL"],
           ylabel="Mean reward contribution",title="Episodes 14–20" if window=="release" else "Final episode 200")
axes[1].legend(fontsize=8,loc="upper left")
fig.suptitle("Why final reward and RMSE can rank differently: the reward also values near-target behavior\nSame original reward parameters; every stored reward and component sum independently reproduced",fontsize=12)
a.image(fig,"11_reward_decomposition")
print(a.json.dumps(rows,indent=2))
