"""Read-only assessment of the October reviewer campaign, never a simulator run.

Only closed completed trajectories and final accepted E1 generations enter.
One low-priority, single-thread process reads one training bundle at a time.
All derived data and figures are new; manuscript and original results stay put.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
import review_numeric_recovery
review_numeric_recovery.install()
import numpy as np
import psutil
psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comparison_studies.common import reference_physical
from comparison_studies.analysis import trajectories
from aspen_retention import open_read

DATA = HERE / "data"
FIG = HERE / "figures"
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
CATS = ["horizon", "weights", "markov", "residual", "combined"]
LABELS = ["Horizon", "Weights", "Markov", "Residual", "Combined"]
WINDOWS = {"release": (14, 20), "prechange": (191, 199),
           "change": (201, 210), "adapted": (291, 299),
           "phase2_training": (201, 299), "evaluation": (300, 300)}
COLORS = {"sg": "#126c91", "plain": "#d45c38", "baseline": "#777777"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.facecolor": "white"})


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(4 * 1024**2):
            h.update(b)
    return h.hexdigest()


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(name, value):
    (DATA / name).write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n",
                             encoding="utf-8")


def save_csv(name, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (DATA / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def csv_rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def image(fig, name):
    fig.savefig(FIG / (name+".png"), dpi=175, bbox_inches="tight")
    fig.savefig(FIG / (name+".svg"), bbox_inches="tight")
    plt.close(fig)


def safe_lines(path):
    with open_read(path) as f:
        lines = f.read().splitlines(keepends=True)
    return [json.loads(x) for x in lines if x.endswith(b"\n") and x.strip()]


def rescored(y, u, ref, data, params, *, details=False):
    """Vectorized existing relative-QR reward, checked against its scalar source."""
    ranges = np.asarray(data["data_max"])-np.asarray(data["data_min"])
    prior = np.vstack([data["steady_states"]["ss_inputs"], u[:-1]])
    e, du = (y[1:]-ref)/ranges[2:], (u-prior)/ranges[:2]
    band = np.maximum(np.abs(ref)*params["k_rel"], params["band_floor_phys"])/ranges[2:]
    q, r = np.asarray(params["Q_diag"]), np.asarray(params["R_diag"])
    s = 1/(1+np.exp(-np.clip((band-np.abs(e))/np.maximum(params["tau_frac"]*band,1e-12),-60,60)))
    if params["gate"] == "geom":
        wi = np.prod(s, axis=1)**.5
    elif params["gate"] == "mean":
        wi = s.mean(axis=1)
    else:
        wi = s.prod(axis=1)
    eq = np.sum(q*e**2, axis=1)
    effective = (1-wi)*eq+wi*params["lam_in"]*eq
    slope = 2*q*band
    linout = (1-wi)*np.sum(params["gamma_out"]*slope*np.maximum(np.abs(e)-band,0),axis=1)
    linin = wi*np.sum(params["gamma_in"]*slope*np.minimum(np.abs(e),band),axis=1)
    z = np.clip(np.abs(e)/np.maximum(band,1e-12),0,1)
    assert params["bonus_kind"] == "exp"
    phi = (np.exp(-params["bonus_k"]*z)-np.exp(-params["bonus_k"]))/(1-np.exp(-params["bonus_k"]))
    bonus = wi*params["beta"]*np.sum(q*band**2*phi,axis=1)
    result = (-(effective+np.sum(r*du**2,axis=1)+linout+linin)+bonus)*params["reward_scale"]
    from utils.rewards import make_reward_fn_relative_QR
    kw = {k:v for k,v in params.items() if k!="band_floor_scaled"}
    _, fn = make_reward_fn_relative_QR(data["data_min"],data["data_max"],2,**kw)
    ix = np.linspace(0,len(result)-1,101,dtype=int)
    np.testing.assert_allclose(result[ix],[fn(e[i],du[i],ref[i]) for i in ix],rtol=1e-11,atol=1e-8)
    if details:
        scale=params["reward_scale"]
        components=dict(quadratic_tracking=-effective*scale,
            input_movement=-np.sum(r*du**2,axis=1)*scale,
            outside_band=-linout*scale,inside_band=-linin*scale,
            tracking_bonus=bonus*scale)
        np.testing.assert_allclose(sum(components.values()),result,rtol=1e-10,atol=1e-8)
        return result,du,components
    return result, du


def load(path, expected=None):
    path = Path(path)
    before = path.stat()
    sha = digest(path)
    if expected and sha != expected:
        raise ValueError("Closed source checksum differs: "+str(path))
    with path.open("rb") as f:
        value = pickle.load(f)
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return value, dict(path=str(path),sha256=sha,bytes=before.st_size)


def control_rows(d, plant, category, mode, seed, common, reference, provenance):
    y, u, stored, ref = trajectories(d)
    assert len(stored)//int(d["time_in_sub_episodes"]) == 300
    np.testing.assert_allclose(ref,reference,rtol=0,atol=1e-7)
    params = common
    reward, du = rescored(y,u,ref,d,params)
    discrepancy = float(np.max(np.abs(reward-stored)))
    if mode != "baseline" and discrepancy > 1e-5:
        raise ValueError("Stored RL reward differs from the shared score: "+str(provenance))
    n = int(d["time_in_sub_episodes"])
    error = y[1:]-ref
    bands = np.maximum(np.abs(ref)*params["k_rel"],params["band_floor_phys"])
    rows = []
    for name,(first,last) in WINDOWS.items():
        a,b = (first-1)*n,last*n
        err = error[a:b]
        nr = err/bands[a:b]
        row = dict(plant=plant,category=category,mode=mode,seed=seed,window=name,
            steps=b-a,mean_reward=float(reward[a:b].mean()),
            rmse_output1=float(np.sqrt(np.mean(err[:,0]**2))),
            rmse_output2=float(np.sqrt(np.mean(err[:,1]**2))),
            max_abs_output1_error=float(np.max(np.abs(err[:,0]))),
            max_abs_output2_error=float(np.max(np.abs(err[:,1]))),
            band_mse=float(np.mean(np.sum(nr**2,axis=1))),
            joint_band_fraction=float(np.mean(np.all(np.abs(nr)<=1,axis=1))),
            normalized_input_movement=float(np.mean(np.sum(np.abs(du[a:b]),axis=1))),
            input1_total_abs_movement=float(np.sum(np.abs(du[a:b,0]))*(d["data_max"][0]-d["data_min"][0])),
            input2_total_abs_movement=float(np.sum(np.abs(du[a:b,1]))*(d["data_max"][1]-d["data_min"][1])))
        rows.append(row)
    ep = reward.reshape(300,n).mean(axis=1)
    np.testing.assert_allclose(stored.reshape(300,n).mean(axis=1),
                               np.asarray(d["avg_rewards"]).reshape(-1)[:300]
                               if "avg_rewards" in d else stored.reshape(300,n).mean(axis=1),
                               rtol=1e-6,atol=1e-5)
    per_episode_error=error.reshape(300,n,2)
    per_episode_rmse=np.sqrt(np.mean(per_episode_error**2,axis=1))
    curve = dict(plant=plant,category=category,mode=mode,seed=seed,reward=ep.tolist(),
        rmse1=per_episode_rmse[:,0].tolist(),rmse2=per_episode_rmse[:,1].tolist(),
        movement=np.abs(du).sum(axis=1).reshape(300,n).mean(axis=1).tolist())
    final = dict(plant=plant,category=category,mode=mode,seed=seed,
                 ref=ref[-n:].tolist(),y=y[-n:].tolist(),u=u[-n:].tolist(),
                 time=np.arange(n).tolist())
    provenance.update(reward_rescore_max_abs_difference=discrepancy,
        finite_primary_arrays=True,reference_matches_common=True,profile=d.get("training_profile_name"),
        stored_reward_kept=True)
    return rows,curve,final


def controls():
    old = read(ROOT/"change-reports/run_recovery_20261020/all_runs_20261020_165926.json")
    new = read(ROOT/"change-reports/hourly_run_checks_20261024/audit.json")
    inventories = {
        "polymer":[dict(x,path=x["path"]) for x in old["completed"]["polymer_repetitions"]["runs"]],
        "column":[dict(x,path=x["result_file"]) for x in new["column_repetitions"]["results"]]}
    baseline_paths = {
        "polymer": ROOT/"Polymer/Data/mpc_results_dist_robustness_200_100.pickle",
        "column": ROOT/"Distillation/Data/mpc_results_disturb_fluctuation_temperature_flip_200_100.pickle"}
    rows, curves, finals, sources = [], [], [], []
    for plant,jobs in inventories.items():
        assert len(jobs)==50
        identities = {(j["category"],j["mode"],j["seed"]) for j in jobs}
        assert len(identities)==50
        assert {j["seed"] for j in jobs}=={7,11,23,37,53}
        first,_ = load(jobs[0]["path"],jobs[0].get("sha256"))
        common = first["reward_params"]
        reference = reference_physical(first)
        reference = reference.copy()
        del first
        baseline,src = load(baseline_paths[plant])
        src.update(plant=plant,category="baseline",mode="baseline",seed=None)
        a,b,c = control_rows(baseline,plant,"baseline","baseline",None,common,reference,src)
        rows.extend(a);curves.append(b);finals.append(c);sources.append(src)
        del baseline;gc.collect()
        for i,job in enumerate(jobs):
            d,src = load(job["path"],job.get("sha256"))
            src.update(plant=plant,category=job["category"],mode=job["mode"],seed=job["seed"])
            if "reward_params" in d:
                assert plain(d["reward_params"])==plain(common), "Different scientific rewards across methods"
            src["reward_parameter_export_present"] = "reward_params" in d
            # Some original family bundles omit this metadata. In that case
            # the full raw-reward equality below establishes score agreement;
            # no missing parameter value is inferred from a filename.
            a,b,c = control_rows(d,plant,job["category"],job["mode"],job["seed"],common,reference,src)
            rows.extend(a);curves.append(b);sources.append(src)
            if job["category"]=="combined":
                finals.append(c)
            del d,c;gc.collect()
            print(f"{plant} {i+1}/50 {job['category']} {job['mode']} seed{job['seed']}",flush=True)
            save_csv("repeated_metrics.csv",rows)
            save_json("repeated_provenance.json",sources)
            time.sleep(.1)
        del reference
    save_json("reward_curves.json",curves)
    save_json("combined_final_trajectories.json",finals)
    paired=[]
    for plant in inventories:
        for cat in CATS:
            for window in WINDOWS:
                for seed in (7,11,23,37,53):
                    a=next(r for r in rows if (r["plant"],r["category"],r["window"],r["mode"],r["seed"])==(plant,cat,window,"sg",seed))
                    b=next(r for r in rows if (r["plant"],r["category"],r["window"],r["mode"],r["seed"])==(plant,cat,window,"plain",seed))
                    paired.append(dict(plant=plant,category=cat,window=window,seed=seed,
                        reward_gain=a["mean_reward"]-b["mean_reward"],
                        rmse1_reduction_percent=100*(1-a["rmse_output1"]/b["rmse_output1"]),
                        rmse2_reduction_percent=100*(1-a["rmse_output2"]/b["rmse_output2"]),
                        band_mse_reduction_percent=100*(1-a["band_mse"]/b["band_mse"]),
                        movement_reduction_percent=100*(1-a["normalized_input_movement"]/b["normalized_input_movement"])))
    save_csv("paired_seed_differences.csv",paired)
    print("Controls complete",flush=True)


def gates():
    old=read(ROOT/"change-reports/run_recovery_20261020/all_runs_20261020_165926.json")
    cutoff=read(ROOT/"change-reports/hourly_run_checks_20261024/audit_20261026T0426.json")
    inputs=[]
    for path,v in old["completed"]["polymer_E1"]["sources"].items():
        inputs.append(("polymer",Path(path),v["records"]))
    for row in cutoff["E1"]:
        inputs.append(("column",Path(row["directory"])/"capture/decisions.jsonl",row["completed_pairs"]))
    rows,sources=[],[]
    seen=set()
    for plant,path,count in inputs:
        records=safe_lines(path)[:count]
        assert len(records)==count
        canonical=json.dumps(records,sort_keys=True,separators=(",",":")).encode()
        sources.append(dict(plant=plant,path=str(path),completed_prefix_records=count,
                            canonical_prefix_sha256=hashlib.sha256(canonical).hexdigest()))
        for record in records:
            identity=(plant,record["category"],record["channel"],record["step"])
            assert identity not in seen
            seen.add(identity)
            assert record["status"]=="completed" and not record["smoke"]
            delta=record["metrics"]["200"]["delta_return"]
            trace=path.parent/record["trace"]
            with np.load(trace,allow_pickle=False) as arrays:
                returns={}
                for branch in ("policy","supervisor"):
                    for name in ("reward","y","u","reference"):
                        assert np.isfinite(arrays[branch+"__"+name]).all()
                    reward=arrays[branch+"__reward"]
                    assert len(reward)==record["rollout_steps"]
                    ret=float(np.dot(record["gamma_per_control_step"]**np.arange(200),reward[:200]))
                    np.testing.assert_allclose(ret,record["metrics"]["200"][branch]["return"],rtol=1e-10,atol=1e-7)
                    actual=np.sqrt(np.mean((arrays[branch+"__y"][:200]-arrays[branch+"__reference"][:200])**2,axis=0))
                    np.testing.assert_allclose(actual,record["metrics"]["200"][branch]["rmse_physical"],rtol=1e-8,atol=1e-8)
                    returns[branch]=ret
            np.testing.assert_allclose(returns["policy"]-returns["supervisor"],delta,rtol=1e-9,atol=1e-7)
            gate=record["gate"];take=int(gate["selected_source"])==2
            score=gate.get("advantage_policy_supervisor")
            rows.append(dict(plant=plant,category=record["category"],context=record["context"],
                channel=record["channel"],window=record["window"],seed=record["seed"],step=record["step"],
                rollout_steps=record["rollout_steps"],delta_return=delta,take_policy=take,
                score_advantage=score,gate_regret=max(0,-delta if take else delta),
                supervisor_regret=max(0,delta),policy_regret=max(0,-delta),
                harmful_acceptance=bool(take and delta<0),material_harmful_acceptance=bool(take and delta < -1),
                beneficial_rejection=bool(not take and delta>0),gate_gain_over_supervisor=delta if take else 0,
                trace=str(trace),trace_sha256=digest(trace)))
        print("Gate traces verified",plant,path.parent.parent.name,count,flush=True)
    assert len(rows)==3200+640
    save_csv("gate_decisions.csv",rows)
    save_json("gate_provenance.json",dict(column_cutoff_utc=cutoff["at_utc"],sources=sources,
        column_complete=True, accepted_combined_completion=str(ROOT/"change-reports/combined_inference_recovery_20261026/completion.json"),
        recovery_replay_note="Current accepted technical replay generations are used; all original alternatives remain independently retained.",
        primary_horizon=200,zero_sign_tolerance=True,material_harm_sensitivity_reward_units=1,
        all_primary_returns_and_rmse_recomputed=True,one_training_seed=True))
    groups=defaultdict(list)
    for row in rows:
        groups[(row["plant"],row["category"],row["channel"],row["window"])].append(row)
    summary=[]
    for key,group in groups.items():
        accepted=sum(r["take_policy"] for r in group)
        summary.append(dict(zip(("plant","category","channel","window"),key),pairs=len(group),
            accepted=accepted,harmful_accepted=sum(r["harmful_acceptance"] for r in group),
            material_harmful_accepted=sum(r["material_harmful_acceptance"] for r in group),
            beneficial_rejected=sum(r["beneficial_rejection"] for r in group),
            gate_regret=float(np.mean([r["gate_regret"] for r in group])),
            supervisor_regret=float(np.mean([r["supervisor_regret"] for r in group])),
            policy_regret=float(np.mean([r["policy_regret"] for r in group])),
            gate_gain_over_supervisor=float(np.mean([r["gate_gain_over_supervisor"] for r in group]))))
    save_csv("gate_groups.csv",summary)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["controls","gates"])
    args=parser.parse_args()
    if args.stage=="controls":
        controls()
    else:
        gates()


if __name__=="__main__":
    main()
