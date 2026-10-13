"""Physical output metrics with the manuscript's original episode windows."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import pickle
import numpy as np
from .common import reference_physical,write_json

WINDOWS = {"release_14_20":(14,20),"postrelease_14_200":(14,200),
           "learning_tail_191_199":(191,199),"last_five_196_200":(196,200),"evaluation_200":(200,200)}


def trajectories(result):
    y = np.asarray(result.get("y",result.get("y_mpc")),float)
    if "u" in result:
        u = np.asarray(result["u"],float)
    else:
        # The canonical publication baseline exports PHYSICAL inputs under
        # u_mpc. Its delta_u_storage remains scaled; verify this in the audit.
        u = np.asarray(result["u_mpc"],float)
    rewards = np.asarray(result["rewards_step"],float).reshape(-1)
    n = int(result["nFE"])
    if y.shape != (n+1,2) or u.shape != (n,2) or len(rewards) != n:
        raise ValueError("Trajectory dimensions or reward alignment differ from original step semantics.")
    if not all(np.all(np.isfinite(a)) for a in (y,u,rewards)):
        raise ValueError("Nonfinite primary trajectory; preserve and diagnose the run.")
    return y,u,rewards,reference_physical(result)


def metrics(result):
    y,u,rewards,reference = trajectories(result)
    size = int(result["time_in_sub_episodes"])
    dt = float(result["delta_t"])
    rows = []
    for label,(first,last) in WINDOWS.items():
        start,stop = (first-1)*size,last*size
        error = y[start+1:stop+1]-reference[start:stop]
        previous = np.vstack([np.asarray(result["steady_states"]["ss_inputs"]),u[:-1]])
        movement = u[start:stop]-previous[start:stop]
        row = {"method":result.get("comparison_method","offsetfree"),"window":label,
                     "cycles":last-first+1,"steps":stop-start,"mean_reward":float(rewards[start:stop].mean()),
                     "eta_rmse_L_per_g":float(np.sqrt(np.mean(error[:,0]**2))),
                     "T_rmse_K":float(np.sqrt(np.mean(error[:,1]**2))),
                     "eta_mae_L_per_g":float(np.mean(np.abs(error[:,0]))),"T_mae_K":float(np.mean(np.abs(error[:,1]))),
                     "eta_max_abs_L_per_g":float(np.max(np.abs(error[:,0]))),"T_max_abs_K":float(np.max(np.abs(error[:,1]))),
                     "eta_iae_L_h_per_g":float(np.sum(np.abs(error[:,0]))*dt),
                     "T_iae_K_h":float(np.sum(np.abs(error[:,1]))*dt),
                     "Qc_total_abs_movement":float(np.sum(np.abs(movement[:,0]))),
                     "Qm_total_abs_movement":float(np.sum(np.abs(movement[:,1])))}
        ls_source = result.get("rl_action_source_log")
        row["ls_execution_fraction"] = (float(np.mean(np.asarray(ls_source)[start:stop]==5))
                                        if result.get("comparison_method") == "ls" and ls_source is not None else None)
        for channel in ("horizon","markov","weight","residual"):
            source = result.get(channel+"_sg_selected_source_log")
            selected = np.asarray(source)[start:stop] if source is not None else np.empty(0)
            valid = np.isin(selected,[1,2])
            row[channel+"_policy_fraction_among_policy_supervisor_sources"] = (float(np.mean(selected[valid]==2))
                                                                             if np.any(valid) else None)
        rows.append(row)
    return rows


def summarize(directory):
    directory = Path(directory)
    rows = []
    paths = list(sorted(directory.glob("*/training.pkl")))
    references = directory/"references.json"
    if references.exists():
        paths.extend(Path(p) for p in json.loads(references.read_text(encoding="utf-8")).values())
    seen = set()
    for path in paths:
        if path.resolve() in seen:
            continue
        seen.add(path.resolve())
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if result.get("comparison_smoke",False):
            continue
        rows.extend(metrics(result))
    if not rows:
        return []
    with (directory/"comparison_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    write_json(directory/"analysis_definitions.json",{
        "windows":WINDOWS,"alignment":"after-action y[k+1] minus reference[k]",
        "reward":"original fixed reward, unsmoothed arithmetic mean",
        "units":"physical outputs and inputs; Ts in hours",
        "source_fractions":"LS: actual source5 per control step; SG: source2 among source1/2, excluding warm/held/fallback; diagnostic only",
        "limitations":"one realization per comparator; no uncertainty interval or causal channel attribution"})
    return rows
