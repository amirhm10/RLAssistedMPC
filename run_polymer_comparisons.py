"""Scenario 1 (200 cycles) polymer comparators without modifying original code.

python run_polymer_comparisons.py --dry-run
python run_polymer_comparisons.py --smoke
python run_polymer_comparisons.py --methods ls
python run_polymer_comparisons.py --methods arx bo threshold
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def baseline_audit(namespace, full):
    """Audit physical schedule/scaling/reward, plus a native nominal cycle replay."""
    import numpy as np
    from comparison_studies.analysis import trajectories
    from comparison_studies.common import reference_physical,window_bundle
    from comparison_studies.native import nominal_run
    path = ROOT/"Polymer/Data/mpc_results_dist.pickle"
    report = {"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"usable":False}
    with path.open("rb") as stream:
        result = pickle.load(stream)
    y,u,rewards,ref = trajectories(result)
    checks = {"steps":result["nFE"] == 160000,"cycle_length":result["time_in_sub_episodes"] == 800,
              "sampling_time":np.isclose(result["delta_t"],namespace["delta_t"]),
              "references":np.allclose(result["y_sp"],full["y_sp"],rtol=0,atol=1e-12),
              "scaling":np.allclose(result["data_min"],namespace["data_min"]) and np.allclose(result["data_max"],namespace["data_max"]),
              "steady_outputs":np.allclose(result["steady_states"]["y_ss"],namespace["steady_states"]["y_ss"]),
              "steady_inputs":np.allclose(result["steady_states"]["ss_inputs"],namespace["steady_states"]["ss_inputs"])}
    # A saved trajectory alone does not reveal historical controller settings.
    replay,_ = nominal_run(namespace,window_bundle(full,0,800),timing=False)
    output_tolerance = .01*np.asarray(namespace["reward_params"]["band_floor_phys"])
    input_tolerance = .0001*(np.asarray(namespace["SYS"]["input_bounds"]["u_max"])-np.asarray(namespace["SYS"]["input_bounds"]["u_min"]))
    checks["nominal_cycle_outputs"] = bool(np.all(np.max(np.abs(replay["y"]-y[:801]),axis=0)<=output_tolerance))
    checks["nominal_cycle_inputs"] = bool(np.all(np.max(np.abs(replay["u"]-u[:800]),axis=0)<=input_tolerance))
    report["replay_output_tolerance"] = output_tolerance.tolist()
    report["replay_input_tolerance"] = input_tolerance.tolist()
    report["replay_tolerance_definition"] = "1% of minimum physical tracking bands; 0.01% of input ranges"
    report["nominal_replay_max_output_difference"] = np.max(np.abs(replay["y"]-y[:801]),axis=0).tolist()
    report["nominal_replay_max_input_difference"] = np.max(np.abs(replay["u"]-u[:800]),axis=0).tolist()
    # Re-score every step under the exact shared reward. Never overwrite the source.
    lo,hi = np.asarray(result["data_min"]),np.asarray(result["data_max"])
    dy = (y[1:]-ref)/(hi[2:]-lo[2:])
    previous = np.vstack([result["steady_states"]["ss_inputs"],u[:-1]])
    du = (u-previous)/(hi[:2]-lo[:2])
    checks["input_units"] = np.allclose(du,result["delta_u_storage"],rtol=1e-7,atol=1e-8)
    bounds = namespace["SYS"]["input_bounds"]
    checks["input_bounds"] = bool(np.all(u>=np.asarray(bounds["u_min"])-1e-6) and np.all(u<=np.asarray(bounds["u_max"])+1e-6))
    rescored = np.array([namespace["reward_fn"](e,d,r) for e,d,r in zip(dy,du,ref)])
    checks["reward"] = np.allclose(rescored,rewards,rtol=1e-7,atol=1e-8)
    disturbance = result.get("disturbance_profile") or {}
    report["disturbance_profile_keys"] = list(disturbance)
    # A nominal replay verifies controller behavior; exogenous arrays must also match.
    from utils.helpers import build_polymer_disturbance_schedule,disturbance_profile_from_schedule
    expected = disturbance_profile_from_schedule(build_polymer_disturbance_schedule(
        qi=full["qi"],qs=full["qs"],ha=full["ha"]))
    def equivalent(a,b):
        if isinstance(a,dict) and isinstance(b,dict):
            return set(a) == set(b) and all(equivalent(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple,np.ndarray)) or isinstance(b,(list,tuple,np.ndarray)):
            try:
                return bool(np.allclose(np.asarray(a,float),np.asarray(b,float),rtol=1e-10,atol=1e-8))
            except (ValueError,TypeError):
                return a == b
        return a == b
    # Descriptive labels may have been added after the saved baseline.
    for key in ("values","schedule","qi","qs","ha"):
        if key in expected and key in disturbance:
            checks["disturbance_"+key] = equivalent(expected[key],disturbance[key])
    if not any(k.startswith("disturbance_") for k in checks):
        checks["disturbance_provenance"] = False
    report.update(checks={k:bool(v) for k,v in checks.items()},usable=bool(all(checks.values())),
                  reward_max_abs_difference=float(np.max(np.abs(rescored-rewards))),
                  historical_config_limitation="one-cycle controller replay plus complete schedule/scaling/reward audit")
    return report,result


def matching_reference(method,seed,hashes,path=None):
    """Only reuse explicit completed production jobs with matching provenance."""
    candidates = [Path(path)] if path else sorted((ROOT/"Polymer/Results/comparisons/scenario1").glob(f"*/{method}"),reverse=True)
    for candidate in candidates:
        if not all((candidate/name).is_file() for name in ("job.json","status.json","training.pkl")):
            continue
        job = json.loads((candidate/"job.json").read_text(encoding="utf-8"))
        status = json.loads((candidate/"status.json").read_text(encoding="utf-8"))
        if (job["method"] == method and job["seed"] == seed and not job["smoke"]
                and job["profile"] == "legacy_gradual_200" and job["source_hashes"] == hashes
                and status["state"] == "completed" and status["steps"] == 160000):
            if status.get("trajectory_sha256") != hashlib.sha256((candidate/"training.pkl").read_bytes()).hexdigest():
                continue
            return candidate/"training.pkl"
    if path:
        raise ValueError(f"{method} reference lacks matching completed Scenario1 provenance.")
    return None


def worker(job_path):
    import numpy as np
    import torch
    from comparison_studies.common import (write_json,source_hashes,prepare_nominal,episode_bundle,
                                          solver_audit,reference_physical)
    from comparison_studies.native import ls_run,arx_run,bo_run,combined_run,nominal_run
    from comparison_studies.analysis import metrics
    from comparison_studies.checkpoints import save_inference_checkpoint
    job_path = Path(job_path)
    job = json.loads(job_path.read_text(encoding="utf-8"))
    output = job_path.parent
    status = {"state":"running","started_at":now(),"pid":os.getpid(),"method":job["method"]}
    write_json(output/"status.json",status)
    try:
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Scientific/comparison sources changed after job preparation.")
        torch.set_num_threads(job["threads"])
        random.seed(job["seed"]);np.random.seed(job["seed"]);torch.manual_seed(job["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(job["seed"])
        namespace = prepare_nominal(job["seed"],output)
        full = episode_bundle(namespace,job["smoke"])
        write_json(output/"resolved_config.json",{
            "method":job["method"],"seed":job["seed"],"profile":"legacy_gradual_200",
            "cycles":200,"steps":full["nFE"],"smoke":job["smoke"],"native_defaults":namespace["NB"],
            "calibration":"BO: unchanged cycles11-12 after nominal prefix; threshold: nominal cycles1-10",
            "ARX":{"forgetting":.995,"initial_covariance":1000,"covariance_eigenvalue_bounds":[1e-8,1e8]}})
        threshold_value = None
        if job["method"] == "threshold":
            from comparison_studies.common import window_bundle
            from comparison_studies.threshold import calibrate
            size = full["time_in_sub_episodes"]
            warm,_ = nominal_run(namespace,window_bundle(full,0,10*size),timing=False)
            threshold_value = calibrate(warm["y"][:-1],reference_physical(warm),namespace["reward_params"])
            write_json(output/"threshold_calibration.json",{
                "threshold":threshold_value,"quantile":.95,"multiplier":1.25,"minimum":1.,
                "source":"nominal original cycles1-10; before-action measurements",
                "steps":10*size,"no_final_cycle_tuning":True,"reward_params":namespace["reward_params"]})
            # Calibration must not change the neural initialization RNG stream.
            random.seed(job["seed"]);np.random.seed(job["seed"]);torch.manual_seed(job["seed"])
            if torch.cuda.is_available():torch.cuda.manual_seed_all(job["seed"])
        began = time.perf_counter()
        agents = None
        with solver_audit() as audit:
            if job["method"] == "ls":
                result = ls_run(namespace,full)
            elif job["method"] == "arx":
                result = arx_run(namespace,full)
            elif job["method"] == "bo":
                result = bo_run(namespace,full,output,seed=job["seed"],smoke=job["smoke"])
            else:
                result,agents,actual_defaults = combined_run(output,seed=job["seed"],threshold=threshold_value,smoke=job["smoke"])
                write_json(output/"resolved_combined_defaults.json",actual_defaults)
        result = {k:v for k,v in result.items() if not k.startswith("_")}
        result.update(comparison_method=job["method"],comparison_seed=job["seed"],comparison_smoke=job["smoke"],
                      comparison_source_hashes=job["source_hashes"],comparison_solver_audit=np.asarray(audit),
                      comparison_worker_seconds=time.perf_counter()-began)
        if result["nFE"] != (800 if job["smoke"] else 160000):
            raise RuntimeError("Incorrect Scenario1 control-step count.")
        # Export raw data before the checkpoint and integrity checks. A failed
        # status still prevents this unvalidated trajectory being reused.
        with (output/"training.pkl").open("xb") as stream:
            pickle.dump(result,stream,protocol=pickle.HIGHEST_PROTOCOL)
        status.update(trajectory_saved=True,
                      trajectory_sha256=hashlib.sha256((output/"training.pkl").read_bytes()).hexdigest())
        if agents is not None:
            expected_agents = {"horizon_agent", "weight_agent", "markov_agent", "residual_agent"}
            if set(agents) != expected_agents:
                raise RuntimeError(f"Incomplete combined policy export: {sorted(agents)}")
            checkpoint = save_inference_checkpoint(output/"final_inference_networks.pt", agents,
                metadata={"method":job["method"],"seed":job["seed"],"profile":job["profile"],
                          "smoke":job["smoke"],"source_hashes":job["source_hashes"],
                          "configuration":"resolved_combined_defaults.json"})
            status["checkpoint"] = checkpoint
        if source_hashes() != job["source_hashes"]:
            raise RuntimeError("Sources changed during the run; raw trajectory retained for diagnosis.")
        if not job["smoke"]:
            write_json(output/"metrics.json",metrics(result))
        write_json(output/"timing.json",result.get("comparison_timing"))
        status.update(state="completed",finished_at=now(),python=sys.executable,steps=int(result["nFE"]),
                      optimizer_calls=len(audit),optimizer_failed_calls=sum(not a[1] for a in audit),
                      duration_seconds=result["comparison_worker_seconds"],
                      numpy=np.__version__,torch=torch.__version__,
                      device="cuda" if job["method"] in ("sg","threshold") and torch.cuda.is_available() else "cpu",
                      trajectory_sha256=hashlib.sha256((output/"training.pkl").read_bytes()).hexdigest())
        write_json(output/"status.json",status)
        return 0
    except BaseException as error:
        status.update(state="failed",finished_at=now(),error=str(error),error_type=type(error).__name__)
        write_json(output/"status.json",status)
        traceback.print_exc()
        return 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods",nargs="+",choices=["ls","arx","bo","threshold","sg"],default=["ls","arx","bo","threshold"])
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--threads",type=int,default=1)
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--sg-reference",type=Path,help="Verified completed comparison SG job directory to reuse.")
    p.add_argument("--ls-reference",type=Path,help="Verified completed comparison LS job directory to reuse.")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--smoke",action="store_true")
    p.add_argument("--audit-baseline",action="store_true")
    p.add_argument("--analyse",type=Path)
    p.add_argument("--worker-job",type=Path,help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_job:
        return worker(args.worker_job)
    from comparison_studies.common import write_json,source_hashes
    if args.analyse:
        from comparison_studies.analysis import summarize
        print(f"Summarized {len(summarize(args.analyse))} method/window rows.")
        return 0
    if args.seed < 0 or args.threads < 1 or len(set(args.methods)) != len(args.methods):
        raise ValueError("Require a nonnegative seed, positive thread count and unique methods.")
    methods = list(args.methods)
    hashes = source_hashes()
    references = {}
    for method,path in (("ls",args.ls_reference),("sg",args.sg_reference)):
        found = matching_reference(method,args.seed,hashes,path=path) if not args.smoke else None
        if found:
            references[method] = str(found)
            if method in methods:
                methods.remove(method)
    if "threshold" in methods and "sg" not in methods and "sg" not in references:
        methods.append("sg")
    if args.dry_run:
        print(json.dumps({"methods":methods,"seed":args.seed,"profile":"legacy_gradual_200","cycles":200,
                          "control_steps_per_main_run":160000,"warm_cycles":10,"critic_only_cycles":[11,12,13],
                          "first_policy_cycle":14,"final_evaluation_cycle":200,"BO_trials":40,
                          "BO_calibration_steps":64000,"python":args.python,"original_sources":"read-only", "reused_references":references},indent=2))
        return 0
    output = (args.output_dir or ROOT/"Polymer/Results/comparisons/scenario1"/
              (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])).resolve()
    output.mkdir(parents=True,exist_ok=False)
    manifest = {"started_at":now(),"methods":methods,"seed":args.seed,"smoke":args.smoke,
                "profile":"legacy_gradual_200","source_hashes":source_hashes(),"python":args.python,
                "sg_reference":str(args.sg_reference) if args.sg_reference else None,
                "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                "git_dirty":subprocess.check_output(["git","status","--short"],cwd=ROOT,text=True)}
    write_json(output/"manifest.json",manifest)
    print(f"Comparison output: {output}",flush=True)
    if args.audit_baseline or not args.smoke:
        from comparison_studies.common import prepare_nominal,episode_bundle
        namespace = prepare_nominal(args.seed,output)
        report,_ = baseline_audit(namespace,episode_bundle(namespace))
        write_json(output/"baseline_audit.json",report)
        if report["usable"]:
            references["offsetfree"] = report["path"]
        if args.audit_baseline:
            print(json.dumps(report,indent=2))
            return int(not report["usable"])
        print(f"Existing nominal MPC baseline verified: {report['usable']}",flush=True)
    write_json(output/"references.json",references)
    for method in methods:
        job_dir = output/method
        job_dir.mkdir()
        job = {**manifest,"method":method,"threads":args.threads}
        write_json(job_dir/"job.json",job)
        env = os.environ.copy()
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            env[name] = str(args.threads)
        env["MPLBACKEND"] = "Agg"
        print(f"Starting {method}",flush=True)
        with (job_dir/"run.log").open("w",encoding="utf-8") as stream:
            completed = subprocess.run([args.python,"-u",str(Path(__file__).resolve()),"--worker-job",str(job_dir/"job.json")],
                                       cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:
            write_json(output/"status.json",{"state":"failed","method":method,"finished_at":now()})
            print(f"Failed {method}; inspect {job_dir/'run.log'}",flush=True)
            return completed.returncode
        print(f"Completed {method}",flush=True)
    if not args.smoke:
        from comparison_studies.analysis import summarize
        summarize(output)
    write_json(output/"status.json",{"state":"completed","finished_at":now(),"methods":methods,"smoke":args.smoke})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
