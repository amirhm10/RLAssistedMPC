"""Invoke original closed loops with isolated controller adapters."""
from __future__ import annotations
import ast
from contextlib import ExitStack
from copy import deepcopy
import inspect
import textwrap
import numpy as np

from .common import PlantTiming, baseline_arguments, configure, episode_bundle, reference_physical, window_bundle


def baseline_with_initial_state(cfg, ctx):
    """Private copy adds only optional calibration observer initial conditions."""
    from utils.mpc_baseline_runner import run_offsetfree_mpc
    source = textwrap.dedent(inspect.getsource(run_offsetfree_mpc))
    root = ast.parse(source)
    class Initialize(ast.NodeTransformer):
        changed = 0
        def visit_Assign(self, node):
            if any(isinstance(t,ast.Name) and t.id == "xhatdhat" for t in node.targets):
                self.changed += 1
                extra = ast.parse("if 'comparison_initial_xhat' in runtime_ctx:\n    xhatdhat[:,0] = runtime_ctx['comparison_initial_xhat']").body[0]
                return [node, ast.copy_location(extra,node)]
            return node
    transformer = Initialize()
    root = transformer.visit(root)
    if transformer.changed != 1:
        raise RuntimeError("Original baseline observer initialization contract changed.")
    namespace = dict(run_offsetfree_mpc.__globals__)
    exec(compile(ast.fix_missing_locations(root),inspect.getsourcefile(run_offsetfree_mpc),"exec"),namespace)
    return namespace["run_offsetfree_mpc"](cfg,ctx)


def nominal_run(namespace, bundle, *, alpha=1., beta=1., system=None, initial_xhat=None, timing=True):
    model = deepcopy(namespace["MPC_obj"])
    model.Q_out = np.array([5.*alpha,1.])
    model.R_in = np.array([beta,beta])
    cfg, ctx = baseline_arguments(namespace,bundle,system=system,model=model,initial_xhat=initial_xhat)
    cfg.update(Q1_penalty=5.*alpha,Q2_penalty=1.,R1_penalty=beta,R2_penalty=beta)
    clock = PlantTiming()
    if timing:
        clock.attach(ctx["system"])
    result = baseline_with_initial_state(cfg,ctx)
    result["comparison_timing"] = clock.summary() if timing else None
    result["comparison_fixed_weights"] = {"alpha":alpha,"beta":beta,"Q":model.Q_out,"R":model.R_in}
    return result,ctx["system"]


def arx_run(namespace, bundle):
    from .arx import ARXMPC, IncrementalARX, nominal_coefficients
    from utils.helpers import apply_min_max
    data = namespace["system_data"]
    plant = namespace["runtime_ctx"]["system_factory"]()
    lo, hi = namespace["data_min"], namespace["data_max"]
    u_ss = apply_min_max(namespace["steady_states"]["ss_inputs"],lo[:2],hi[:2])
    y_ss = apply_min_max(namespace["steady_states"]["y_ss"],lo[2:],hi[2:])
    theta = nominal_coefficients(data["A"],data["B"],data["C"])
    predictor = IncrementalARX(theta,apply_min_max(plant.current_output,lo[2:],hi[2:])-y_ss,
                              apply_min_max(plant.current_input,lo[:2],hi[:2])-u_ss)
    model = ARXMPC(deepcopy(namespace["MPC_obj"]),predictor)
    cfg,ctx = baseline_arguments(namespace,bundle,system=plant,model=model)
    def observe(plant):
        predictor.observe(apply_min_max(plant.current_output,lo[2:],hi[2:])-y_ss,
                          apply_min_max(plant.current_input,lo[:2],hi[:2])-u_ss)
    clock = PlantTiming()
    clock.attach(plant,observe)
    result = baseline_with_initial_state(cfg,ctx)
    result.update(comparison_arx_initial_theta=theta,comparison_arx_theta=np.asarray(predictor.coefficient_log),
                  comparison_arx_update_accepted=np.asarray(predictor.update_log),
                  comparison_arx_forgetting=predictor.forgetting,comparison_timing=clock.summary(),
                  comparison_arx_state="measured incremental history; nominal observer is diagnostic only")
    return result


def ls_run(namespace, bundle):
    from utils.markov_runner import run_markov_correction_supervisor
    cfg,ctx = deepcopy(namespace["markov_cfg"]),dict(namespace["runtime_ctx"])
    cfg["run_rl_proposal"] = False
    cfg["rl_save_agent_checkpoint"] = False
    ctx["episode_bundle"] = bundle
    clock = PlantTiming()
    factory = ctx["system_factory"]
    ctx["system_factory"] = lambda: clock.attach(factory())
    result = run_markov_correction_supervisor(cfg,ctx)
    result["comparison_timing"] = clock.summary()
    if result.get("_rl_agent") is not None:
        raise RuntimeError("LS-only unexpectedly instantiated an RL agent.")
    result["comparison_native_agent_kind_setting"] = cfg["agent_kind"]
    result["agent_kind"] = "none"
    result["supervisor_gated_td3_enabled"] = False
    result["supervisor_gated_algorithm"] = None
    return result


def bo_run(namespace, full, output, *, seed=7, smoke=False):
    from .bayes import tune
    from .common import solver_audit,write_json
    size = full["time_in_sub_episodes"]
    prefix,plant = nominal_run(namespace,window_bundle(full,0,10*size),timing=False)
    initial_xhat = prefix["xhatdhat"][:,-1].copy()
    calibration = window_bundle(full,10*size,12*size)
    def evaluate(alpha,beta):
        with solver_audit() as audit:
            result,_ = nominal_run(namespace,calibration,alpha=alpha,beta=beta,
                                 system=deepcopy(plant),initial_xhat=initial_xhat)
        failures = sum(not row[1] for row in audit)
        cost = -float(np.mean(result["rewards_step"]))
        return cost,{"valid":failures == 0 and np.all(np.isfinite(result["y"])),
                     "solver_failures":int(failures),"steps":int(result["nFE"])}
    best,records = tune(evaluate,seed=seed,trials=5 if smoke else 40,
                       callback=lambda records:write_json(output/"bo_trials.json",records))
    result,_ = nominal_run(namespace,full,alpha=best["alpha"],beta=best["beta"])
    result.update(comparison_bo_trials=records,comparison_bo_best=best,
                  comparison_bo_calibration={"original_cycles":[11,12],"steps_per_trial":2*size,
                                             "prefix_steps":10*size,"main_initial_condition_reset":True})
    return result


def combined_run(output, *, seed=7, threshold=None, smoke=False):
    from run_polymer_e1 import execute_entrypoint
    from review_experiments.e1.runtime import install_runner
    from .common import smoke_runner
    from .threshold import ThresholdSession,acceptance_override
    import Simulation.system_functions as plants
    sessions,context,clock = [],{},PlantTiming()
    original_plant = plants.PolymerCSTR
    def factory(*args,**kwargs):
        return clock.attach(original_plant(*args,**kwargs))
    plants.PolymerCSTR = factory
    def session_factory(compiled,args,kwargs):
        session = ThresholdSession(compiled,context,threshold=threshold)
        sessions.append(session)
        return session
    try:
        with ExitStack() as stack:
            if smoke:
                stack.enter_context(smoke_runner("combined"))
            if threshold is not None:
                stack.enter_context(acceptance_override(context))
            stack.enter_context(install_runner("combined",session_factory))
            ns = execute_entrypoint("combined",lambda defaults:configure(defaults,"sg",seed,output))
    finally:
        plants.PolymerCSTR = original_plant
    result = ns["result_bundle"]
    result["comparison_timing"] = clock.summary()
    result["comparison_threshold"] = threshold
    result["comparison_threshold_decisions"] = sessions[0].records
    return result,sessions[0].agents,ns["NB"]
