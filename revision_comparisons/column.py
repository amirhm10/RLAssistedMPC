"""Column comparators built around the unchanged distillation entrypoints.

No COM session is opened on import or while printing a plan. Live execution
uses the same per-case locks as the active column repeat/E3 campaigns.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import hashlib
from pathlib import Path
import time
import pickle

import numpy as np

from comparison_studies.common import PlantTiming, reference_physical, window_bundle
from review_experiments.e3.common import ROOT, SEED_KEYS, source_hashes as e3_hashes
from .runtime import seed_all

PROFILE = "legacy_200"
METHODS = ("ls", "bo", "threshold", "sg", "nominal")


def calibration_window(full, start, stop):
    value = window_bundle(full, start, stop)
    # The shared polymer helper intentionally drops its derived disturbance
    # schedule. Column feed flow is independent data and must be sliced too.
    value["disturbance_schedule"] = np.asarray(full["disturbance_schedule"])[start:stop].copy()
    return value


def source_hashes():
    from run_polymer_e1 import source_hashes as e1_hashes
    result = e3_hashes()
    result.update(e1_hashes())
    paths = list((ROOT/"comparison_studies").glob("*.py"))
    # Progress viewers and ARX diagnostics are deliberately independent of this
    # execution dependency list, so adding diagnostics cannot invalidate a run.
    paths += [ROOT/"revision_comparisons"/"column.py", ROOT/"revision_comparisons"/"__init__.py",
              ROOT/"revision_comparisons"/"runtime.py",
              ROOT/"run_distillation_comparisons.py",
              ROOT/"Distillation/Data/mpc_results_disturb_fluctuation.pickle"]
    for path in paths:
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def configure(defaults, method, seed, output, aspen):
    value = deepcopy(defaults)
    value.update(run_mode="disturb", disturbance_profile="fluctuation",
                 training_profile_override=PROFILE, n_tests_override=200,
                 set_points_len_override=200, warm_start_override=10,
                 results_dir_override=str(output), result_prefix_override="column_comparison",
                 compare_prefix_override="column_comparison",
                 aspen_path_override=aspen["aspen_path"], snaps_path_override=aspen["snaps_path"])
    if method in ("sg", "threshold"):
        value["combined_agent_mode"] = "sg"
    else:
        value["agent_mode"] = "sg"
        value["controller"]["run_rl_proposal"] = False
        value["controller"]["rl_save_agent_checkpoint"] = False
    for key in SEED_KEYS:
        if isinstance(value.get(key), dict):
            value[key]["seed"] = seed
    return value


def audit_baseline(namespace, output):
    """Audit/reuse the saved Scenario 1 MPC bundle; never regenerate it."""
    from comparison_studies.analysis import trajectories
    from comparison_studies.common import write_json
    from review_experiments.e3.common import sha256
    from systems.distillation.data_io import canonical_baseline_path
    path = canonical_baseline_path(ROOT, "disturb", "fluctuation", training_profile_name=PROFILE)
    with path.open("rb") as stream:
        saved = pickle.load(stream)
    y, u, rewards, reference = trajectories(saved)
    full, cfg = namespace["EPISODE_BUNDLE"], namespace["markov_cfg"]
    if saved["nFE"] != 80000 or saved["time_in_sub_episodes"] != 400:
        raise ValueError("The saved column MPC reference is not the original 200-cycle Scenario 1.")
    if tuple(saved["mpc_horizons"]) != (cfg["predict_h"], cfg["cont_h"]):
        raise ValueError("Existing MPC reference has different fixed horizons.")
    if saved.get("observer_update_mode", "legacy_previous_measurement") != cfg["observer_update_alignment"]:
        raise ValueError("Existing MPC reference uses different observer alignment.")
    for key in ("data_min", "data_max"):
        np.testing.assert_allclose(saved[key], namespace[key], rtol=0, atol=1e-12,
                                   err_msg="MPC reference scaling differs.")
    expected_reference = dict(saved, y_sp=full["y_sp"], steady_states=namespace["steady_states"])
    np.testing.assert_allclose(reference, reference_physical(expected_reference), rtol=0, atol=1e-7,
                               err_msg="MPC reference setpoints differ.")
    profile = saved["disturbance_profile"]
    if len(profile) != 1:
        raise ValueError("Expected exactly the original feed-flow disturbance.")
    np.testing.assert_allclose(next(iter(profile.values())), full["disturbance_schedule"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(saved["delta_t"], namespace["system"].delta_t, rtol=0, atol=1e-12)
    bounds = namespace["SYS"]["input_bounds"]
    if np.any(u < np.asarray(bounds["u_min"])-1e-6) or np.any(u > np.asarray(bounds["u_max"])+1e-6):
        raise ValueError("Saved reference input units/bounds are inconsistent.")
    ranges = np.asarray(saved["data_max"])-np.asarray(saved["data_min"])
    prior = np.vstack([saved["steady_states"]["ss_inputs"], u[:-1]])
    dy, du = (y[1:]-reference)/ranges[2:], (u-prior)/ranges[:2]
    recomputed = np.array([namespace["reward_fn"](a, b, r) for a, b, r in zip(dy, du, reference)])
    np.testing.assert_allclose(dy, saved["delta_y_storage"], rtol=1e-8, atol=1e-8,
                               err_msg="Saved reference tracking-error alignment differs.")
    np.testing.assert_allclose(du, saved["delta_u_storage"], rtol=1e-8, atol=1e-8,
                               err_msg="Saved reference input movement/units differ.")
    difference = float(np.max(np.abs(recomputed-rewards)))
    # Canonical cache predates the active reward parameters. Re-score its
    # verified trajectory with exactly the new arms' fixed reward, in sidecars
    # only. The cached simulation and its original reward remain untouched.
    rescored = dict(saved, rewards_step=recomputed, reward_params=namespace["reward_params"])
    write_json(output/"baseline_audit.json", {
        "state": "verified_configuration_schedule_scaling_and_saved_reward", "source": str(path),
        "source_sha256": sha256(path), "steps": len(rewards),
        "reward_max_abs_difference_from_cached_values": difference,
        "reference_metric_reward": "recomputed with the unchanged active column reward for all arms",
        "cached_reward_matches_active_reward": bool(np.allclose(recomputed, rewards, rtol=1e-8, atol=1e-8)),
        "reward_params": namespace["reward_params"],
        "simulation_replayed": False,
        "scope": "saved publication baseline; no same-environment bitwise trajectory reproduction claim"})
    write_json(output/"baseline_metrics.json", metrics(rescored))


def prepare(seed, output, aspen):
    """Original Markov setup, stopping immediately before its closed-loop call.

    Call only inside aspen_resources, which owns and closes the resulting case.
    """
    import systems.distillation as distillation
    path = ROOT/"distillation_RL_assisted_MPC_markov_unified.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "result_bundle" for t in node.targets))
    original = distillation.get_distillation_notebook_defaults
    distillation.get_distillation_notebook_defaults = lambda family: configure(original(family), "ls", seed, output, aspen)
    namespace = {"__name__": "__column_comparison_setup__", "__file__": str(path)}
    try:
        exec(compile(ast.Module(body=tree.body[:end], type_ignores=[]), str(path), "exec"), namespace)
    finally:
        distillation.get_distillation_notebook_defaults = original
    bundle = namespace["EPISODE_BUNDLE"]
    if bundle["nFE"] != 80000 or bundle["time_in_sub_episodes"] != 400:
        raise RuntimeError("Column Scenario 1 must have 200 original 400-step cycles.")
    if not (Path(output)/"baseline_audit.json").exists():
        audit_baseline(namespace, output)
    return namespace


def baseline(namespace, bundle=None, *, alpha=1., beta=1.):
    from utils.mpc_baseline_runner import run_offsetfree_mpc
    cfg, ctx = deepcopy(namespace["markov_cfg"]), dict(namespace["runtime_ctx"])
    bundle = deepcopy(bundle if bundle is not None else namespace["EPISODE_BUNDLE"])
    model = deepcopy(namespace["MPC_obj"])
    q0, r0 = np.asarray(model.Q_out, float).copy(), np.asarray(model.R_in, float).copy()
    model.Q_out = q0 * np.array([alpha, 1.])
    model.R_in = r0 * beta
    cfg.update(n_tests=int(bundle["nFE"]//bundle["time_in_sub_episodes"]),
               set_points_len=int(bundle["time_in_sub_episodes"]//2),
               Q1_penalty=float(model.Q_out[0]), Q2_penalty=float(model.Q_out[1]),
               R1_penalty=float(model.R_in[0]), R2_penalty=float(model.R_in[1]))
    # Match the column's own baseline alignment, bounds and SLSQP settings.
    ctx.update(episode_bundle=bundle, MPC_obj=model, disturbance_schedule=bundle["disturbance_schedule"],
               u_lower_phys=namespace["SYS"]["input_bounds"]["u_min"],
               u_upper_phys=namespace["SYS"]["input_bounds"]["u_max"],
               observer_update_mode=cfg["observer_update_alignment"])
    clock = PlantTiming()
    clock.attach(ctx["system"])
    result = run_offsetfree_mpc(cfg, ctx)
    result["comparison_timing"] = clock.summary()
    result["comparison_fixed_weights"] = {"alpha": alpha, "beta": beta,
                                           "Q": model.Q_out, "R": model.R_in, "Q0": q0, "R0": r0}
    result["reward_params"] = namespace["reward_params"]
    return result


@contextmanager
def timed_columns(clock):
    """Attach timing to the owned case; no simulator files or shared code edits."""
    import systems.distillation.plant as plant
    cls = plant.DistillationColumnAspen
    original = cls.__init__
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        clock.attach(self)
    cls.__init__ = initialize
    try:
        yield
    finally:
        cls.__init__ = original


def run_method(method, output, aspen, seed, *, bo_trials=40):
    """A single main controller, with explicit calibration on separate resets."""
    from comparison_studies.bayes import tune
    from comparison_studies.common import solver_audit, write_json
    from comparison_studies.threshold import calibrate, ThresholdSession, acceptance_override
    from review_experiments.e1.runtime import install_runner
    from review_experiments.e3.runtime import aspen_resources, execute_entrypoint
    seed_all(seed)
    agents = None
    if method in ("ls", "nominal"):
        with aspen_resources(aspen):
            namespace = prepare(seed, output, aspen)
            write_json(output/"resolved_defaults.json", namespace["NB"])
            if method == "ls":
                from utils.markov_runner import run_markov_correction_supervisor
                clock = PlantTiming()
                clock.attach(namespace["system"])
                result = run_markov_correction_supervisor(namespace["markov_cfg"], namespace["runtime_ctx"])
                result["comparison_timing"] = clock.summary()
                if result.get("_rl_agent") is not None:
                    raise RuntimeError("LS-only must not instantiate an RL agent.")
                result.update(agent_kind="none", supervisor_gated_td3_enabled=False)
                result["reward_params"] = namespace["reward_params"]
            else:
                result = baseline(namespace)
        return result, agents

    threshold = None
    if method == "threshold":
        with aspen_resources(aspen):
            namespace = prepare(seed, output, aspen)
            prefix = calibration_window(namespace["EPISODE_BUNDLE"], 0, 4000)
            warm = baseline(namespace, prefix)
            threshold = calibrate(warm["y"][:-1], reference_physical(warm), namespace["reward_params"])
            write_json(output/"threshold_calibration.json", {
                "threshold": threshold, "quantile": .95, "multiplier": 1.25, "minimum": 1.,
                "source": "nominal cycles 1--10, physical before-action tracking error",
                "steps": 4000, "reward_params": namespace["reward_params"], "final_evaluation_used": False})

    if method in ("threshold", "sg"):
        context, sessions, clock = {}, [], PlantTiming()
        def session_factory(compiled, args, kwargs):
            session = ThresholdSession(compiled, context, threshold=threshold)
            sessions.append(session)
            return session
        seed_all(seed)  # calibration cannot change neural initialization
        with ExitStack() as stack:
            stack.enter_context(aspen_resources(aspen))
            stack.enter_context(timed_columns(clock))
            if threshold is not None:
                stack.enter_context(acceptance_override(context))
            stack.enter_context(install_runner("combined", session_factory))
            namespace = execute_entrypoint("distillation", "combined",
                lambda defaults: configure(defaults, method, seed, output, aspen))
            result = namespace["result_bundle"]
            agents = sessions[0].agents
            write_json(output/"resolved_defaults.json", namespace["NB"])
        result.update(comparison_threshold=threshold, comparison_threshold_decisions=sessions[0].records,
                      comparison_timing=clock.summary())
        return result, agents

    if method != "bo":
        raise ValueError(f"Unknown column comparator: {method}")

    calibration_began = time.perf_counter()
    trial_count = 0
    def evaluate(alpha, beta):
        nonlocal trial_count
        trial_count += 1
        print(f"BO calibration trial {trial_count}/{bo_trials}: alpha={alpha:.6g}, beta={beta:.6g}", flush=True)
        # COM objects cannot be deep-copied. Reinitialize a new owned column,
        # replay the identical nominal prefix and then apply trial weights.
        # This deliberately avoids an unverified partial-state snapshot.
        with aspen_resources(aspen):
            namespace = prepare(seed, output, aspen)
            full = namespace["EPISODE_BUNDLE"]
            prefix = calibration_window(full, 0, 4000)
            prefix_result = baseline(namespace, prefix)
            trial = calibration_window(full, 4000, 4800)
            from comparison_studies.native import baseline_with_initial_state
            cfg, ctx = deepcopy(namespace["markov_cfg"]), dict(namespace["runtime_ctx"])
            model = deepcopy(namespace["MPC_obj"])
            model.Q_out = np.asarray(model.Q_out, float) * np.array([alpha, 1.])
            model.R_in = np.asarray(model.R_in, float) * beta
            cfg.update(n_tests=2, set_points_len=200,
                       Q1_penalty=float(model.Q_out[0]), Q2_penalty=float(model.Q_out[1]),
                       R1_penalty=float(model.R_in[0]), R2_penalty=float(model.R_in[1]))
            ctx.update(episode_bundle=trial, MPC_obj=model, disturbance_schedule=trial["disturbance_schedule"],
                       comparison_initial_xhat=prefix_result["xhatdhat"][:, -1].copy(),
                       u_lower_phys=namespace["SYS"]["input_bounds"]["u_min"],
                       u_upper_phys=namespace["SYS"]["input_bounds"]["u_max"],
                       observer_update_mode=cfg["observer_update_alignment"])
            with solver_audit() as audit:
                result = baseline_with_initial_state(cfg, ctx)
            failures = sum(not row[1] for row in audit)
            return -float(np.mean(result["rewards_step"])), {
                "valid": failures == 0 and np.all(np.isfinite(result["y"])),
                "solver_failures": int(failures), "trial_steps": 800, "replayed_prefix_steps": 4000}

    best, records = tune(evaluate, seed=seed, trials=bo_trials,
                         callback=lambda rows: write_json(output/"bo_trials.json", rows))
    calibration_seconds = time.perf_counter()-calibration_began
    with aspen_resources(aspen):
        namespace = prepare(seed, output, aspen)
        write_json(output/"resolved_defaults.json", namespace["NB"])
        result = baseline(namespace, alpha=best["alpha"], beta=best["beta"])
    result.update(comparison_bo_trials=records, comparison_bo_best=best,
                  comparison_bo_calibration={"objective_cycles": [11, 12], "trials": bo_trials,
                    "steps_per_trial": 800, "nominal_prefix_steps_per_trial": 4000,
                    "total_calibration_steps": bo_trials*4800, "seconds": calibration_seconds,
                    "reset": "new owned Aspen case plus replayed unchanged nominal prefix",
                    "main_initial_condition_reset": True})
    return result, agents


def metrics(result):
    """Physical column labels and after-action alignment, with the fixed reward."""
    from comparison_studies.analysis import trajectories, WINDOWS
    y, u, rewards, reference = trajectories(result)
    size, dt = int(result["time_in_sub_episodes"]), float(result["delta_t"])
    previous = np.vstack([result["steady_states"]["ss_inputs"], u[:-1]])
    params = result.get("reward_params")
    rows = []
    for label, (first, last) in WINDOWS.items():
        start, stop = (first-1)*size, last*size
        if stop > len(rewards):
            raise ValueError("A partial trajectory cannot enter production comparison metrics.")
        error = y[start+1:stop+1]-reference[start:stop]
        row = {"method": result.get("comparison_method", "nominal"), "window": label,
               "steps": stop-start, "mean_reward": float(rewards[start:stop].mean()),
               "ethane_composition_rmse_mole_fraction": float(np.sqrt(np.mean(error[:, 0]**2))),
               "tray85_temperature_rmse_K": float(np.sqrt(np.mean(error[:, 1]**2))),
               "ethane_composition_iae_mole_fraction_h": float(np.sum(np.abs(error[:, 0]))*dt),
               "tray85_temperature_iae_K_h": float(np.sum(np.abs(error[:, 1]))*dt),
               "reflux_total_abs_movement": float(np.sum(np.abs(u[start:stop, 0]-previous[start:stop, 0]))),
               "reboiler_total_abs_movement": float(np.sum(np.abs(u[start:stop, 1]-previous[start:stop, 1])))}
        if params is not None:
            bands = np.maximum(np.abs(reference[start:stop])*np.asarray(params["k_rel"]), params["band_floor_phys"])
            normalized = np.abs(error)/bands
            row.update(joint_band_fraction=float(np.mean(np.all(normalized <= 1, axis=1))),
                       mean_band_normalized_squared_error=float(np.mean(np.sum(normalized**2, axis=1))))
        rows.append(row)
    return rows
