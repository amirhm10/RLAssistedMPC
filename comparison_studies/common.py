"""Reuse authoritative setup and record explicit comparison provenance."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "legacy_gradual_200"
METHODS = ("ls", "arx", "bo", "threshold", "sg")


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    from run_polymer_e1 import source_hashes as scientific_hashes
    result = scientific_hashes()
    paths = list((ROOT / "comparison_studies").glob("*.py")) + [ROOT / "run_polymer_comparisons.py"]
    for path in paths:
        result[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def configure(defaults, method, seed, output):
    from run_polymer_e1 import SEED_KEYS
    value = deepcopy(defaults)
    value.update(run_mode="disturb", training_profile_override=PROFILE, n_tests_override=200,
                 results_dir_override=str(output), result_prefix_override="comparison", compare_prefix_override="comparison")
    if method in ("threshold", "sg"):
        value["combined_agent_mode"] = "sg"
    else:
        value["agent_kind"] = "sg_td3"
        value["controller"]["run_rl_proposal"] = False
        value["controller"]["rl_save_agent_checkpoint"] = False
    for key in SEED_KEYS:
        if isinstance(value.get(key), dict):
            value[key]["seed"] = seed
    return value


def prepare_nominal(seed=7, output=None):
    """Execute the unchanged Markov entrypoint up to, but excluding, its run."""
    import systems.polymer as polymer
    path = ROOT / "RL_assisted_MPC_markov_unified.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    end = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "result_bundle" for t in node.targets))
    original = polymer.get_polymer_notebook_defaults
    polymer.get_polymer_notebook_defaults = lambda family: configure(original(family), "ls", seed, output or ROOT / "Polymer/Results/comparisons")
    namespace = {"__name__": "__comparison_setup__", "__file__": str(path)}
    try:
        exec(compile(ast.Module(body=tree.body[:end], type_ignores=[]), str(path), "exec"), namespace)
    finally:
        polymer.get_polymer_notebook_defaults = original
    return namespace


def episode_bundle(namespace, smoke=False):
    from systems.polymer.scenarios import build_polymer_training_profile
    cfg = namespace["markov_cfg"]
    result = build_polymer_training_profile(
        profile_name=PROFILE, y_sp_scenario=namespace["y_sp_scenario"], n_tests=200,
        set_points_len=400, warm_start=10, test_cycle=list(cfg["test_cycle"]),
        **{k: float(cfg[k]) for k in ("nominal_qi", "nominal_qs", "nominal_ha", "qi_change", "qs_change", "ha_change")},
        steady_outputs=namespace["steady_states"]["y_ss"], data_min=namespace["data_min"],
        data_max=namespace["data_max"], n_inputs=2)
    if smoke:
        result = compress_bundle(result)
    return result


def compress_bundle(original, stride=200):
    """Technical clock only. Never regenerate or compress a production ramp."""
    value = deepcopy(original)
    n = int(value["nFE"])
    for key, item in list(value.items()):
        if isinstance(item, np.ndarray) and item.shape and item.shape[0] == n:
            value[key] = item[::stride].copy()
    # The linear schedules are built from these same sliced arrays.
    value.pop("disturbance_schedule", None)
    for key in ("sub_episodes_changes_dict", "sub_episode_changes_dict"):
        value[key] = {(int(k) + 1) // stride - 1: v for k, v in value[key].items()}
    value["test_train_dict"] = {int(k) // stride: v for k, v in value["test_train_dict"].items()}
    for key in ("nFE", "time_in_sub_episodes", "warm_start_step", "phase_switch_step", "exploration_freeze_step"):
        if value.get(key) is not None:
            value[key] = int(value[key]) // stride
    for window in value.get("experiment_phase_windows", []):
        for key in ("step_start", "step_end_exclusive"):
            if window.get(key) is not None:
                window[key] = int(window[key]) // stride
    value["comparison_smoke_stride"] = stride
    return value


@contextmanager
def smoke_runner(category):
    """Scenario1 technical clock, including its explicitly absent phase switch."""
    import importlib
    from review_experiments.e1.runtime import RUNNER_TARGETS
    from utils.episode_profiles import validate_episode_bundle
    module = importlib.import_module(RUNNER_TARGETS[category][0])
    original = module.resolve_episode_bundle
    def resolve(*args, **kwargs):
        return validate_episode_bundle(compress_bundle(original(*args, **kwargs)),
                                       expected_n_tests=kwargs["expected_n_tests"],
                                       expected_n_outputs=kwargs["expected_n_outputs"])
    module.resolve_episode_bundle = resolve
    try:
        yield
    finally:
        module.resolve_episode_bundle = original


def window_bundle(full, start, stop):
    """Slice the ORIGINAL full-scenario arrays; do not regenerate shorter ramps."""
    value = deepcopy(full)
    size = int(full["time_in_sub_episodes"])
    if start % size or stop % size or not 0 <= start < stop <= full["nFE"]:
        raise ValueError("Calibration windows must contain complete original cycles.")
    for key, item in list(value.items()):
        if isinstance(item, np.ndarray) and item.shape and item.shape[0] == full["nFE"]:
            value[key] = item[start:stop].copy()
    value.pop("disturbance_schedule", None)
    value["nFE"] = stop - start
    value["warm_start_step"] = 0
    ends = {k: k // size + 1 for k in range(size - 1, stop - start, size)}
    value["sub_episodes_changes_dict"] = ends
    value["sub_episode_changes_dict"] = ends.copy()
    value["test_train_dict"] = {k: False for k in range(0, stop - start, size)}
    value["comparison_original_step_start"] = start
    return value


def baseline_arguments(namespace, bundle, *, system=None, model=None, initial_xhat=None):
    cfg = deepcopy(namespace["markov_cfg"])
    cfg.update(n_tests=bundle["nFE"] // bundle["time_in_sub_episodes"],
               set_points_len=bundle["time_in_sub_episodes"] // 2,
               warm_start=10 if bundle["nFE"] == 160000 else 0)
    ctx = dict(namespace["runtime_ctx"])
    ctx.update(system=system or ctx["system_factory"](), MPC_obj=model or deepcopy(namespace["MPC_obj"]),
               episode_bundle=deepcopy(bundle), u_lower_phys=namespace["SYS"]["input_bounds"]["u_min"],
               u_upper_phys=namespace["SYS"]["input_bounds"]["u_max"],
               observer_update_mode=cfg["observer_update_alignment"])
    if initial_xhat is not None:
        ctx["comparison_initial_xhat"] = np.asarray(initial_xhat, float).copy()
    return cfg, ctx


def reference_physical(bundle):
    lo, hi = np.asarray(bundle["data_min"])[2:], np.asarray(bundle["data_max"])[2:]
    ss = (np.asarray(bundle["steady_states"]["y_ss"]) - lo) / (hi - lo)
    return (np.asarray(bundle["y_sp"]) + ss) * (hi - lo) + lo


class PlantTiming:
    """Time inter-step controller work separately from nonlinear integration.

    First interval contains setup and is excluded from timing summaries. Replay,
    reward and learning after a step are attributed to the next interval.
    """
    def __init__(self):
        self.control_seconds = []
        self.plant_seconds = []
        self.last_return = time.perf_counter()

    def attach(self, plant, callback=None):
        original = plant.step
        owner = self
        def timed(self, *args, **kwargs):
            began = time.perf_counter()
            owner.control_seconds.append(began - owner.last_return)
            result = original(*args, **kwargs)
            integration_finished = time.perf_counter()
            owner.plant_seconds.append(integration_finished - began)
            owner.last_return = integration_finished
            if callback is not None:
                callback(self)
            return result
        plant.step = types.MethodType(timed, plant)
        return plant

    def summary(self):
        a = np.asarray(self.control_seconds[1:], float)
        return {"intervals": int(a.size), "controller_mean_ms": float(a.mean() * 1000) if a.size else None,
                "controller_p95_ms": float(np.quantile(a, .95) * 1000) if a.size else None,
                "plant_total_seconds": float(sum(self.plant_seconds)),
                "definition": "inter-plant-step work excluding integration; first/setup interval excluded"}


@contextmanager
def solver_audit():
    import scipy.optimize as spo
    original = spo.minimize
    records = []
    def minimize(*args, **kwargs):
        began = time.perf_counter()
        result = original(*args, **kwargs)
        records.append((time.perf_counter() - began, bool(result.success), int(result.status)))
        return result
    spo.minimize = minimize
    try:
        yield records
    finally:
        spo.minimize = original
