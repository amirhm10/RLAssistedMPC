"""Decision sampling, paired rollout logging, and reusable snapshot export."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import io
from contextlib import redirect_stdout
from pathlib import Path
import time
import types

import numpy as np
import cloudpickle

from .runtime import (
    agents_in, clone_environment, execute_rollout, network_modes, rng_state, restore_rng,
    RestorationError,
)


WINDOWS = (
    ("release", 14, 20), ("prechange_mature", 191, 199),
    ("plant_change", 201, 210), ("adapted", 291, 299),
)


@dataclass(frozen=True)
class StudyOptions:
    seed: int = 7
    decisions_per_window: int = 100
    rollout_steps: int = 200
    horizons: tuple[int, ...] = (20, 80, 200)
    long_rollout_steps: int = 800
    long_fraction: float = 0.1
    snapshots_per_window: int = 1
    support_samples: int = 4096
    continuation: str = "gated"
    practical_tolerance: float = 0.0
    smoke: bool = False

    def validate(self):
        if self.decisions_per_window < 1 or self.rollout_steps < 1:
            raise ValueError("Decision count and rollout length must be positive.")
        if not self.horizons or min(self.horizons) < 1 or max(self.horizons) > self.rollout_steps:
            raise ValueError("Report horizons must lie within the primary rollout.")
        if not 0 <= self.long_fraction <= 1 or self.long_rollout_steps < self.rollout_steps:
            raise ValueError("Invalid long-rollout settings.")
        if self.snapshots_per_window < 0 or self.support_samples < 1:
            raise ValueError("Invalid snapshot/support sampling settings.")
        if self.practical_tolerance < 0:
            raise ValueError("Practical return tolerance cannot be negative.")
        if self.continuation not in {"gated", "policy", "supervisor"}:
            raise ValueError("Unknown continuation rule.")


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(child) for child in value]
    return value


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def independent_rng(seed, *labels):
    digest = hashlib.sha256(":".join(map(str, (seed,) + labels)).encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def replay_support(agent, state, decision, limit, seed, label):
    """Distances to executed-action replay available at this exact decision.

    State coordinates are standardized using this buffer sample only. Continuous
    actions are already in [-1,1]; DQN action IDs are treated categorically.
    Distances are descriptive support diagnostics, not probabilities/confidence.
    """
    buffer = agent.buffer
    size = int(buffer.size)
    if size == 0:
        return {"buffer_size": 0, "sample_size": 0}
    generator = independent_rng(seed, "support", label)
    indices = generator.choice(size, min(size, limit), replace=False)
    states = np.asarray(buffer.states[indices], float)
    actions = np.asarray(buffer.actions[indices])
    mean = states.mean(axis=0)
    scale = np.maximum(states.std(axis=0), 1e-6)
    state_distance_squared = np.mean(((states - np.asarray(state)) / scale) ** 2, axis=1)
    result = {
        "buffer_size": size, "sample_size": int(len(indices)),
        "state_nearest_rms_standardized": float(np.sqrt(state_distance_squared.min())),
        "state_mean": mean, "state_scale": scale,
        "normalization": "state standard deviations from decision-time replay; actions in normalized coordinates",
    }
    for name, action in (("policy", decision.policy_action), ("supervisor", decision.supervisor_action)):
        if actions.ndim == 1:
            matching = actions == int(action)
            result[f"{name}_action_sample_count"] = int(matching.sum())
            result[f"{name}_joint_nearest"] = (
                float(np.sqrt(state_distance_squared[matching].min())) if matching.any() else None
            )
        else:
            action_distance_squared = np.mean((actions - np.asarray(action)) ** 2, axis=1)
            result[f"{name}_joint_nearest"] = float(np.sqrt(
                (state_distance_squared + action_distance_squared).min()
            ))
    if hasattr(buffer, "selected_sources"):
        counts = Counter(int(item) for item in buffer.selected_sources[indices])
        result["sample_source_counts"] = dict(counts)
    return result


def _state_dimension_record(environment, compiled):
    result = {}
    for name in ("warm_start_step", "post_warm_start_action_freeze_end_step", "decision_interval",
                 "post_warm_action_freeze_steps", "time_in_sub_episodes", "nFE"):
        if name in environment:
            result[name] = environment[name]
    if "ctx" in environment:
        result["markov_timing"] = {name: environment["ctx"].get(name) for name in (
            "warm_start_step", "time_in_sub_episodes", "nFE", "run_mode", "observer_alignment"
        )}
    for name in ("phase1", "horizon_phase1", "weight_phase1", "markov_phase1", "residual_phase1"):
        value = environment.get(name)
        if isinstance(value, dict):
            result[name] = {key: item for key, item in value.items()
                            if not isinstance(item, np.ndarray) and isinstance(item, (bool, int, float, str))}
    result["training_freeze"] = "none in the main trajectory; all network/replay updates disabled only in branches"
    profile = environment.get("episode_bundle", environment.get("ctx", {}).get("episode_bundle", {}))
    if isinstance(profile, dict):
        result["original_profile"] = {key: profile[key] for key in (
            "training_profile_name", "phase_switch_step", "phase_switch_episode", "exploration_freeze_step",
            "phase2_learning_enabled", "phase2_final_episode_evaluation_only"
        ) if key in profile}
    return result


def _trace_row(environment, step):
    system = environment["system"]
    history = environment.get("history")
    reward = float(history["rewards"][step]) if isinstance(history, dict) and "rewards" in history else float(environment["rewards"][step])
    row = {
        "reward": reward,
        "y": np.asarray(system.current_output, float).copy(),
        "u": np.asarray(system.current_input, float).copy(),
        "reference": np.asarray(environment["y_sp_phys"], float).copy(),
    }
    yhat = environment.get("yhat_next_pred")
    if yhat is not None:
        row["existing_observer_yhat_next_scaled_deviation"] = np.asarray(yhat, float).copy()
    if isinstance(history, dict):
        for name in ("prediction_error_nominal_log", "prediction_error_markov_log",
                     "s_pred_log", "fallback_log", "action_source_log"):
            if name in history:
                row[name] = float(history[name][step])
    for name in ("current_Hp", "current_Hc", "Hp", "Hc", "weight_mult", "z_exec", "delta_u_res_exec"):
        if name in environment:
            row[name] = deepcopy(environment[name])
    for name in ("executed_action_raw_log", "weight_executed_action_raw_log", "sg_executed_action_raw_log",
                 "horizon_action_source_log", "sg_selected_source_log", "markov_sg_selected_source_log",
                 "weight_sg_selected_source_log", "residual_sg_selected_source_log"):
        value = environment.get(name)
        if isinstance(value, np.ndarray):
            row[name] = deepcopy(value[step])
    for name in ("sol", "sol0"):
        solution = environment.get(name)
        if hasattr(solution, "success"):
            row[f"{name}_success"] = int(solution.success)
    return row


def run_branch(snapshot, compiled, site, record, source, steps, continuation):
    environment = clone_environment(
        snapshot, compiled, selected_agent=snapshot[site.agent_name], initial=record,
        initial_source=source, continuation=continuation,
    )
    previous_input = np.asarray(environment["system"].current_input, float).copy()
    start = int(environment[compiled.step_name])
    if start + steps > int(environment["nFE"]):
        raise ValueError("Rollout exceeds the original reference/disturbance schedule.")
    rows = []
    from .runtime import freeze_normalization
    freeze_normalization(environment, compiled)
    branch_stdout = io.StringIO()
    try:
        with redirect_stdout(branch_stdout):
            exec(site.suffix, environment, environment)
            rows.append(_trace_row(environment, start))
            for step in range(start + 1, start + steps):
                environment[compiled.step_name] = step
                exec(compiled.step_code, environment, environment)
                rows.append(_trace_row(environment, step))
        trace = {name: np.asarray([row[name] for row in rows])
                 for name in rows[0] if all(name in row for row in rows)}
        trace["previous_input"] = previous_input
        if not all(np.all(np.isfinite(trace[name])) for name in ("reward", "y", "u", "reference")):
            raise FloatingPointError("Nonfinite reward, output, input or reference in branch.")
        return {"state": "completed", "trace": trace, "completed_steps": len(rows),
                "controller_stdout": branch_stdout.getvalue()[:4000]}
    except (Exception, KeyboardInterrupt) as error:
        if isinstance(error, (KeyboardInterrupt, RestorationError)):
            raise
        trace = {name: np.asarray([row[name] for row in rows]) for name in (
            "reward", "y", "u", "reference"
        )} if rows else {}
        return {"state": "failed", "error_type": type(error).__name__, "error": str(error),
                "completed_steps": len(rows), "trace": trace, "controller_stdout": branch_stdout.getvalue()[:4000]}


def branch_metrics(trace, horizon, gamma):
    reward = trace["reward"][:horizon]
    error = trace["y"][:horizon] - trace["reference"][:horizon]
    inputs = trace["u"][:horizon]
    movement = np.diff(np.vstack([trace["previous_input"], inputs]), axis=0)
    return {
        "return": float(np.dot(np.power(gamma, np.arange(horizon)), reward)),
        "rmse_physical": np.sqrt(np.mean(error ** 2, axis=0)),
        "max_abs_error_physical": np.max(np.abs(error), axis=0),
        "sum_abs_input_movement_physical": np.sum(np.abs(movement), axis=0),
        "solver_failure_steps": int(np.sum(trace["sol_success"][:horizon] == 0)) if "sol_success" in trace else None,
    }


class StudySession:
    def __init__(self, compiled, output, options, *, records=None, source_hashes=None):
        options.validate()
        self.compiled, self.output, self.options = compiled, Path(output), options
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "traces").mkdir(exist_ok=True)
        self.counts = Counter()
        self.schedules = {}
        self.window_for_step = {}
        self.long_steps = set()
        self.snapshot_steps = set()
        self.active_key = None
        self.record = None
        self.snapshot = None
        self.method_restore = None
        self.all_records = records if records is not None else []
        self.started = time.perf_counter()
        self.source_hashes = source_hashes or {}

    def _initialize(self, environment):
        if self.schedules:
            return
        substeps = int(environment.get("time_in_sub_episodes", environment.get("ctx", {}).get("time_in_sub_episodes", 0)))
        nfe = int(environment.get("nFE", 0))
        if substeps <= 0 or nfe <= 0:
            raise ValueError("Missing original episode clock in E1 capture.")
        warm = int(environment.get("warm_start_step", environment.get("ctx", {}).get("warm_start_step", 0)))
        freeze_end = int(environment.get("post_warm_start_action_freeze_end_step", warm))
        for name in ("phase1", "horizon_phase1", "weight_phase1", "markov_phase1", "residual_phase1"):
            phase = environment.get(name)
            if isinstance(phase, dict):
                freeze_end = max(freeze_end, int(phase.get("action_freeze_end_step", warm)))
        longest = self.options.long_rollout_steps if self.options.long_fraction else self.options.rollout_steps
        for site in self.compiled.sites.values():
            interval = int(environment.get("decision_interval", 4)) if site.channel == "horizon" else 1
            chosen = set()
            for window, first, last in WINDOWS:
                pool = np.arange(max((first - 1) * substeps, freeze_end + 1), min(last * substeps, nfe - longest + 1))
                if site.channel == "horizon":
                    pool = pool[pool % interval == 0]
                if len(pool) < self.options.decisions_per_window:
                    raise ValueError(f"Too few eligible {site.channel}/{window} decisions: {len(pool)}.")
                generator = independent_rng(self.options.seed, self.compiled.category, site.channel, window)
                steps = np.sort(generator.choice(pool, self.options.decisions_per_window, replace=False))
                chosen.update(map(int, steps))
                for step in steps:
                    self.window_for_step[(site.key, int(step))] = window
                long_count = int(round(len(steps) * self.options.long_fraction))
                if long_count:
                    self.long_steps.update((site.key, int(step)) for step in generator.choice(steps, long_count, replace=False))
                saved_count = min(len(steps), self.options.snapshots_per_window)
                if saved_count:
                    self.snapshot_steps.update((site.key, int(step)) for step in generator.choice(steps, saved_count, replace=False))
            self.schedules[site.key] = chosen
        write_json(self.output / "sampling.json", {
            "options": asdict(self.options), "windows": WINDOWS,
            "timing": _state_dimension_record(environment, self.compiled),
            "selected_steps": {key: sorted(value) for key, value in self.schedules.items()},
            "long_steps": sorted(self.long_steps), "snapshot_steps": sorted(self.snapshot_steps),
            "sampling": "independent local RNG; all eligible gate proposals regardless of acceptance",
        })

    def wants(self, key, environment):
        self._initialize(environment)
        step = int(environment[self.compiled.step_name])
        wanted = step in self.schedules[key]
        if wanted:
            self.active_key = key
        return wanted

    def is_active(self, key):
        return self.active_key == key

    def before(self, site, environment):
        self.record = None
        self.snapshot = clone_environment(environment, self.compiled)
        agent = environment[site.agent_name]
        previous = vars(agent).get("select_action_with_supervisor")
        existed = "select_action_with_supervisor" in vars(agent)
        original = agent.select_action_with_supervisor

        def record(self_agent, state, *args, **kwargs):
            result = original(state, *args, **kwargs)
            self.record = {"state": np.asarray(state, float).copy(), "decision": deepcopy(result),
                           "kwargs": deepcopy(kwargs)}
            return result

        agent.select_action_with_supervisor = types.MethodType(record, agent)
        self.method_restore = (agent, existed, previous)

    def detach(self):
        if self.method_restore:
            agent, existed, previous = self.method_restore
            if existed:
                agent.select_action_with_supervisor = previous
            else:
                del agent.select_action_with_supervisor
            self.method_restore = None

    def after(self, site, environment):
        self.detach()
        if self.record is None:
            raise RuntimeError("Sampled point was not an actual gate decision; check hand-off/interval semantics.")
        state = rng_state()
        modes = network_modes(environment, self.compiled)
        step = int(environment[self.compiled.step_name])
        window = self.window_for_step[(site.key, step)]
        identity = f"{site.channel}_{window}_{step:07d}"
        agent = environment[site.agent_name]
        decision = self.record["decision"]
        support = replay_support(agent, self.record["state"], decision,
                                 self.options.support_samples, self.options.seed, identity)
        extended = (site.key, step) in self.long_steps
        steps = self.options.long_rollout_steps if extended else self.options.rollout_steps
        try:
            # Each branch begins with the same random state. Main training gets
            # back the exact state immediately after its original gate call.
            branches = {}
            for source in ("policy", "supervisor"):
                restore_rng(state)
                branches[source] = run_branch(self.snapshot, self.compiled, site, self.record,
                                              source, steps, self.options.continuation)
            snapshot_path = None
            if (site.key, step) in self.snapshot_steps:
                directory = self.output / "snapshots"
                directory.mkdir(exist_ok=True)
                snapshot_path = directory / f"{identity}.pkl.gz"
                from .snapshots import portable_environment
                exported = portable_environment(self.snapshot, self.compiled)
                payload = {
                    "schema": 1, "category": self.compiled.category, "site": site.key,
                    "local_names": sorted(self.compiled.local_names),
                    "environment": {key: exported[key] for key in self.compiled.local_names if key in exported},
                    "record": self.record, "rng": state, "options": asdict(self.options),
                    "source_sha256": hashlib.sha256(self.compiled.source.encode()).hexdigest(),
                    "scientific_hashes": {key: value for key, value in self.source_hashes.items()
                                          if not key.startswith("review_experiments/") and key != "run_polymer_e1.py"},
                    "continuation": self.options.continuation,
                }
                with gzip.open(snapshot_path, "xb", compresslevel=3) as stream:
                    cloudpickle.dump(payload, stream)
                del exported, payload
            values = {}
            horizons = sorted(set(self.options.horizons + ((400, self.options.long_rollout_steps) if extended else ())))
            gamma = float(agent.gamma)
            for horizon in horizons:
                if horizon > steps:
                    continue
                if all(branch["state"] == "completed" for branch in branches.values()):
                    metrics = {name: branch_metrics(branch["trace"], horizon, gamma) for name, branch in branches.items()}
                    delta = metrics["policy"]["return"] - metrics["supervisor"]["return"]
                    selected = "policy" if int(decision.selected_source) == 2 else "supervisor"
                    values[str(horizon)] = {
                        "policy": metrics["policy"], "supervisor": metrics["supervisor"],
                        "delta_return": delta,
                        "selected": selected,
                        "regret": max(metrics["policy"]["return"], metrics["supervisor"]["return"]) - metrics[selected]["return"],
                    }
            arrays = {f"{name}__{key}": value for name, branch in branches.items() for key, value in branch["trace"].items()}
            prediction = {}
            if site.channel == "markov":
                from .prediction import prediction_probe
                for name, branch in branches.items():
                    if branch["state"] == "completed":
                        diagnostic, forecasts = prediction_probe(self.snapshot, decision, branch["trace"], self.compiled.category)
                        prediction[name] = diagnostic
                        arrays.update({f"{name}__forecast_{predictor}": value for predictor, value in forecasts.items()})
            trace_path = self.output / "traces" / f"{identity}.npz"
            if trace_path.exists():
                raise FileExistsError(trace_path)
            np.savez_compressed(trace_path, **arrays)
            result = {
                "schema": 1, "seed": self.options.seed, "category": self.compiled.category,
                "channel": site.channel, "context": "combined" if self.compiled.category == "combined" else "standalone",
                "window": window, "step": step, "episode": int(step // int(environment.get("time_in_sub_episodes", environment.get("ctx", {}).get("time_in_sub_episodes")))) + 1,
                "gate_source_line": site.line, "observation": self.record["state"],
                "previous_action": self.record["kwargs"].get("previous_action"),
                "initial_tracking_error_physical": np.asarray(self.snapshot["system"].current_output) - np.asarray(self.snapshot["y_sp_phys"]),
                "gate": asdict(decision), "gate_config": asdict(agent.supervisor_gate_config),
                "critic_train_steps_at_decision": int(agent.train_steps),
                "support": support, "gamma_per_control_step": gamma,
                "rollout_steps": steps, "continuation": self.options.continuation,
                "branches": {name: {key: value for key, value in branch.items() if key != "trace"} for name, branch in branches.items()},
                "metrics": values, "trace": str(trace_path.relative_to(self.output)),
                "prediction_validation": prediction,
                "snapshot": str(snapshot_path.relative_to(self.output)) if snapshot_path else None,
                "status": "completed" if values else "branch_failed",
                "smoke": self.options.smoke,
            }
            with (self.output / "decisions.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(result), allow_nan=False) + "\n")
            self.all_records.append(json_safe(result))
            self.counts[(site.channel, window)] += 1
            if sum(self.counts.values()) % 10 == 0 or self.options.smoke:
                print(f"E1 {self.compiled.category}: {sum(self.counts.values())} matched decisions saved; step {step}", flush=True)
        finally:
            restore_rng(state)
            for module, training in modes:
                module.training = training
            self.active_key, self.snapshot, self.record = None, None, None

    def finish(self, environment):
        expected = len(self.compiled.sites) * len(WINDOWS) * self.options.decisions_per_window
        actual = sum(self.counts.values())
        write_json(self.output / "capture_status.json", {
            "state": "completed" if actual == expected else "incomplete",
            "expected_decisions": expected, "saved_decisions": actual,
            "counts": {f"{channel}/{window}": count for (channel, window), count in self.counts.items()},
            "elapsed_seconds": time.perf_counter() - self.started,
            "evidence_scope": "single training seed" if self.options.seed is not None else "unknown",
        })
        if actual != expected:
            raise RuntimeError(f"E1 captured {actual} of {expected} required decisions.")
