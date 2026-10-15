"""Passive numeric capture: no agent calls, branch rollouts, or training changes."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from utils.markov_runner import raw_action_to_z
from .common import PREDICTORS, WINDOWS, json_safe, write_json
from .prediction import forecasts


@dataclass(frozen=True)
class CaptureOptions:
    seed: int = 7
    horizons: tuple[int, ...] = (1, 4, 8)
    samples_per_window: int = 100
    smoke: bool = False

    def validate(self):
        if self.seed < 0 or self.samples_per_window < 1:
            raise ValueError("Seed must be nonnegative and sampling budget positive.")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons or min(self.horizons) < 1:
            raise ValueError("Prediction horizons must be distinct, increasing positive integers.")


def sampling_schedule(episode_steps, nfe, options, label):
    """Select non-overlapping origins without consuming the training RNG."""
    options.validate()
    length = max(options.horizons)
    schedule = {}
    for window, first, last in WINDOWS:
        start, end = (first - 1) * episode_steps, min(last * episode_steps, nfe)
        pool = np.arange(start, end - length + 1, length, dtype=int)
        if len(pool) < options.samples_per_window:
            raise ValueError(f"Insufficient complete futures in {label}/{window}: {len(pool)}.")
        digest = hashlib.sha256(f"{options.seed}:{label}:{window}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        schedule.update({int(k): window for k in rng.choice(pool, options.samples_per_window, replace=False)})
    return dict(sorted(schedule.items()))


def numeric_snapshot(env, category):
    """Copy only predictor inputs after guards/fallback and before the plant step."""
    if category == "markov":
        cfg, ctx, step = env["config"], env["ctx"], int(env["step"])
        A, B, C, L = (env[key] for key in ("A", "B", "C", "L"))
        initial = env["x_model"]
        bound = cfg["z_bound"]
        policy = env["history"]["sg_policy_action_raw_log"][step]
        executed = env["z_exec"]
        episode_steps, nfe = int(ctx["time_in_sub_episodes"]), int(ctx["nFE"])
        mode = ctx["observer_alignment"]
        source, gate = int(env["action_source"]), int(env["history"]["sg_selected_source_log"][step])
        hp, hc = int(cfg["predict_h"]), int(cfg["cont_h"])
        extra = {}
    else:
        cfg, ctx, step = env["markov_cfg"], env, int(env["i"])
        A, B, C, L = (env[key] for key in ("A_base", "B_base", "C_aug", "L_nom"))
        initial = env["xhatdhat"][:, step]
        bound = env["markov_z_bound"]
        policy = env["markov_sg_policy_action_raw_log"][step]
        executed = env["markov_z_executed_log"][step]
        episode_steps, nfe = int(env["time_in_sub_episodes"]), int(env["nFE"])
        mode = env["observer_update_alignment"]
        source, gate = int(env["markov_action_source_log"][step]), int(env["markov_sg_selected_source_log"][step])
        hp, hc = int(env["current_Hp"]), int(env["current_Hc"])
        extra = {key: np.asarray(env[key]).copy() for key in ("weight_mult", "delta_u_res_exec") if key in env}
        if "residual_projection" in env:
            extra["delta_u_res_exec"] = np.asarray(env["residual_projection"]["delta_u_res_exec"]).copy()
    ls_accepted = bool(env["ls_accepted"])
    zero = np.zeros_like(np.asarray(executed, float))
    corrections = np.stack([zero, np.asarray(env["z_ls"]) if ls_accepted else zero,
                            raw_action_to_z(policy, bound), executed])
    model = {"A": A, "B": B, "C": C, "L": L, "x0": initial,
             "data_min": ctx["data_min"], "data_max": ctx["data_max"],
             "ss_scaled_inputs": ctx["ss_scaled_inputs"], "y_ss_scaled": ctx["y_ss_scaled"],
             "basis_family": cfg["basis_family"], "corrections": corrections,
             "observer_mode": mode, "observer_float32": category == "combined", **extra}
    model = {key: np.asarray(value).copy() for key, value in model.items()}
    record = {"step": step, "episode": step // episode_steps + 1, "ls_accepted": ls_accepted,
              "executed_source_code": source, "gate_source_code": gate, "controller_Hp": hp,
              "controller_Hc": hc, "correction_bound": float(bound)}
    return model, record, episode_steps, nfe


class CaptureSession:
    def __init__(self, directory, plant, category, options, stop_file=None):
        self.output, self.plant, self.category, self.options = Path(directory), plant, category, options
        options.validate()
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "traces").mkdir(exist_ok=True)
        self.schedule = None
        self.pending = None
        self.records = []
        self.stop_file = Path(stop_file) if stop_file else None

    def before(self, env):
        if self.stop_file is not None and self.stop_file.exists():
            raise KeyboardInterrupt("E3 parent requested a graceful stop.")
        step = int(env["step"] if self.category == "markov" else env["i"])
        if self.schedule is None:
            clock = env["ctx"] if self.category == "markov" else env
            self.schedule = sampling_schedule(int(clock["time_in_sub_episodes"]), int(clock["nFE"]),
                                              self.options, f"{self.plant}/{self.category}")
            write_json(self.output / "sampling.json", {
                "windows": WINDOWS, "origins": self.schedule, "horizons": self.options.horizons,
                "samples_per_window": self.options.samples_per_window,
                "scope": "one training seed; independent local sampling RNG; no outcome selection",
                "smoke": self.options.smoke,
            })
        if step in self.schedule:
            if self.pending is not None:
                raise RuntimeError("Overlapping E3 captures indicate an invalid sampling schedule.")
            model, record, _, _ = numeric_snapshot(env, self.category)
            record.update(plant=self.plant, category=self.category, window=self.schedule[step],
                          dataset="training", seed=self.options.seed, smoke=self.options.smoke)
            self.pending = {"model": model, "record": record, "u": [], "y": []}

    def after(self, env):
        if self.pending is None:
            return
        system = env["system"]
        self.pending["u"].append(np.asarray(system.current_input, float).copy())
        self.pending["y"].append(np.asarray(system.current_output, float).copy())
        if len(self.pending["u"]) == max(self.options.horizons):
            item = self.pending
            model, record = item["model"], item["record"]
            u, y = np.asarray(item["u"]), np.asarray(item["y"])
            predicted, failures = forecasts(model, model["x0"], u)
            path = self.output / "traces" / f"{record['window']}_{record['step']:07d}.npz"
            if path.exists():
                raise FileExistsError(path)
            np.savez_compressed(path, **model, u_future=u, y_future=y,
                                **{f"forecast_{name}": predicted[name] for name in PREDICTORS})
            record.update(trace=str(path.relative_to(self.output)), predictor_failures=failures,
                          nonfinite_truth=not bool(np.all(np.isfinite(u)) and np.all(np.isfinite(y))))
            with (self.output / "records.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(record), allow_nan=False) + "\n")
            self.records.append(record)
            self.pending = None
            if len(self.records) % 25 == 0 or self.options.smoke:
                print(f"E3 {self.plant}/{self.category}: {len(self.records)} prediction origins saved", flush=True)

    def finish(self):
        expected = len(WINDOWS) * self.options.samples_per_window
        complete = len(self.records) == expected and self.pending is None
        checkpoints = {window: next((record["trace"] for record in reversed(self.records)
                                    if record["window"] == window), None)
                       for window in ("prechange_mature", "adapted")}
        write_json(self.output / "capture_status.json", {
            "state": "completed" if complete else "incomplete", "expected": expected,
            "saved": len(self.records), "checkpoints": checkpoints,
            "failed_predictors": sum(len(r["predictor_failures"]) for r in self.records),
        })
        if not complete:
            raise RuntimeError(f"E3 capture incomplete: {len(self.records)}/{expected} origins.")
