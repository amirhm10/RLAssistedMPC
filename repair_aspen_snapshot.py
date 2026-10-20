"""Isolated case-16 snapshot experiments; never edits a live job dependency."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import pickle
import shutil
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parent


def verify_continuation_arrays(runs):
    """Array-valued physical tolerances work across the pinned NumPy builds."""
    import numpy as np
    for run in runs[1:]:
        if not np.allclose(run["outputs"], runs[0]["outputs"], rtol=0, atol=[1e-8, 1e-4]):
            raise RuntimeError("Repeated native continuations disagree in physical outputs.")
        tolerance = np.array([1e-8, 1e-3, 1e-3, 1e-6] + [v for _ in range(6) for v in (1e-4, 1e-8)])
        if np.any(np.abs(run["internal"] - runs[0]["internal"]) > tolerance):
            raise RuntimeError("Restored continuation internal profiles do not repeat.")
        np.testing.assert_array_equal(run["controller"]["observer"], runs[0]["controller"]["observer"])


class ReadyNativeSnapshot:
    """Short native name plus one verified zero-time restore initialization.

    In installed Aspen Dynamics V14, the first Step after LoadSnapshot is an
    initialization operation with zero time advance, regardless of script flag.
    Complete that operation before applying a counterfactual action. Refuse to
    proceed if it advances time or changes the saved physical diagnostics.
    """
    def __init__(self, plant, controller, output):
        from column_review.pilot import NativeSnapshot
        short_id = uuid.uuid4().hex[:12]
        with patch("column_review.pilot.uuid.uuid4", return_value=SimpleNamespace(hex=short_id)):
            self.snapshot = NativeSnapshot(plant, controller, output)
        self.plant = plant
        self.priming_records = []

    def restore(self):
        import numpy as np
        from column_review.pilot import profile
        controller = self.snapshot.restore()
        before = profile(self.plant)[1]
        self.plant.sim.Step(True, False)
        after = profile(self.plant)[1]
        # Reuse the existing immediate-state tolerances; time is tighter.
        tolerance = np.array([1e-9, 1e-3, 1e-3, 1e-6] + [v for _ in range(6) for v in (1e-5, 1e-9)])
        if np.any(np.abs(after-before) > tolerance):
            raise RuntimeError("Restore initialization advanced time or changed native physical diagnostics.")
        self.priming_records.append({"before": before.tolist(), "after": after.tolist()})
        return controller


def continuation_pilot(namespace, output):
    import numpy as np
    from column_review.common import write, digest
    from column_review.pilot import profile
    from revision_comparisons.column import baseline, calibration_window
    full = namespace["EPISODE_BUNDLE"]
    prefix = baseline(namespace, calibration_window(full, 0, 400))
    if prefix.get("use_shifted_mpc_warm_start"):
        raise RuntimeError("Shifted MPC warm-start extraction is required before this pilot.")
    plant = namespace["system"]
    controller = {"observer": prefix["xhatdhat"][:, -1].copy(),
                  "previous_input": plant.current_input.copy(),
                  "mpc_initial_guess": np.zeros(2 * int(namespace["markov_cfg"]["cont_h"]))}
    snapshot = ReadyNativeSnapshot(plant, controller, output / "preserved_snapshots")
    with (output / "python_checkpoint.pkl").open("xb") as stream:
        pickle.dump(snapshot.snapshot.python_state, stream, protocol=5)
    controls = np.repeat(plant.current_input[None], 24, axis=0)
    bounds = namespace["SYS"]["input_bounds"]
    span = np.asarray(bounds["u_max"]) - np.asarray(bounds["u_min"])
    controls[8:16, 0] += .005 * span[0]
    controls[16:, 1] += .005 * span[1]
    controls = np.clip(controls, bounds["u_min"], bounds["u_max"])
    disturbances = full["disturbance_schedule"][400:424].copy()
    runs, clocks = [], []
    for repeat in range(3):
        restored = snapshot.restore()
        outputs, internal, elapsed = [], [], []
        for control, disturbance in zip(controls, disturbances):
            plant.current_input = control.copy()
            previous = float(plant.sim.Time)
            plant.step(disturbances=np.atleast_1d(disturbance))
            delta = float(plant.sim.Time) - previous
            # The COM clock is quantized to micro-hours. This does not alter
            # the sampling interval, integrator settings, inputs or plant step.
            if abs(delta - plant.delta_t) > 1.1e-6:
                raise RuntimeError(f"Restored clock did not advance one original interval: {delta}.")
            elapsed.append(delta)
            outputs.append(plant.current_output.copy())
            internal.append(profile(plant)[1])
        runs.append({"outputs": np.asarray(outputs), "internal": np.asarray(internal), "controller": restored})
        clocks.append(elapsed)
        np.savez_compressed(output / f"continuation_{repeat}.npz", outputs=runs[-1]["outputs"],
                            internal=runs[-1]["internal"], inputs=controls, disturbances=disturbances,
                            clock_intervals=np.asarray(elapsed))
        write(output / "restore_initializations.json", snapshot.priming_records)
        print(f"Snapshot continuation {repeat+1}/3: 24 advancing samples.", flush=True)
    np.savez_compressed(output / "continuations.npz", inputs=controls, disturbances=disturbances, clock_intervals=np.asarray(clocks),
                        **{f"repeat_{i}_{key}": row[key] for i,row in enumerate(runs) for key in ("outputs", "internal")})
    verify_continuation_arrays(runs)
    snapshot.restore()
    write(output / "restore_initializations.json", snapshot.priming_records)
    result = {"state": "completed", "pilot_passed": True, "prefix_steps": 400,
              "continuation_steps_each": 24, "identical_continuations": 3,
              "clock_tolerance_hours": 1.1e-6, "diagnostic_labels": snapshot.snapshot.labels,
              "max_output_difference": np.max(np.abs(runs[1]["outputs"]-runs[0]["outputs"]), axis=0).tolist(),
              "native_snapshot": str(snapshot.snapshot.path), "native_snapshot_sha256": digest(snapshot.snapshot.path),
              "adapter": "Short snapshot name; verify one zero-time native initialization before the original plant step.",
              "e1_production_ready": False, "e1_critic_validation_performed": False,
              "remaining": "E1 adapter still must preserve learners, gates, action histories, observer/MPC state and select the matched-rollout budget."}
    write(output / "pilot_result.json", result)
    return result


def native_state(sim):
    result = {}
    for key in ("Time", "State", "RunMode", "Successful", "Termination", "EndTime", "CommunicationInterval"):
        try:
            result[key] = getattr(sim, key)
        except Exception as error:
            result[key] = str(error)
    settings = sim.Options.TimeSettings
    for key in ("EnablePauseAt", "PauseAt", "EnableStepFor", "StepFor", "CommunicationInterval", "RecordHistory", "RealTimeSyncFactor"):
        try:
            result["TimeSettings." + key] = getattr(settings, key)
        except Exception as error:
            result["TimeSettings." + key] = str(error)
    return result


def diagnose(plant, output):
    import numpy as np
    from column_review.common import write
    from column_review.pilot import profile
    sim, rows = plant.sim, []

    def record(label, action=None):
        before = native_state(sim)
        error = None
        if action:
            try:
                action()
            except Exception as caught:
                error = repr(caught)
        row = {"label": label, "before": before, "after": native_state(sim),
               "profile": profile(plant)[1].tolist(), "error": error}
        rows.append(row)
        write(output / "diagnosis.json", rows)
        print(label, row["after"]["Time"], error, flush=True)
        return row

    for i in range(3):
        record(f"ordinary_step_{i}", lambda: sim.Step(True))
    results = sim.Results
    results.Refresh()
    name = "e1_" + uuid.uuid4().hex[:12]
    results.TakeSnapshot(name)
    results.Refresh()
    snap = results.FindSnapshot(name)
    if isinstance(snap, (int, np.integer)):
        snap = results.GetSnapshot(int(snap))
    snap.ExportasBinary(str(output / "diagnostic.snp"))
    saved_profile = profile(plant)[1]
    record("snapshot_saved")
    record("advance_before_restore", lambda: sim.Step(True))
    record("load", lambda: results.LoadSnapshot(snap))
    np.testing.assert_allclose(profile(plant)[1], saved_profile, rtol=0, atol=1e-8)
    record("default_step_after_load", lambda: sim.Step(True))
    record("step_without_script", lambda: sim.Step(True, False))
    def rearm():
        sim.Options.TimeSettings.EnablePauseAt = False
        sim.Options.TimeSettings.EnableStepFor = False
        sim.RunMode = "Dynamic"
        sim.CommunicationInterval = float(plant.delta_t)
        sim.EndTime = float(sim.Time) + 3 * plant.delta_t
    record("rearm", rearm)
    record("rearmed_step", lambda: sim.Step(True, False))
    record("bounded_run_without_script", lambda: sim.Run(True, False))
    record("step_after_run", lambda: sim.Step(True, False))
    # Distinguish a first-call no-op from the effect of the optional script.
    for script in (True, False, True, False):
        record(f"reload_for_script_{script}", lambda: results.LoadSnapshot(snap))
        for i in range(3):
            record(f"restored_script_{script}_step_{i}", lambda: sim.Step(True, script))
    record("reload", lambda: results.LoadSnapshot(snap))
    record("run_initialization", lambda: (setattr(sim, "RunMode", "Initialization"), sim.Run(True, False)))
    record("dynamic_after_initialization", lambda: (setattr(sim, "RunMode", "Dynamic"), sim.Step(True, False)))
    return {"state": "completed", "diagnostic_only": True, "snapshot_name": name, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.diagnose == args.pilot:
        raise ValueError("Select exactly one bounded diagnosis or continuation pilot.")
    import review_blas_runtime
    review_blas_runtime.activate()
    import torch
    torch.set_num_threads(1)
    from column_review.common import read, write, digest, now, record_owned_apps, owned_apps_remaining
    from review_experiments.e3.runtime import aspen_resources
    from revision_comparisons.column import prepare
    from run_distillation_phase2_repeats import aspen_file_lock
    import continue_column_queue as queue
    original = read(ROOT / "Distillation/Results/column_review/20261018_recovered/e1_pilot/job.json")["aspen"]
    assert Path(original["aspen_path"]).name == "C2S_SS_simulation16.dynf"
    assert digest(original["aspen_path"]) == original["aspen_sha256"]
    queue_root = Path(read(ROOT / "Distillation/Results/review_recovery/latest.json")["repeats"])
    queue.validate_group(read(queue_root / "combined/manifest.json"))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model_dir = output / "private_model"
    model_dir.mkdir()
    model = model_dir / Path(original["aspen_path"]).name
    shutil.copyfile(original["aspen_path"], model)
    paths = dict(original, aspen_path=str(model), snaps_path=str(model_dir / "AM_C2S_SS_simulation16"))
    Path(paths["snaps_path"]).mkdir()
    status = {"state": "running", "pid": os.getpid(), "started_at": now(), "aspen": paths,
              "source_case": original, "scope": "Isolated technical snapshot repair; no live-job files changed."}
    write(output / "status.json", status)
    try:
        with aspen_file_lock(Path(original["aspen_path"]), ROOT / "Distillation/Results/_aspen_file_locks"):
            with record_owned_apps(output), aspen_resources(paths):
                ns = prepare(7, output, paths)
                sim = ns["system"].sim
                logger = sim.OutputLogger
                logger.Path = str(output / "native_messages.txt")
                logger.WriteFileHeader = False
                logger.FileOutput = True
                result = diagnose(ns["system"], output) if args.diagnose else continuation_pilot(ns, output)
        status.update(state="completed", result=result, finished_at=now(), aspen_closed=True)
    except BaseException as error:
        status.update(state="failed", error=str(error), traceback=traceback.format_exc(), finished_at=now())
        traceback.print_exc()
    status["owned_apps_remaining"] = owned_apps_remaining(output / "owned_aspen_apps.json")
    write(output / "status.json", status)
    return 0 if status["state"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
