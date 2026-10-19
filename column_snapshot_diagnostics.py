"""Bounded case-16 snapshot diagnosis, separate from production fingerprints."""
from pathlib import Path
import argparse
import json
import sys
import traceback
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("short", "clock"), required=True)
    args = parser.parse_args()
    from run_polymer_e3_recovery import environment
    import os
    os.environ.update(environment())
    from column_review.common import read, write, record_owned_apps, now
    from column_review.pilot import NativeSnapshot, paired_continuations, profile
    from revision_comparisons.column import prepare
    from review_experiments.e3.runtime import aspen_resources
    import numpy as np
    stage = ROOT / "Distillation/Results/column_review/20261018_recovered/e1_pilot"
    job = read(stage / "job.json")
    output = ROOT / "Distillation/Results/review_recovery/20261018" / ("snapshot_diag_" + args.mode)
    output.mkdir(exist_ok=False)
    paths = dict(job["aspen"], snaps_path=str(output / "cleanup/AM_owned"))
    Path(paths["snaps_path"]).mkdir(parents=True)
    status = {"state": "running", "mode": args.mode, "started_at": now(), "pid": os.getpid(),
              "scope": "Technical startup snapshot diagnosis; no 400-step prefix or trained-agent/E1 validation."}
    write(output / "status.json", status)
    try:
        with record_owned_apps(output), aspen_resources(paths):
            ns = prepare(7, output, paths)
            plant = ns["system"]
            sim = plant.sim
            logger = sim.OutputLogger
            logger.WriteFileHeader = False
            logger.Path = str(output / "native_messages.txt")
            logger.FileOutput = True
            diagnostic = {key: str(getattr(sim, key)) for key in ("State", "RunMode", "Successful", "FullName", "Name", "Termination")}
            diagnostic["Time"] = float(sim.Time)
            diagnostic["RecordHistory"] = bool(sim.Options.TimeSettings.RecordHistory)
            diagnostic["CommunicationInterval"] = float(sim.CommunicationInterval)
            diagnostic["EndTime"] = float(sim.EndTime)
            sim.Results.Refresh()
            diagnostic["SnapshotCount"] = int(sim.Results.SnapshotCount)
            diagnostic["ResultCount"] = int(sim.Results.ResultCount)
            write(output / "native_before.json", diagnostic)
            steps = []
            original_step = plant.step
            def traced_step(*args, **kwargs):
                row = {"before": float(sim.Time), "run_mode": str(sim.RunMode),
                       "communication_interval": float(sim.CommunicationInterval),
                       "configured_delta_t": float(plant.delta_t), "end_time": float(sim.EndTime)}
                original_step(*args, **kwargs)
                row.update(after=float(sim.Time), successful=bool(sim.Successful), termination=str(sim.Termination))
                row["actual_delta_t"] = row["after"] - row["before"]
                steps.append(row)
                write(output / "step_clock.json", steps)
            plant.step = traced_step
            if args.mode == "clock":
                plant.step(disturbances=np.array([float(plant.feed.FmR.Value)]))
            sim.Pause()
            before = profile(plant)[1]
            # Isolate description-length and paused-state behavior without
            # editing the shared adapter used by the live comparison workers.
            with patch("column_review.pilot.uuid.uuid4", return_value=SimpleNamespace(hex="r1018")):
                snapshot = NativeSnapshot(plant, {"observer": np.zeros(2)}, output / "snapshots")
            status.update(snapshot_taken=True, native_snapshot=str(snapshot.path))
            write(output / "status.json", status)
            np.testing.assert_allclose(profile(plant)[1], before, rtol=0, atol=1e-9)
            inputs = np.repeat(plant.current_input[None], 3, axis=0)
            disturbances = np.full(3, float(plant.feed.FmR.Value))
            try:
                runs = paired_continuations(plant, snapshot, inputs, disturbances)
            except RuntimeError as error:
                if args.mode != "clock":
                    raise
                write(output / "original_restore_error.json", {"error": str(error), "steps": steps})
                def rearmed_step(*args, **kwargs):
                    sim.RunMode = "Dynamic"
                    sim.CommunicationInterval = float(plant.delta_t)
                    sim.EndTime = float(sim.Time) + 2 * float(plant.delta_t)
                    traced_step(*args, **kwargs)
                plant.step = rearmed_step
                runs = paired_continuations(plant, snapshot, inputs, disturbances)
            np.savez_compressed(output / "continuations.npz", **{f"repeat_{i}": row["outputs"] for i, row in enumerate(runs)})
            status.update(state="completed", snapshot_taken=True, repeated_continuations=3,
                          steps_each=3, native_snapshot=str(snapshot.path), finished_at=now())
        write(output / "status.json", status)
        print(json.dumps(status, indent=2))
        return 0
    except BaseException as error:
        status.update(state="failed", error=str(error), traceback=traceback.format_exc(), finished_at=now())
        write(output / "status.json", status)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
