"""Reversible OS scheduling change for three exact accepted Python generations.

No simulator calls, process signals, training edits, thread changes or restart.
Never apply this saved plan to a replacement process or an unrecognized state.
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "change-reports/combined_parallel_20261024"))
import parallel_branches as p
import psutil


def current(target):
    actual = p.h.process(target["pid"])
    if not p.same_generation(target, actual):
        raise RuntimeError("An accepted process generation changed; do not reuse this plan")
    process = psutil.Process(target["pid"])
    if process.name().lower() != "python.exe":
        raise RuntimeError("Only the three accepted Python workers are in scope")
    return process


def main(rollback=False):
    plan = p.closed_json(HERE / "priority_plan.json")
    if {t["pid"] for t in plan["targets"]} != {59504, 38188, 46276}:
        raise RuntimeError("The saved scope differs from the accepted production generations")
    receipt = HERE / ("rollback.json" if rollback else "applied.json")
    if receipt.exists():
        raise FileExistsError("Inspect the existing scheduling receipt before another action")
    status = p.h.stable_batch({"manifest": Path(plan["manifest"])})["manifest"]
    if status.get("state") != "running" or not status.get("validation_passed"):
        raise RuntimeError("Scheduling requires the current accepted healthy production stage")
    for target in plan["targets"]:
        process = current(target)
        expected = plan["normal_priority"] if rollback else target["old_priority"]
        if int(process.nice()) != expected or process.cpu_affinity() != target["affinity"]:
            raise RuntimeError("Priority or affinity changed since the guarded plan")
    changed = []
    try:
        for target in plan["targets"]:
            process = current(target)
            desired = target["old_priority"] if rollback else plan["normal_priority"]
            process.nice(desired)
            changed.append(target)
            if int(process.nice()) != desired or process.cpu_affinity() != target["affinity"]:
                raise RuntimeError("Scheduling readback differs")
    except BaseException:
        for target in reversed(changed):
            process = current(target)
            process.nice(plan["normal_priority"] if rollback else target["old_priority"])
        raise
    value = dict(at_utc=p.now(),passed=True,rollback=rollback,
        workers=[dict(identity=p.h.process(t["pid"]),priority=int(current(t).nice()),
                      affinity=current(t).cpu_affinity()) for t in plan["targets"]],
        actions=dict(priority_changes=len(changed),affinity_changes=0,process_signals=0,
                     restarts=0,simulator_calls=0,training_or_solver_changes=0,data_deletions=0))
    p.immutable(receipt,value)
    print(json.dumps(value,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    main(args.rollback)
