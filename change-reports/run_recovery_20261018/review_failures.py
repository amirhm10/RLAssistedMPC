"""Preserve old crash signatures and diagnostic timing without rerunning jobs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def tail(path, limit=16000):
    with Path(path).open("rb") as stream:
        stream.seek(max(0, Path(path).stat().st_size - limit))
        return stream.read().decode("utf-8", errors="replace")


def main():
    old = ROOT / "Distillation/Results/phase2_repeats/20261014_154252_a2ed3b"
    result = {"repeat_crashes": [], "comparison_crashes": [],
              "confirmed_module": "Windows Application event 1000 identifies openblas.dll for SG seed 23; other repeat modules unconfirmed.",
              "timer_causality": "Four crashes coincide with a scheduled all-thread stack dump. Correlation is strong, but no local reproducer established causation.",
              "upstream_context": {"url": "https://github.com/python/cpython/issues/140815",
                                   "limitation": "Upstream report concerns a different platform/extremely small timeout. It does not prove the cause of these Windows 30-minute-monitor failures."}}
    for job in read(old / "manifest.json")["jobs"]:
        if job["state"] != "failed":
            continue
        attempt = job["attempts"][-1]
        log = Path(attempt["directory"]) / "run.log"
        cycles = re.findall(r"Sub_Episode:\s*(\d+)", tail(log))
        result["repeat_crashes"].append({k: job[k] for k in ("category", "mode", "seed")} |
                                        {"returncode": attempt.get("returncode"), "attempt": attempt,
                                         "last_completed_cycle": int(cycles[-1]) if cycles else None,
                                         "python_traceback_in_tail": "Traceback (most recent call last)" in tail(log)})
    column = ROOT / "Distillation/Results/column_review/20261016_prepared_cases12_16"
    for method in ("sg", "threshold", "ls", "bo"):
        attempt = read(column / method / "attempts.json")[-1]
        directory = Path(attempt["directory"])
        stacks = directory.parent / (directory.name + "_stall_stacks.log")
        started = datetime.fromisoformat(attempt["started_at"])
        dumped = datetime.fromtimestamp(stacks.stat().st_mtime, timezone.utc)
        seconds = (dumped - started).total_seconds()
        result["comparison_crashes"].append({"method": method, "attempt": attempt,
                                             "last_stack_dump_at": dumped.isoformat(),
                                             "seconds_after_worker_started": seconds,
                                             "seconds_from_nearest_1800_second_interval": abs(seconds - round(seconds/1800)*1800),
                                             "last_stack_dump_tail": tail(stacks, 500)})
    result["pilot_error"] = read(column / "e1_pilot/a2/status.json")
    path = Path(__file__).with_name("failure_review.json")
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"repeat_crashes": len(result["repeat_crashes"]),
                      "timer_offsets_seconds": {r["method"]: r["seconds_from_nearest_1800_second_interval"] for r in result["comparison_crashes"]}}, indent=2))


if __name__ == "__main__":
    main()
