"""Apply the author's explicit temporary OneDrive quit; never force termination."""
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("memory_capacity", HERE / "extend_pagefile_online.py")
capacity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capacity)


def identity_if_present(pid):
    try:
        return capacity.process_identity(pid)
    except OSError as error:
        if error.winerror == 87:
            return dict(pid=pid, alive=False, created=None)
        raise


def main():
    plan = json.loads((HERE / "onedrive_quit_plan.json").read_text(encoding="utf-8-sig"))
    protection = json.loads((HERE / "console_cleanup_plan.json").read_text(encoding="utf-8"))["protected"]
    assert plan["signature_status"] == "Valid" and "Microsoft Corporation" in plan["signer"]
    assert plan["executable"] == r"C:\Program Files\Microsoft OneDrive\OneDrive.exe"
    assert plan["processes"] and all(p["owner"] == plan["current_user"] for p in plan["processes"])
    assert all(p["path"] == plan["executable"] and str(p["pid"]) not in protection for p in plan["processes"])
    identities = [capacity.process_identity(p["pid"]) for p in plan["processes"]]
    assert all(i["alive"] for i in identities)
    before = capacity.measure()
    protected = capacity.protected_unchanged()
    capacity.save("onedrive_before.json", dict(memory=before, identities=identities, protected_processes=protected))
    outcome = dict(command=[plan["executable"], "/shutdown"], approved="Temporarily quit OneDrive",
                   before=before, previous_private_bytes=sum(p["private_bytes"] for p in plan["processes"]))
    try:
        completed = subprocess.run(outcome["command"], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        outcome["returncode"] = completed.returncode
        outcome["stdout"] = completed.stdout.decode("utf-8", errors="replace")
        outcome["stderr"] = completed.stderr.decode("utf-8", errors="replace")
        for _ in range(30):
            states = [identity_if_present(i["pid"]) for i in identities]
            if all(not state["alive"] or state["created"] != original["created"]
                   for state, original in zip(states, identities)):
                break
            time.sleep(1)
        outcome["previous_processes_after"] = states
        outcome["original_onedrive_processes_closed"] = all(
            not state["alive"] or state["created"] != original["created"]
            for state, original in zip(states, identities))
    except Exception as error:
        outcome["error"] = repr(error)
    outcome["after"] = capacity.measure()
    outcome["protected_processes_unchanged"] = capacity.protected_unchanged()
    outcome["commitment_reduction_bytes"] = before["committed_bytes"] - outcome["after"]["committed_bytes"]
    outcome["persistent_paging_configuration_unchanged"] = before["configuration"] == outcome["after"]["configuration"]
    capacity.save("onedrive_quit_result.json", outcome)
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
