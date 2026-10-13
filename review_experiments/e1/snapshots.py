"""Portable inference snapshots for subsequent reviewer diagnostics."""
from copy import deepcopy
import gzip
import hashlib
import importlib
from pathlib import Path

import cloudpickle
import torch

from .runtime import CompiledRunner, RUNNER_TARGETS, agents_in, clone_environment, rng_state, restore_rng
from .study import run_branch, write_json


def portable_environment(environment, compiled):
    """Export CPU weights without touching the suspended training networks."""
    cloned = clone_environment(environment, compiled)
    modules = {}
    for agent in agents_in(cloned, compiled.local_names).values():
        for name, value in list(vars(agent).items()):
            if isinstance(value, torch.nn.Module):
                if id(value) not in modules:
                    modules[id(value)] = deepcopy(value).to("cpu")
                setattr(agent, name, modules[id(value)])
        if hasattr(agent, "device"):
            agent.device = torch.device("cpu")
    return cloned


def load_snapshot(path, scientific_hashes=None, device="auto"):
    # These are locally generated trusted experiment artifacts, not uploads.
    with gzip.open(path, "rb") as stream:
        payload = cloudpickle.load(stream)
    if payload["schema"] != 1:
        raise ValueError("Unsupported E1 snapshot schema.")
    if scientific_hashes is not None:
        changed = [name for name, value in payload["scientific_hashes"].items()
                   if scientific_hashes.get(name) != value]
        if changed:
            raise ValueError(f"Snapshot scientific sources changed: {changed[:5]}")
    module, name = RUNNER_TARGETS[payload["category"]]
    compiled = CompiledRunner(getattr(importlib.import_module(module), name), payload["category"])
    if hashlib.sha256(compiled.source.encode()).hexdigest() != payload["source_sha256"]:
        raise ValueError("Snapshot runner source differs from the current runner.")
    environment = dict(compiled.original.__globals__)
    environment.update(payload["environment"])
    environment = clone_environment(environment, compiled)
    selected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    for agent in agents_in(environment, compiled.local_names).values():
        for value in vars(agent).values():
            if isinstance(value, torch.nn.Module):
                value.to(selected_device)
        if hasattr(agent, "device"):
            agent.device = selected_device
    return payload, compiled, environment


def replay_snapshot(path, output, *, steps=None, sources=("policy", "supervisor"), scientific_hashes=None, device="auto"):
    import numpy as np
    payload, compiled, environment = load_snapshot(path, scientific_hashes, device)
    steps = int(steps or payload["options"]["rollout_steps"])
    site = compiled.sites[payload["site"]]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous_rng = rng_state()
    results = {}
    try:
        for source in sources:
            restore_rng(payload["rng"])
            branch = run_branch(environment, compiled, site, payload["record"], source, steps, payload["continuation"])
            np.savez_compressed(output / f"{source}.npz", **branch.pop("trace"))
            results[source] = branch
    finally:
        restore_rng(previous_rng)
    write_json(output / "replay.json", {"snapshot": str(Path(path).resolve()), "steps": steps, "branches": results})
    return results
