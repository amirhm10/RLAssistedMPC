"""Validated network-weight exports; these are not training-resume checkpoints."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
from pathlib import Path

import torch


def load_inference_checkpoint(path, agents=None):
    """Validate an export and optionally restore identically configured agents.

    Construct agents from the recorded configuration before calling this.
    Optimizers, replay buffers, plant/observer state and RNG are not included.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != 1 or payload.get("kind") != "inference_network_weights":
        raise ValueError("Unsupported inference checkpoint (legacy empty exports cannot be restored).")
    records = payload.get("agents")
    if not isinstance(records, dict) or not records:
        raise ValueError("Checkpoint has no agents.")
    for name, record in records.items():
        state = record.get("state_dict")
        if not isinstance(state, dict) or not state or not any(
                isinstance(t, torch.Tensor) and t.numel() for t in state.values()):
            raise ValueError(f"Checkpoint contains no network tensors for {name}.")
        if any(not isinstance(t, torch.Tensor) or not bool(torch.isfinite(t).all()) for t in state.values()):
            raise ValueError(f"Invalid or nonfinite network tensor for {name}.")
    if agents is not None:
        if set(agents) != set(records):
            raise ValueError("Checkpoint agent names differ from the requested agents.")
        for name, agent in agents.items():
            agent.load_state_dict(records[name]["state_dict"], strict=True)
    return payload


def save_inference_checkpoint(path, agents, *, metadata=None):
    """Save all registered parameters/buffers and verify the disk round trip.

    state_dict() includes children registered through nn.Module.__setattr__,
    which do not appear as ordinary vars(agent) entries. CPU copies leave the
    live networks, device placement, modes and random streams unchanged.
    """
    path = Path(path)
    records = {}
    for name, agent in agents.items():
        if not isinstance(agent, torch.nn.Module):
            raise TypeError(f"Expected a PyTorch agent module: {name}")
        attributes = {key: getattr(agent, key) for key in ("steps", "train_steps", "total_it")
                      if isinstance(getattr(agent, key, None), (int, float, bool))}
        gate = getattr(agent, "supervisor_gate_config", None)
        if is_dataclass(gate):
            attributes["supervisor_gate_config"] = asdict(gate)
        records[name] = {
            "class": f"{type(agent).__module__}.{type(agent).__qualname__}",
            "attributes": attributes,
            "state_dict": {key: value.detach().cpu().clone() for key, value in agent.state_dict().items()},
        }
    payload = {"schema": 1, "kind": "inference_network_weights",
               "agents": records, "metadata": metadata or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite a checkpoint: {path}")
    try:
        with temporary.open("xb") as stream:
            torch.save(payload, stream)
        loaded = load_inference_checkpoint(temporary)
        for name, record in records.items():
            actual = loaded["agents"][name]["state_dict"]
            if set(actual) != set(record["state_dict"]) or any(
                    actual[key].dtype != tensor.dtype or not torch.equal(actual[key], tensor)
                    for key, tensor in record["state_dict"].items()):
                raise RuntimeError(f"Network checkpoint round trip differs for {name}.")
        # Exclusive destination keeps an accidental second launch from
        # replacing an earlier result. Rename only after validation succeeds.
        temporary.rename(path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tensor_counts": {name: len(record["state_dict"]) for name, record in records.items()},
            "parameter_elements": {name: sum(t.numel() for t in record["state_dict"].values())
                                   for name, record in records.items()},
            "round_trip_verified": True, "training_resume": False}
