from __future__ import annotations


def capture_agent_replay_snapshot(agent):
    if agent is None:
        return None
    replay_buffer = getattr(agent, "buffer", None)
    if replay_buffer is None or not hasattr(replay_buffer, "export_snapshot"):
        return None
    return replay_buffer.export_snapshot(ordered=True)


def attach_single_agent_replay_snapshot(result_bundle: dict, agent, key: str = "replay_buffer_snapshot") -> None:
    snapshot = capture_agent_replay_snapshot(agent)
    if snapshot is not None:
        result_bundle[key] = snapshot


def capture_named_agent_replay_snapshots(agent_map: dict) -> dict:
    snapshots = {}
    for name, agent in agent_map.items():
        snapshot = capture_agent_replay_snapshot(agent)
        if snapshot is not None:
            snapshots[str(name)] = snapshot
    return snapshots
