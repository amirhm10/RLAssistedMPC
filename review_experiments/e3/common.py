"""Standard-library-only configuration, provenance and artifact helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILES = {"polymer": "robustness_200_100", "distillation": "temperature_flip_200_100"}
ENTRIES = {
    "polymer": {"markov": "RL_assisted_MPC_markov_unified.py",
                "combined": "RL_assisted_MPC_combined_unified.py", "nominal": "MPCOffsetFree_unified.py"},
    "distillation": {"markov": "distillation_RL_assisted_MPC_markov_unified.py",
                     "combined": "distillation_RL_assisted_MPC_combined_unified.py",
                     "nominal": "distillation_MPCOffsetFree_unified.py"},
}
WINDOWS = (("release", 14, 20), ("prechange_mature", 191, 199),
           ("plant_change", 201, 210), ("adapted", 291, 299))
PREDICTORS = ("nominal", "accepted_ls_or_zero", "rl_proposed", "executed")
SEED_KEYS = ("agent", "td3_agent", "horizon_agent", "markov_td3_agent",
             "weights_td3_agent", "residual_td3_agent")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_safe(value):
    # numpy is deliberately not imported by the launcher/dry run.
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        import math
        return value if math.isfinite(value) else None
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(root=ROOT):
    root = Path(root)
    files = {root / entry for entries in ENTRIES.values() for entry in entries.values()}
    files.update(root / name for name in ("run_e3_prediction_quality.py", "run_distillation_phase2_repeats.py",
                                        "environment.rl-assisted-env.yml", "environment.rl-assisted-env.win-64.lock.yml"))
    for directory in ("Simulation", "DQN", "TD3Agent", "utils", "systems", "review_experiments/e3"):
        files.update((root / directory).rglob("*.py"))
    for plant, model in (("Polymer", "system_dict"), ("Distillation", "system_dict.pickle")):
        files.update(root / plant / "Data" / name for name in (model, "scaling_factor.pickle", "min_max_states.pickle"))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files) if p.is_file()}


def check_sources(expected):
    actual = source_hashes()
    if actual != expected:
        changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
        raise RuntimeError(f"E3 scientific sources/identification assets changed: {changed}. Start a new study.")
