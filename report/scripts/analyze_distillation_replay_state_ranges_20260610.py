from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "report" / "figures" / "distillation_replay_state_ranges_20260610"

DISTILLATION_BUNDLES = {
    "weights": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_weights_sg_td3_critic_warm3_margin0_sup001_gauss015_003_manual_off_disturb_fluctuation_mismatch"
    / "20260608_181133"
    / "input_data.pkl",
    "residual": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_residual_sg_td3_critic_warm3_margin05_paramnoise_manual_off_disturb_fluctuation_mismatch_no_rho"
    / "20260608_183743"
    / "input_data.pkl",
    "markov": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_markov_sg_td3_critic_warm3_margin05_softparamnoise_ls_else_mpc_shadow_disturb_fluctuation_mismatch"
    / "20260608_194745"
    / "input_data.pkl",
    "horizon": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11"
    / "20260608_180237"
    / "input_data.pkl",
    "dueling": REPO_ROOT
    / "Distillation"
    / "Results"
    / "distillation_dueling_horizon_sg_dqn_critic_warm3_default_ofmpc_eps02_002_disturb_fluctuation_mismatch_np6_11_nc3_11"
    / "20260608_180736"
    / "input_data.pkl",
}

STATE_LABELS_15 = [
    "base_0",
    "base_1",
    "base_2",
    "base_3",
    "base_4",
    "base_5",
    "base_6",
    "ysp_x24",
    "ysp_T85",
    "u_reflux",
    "u_reboiler",
    "innov_x24",
    "innov_T85",
    "track_x24",
    "track_T85",
]


def _load_bundle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _segment_masks(n_rows: int, n_fe: int, time_in_subepisodes: int) -> dict[str, np.ndarray]:
    start_step = int(n_fe) - int(n_rows)
    steps = np.arange(start_step, int(n_fe), dtype=int)
    steady_start = max(0, int(time_in_subepisodes) - 100)
    return {
        "all": np.ones(n_rows, dtype=bool),
        "tail20": steps >= int(n_fe) - 20 * int(time_in_subepisodes),
        "steady_last100_each_ep": (steps % int(time_in_subepisodes)) >= steady_start,
        "tail20_steady_last100": (
            (steps >= int(n_fe) - 20 * int(time_in_subepisodes))
            & ((steps % int(time_in_subepisodes)) >= steady_start)
        ),
    }


def _summary_row(name: str, source: str, segment: str, states: np.ndarray, rewards, sources) -> dict:
    states = np.asarray(states, float)
    rewards = None if rewards is None else np.asarray(rewards, float).reshape(-1)
    sources = None if sources is None else np.asarray(sources, float).reshape(-1)
    feature_range = np.nanmax(states, axis=0) - np.nanmin(states, axis=0)
    feature_std = np.nanstd(states, axis=0)
    unique_fraction = np.unique(np.round(states, 3), axis=0).shape[0] / max(1, states.shape[0])

    row = {
        "run": name,
        "source": source,
        "segment": segment,
        "n": int(states.shape[0]),
        "state_dim": int(states.shape[1]),
        "median_range": float(np.nanmedian(feature_range)),
        "max_range": float(np.nanmax(feature_range)),
        "median_std": float(np.nanmedian(feature_std)),
        "max_std": float(np.nanmax(feature_std)),
        "low_range_lt_0p01": int(np.sum(feature_range < 0.01)),
        "low_range_lt_0p05": int(np.sum(feature_range < 0.05)),
        "rounded3_unique_fraction": float(unique_fraction),
    }
    if states.shape[1] == 15:
        track_abs = np.abs(states[:, 13:15])
        innov_abs = np.abs(states[:, 11:13])
        row.update(
            {
                "pct_max_track_lt_0p05": float(100.0 * np.mean(np.max(track_abs, axis=1) < 0.05)),
                "pct_max_track_lt_0p10": float(100.0 * np.mean(np.max(track_abs, axis=1) < 0.10)),
                "pct_max_innov_lt_0p01": float(100.0 * np.mean(np.max(innov_abs, axis=1) < 0.01)),
                "pct_track0p05_innov0p01": float(
                    100.0
                    * np.mean((np.max(track_abs, axis=1) < 0.05) & (np.max(innov_abs, axis=1) < 0.01))
                ),
                "track_abs_median": float(np.nanmedian(track_abs)),
                "track_abs_q90": float(np.nanquantile(track_abs, 0.90)),
                "track_abs_q99": float(np.nanquantile(track_abs, 0.99)),
                "innov_abs_median": float(np.nanmedian(innov_abs)),
                "innov_abs_q90": float(np.nanquantile(innov_abs, 0.90)),
                "innov_abs_q99": float(np.nanquantile(innov_abs, 0.99)),
            }
        )
    if rewards is not None and rewards.size == states.shape[0]:
        row["mean_replay_reward"] = float(np.nanmean(rewards))
    if sources is not None and sources.size == states.shape[0]:
        row["policy_source_fraction"] = float(np.nanmean(sources == 2))
    return row


def _feature_rows(name: str, source: str, segment: str, states: np.ndarray) -> list[dict]:
    states = np.asarray(states, float)
    labels = STATE_LABELS_15 if states.shape[1] == 15 else [f"state_{idx}" for idx in range(states.shape[1])]
    rows = []
    for idx, label in enumerate(labels):
        values = states[:, idx]
        rows.append(
            {
                "run": name,
                "source": source,
                "segment": segment,
                "feature_index": idx,
                "feature": label,
                "min": float(np.nanmin(values)),
                "q05": float(np.nanquantile(values, 0.05)),
                "median": float(np.nanmedian(values)),
                "q95": float(np.nanquantile(values, 0.95)),
                "max": float(np.nanmax(values)),
                "range": float(np.nanmax(values) - np.nanmin(values)),
                "std": float(np.nanstd(values)),
            }
        )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict] = []
    feature_rows: list[dict] = []
    availability: dict[str, dict] = {}

    for name, path in DISTILLATION_BUNDLES.items():
        bundle = _load_bundle(path)
        n_fe = int(bundle.get("nFE", 0))
        time_in_subepisodes = int(bundle.get("time_in_sub_episodes", 400))
        snapshot = bundle.get("replay_buffer_snapshot")
        if snapshot is not None:
            states = np.asarray(snapshot["states"], float)
            rewards = np.asarray(snapshot.get("rewards"), float)
            sources = snapshot.get("selected_sources")
            masks = _segment_masks(states.shape[0], n_fe, time_in_subepisodes)
            availability[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "source": "replay_buffer_snapshot",
                "state_shape": list(states.shape),
            }
            for segment, mask in masks.items():
                summary_rows.append(
                    _summary_row(name, "replay_buffer_snapshot", segment, states[mask], rewards[mask], None if sources is None else np.asarray(sources)[mask])
                )
            feature_rows.extend(
                _feature_rows(name, "replay_buffer_snapshot", "tail20_steady_last100", states[masks["tail20_steady_last100"]])
            )
            continue

        state_log = bundle.get("rl_state_log")
        if state_log is not None:
            states = np.asarray(state_log, float)
            pushed = np.asarray(bundle.get("rl_replay_pushed_log", np.ones(states.shape[0])), bool)
            masks = {
                "all": np.ones(states.shape[0], dtype=bool),
                "pushed": pushed,
                "tail20": np.arange(states.shape[0]) >= states.shape[0] - 20 * time_in_subepisodes,
                "steady_last100_each_ep": (np.arange(states.shape[0]) % time_in_subepisodes) >= max(0, time_in_subepisodes - 100),
                "tail20_steady_last100": (
                    (np.arange(states.shape[0]) >= states.shape[0] - 20 * time_in_subepisodes)
                    & ((np.arange(states.shape[0]) % time_in_subepisodes) >= max(0, time_in_subepisodes - 100))
                ),
            }
            availability[name] = {
                "path": str(path.relative_to(REPO_ROOT)),
                "source": "rl_state_log_not_exact_replay_snapshot",
                "state_shape": list(states.shape),
                "pushed_fraction": float(np.mean(pushed)),
            }
            for segment, mask in masks.items():
                summary_rows.append(_summary_row(name, "rl_state_log", segment, states[mask], None, None))
            feature_rows.extend(_feature_rows(name, "rl_state_log", "tail20_steady_last100", states[masks["tail20_steady_last100"]]))
            continue

        availability[name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "source": "missing",
            "state_shape": None,
        }

    summary_path = OUT_DIR / "distillation_replay_state_range_summary.csv"
    feature_path = OUT_DIR / "distillation_replay_state_feature_detail_tail20_steady.csv"
    json_path = OUT_DIR / "distillation_replay_state_range_summary.json"

    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary_rows for key in row}))
        writer.writeheader()
        writer.writerows(summary_rows)

    with feature_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in feature_rows for key in row}))
        writer.writeheader()
        writer.writerows(feature_rows)

    with json_path.open("w") as handle:
        json.dump(
            {
                "availability": availability,
                "summary_csv": str(summary_path.relative_to(REPO_ROOT)),
                "feature_detail_csv": str(feature_path.relative_to(REPO_ROOT)),
            },
            handle,
            indent=2,
        )

    print(f"Wrote {summary_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {feature_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {json_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
