from __future__ import annotations

import argparse
import csv
import json
import pickle
import pathlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
POLYMER_RESULTS = REPO_ROOT / "Polymer" / "Results"
FROZEN_LEGACY_RUN = REPO_ROOT / "Polymer" / "Results" / "polymer_markov_corrected_mpc" / "20260510_204243"
FROZEN_UNIFIED_RUN = REPO_ROOT / "Polymer" / "Results" / "td3_markov_disturb" / "20260510_204834"
FIG_ROOT = REPO_ROOT / "report" / "figures" / "polymer_markov_legacy_mimic_ladder_20260510"

STEP_INFO = {
    "step0_unified_clone": {
        "title": "Step 0",
        "code_delta": "Unified clone baseline.",
        "legacy_difference": "No legacy difference applied yet; verifies the fork reproduces unified behavior.",
    },
    "step1_runtime_context_parity": {
        "title": "Step 1",
        "code_delta": "Use the legacy-script bounds source from system-identification artifacts inside runtime-context construction.",
        "legacy_difference": "Legacy builds input bounds from system_data b_min/b_max rather than notebook-local bounds reconstruction.",
    },
    "step2_nominal_reference_parity": {
        "title": "Step 2",
        "code_delta": "Use the exact legacy nominal lifted G0 solve path instead of the generic shared nominal-reference dispatcher.",
        "legacy_difference": "Legacy compares LS and TD3 candidates against a nominal action/cost from solve_lifted_mpc(..., G0, ...).",
    },
    "step3_plant_step_parity": {
        "title": "Step 3",
        "code_delta": "Apply polymer disturbances and step the plant exactly like the legacy script inside the live loop.",
        "legacy_difference": "Legacy writes Qi/Qs/hA directly onto PolymerCSTR before system.step().",
    },
    "step4_td3_construction_parity": {
        "title": "Step 4",
        "code_delta": "Switch the TD3 construction path to the legacy-style seed semantics while keeping deterministic defaults for attribution.",
        "legacy_difference": "Legacy creates TD3Agent without forwarding a dedicated seed into the constructor.",
    },
    "step5_comparator_reporting_parity": {
        "title": "Step 5",
        "code_delta": "Always save an internal nominal rerun bundle for legacy-style report comparisons.",
        "legacy_difference": "Legacy compares against its own internal nominal rerun, not only the canonical saved baseline bundle.",
    },
    "step6_residual_delta_hunt": {
        "title": "Step 6",
        "code_delta": "Residual low-level delta hunt.",
        "legacy_difference": "Use only if Steps 1-5 still leave a material gap to frozen legacy behavior.",
    },
}


@dataclass
class RunArtifacts:
    label: str
    run_dir: Path
    bundle: dict
    stage_arrays: dict[str, np.ndarray]


class DummyPandasObject:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        return DummyPandasObject(*args, **kwargs)

    def __setstate__(self, state):
        self.state = state

    def __getattr__(self, name):
        return DummyPandasObject()


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib._local":
            if name == "Path":
                return pathlib.Path
            if name == "WindowsPath":
                return pathlib.WindowsPath
            if name == "PosixPath":
                return pathlib.PosixPath
        if module.startswith("pandas"):
            return DummyPandasObject
        return super().find_class(module, name)


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_stage_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    arrays: dict[str, np.ndarray] = {}
    for key in rows[0].keys():
        if key == "action_source_name":
            arrays[key] = np.asarray([row[key] for row in rows], dtype=object)
        else:
            arrays[key] = np.asarray([_parse_float(row[key]) for row in rows], dtype=float)
    return arrays


def _load_bundle(path: Path) -> dict:
    with path.open("rb") as handle:
        return CompatUnpickler(handle).load()


def _load_run(label: str, run_dir: Path) -> RunArtifacts:
    stage_csv = run_dir / "markov_stage_diagnostics.csv"
    bundle_pkl = run_dir / "input_data.pkl"
    if not stage_csv.exists():
        raise FileNotFoundError(f"Missing stage diagnostics: {stage_csv}")
    if not bundle_pkl.exists():
        raise FileNotFoundError(f"Missing input_data bundle: {bundle_pkl}")
    return RunArtifacts(
        label=label,
        run_dir=run_dir,
        bundle=_load_bundle(bundle_pkl),
        stage_arrays=_load_stage_csv(stage_csv),
    )


def _all_mimic_runs() -> list[Path]:
    candidates = []
    for path in POLYMER_RESULTS.rglob("markov_stage_diagnostics.csv"):
        run_dir = path.parent
        bundle_pkl = run_dir / "input_data.pkl"
        if not bundle_pkl.exists():
            continue
        try:
            bundle = _load_bundle(bundle_pkl)
        except Exception:
            continue
        if bundle.get("legacy_mimic_step"):
            candidates.append(run_dir)
    return sorted(set(candidates))


def _latest_mimic_run(step_name: str | None) -> Path:
    candidates = _all_mimic_runs()
    if step_name is not None:
        filtered = []
        for run_dir in candidates:
            try:
                bundle = _load_bundle(run_dir / "input_data.pkl")
            except Exception:
                continue
            if str(bundle.get("legacy_mimic_step")) == str(step_name):
                filtered.append(run_dir)
        candidates = filtered
    if not candidates:
        suffix = f" for step '{step_name}'" if step_name else ""
        raise FileNotFoundError(f"No legacy-mimic Markov runs found{suffix}.")
    return candidates[-1]


def _latest_runs_by_step() -> list[Path]:
    grouped: dict[str, Path] = {}
    for run_dir in _all_mimic_runs():
        bundle = _load_bundle(run_dir / "input_data.pkl")
        step_name = _infer_step_name(run_dir, bundle)
        prev = grouped.get(step_name)
        if prev is None or run_dir.name > prev.name:
            grouped[step_name] = run_dir
    return [grouped[key] for key in sorted(grouped)]


def _infer_step_name(run_dir: Path, bundle: dict) -> str:
    step_name = bundle.get("legacy_mimic_step")
    if step_name:
        return str(step_name)
    prefix = "td3_markov_disturb_legacy_mimic_"
    parent_name = run_dir.parent.name
    if parent_name.startswith(prefix):
        return parent_name[len(prefix) :]
    return "unknown_step"


def _episode_len(run: RunArtifacts) -> int:
    n_steps = int(len(run.stage_arrays["step"]))
    bundle = run.bundle
    step_changes = bundle.get("sub_episodes_changes_dict", {})
    n_episodes = len(step_changes) if isinstance(step_changes, dict) and step_changes else 50
    return max(1, n_steps // max(1, int(n_episodes)))


def _trajectory_distance(bundle_a: dict, bundle_b: dict) -> dict[str, float]:
    y_a = np.asarray(bundle_a["y"], float)
    y_b = np.asarray(bundle_b["y"], float)
    u_a = np.asarray(bundle_a["u"], float)
    u_b = np.asarray(bundle_b["u"], float)

    y_len = min(len(y_a), len(y_b))
    u_len = min(len(u_a), len(u_b))
    y_diff = y_a[:y_len] - y_b[:y_len]
    u_diff = u_a[:u_len] - u_b[:u_len]

    return {
        "output_1_rmse": float(np.sqrt(np.mean(y_diff[:, 0] ** 2))),
        "output_2_rmse": float(np.sqrt(np.mean(y_diff[:, 1] ** 2))),
        "input_1_rmse": float(np.sqrt(np.mean(u_diff[:, 0] ** 2))),
        "input_2_rmse": float(np.sqrt(np.mean(u_diff[:, 1] ** 2))),
        "max_output_abs_diff": float(np.max(np.abs(y_diff))),
        "max_input_abs_diff": float(np.max(np.abs(u_diff))),
    }


def _source_fraction(arrays: dict[str, np.ndarray], source: int) -> float:
    src = arrays["action_source"].astype(int)
    return float(np.mean(src == int(source)))


def _mean(arrays: dict[str, np.ndarray], key: str) -> float:
    return float(np.nanmean(arrays[key]))


def _summary(run: RunArtifacts) -> dict:
    stage = run.stage_arrays
    bundle = run.bundle
    step_name = _infer_step_name(run.run_dir, bundle)
    step_info = STEP_INFO.get(step_name, {})
    return {
        "label": run.label,
        "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
        "legacy_mimic_step": step_name,
        "legacy_mimic_level": bundle.get("legacy_mimic_level"),
        "legacy_mimic_title": bundle.get("legacy_mimic_title", step_info.get("title")),
        "legacy_mimic_code_delta": bundle.get("legacy_mimic_code_delta", step_info.get("code_delta")),
        "legacy_mimic_legacy_difference": bundle.get("legacy_mimic_legacy_difference", step_info.get("legacy_difference")),
        "legacy_mimic_bounds_source": bundle.get("legacy_mimic_bounds_source"),
        "legacy_mimic_td3_seed_mode": bundle.get("legacy_mimic_td3_seed_mode"),
        "legacy_mimic_seed": bundle.get("legacy_mimic_seed"),
        "nominal_solver_mode": bundle.get("nominal_solver_mode"),
        "td3_fraction": _source_fraction(stage, 2),
        "ls_fraction": _source_fraction(stage, 3),
        "nominal_fraction": _source_fraction(stage, 4),
        "warm_start_ls_fraction": _source_fraction(stage, 1),
        "mean_executed_z_norm": _mean(stage, "executed_z_norm"),
        "mean_executed_gain_drift": _mean(stage, "executed_gain_drift"),
        "mean_first_move_diff_norm": _mean(stage, "executed_first_move_diff_norm"),
        "mean_full_sequence_diff_norm": _mean(stage, "executed_full_sequence_diff_norm"),
        "mean_nominal_cost_margin": _mean(stage, "executed_cost_margin"),
        "mean_requested_prediction_score": _mean(stage, "requested_prediction_score"),
        "mean_ls_prediction_score": _mean(stage, "ls_prediction_score"),
        "mean_executed_prediction_score": _mean(stage, "executed_prediction_score"),
        "internal_nominal_available": bool(bundle.get("debug_nominal") is not None),
    }


def _plot_step_comparison(
    mimic: RunArtifacts,
    legacy: RunArtifacts,
    unified: RunArtifacts,
    legacy_gap: dict[str, float],
    unified_gap: dict[str, float],
    out_path: Path,
) -> None:
    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.5))

    ax = axs[0, 0]
    labels = ["TD3", "LS", "Nominal"]
    legacy_vals = [_source_fraction(legacy.stage_arrays, 2), _source_fraction(legacy.stage_arrays, 3), _source_fraction(legacy.stage_arrays, 4)]
    unified_vals = [_source_fraction(unified.stage_arrays, 2), _source_fraction(unified.stage_arrays, 3), _source_fraction(unified.stage_arrays, 4)]
    mimic_vals = [_source_fraction(mimic.stage_arrays, 2), _source_fraction(mimic.stage_arrays, 3), _source_fraction(mimic.stage_arrays, 4)]
    x = np.arange(len(labels))
    width = 0.25
    ax.bar(x - width, legacy_vals, width, label="Frozen legacy", color="#0B6E4F")
    ax.bar(x, unified_vals, width, label="Frozen unified", color="#6E7781")
    ax.bar(x + width, mimic_vals, width, label="Mimic run", color="#C84C09")
    ax.set_xticks(x, labels)
    ax.set_title("Action-source fractions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = axs[0, 1]
    metric_labels = ["||z||", "drift", "u0 diff", "Useq diff"]
    legacy_metrics = [
        _mean(legacy.stage_arrays, "executed_z_norm"),
        _mean(legacy.stage_arrays, "executed_gain_drift"),
        _mean(legacy.stage_arrays, "executed_first_move_diff_norm"),
        _mean(legacy.stage_arrays, "executed_full_sequence_diff_norm"),
    ]
    unified_metrics = [
        _mean(unified.stage_arrays, "executed_z_norm"),
        _mean(unified.stage_arrays, "executed_gain_drift"),
        _mean(unified.stage_arrays, "executed_first_move_diff_norm"),
        _mean(unified.stage_arrays, "executed_full_sequence_diff_norm"),
    ]
    mimic_metrics = [
        _mean(mimic.stage_arrays, "executed_z_norm"),
        _mean(mimic.stage_arrays, "executed_gain_drift"),
        _mean(mimic.stage_arrays, "executed_first_move_diff_norm"),
        _mean(mimic.stage_arrays, "executed_full_sequence_diff_norm"),
    ]
    x = np.arange(len(metric_labels))
    ax.bar(x - width, legacy_metrics, width, color="#0B6E4F")
    ax.bar(x, unified_metrics, width, color="#6E7781")
    ax.bar(x + width, mimic_metrics, width, color="#C84C09")
    ax.set_xticks(x, metric_labels)
    ax.set_title("Mechanism metrics")
    ax.grid(alpha=0.25, axis="y")

    ax = axs[1, 0]
    score_labels = ["Req score", "LS score", "Exec score", "Cost margin"]
    legacy_scores = [
        _mean(legacy.stage_arrays, "requested_prediction_score"),
        _mean(legacy.stage_arrays, "ls_prediction_score"),
        _mean(legacy.stage_arrays, "executed_prediction_score"),
        _mean(legacy.stage_arrays, "executed_cost_margin"),
    ]
    unified_scores = [
        _mean(unified.stage_arrays, "requested_prediction_score"),
        _mean(unified.stage_arrays, "ls_prediction_score"),
        _mean(unified.stage_arrays, "executed_prediction_score"),
        _mean(unified.stage_arrays, "executed_cost_margin"),
    ]
    mimic_scores = [
        _mean(mimic.stage_arrays, "requested_prediction_score"),
        _mean(mimic.stage_arrays, "ls_prediction_score"),
        _mean(mimic.stage_arrays, "executed_prediction_score"),
        _mean(mimic.stage_arrays, "executed_cost_margin"),
    ]
    x = np.arange(len(score_labels))
    ax.bar(x - width, legacy_scores, width, color="#0B6E4F")
    ax.bar(x, unified_scores, width, color="#6E7781")
    ax.bar(x + width, mimic_scores, width, color="#C84C09")
    ax.set_xticks(x, score_labels)
    ax.set_title("Prediction and nominal-margin metrics")
    ax.grid(alpha=0.25, axis="y")

    ax = axs[1, 1]
    gap_labels = ["y1 RMSE", "y2 RMSE", "u1 RMSE", "u2 RMSE"]
    legacy_gap_vals = [
        legacy_gap["output_1_rmse"],
        legacy_gap["output_2_rmse"],
        legacy_gap["input_1_rmse"],
        legacy_gap["input_2_rmse"],
    ]
    unified_gap_vals = [
        unified_gap["output_1_rmse"],
        unified_gap["output_2_rmse"],
        unified_gap["input_1_rmse"],
        unified_gap["input_2_rmse"],
    ]
    x = np.arange(len(gap_labels))
    ax.bar(x - 0.15, legacy_gap_vals, 0.3, label="Mimic vs frozen legacy", color="#C84C09")
    ax.bar(x + 0.15, unified_gap_vals, 0.3, label="Mimic vs frozen unified", color="#6E7781")
    ax.set_xticks(x, gap_labels)
    ax.set_title("Trajectory distance of mimic run")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Legacy mimic ladder: {_infer_step_name(mimic.run_dir, mimic.bundle)}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_all_steps_progression(rows: list[dict], out_path: Path) -> None:
    labels = [row["mimic"]["legacy_mimic_step"].replace("step", "S").replace("_", "\n", 1) for row in rows]
    x = np.arange(len(labels))

    fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.0))

    ax = axs[0, 0]
    ax.plot(x, [row["mimic_vs_frozen_legacy"]["output_1_rmse"] for row in rows], marker="o", label="y1 RMSE")
    ax.plot(x, [row["mimic_vs_frozen_legacy"]["output_2_rmse"] for row in rows], marker="o", label="y2 RMSE")
    ax.set_title("Distance to frozen legacy")
    ax.set_xticks(x, labels)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[0, 1]
    ax.plot(x, [row["mimic"]["td3_fraction"] for row in rows], marker="o", label="TD3")
    ax.plot(x, [row["mimic"]["ls_fraction"] for row in rows], marker="o", label="LS")
    ax.plot(x, [row["mimic"]["nominal_fraction"] for row in rows], marker="o", label="Nominal")
    ax.set_title("Action-source fractions")
    ax.set_xticks(x, labels)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    ax.plot(x, [row["mimic"]["mean_executed_z_norm"] for row in rows], marker="o", label="||z||")
    ax.plot(x, [row["mimic"]["mean_executed_gain_drift"] for row in rows], marker="o", label="drift")
    ax.set_title("Mechanism shift")
    ax.set_xticks(x, labels)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    ax.plot(x, [row["mimic"]["mean_executed_prediction_score"] for row in rows], marker="o", label="executed score")
    ax.plot(x, [row["mimic"]["mean_ls_prediction_score"] for row in rows], marker="o", label="LS score")
    ax.plot(x, [row["mimic"]["mean_requested_prediction_score"] for row in rows], marker="o", label="requested score")
    ax.set_title("Prediction-score progression")
    ax.set_xticks(x, labels)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    for ax in axs.reshape(-1):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Legacy mimic ladder progression", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the latest polymer Markov legacy-mimic ladder run.")
    parser.add_argument("--step", default=None, help="Optional legacy_mimic_step to filter for.")
    parser.add_argument("--all-steps", action="store_true", help="Analyze the latest run for every mimic step.")
    args = parser.parse_args()
    legacy = _load_run("Frozen legacy", FROZEN_LEGACY_RUN)
    unified = _load_run("Frozen unified", FROZEN_UNIFIED_RUN)

    if args.all_steps:
        out_dir = FIG_ROOT / "all_steps"
        out_dir.mkdir(parents=True, exist_ok=True)
        summaries = []
        for mimic_dir in _latest_runs_by_step():
            mimic = _load_run("Mimic", mimic_dir)
            step_name = _infer_step_name(mimic.run_dir, mimic.bundle)
            step_dir = FIG_ROOT / step_name
            step_dir.mkdir(parents=True, exist_ok=True)
            legacy_gap = _trajectory_distance(mimic.bundle, legacy.bundle)
            unified_gap = _trajectory_distance(mimic.bundle, unified.bundle)
            figure_path = step_dir / f"{step_name}_comparison.png"
            _plot_step_comparison(mimic, legacy, unified, legacy_gap, unified_gap, figure_path)
            summaries.append(
                {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "frozen_legacy_run": str(FROZEN_LEGACY_RUN.relative_to(REPO_ROOT)),
                    "frozen_unified_run": str(FROZEN_UNIFIED_RUN.relative_to(REPO_ROOT)),
                    "episode_len": int(_episode_len(mimic)),
                    "mimic": _summary(mimic),
                    "frozen_legacy": _summary(legacy),
                    "frozen_unified": _summary(unified),
                    "mimic_vs_frozen_legacy": legacy_gap,
                    "mimic_vs_frozen_unified": unified_gap,
                    "figure_path": str(figure_path.relative_to(REPO_ROOT)),
                }
            )
            with (step_dir / f"{step_name}_summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summaries[-1], handle, indent=2)

        progression_path = out_dir / "legacy_mimic_all_steps_progression.png"
        _plot_all_steps_progression(summaries, progression_path)
        aggregate = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "frozen_legacy_run": str(FROZEN_LEGACY_RUN.relative_to(REPO_ROOT)),
            "frozen_unified_run": str(FROZEN_UNIFIED_RUN.relative_to(REPO_ROOT)),
            "figure_path": str(progression_path.relative_to(REPO_ROOT)),
            "steps": summaries,
        }
        with (out_dir / "legacy_mimic_all_steps_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2)
        print(json.dumps(aggregate, indent=2))
        return

    mimic_dir = _latest_mimic_run(args.step)
    mimic = _load_run("Mimic", mimic_dir)

    step_name = _infer_step_name(mimic.run_dir, mimic.bundle)
    out_dir = FIG_ROOT / step_name
    out_dir.mkdir(parents=True, exist_ok=True)

    legacy_gap = _trajectory_distance(mimic.bundle, legacy.bundle)
    unified_gap = _trajectory_distance(mimic.bundle, unified.bundle)

    figure_path = out_dir / f"{step_name}_comparison.png"
    _plot_step_comparison(mimic, legacy, unified, legacy_gap, unified_gap, figure_path)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_legacy_run": str(FROZEN_LEGACY_RUN.relative_to(REPO_ROOT)),
        "frozen_unified_run": str(FROZEN_UNIFIED_RUN.relative_to(REPO_ROOT)),
        "episode_len": int(_episode_len(mimic)),
        "mimic": _summary(mimic),
        "frozen_legacy": _summary(legacy),
        "frozen_unified": _summary(unified),
        "mimic_vs_frozen_legacy": legacy_gap,
        "mimic_vs_frozen_unified": unified_gap,
        "figure_path": str(figure_path.relative_to(REPO_ROOT)),
    }

    summary_path = out_dir / f"{step_name}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
