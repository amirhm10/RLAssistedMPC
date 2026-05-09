from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_PATH = REPO_ROOT / "report" / "polymer_markov_correction_progress.md"


def resolve_bundle_path(path_or_dir):
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        candidate = path / "input_data.pkl"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Could not find input_data.pkl under {path}.")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_markov_bundle(path_or_dir):
    bundle_path = resolve_bundle_path(path_or_dir)
    with open(bundle_path, "rb") as handle:
        bundle = pickle.load(handle)
    return bundle_path, bundle


def _fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "n/a"
        return f"{float(value):.6g}"
    return str(value)


def build_markov_report_text(bundle_path, bundle):
    summary = dict(bundle.get("summary_metrics") or {})
    debug_phase1 = bundle.get("debug_phase1_metrics")
    debug_shadow = bundle.get("debug_shadow")
    debug_nominal = bundle.get("debug_nominal")

    lines = [
        "# Polymer Markov Correction Progress",
        "",
        "## Summary",
        "",
        "This note was generated from an existing saved Markov result bundle.",
        "",
        f"- Bundle: `{bundle_path.as_posix()}`",
        f"- Run mode: `{bundle.get('run_mode', 'unknown')}`",
        f"- Agent kind: `{bundle.get('agent_kind', 'unknown')}`",
        f"- Result episodes: `{len(np.asarray(bundle.get('avg_rewards', []), float))}`",
        "",
        "## Main Metrics",
        "",
        f"- Accepted fraction: `{_fmt(summary.get('accepted_fraction'))}`",
        f"- TD3 accepted fraction: `{_fmt(summary.get('td3_accepted_fraction'))}`",
        f"- LS fallback fraction: `{_fmt(summary.get('ls_fallback_fraction'))}`",
        f"- Nominal fallback fraction: `{_fmt(summary.get('nominal_fallback_fraction'))}`",
        f"- Mean reward: `{_fmt(summary.get('reward_mean'))}`",
        f"- Final episode reward: `{_fmt(summary.get('reward_final_episode'))}`",
        f"- Mean prediction score: `{_fmt(summary.get('prediction_score_mean'))}`",
        f"- Mean gain drift: `{_fmt(summary.get('gain_drift_mean'))}`",
        f"- Replay pushes: `{_fmt(summary.get('rl_replay_push_count'))}`",
        f"- Train updates: `{_fmt(summary.get('rl_train_update_count'))}`",
        "",
        "## Optional Diagnostics Present",
        "",
        f"- Phase-1 lifted validation: `{bool(debug_phase1)}`",
        f"- Shadow / LS diagnostics: `{bool(debug_shadow)}`",
        f"- Debug nominal rollout: `{bool(debug_nominal)}`",
    ]

    checkpoint_path = bundle.get("rl_agent_checkpoint_path")
    if checkpoint_path:
        lines.extend(["", "## Checkpoint", "", f"- `{Path(checkpoint_path).as_posix()}`"])

    return "\n".join(lines) + "\n"


def write_markov_bundle_report(path_or_dir, report_path=DEFAULT_REPORT_PATH):
    bundle_path, bundle = load_markov_bundle(path_or_dir)
    report_path = Path(report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_markov_report_text(bundle_path, bundle), encoding="utf-8")
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description="Write a polymer Markov progress note from an existing saved bundle.")
    parser.add_argument("bundle", help="Path to a Markov result directory or its input_data.pkl bundle.")
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_PATH),
        help="Output markdown report path. Defaults to report/polymer_markov_correction_progress.md.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report_path = write_markov_bundle_report(args.bundle, args.report)
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
