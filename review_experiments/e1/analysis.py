"""Transparent descriptive summaries; one seed never receives seed-level CIs."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .study import write_json


def summarize(records, output: Path, tolerance=0.0, margin_offsets=(0.0,)):
    groups = defaultdict(list)
    for record in records:
        # Smoke records are technical checks and always retain that label.
        for horizon in record["metrics"]:
            groups[(record["seed"], record["context"], record["channel"], record["window"], int(horizon), record["smoke"])].append(record)
    rows = []
    for (seed, context, channel, window, horizon, smoke), group in sorted(groups.items()):
        for offset in margin_offsets:
            accepted, rejected, harmful, missed, ties = 0, 0, 0, 0, 0
            regrets, delta_returns = [], []
            supervisor_regrets, policy_regrets = [], []
            raw_correct, raw_total = 0, 0
            for record in group:
                gate = record["gate"]
                advantage = gate["advantage_policy_supervisor"]
                if offset == 0:
                    # Preserve warm/fallback/critic-training restrictions in the
                    # actual source decision, rather than inferring from score.
                    take_policy = int(gate["selected_source"]) == 2
                else:
                    if advantage is None:
                        continue
                    margin = float(record["gate_config"]["advantage_margin"]) + offset
                    ready = record.get("critic_train_steps_at_decision", 0) >= record["gate_config"].get("min_train_steps_before_policy_gate", 0)
                    take_policy = ready and float(advantage) > margin
                metrics = record["metrics"][str(horizon)]
                delta = float(metrics["delta_return"])
                accepted += int(take_policy)
                rejected += int(not take_policy)
                harmful += int(take_policy and delta < -tolerance)
                missed += int(not take_policy and delta > tolerance)
                ties += int(abs(delta) <= tolerance)
                regrets.append(max(0.0, -delta if take_policy else delta))
                delta_returns.append(delta)
                supervisor_regrets.append(max(0., delta))
                policy_regrets.append(max(0., -delta))
                if abs(delta) > tolerance:
                    if gate.get("q_policy") is not None and gate.get("q_supervisor") is not None:
                        raw_delta = float(gate["q_policy"]) - float(gate["q_supervisor"])
                    elif all(gate.get(key) is not None for key in ("q1_policy", "q2_policy", "q1_supervisor", "q2_supervisor")):
                        raw_delta = min(gate["q1_policy"], gate["q2_policy"]) - min(gate["q1_supervisor"], gate["q2_supervisor"])
                    else:
                        raw_delta = None
                    if raw_delta is not None and raw_delta != 0:
                        raw_total += 1
                        raw_correct += int((raw_delta > 0) == (delta > 0))
            rows.append({
                "seed": seed, "context": context, "channel": channel, "window": window, "horizon": horizon,
                "smoke": smoke, "training_seeds": len({r["seed"] for r in group}),
                "paired_decisions": len(regrets), "accepted": accepted, "rejected": rejected,
                "harmful_acceptances": harmful, "harmful_acceptance_rate": harmful / accepted if accepted else None,
                "beneficial_rejections": missed, "beneficial_rejection_rate": missed / rejected if rejected else None,
                "ties": ties, "mean_regret": float(np.mean(regrets)) if regrets else None,
                "raw_critic_non_tie_rank_cases": raw_total,
                "raw_critic_ranking_agreement": raw_correct / raw_total if raw_total else None,
                "q95_regret": float(np.quantile(regrets, 0.95)) if regrets else None,
                "mean_regret_always_supervisor_initial_choice": float(np.mean(supervisor_regrets)) if regrets else None,
                "mean_regret_always_policy_initial_choice": float(np.mean(policy_regrets)) if regrets else None,
                "mean_delta_return": float(np.mean(delta_returns)) if delta_returns else None,
                "margin_offset_reward_units": offset, "return_tolerance": tolerance,
                "failed_pairs_in_window": sum(r["status"] != "completed" for r in records
                                               if (r["seed"], r["context"], r["channel"], r["window"]) == (seed, context, channel, window)),
            })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    failed = [record for record in records if record["status"] != "completed"]
    predictions = defaultdict(list)
    for record in records:
        for branch, diagnostic in record.get("prediction_validation", {}).items():
            for predictor, horizons in diagnostic["metrics"].items():
                for horizon, values in horizons.items():
                    predictions[(record["seed"], record["context"], record["window"], branch, predictor, int(horizon), record["smoke"])].append(values)
    prediction_rows = []
    for (seed, context, window, branch, predictor, horizon, smoke), values in sorted(predictions.items()):
        pooled_rmse = np.sqrt(np.mean(np.square([v["rmse_physical"] for v in values]), axis=0))
        prediction_rows.append({"seed": seed, "context": context, "window": window, "branch": branch,
                                "predictor": predictor, "horizon": horizon, "smoke": smoke,
                                "decision_contexts": len(values), "pooled_rmse_viscosity_L_per_g": pooled_rmse[0],
                                "pooled_rmse_temperature_K": pooled_rmse[1]})
    with (output / "prediction_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        if prediction_rows:
            writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    write_json(output / "summary.json", {
        "schema": 1, "training_seeds": sorted({r["seed"] for r in records}),
        "total_decisions": len(records), "failed_pairs": len(failed),
        "failure_records": [{key: r[key] for key in ("category", "channel", "window", "step", "branches")} for r in failed],
        "groups": rows,
        "prediction_groups": prediction_rows,
        "interpretation": "descriptive matched-decision evidence; repeated decisions are not independent training seeds",
        "margin_sensitivity_scope": "initial decision re-selection with the original frozen continuation; not a closed-loop gate-margin sweep",
        "prediction_scope": "Markov decision probes compare nominal, accepted LS/zero and RL-proposed forecasts using common observed future inputs and held-out future outputs",
        "continuation": sorted({r["continuation"] for r in records}),
    })
    return rows


def load_records(directory):
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # A restarted training job has a fresh trajectory. Only the latest
        # completed attempt contributes, rather than double-counting partials.
        paths = [directory / job["attempts"][-1] / "capture" / "decisions.jsonl"
                 for job in manifest["jobs"] if job["state"] == "completed"]
    else:
        paths = list(directory.rglob("decisions.jsonl"))
    records = []
    for path in sorted(paths):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    identities = [(r["seed"], r["category"], r["channel"], r["step"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate decision identities; choose one completed attempt per training job.")
    return records
