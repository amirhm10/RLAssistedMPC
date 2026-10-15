"""Per-output lead-specific errors; tracking rewards remain a separate artifact."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from .common import PREDICTORS, sha256, write_json

OUTPUTS = {"polymer": (("viscosity", "L/g"), ("reactor_temperature", "K")),
           "distillation": (("tray24_ethane", "fraction"), ("tray85_temperature", "native simulator unit; unverified"))}


def read_records(directory):
    path = Path(directory) / "records.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(directories, output, horizons=(1, 4, 8), *, plots=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    errors, totals, source_counts, examples = defaultdict(list), Counter(), Counter(), {}
    residual_rows = []
    for directory in map(Path, directories):
        for row in read_records(directory):
            group = tuple(row[key] for key in ("plant", "category", "dataset", "window", "smoke"))
            source_counts[group + (row["executed_source_code"],)] += 1
            with np.load(directory / row["trace"], allow_pickle=False) as arrays:
                measured = arrays["y_future"]
                ranges = arrays["data_max"][-measured.shape[1]:] - arrays["data_min"][-measured.shape[1]:]
                for name in PREDICTORS:
                    forecast = arrays[f"forecast_{name}"]
                    for h in horizons:
                        if h > len(measured):
                            raise ValueError("Stored future is shorter than the analysis horizon.")
                        for index, (label, unit) in enumerate(OUTPUTS[row["plant"]]):
                            key = group + (name, h, label, unit)
                            totals[key] += 1
                            error = float(forecast[h - 1, index] - measured[h - 1, index])
                            valid = np.isfinite(error) and np.isfinite(ranges[index]) and ranges[index] > 0
                            if valid:
                                errors[key].append((error, float(ranges[index])))
                            residual_rows.append({**dict(zip(("plant", "category", "dataset", "window", "smoke"), group)),
                                                  "predictor": name, "horizon": h, "output": label,
                                                  "origin": row["step"], "checkpoint_step": row.get("checkpoint_step"),
                                                  "residual_physical": error if valid else None, "valid": bool(valid),
                                                  "nominal_dataset_sha256": row.get("nominal_dataset_sha256", "")})
                # First origin is fixed by the recorded sampling schedule, not fit.
                if group not in examples:
                    examples[group] = {"truth": measured.copy(),
                                       **{name: arrays[f"forecast_{name}"].copy() for name in PREDICTORS}}
    rows = []
    for key, total in sorted(totals.items()):
        valid = np.asarray(errors[key], float).reshape(-1, 2)
        raw, normalized = (valid[:, 0], valid[:, 0] / valid[:, 1]) if len(valid) else (np.array([]), np.array([]))
        rows.append({**dict(zip(("plant", "category", "dataset", "window", "smoke", "predictor", "horizon", "output", "unit"), key)),
                     "total_origins": total, "valid_origins": len(valid), "failed_origins": total - len(valid),
                     "rmse_physical": float(np.sqrt(np.mean(raw ** 2))) if len(raw) else None,
                     "mae_physical": float(np.mean(np.abs(raw))) if len(raw) else None,
                     "bias_physical": float(np.mean(raw)) if len(raw) else None,
                     "max_abs_error_physical": float(np.max(np.abs(raw))) if len(raw) else None,
                     "nrmse_identification_range": float(np.sqrt(np.mean(normalized ** 2))) if len(raw) else None})
    write_csv(output / "prediction_errors.csv", rows)
    write_csv(output / "residuals.csv", residual_rows)
    sources = []
    denominators = Counter()
    for key, count in source_counts.items():
        denominators[key[:-1]] += count
    for key, count in sorted(source_counts.items()):
        sources.append({**dict(zip(("plant", "category", "dataset", "window", "smoke", "executed_source_code"), key)),
                        "origins": count, "fraction": count / denominators[key[:-1]],
                        "interpretation": "training action source; frozen checkpoint source on nominal data"})
    write_csv(output / "source_fractions.csv", sources)
    write_json(output / "analysis_metadata.json", {
        "horizons": horizons, "aggregation": "one signed error at lead h per origin, separately for each output",
        "normalization": "fixed identification output range; no validation-statistic normalization",
        "forecast_scope": "known actual future inputs; decision-time fixed coefficients; no future output updates",
        "evidence_scope": "single training seed; dependent windows; descriptive results without seed confidence intervals",
        "nominal_scope": "shared OF-MPC dataset; checkpoint corrections only; not full-controller evaluation",
        "limitations": ["Dynamic-matrix corrections retain nominal observer/free response.",
                        "Combined tracking improvement does not establish better predictions or causal channel contributions.",
                        "Native distillation temperature unit has not been verified; no Kelvin conversion is inferred."],
        "input_directories": [str(Path(p).resolve()) for p in directories], "summary_rows": len(rows),
        "analysis_source_sha256": sha256(__file__),
        "numpy_version": np.__version__,
    })
    if plots and rows:
        _plots(output, rows, examples)
    return rows


def _plots(output, rows, examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Each figure keeps plant, dataset and checkpoint/condition separate.
    groups = sorted({(r["plant"], r["dataset"], r["window"]) for r in rows})
    colors = dict(zip(PREDICTORS, ("#555555", "#238b45", "#2166ac", "#b2182b")))
    for plant, dataset, window in groups:
        matching = [r for r in rows if (r["plant"], r["dataset"], r["window"]) == (plant, dataset, window)]
        categories = sorted({r["category"] for r in matching})
        fig, axes = plt.subplots(2, len(categories), figsize=(5 * len(categories), 6), squeeze=False, constrained_layout=True)
        for column, category in enumerate(categories):
            for index, (label, unit) in enumerate(OUTPUTS[plant]):
                ax = axes[index, column]
                for name in PREDICTORS:
                    selected = [r for r in rows if (r["plant"], r["category"], r["dataset"], r["window"], r["output"], r["predictor"]) ==
                                (plant, category, dataset, window, label, name)]
                    if selected:
                        ax.plot([r["horizon"] for r in selected], [r["rmse_physical"] for r in selected],
                                "o-", label=name.replace("_", " "), color=colors[name])
                ax.set(title=f"{category}: {label}", xlabel="Forecast lead (steps)", ylabel=f"RMSE ({unit})")
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{plant}: {dataset}, {window}" + (" (technical smoke)" if any(r["smoke"] for r in matching) else ""))
        fig.savefig(output / f"rmse_{plant}_{dataset}_{window}.png", dpi=160)
        plt.close(fig)
    for group, values in sorted(examples.items()):
        plant, category, dataset, window, smoke = group
        fig, axes = plt.subplots(2, 2, figsize=(9, 5), constrained_layout=True)
        for index, (label, unit) in enumerate(OUTPUTS[plant]):
            lead = np.arange(1, len(values["truth"]) + 1)
            axes[index, 0].plot(lead, values["truth"][:, index], "k.-", label="plant")
            for name in PREDICTORS:
                axes[index, 0].plot(lead, values[name][:, index], label=name.replace("_", " "), color=colors[name])
                axes[index, 1].plot(lead, values[name][:, index] - values["truth"][:, index], color=colors[name])
            axes[index, 0].set(ylabel=f"{label} ({unit})", xlabel="Forecast lead (steps)")
            axes[index, 1].set(ylabel="Prediction residual", xlabel="Forecast lead (steps)")
            for ax in axes[index]:
                ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"First preselected origin: {plant}/{category}/{dataset}/{window}" + (" (technical smoke)" if smoke else ""))
        fig.savefig(output / f"prediction_{plant}_{category}_{dataset}_{window}.png", dpi=160)
        plt.close(fig)
