"""Assemble completed evidence only; exclude superseded partial trajectories."""
from __future__ import annotations

import json
from pathlib import Path

from .checkpoints import digest, write_json


def finalize(stage, batch):
    batch = Path(batch)
    plan = json.loads((batch / "concurrent_plan.json").read_text(encoding="utf-8"))
    if plan["smoke"]:
        return {"state": "skipped", "reason": "Technical smoke is not scientific evidence."}
    retained = plan["retained_evidence"]
    if stage == "e1_combined":
        from review_experiments.e1.analysis import summarize
        paths = [Path(retained["e1_" + category]["directory"]) / "capture/decisions.jsonl"
                 for category in ("horizon", "weights", "markov", "residual")]
        paths.append(batch / stage / "capture/decisions.jsonl")
        records = [json.loads(line) for path in paths for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        identities = {(r["seed"], r["category"], r["channel"], r["step"]) for r in records}
        if len(records) != 3200 or len(identities) != 3200 or any(r["status"] != "completed" for r in records):
            raise ValueError("Merged E1 requires 3200 unique, completed matched decisions.")
        output = batch / "analysis/e1"
        summarize(records, output, 0.)
        result = {"state": "completed", "matched_decisions": 3200,
                  "sources": [{"path": str(path), "sha256": digest(path)} for path in paths],
                  "excluded": "793 pairs from the superseded crashed combined attempt are preserved but not pooled.",
                  "scope": "one seed; standalone and combined contexts reported separately"}
        write_json(output / "provenance.json", result)
        return result
    for name in ("e2_sg", "e2_arx_repaired"):
        path = batch / name / "status.json"
        if not path.exists() or json.loads(path.read_text(encoding="utf-8")).get("state") != "completed":
            return {"state": "waiting", "reason": f"Waiting for {name}."}
    from comparison_studies.analysis import summarize
    from run_polymer_review_recovery import ROOT
    references = {name: str(Path(retained["e2_" + name]["directory"]) / "training.pkl")
                  for name in ("ls", "bo", "threshold")}
    references.update(offsetfree=str(ROOT / "Polymer/Data/mpc_results_dist.pickle"),
                      sg=str(batch / "e2_sg/training.pkl"),
                      arx_regularized=str(batch / "e2_arx_repaired/training.pkl"))
    output = batch / "analysis/e2"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "references.json", references)
    rows = summarize(output)
    if len(rows) != 30:
        raise ValueError("Expected six controllers and five E2 evaluation windows.")
    result = {"state": "completed", "metric_rows": len(rows),
              "sources": {name: {"path": path, "sha256": digest(path)} for name, path in references.items()},
              "excluded": "Original drifting ARX retained as a diagnosed failed comparator, replaced by separately validated regularized ARX.",
              "scope": "one polymer realization per comparator; descriptive evidence"}
    write_json(output / "provenance.json", result)
    return result
