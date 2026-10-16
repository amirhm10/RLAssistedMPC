"""Audit current E1--E3 artifacts and source freezes without launching studies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

from revision_comparisons.progress import ROOT, STUDIES, read_json, snapshot


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(output):
    import numpy as np
    from comparison_studies.common import source_hashes as e2_sources
    from review_experiments.e3.common import source_hashes as e3_sources
    from run_polymer_e1 import source_hashes as e1_sources
    from run_e3_prediction_quality import completed_job
    from run_polymer_review_recovery import COMPARISONS, audit_historical_sources
    result = snapshot()
    result["source_integrity"] = {}
    for name in ("E3_polymer", "E3_column", "E1_combined_recovery", "column_repeats"):
        path = ROOT/STUDIES[name]
        manifest = read_json(path/"manifest.json")
        expected = manifest.get("source_hashes", {})
        if name.startswith("E3"):
            actual = e3_sources()
        elif name.startswith("E1"):
            actual = e1_sources()
        else:
            from run_distillation_phase2_repeats import source_hashes
            actual = source_hashes(ROOT)
        changed = sorted(k for k in expected.keys() | actual.keys() if expected.get(k) != actual.get(k))
        result["source_integrity"][name] = {"files": len(expected), "unchanged": bool(expected) and not changed, "changed": changed}
    result["polymer_E2_comparisons"] = {}
    for method, relative in COMPARISONS.items():
        directory = ROOT/"Polymer/Results/comparisons/scenario1"/relative
        status, job = read_json(directory/"status.json"), read_json(directory/"job.json")
        path = directory/"training.pkl"
        digest = sha256(path)
        with path.open("rb") as stream:
            data = pickle.load(stream)
        finite = all(np.all(np.isfinite(data[k])) for k in ("y", "u", "rewards_step"))
        result["polymer_E2_comparisons"][method] = {
            "directory": str(directory), "state": status.get("state"), "steps": data["nFE"],
            "sha256_verified": digest == status.get("trajectory_sha256"), "finite_primary_arrays": bool(finite),
            "source_audit": audit_historical_sources(job["source_hashes"]),
            "scientific_note": "Completed trajectory with identified model-admission failure; revised production rerun pending" if method == "arx"
                               else "Completed comparator trajectory; threshold old policy export is empty, preserved and not reused as a trained policy" if method == "threshold"
                               else "Completed comparator trajectory"}
        del data
    result["polymer_E1_pairs"] = {}
    directories = {"horizon": "horizon_s7/a1", "weights": "weights_s7/a1",
                   "markov": "markov_s7/a2", "residual": "residual_s7/a1"}
    for method, relative in directories.items():
        directory = ROOT/STUDIES["E1_polymer"]/relative
        path = directory/"capture/decisions.jsonl"
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        result["polymer_E1_pairs"][method] = {"directory": str(directory), "records": len(records),
            "all_pairs_completed": len(records) == 400 and all(r.get("status") == "completed" for r in records),
            "training_export_present": (directory/"training.pkl").exists(),
            "capture_sha256": sha256(path)}
    result["E3_completed_artifacts"] = {}
    for name in ("E3_polymer", "E3_column"):
        manifest = read_json(ROOT/STUDIES[name]/"manifest.json")
        for job in manifest["jobs"]:
            if job["state"] == "completed":
                result["E3_completed_artifacts"][job["id"]] = {"full_artifact_inventory_verified": completed_job(job)}
    paper = ROOT.parent/"Rl_assisted_paper/RL_assisted_paper"
    freeze = read_json(paper/"manuscript_revised/frozen_original_manifest.json")
    changed = [f["path"] for f in freeze["files"] if sha256(paper/"manuscript"/f["path"]) != f["sha256"]]
    result["original_manuscript"] = {"files": len(freeze["files"]), "unchanged": not changed, "changed": changed}
    result["audit_script_sha256"] = sha256(__file__)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"observed_at": result["observed_at_utc"], "counts": result["counts"],
                      "source_integrity": result["source_integrity"],
                      "E3_completed_artifacts": result["E3_completed_artifacts"],
                      "original_manuscript": result["original_manuscript"]}, indent=2))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(audit(parser.parse_args().output))
