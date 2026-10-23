"""Read-only current storage/run assessment; archival remains approval-gated."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from column_review.common import process_identity, owned_apps_remaining
from revision_comparisons.progress import snapshot


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    progress = snapshot()
    current = [r for r in progress["jobs"] if r["study"] not in progress["historical_studies"]]
    running = [r for r in current if r["state"] == "running"]
    failed = [r for r in current if r["state"] in {"failed", "stopped_without_final_status"}]
    issues = []
    workers = []
    protected = {}
    for row in running:
        identity = process_identity(row["pid"])
        native = owned_apps_remaining(Path(row["directory"]) / "owned_aspen_apps.json")
        if not identity["alive"]:
            issues.append(row["job"] + ": worker not alive")
        protected[str(row["pid"])] = identity
        for item in native:
            expected = {key: item["recorded"][key] for key in ("pid", "alive", "created")}
            if item["current"] != expected:
                issues.append(row["job"] + ": native ownership differs")
            protected[str(expected["pid"])] = item["current"]
        workers.append(dict(row, identity=identity, owned_aspen=native))
    failures = []
    for row in failed:
        status = read(Path(row["directory"]) / "status.json")
        checkpoint = read(Path(row["directory"]) / "training_checkpoints/latest.json")
        failures.append(dict(job=row["job"], directory=row["directory"], state=row["state"],
                             finished_at=status.get("finished_at"), error=status.get("error"),
                             traceback=status.get("traceback"), checkpoint_cycles=checkpoint["completed_cycles"],
                             checkpoint_file=checkpoint["file"], checkpoint_sha256=checkpoint["sha256"]))
    queue = progress["repeat_queue_coordinator"]
    protected[str(queue["pid"])] = process_identity(queue["pid"])
    candidates = read(HERE / "archive_candidates.json")
    inventory = read(HERE / "inventory.json")
    scientific = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
    frozen = read(scientific / "manuscript_revised/frozen_original_manifest.json")
    frozen_ok = all(digest(scientific / "manuscript" / row["path"]) == row["sha256"] for row in frozen["files"])
    assessment = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                      scope="Storage diagnosis and proposed local archival; no file move/deletion or run restart",
                      counts=progress["counts"], active_workers=len(running), current_failures=failures,
                      workers=workers, protected_identities=protected, queue=queue,
                      current_state_counts=dict(Counter(r["state"] for r in current)),
                      issues=issues, frozen_original_verified=frozen_ok,
                      storage_inventory=inventory, proposed_archive_bytes=candidates["total_candidate_bytes"],
                      proposed_archive_folders=candidates["candidate_folders"],
                      proposed_destination=candidates["proposed_destination"],
                      archive_authorization="Awaiting author choice in the pending question",
                      archive_applied=False, cloud_changes_applied=False,
                      cloud_observation=dict(site="https://mcmasteru365-my.sharepoint.com/largestfiles",
                          visible_storage="1 TB used of 1 TB (100%)", create_upload_disabled=True,
                          checkpoint_example="Polymer/Results/review_recovery/20261016_remaining_polymer_concurrent/e1_combined/training_checkpoints/training_state_0.pkl",
                          visible_version_count=16, current_visible_size="1.06 GB",
                          historical_visible_sizes=["1.04 GB", "1.03 GB", "1.02 GB", "1.00 GB", "0.99 GB", "995 MB", "981 MB", "967 MB", "953 MB", "939 MB", "925 MB", "911 MB", "897 MB", "883 MB", "786 MB"],
                          limitations="No account-wide version-history/recycle-bin size was measured; candidate logical bytes are not verified cloud quota savings"))
    with (HERE / "assessment.json").open("x", encoding="utf-8") as stream:
        json.dump(assessment, stream, indent=2)
        stream.write("\n")
    print(json.dumps(dict(active_workers=len(running), current_failed=[r["job"] for r in failures],
                          column_repetitions=progress["counts"]["column_repeats"],
                          native_apps=sum(len(w["owned_aspen"]) for w in workers),
                          queue_alive=queue["alive"], issues=issues, frozen_original_verified=frozen_ok), indent=2))


if __name__ == "__main__":
    main()
