"""Record completed archival and a fresh, read-only production inventory."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"
sys.path.insert(0, str(ROOT))
from column_review.common import process_identity, owned_apps_remaining
from revision_comparisons.progress import snapshot


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit():
    before = read(HERE / "assessment.json")
    after_archive = read(HERE / "postarchive_verification.json")
    archive = read(HERE / "archive_result.json")
    progress = snapshot()
    rows = [r for r in progress["jobs"] if r["study"] not in progress["historical_studies"]]
    active = [r for r in rows if r["state"] == "running"]
    failed = [r for r in rows if r["state"] in {"failed", "stopped_without_final_status"}]
    protected = {pid: process_identity(int(pid)) for pid in before["protected_identities"]}
    issues = [f"Protected identity changed: {pid}" for pid, value in protected.items()
              if value != before["protected_identities"][pid]]
    actual = read(HERE / "aspen_inventory_final.json")
    owned = {}
    activity = []
    for row in active:
        previous = next(r for r in before["workers"] if r["pid"] == row["pid"])
        delta = row["cpu_seconds"] - previous["cpu_seconds"]
        activity.append(dict(job=row["job"], pid=row["pid"], python_cpu_delta_seconds=delta))
        if delta <= 0:
            issues.append(f"No Python CPU increase: {row['pid']}")
        for item in owned_apps_remaining(Path(row["directory"]) / "owned_aspen_apps.json"):
            if "recorded" not in item:
                issues.append(f"Unverified native owner for {row['job']}")
                continue
            owned[item["recorded"]["pid"]] = dict(job=row["job"], identity=item["current"])
    actual_pids = {r["pid"] for r in actual["aspen"]}
    unmatched = sorted(actual_pids - set(owned))
    missing = sorted(set(owned) - actual_pids)
    if unmatched or missing:
        issues.append("Native inventory/ownership differs")
    manifest = read(PAPER / "manuscript_revised/frozen_original_manifest.json")
    frozen = all(digest(PAPER / "manuscript" / r["path"]) == r["sha256"] for r in manifest["files"])
    if not frozen:
        issues.append("Frozen original differs")
    report = dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),
                  archive=archive, archive_postcheck=after_archive,
                  current_state_counts=dict(Counter(r["state"] for r in rows)),
                  running_jobs=active, failed_jobs=failed,
                  failed_job_diagnostics=before["current_failures"],
                  current_counts={key: value for key, value in progress["counts"].items()
                                  if key not in progress["historical_studies"]},
                  historical_studies=progress["historical_studies"],
                  protected_identities_unchanged=len(protected) if not issues else None,
                  protected_identities=protected, activity=activity,
                  activity_baseline_utc=before["observed_at_utc"],
                  aspen=dict(inventory=actual, owned=owned, unmatched=unmatched, missing=missing,
                             closed=0, reason="Every native application belongs to a running job"),
                  queue=progress["repeat_queue_coordinator"],
                  frozen_original_verified=frozen, frozen_file_count=len(manifest["files"]),
                  cloud_cleanup=read(HERE / "cloud_cleanup_prepared.json"),
                  cloud_cleanup_applied=False, scientific_sources_modified=False,
                  scientific_jobs_restarted=False, issues=issues)
    assert len(active) == 12 and len(failed) == 3 and not issues, issues
    assert archive["verified_folders"] == 910 and archive["protected_identities_unchanged"]
    write(HERE / "final_assessment.json", report)
    print(json.dumps(dict(running=len(active), failed=[r["job"] for r in failed],
                          aspen=len(actual_pids), stale=unmatched,
                          all_workers_have_cpu_increases=True, issues=issues), indent=2))


def prepend(path, block):
    heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
    assert "## Storage archive and run audit - October 23, 2026" not in rest
    if path == ROOT / "RUN_RECOVERY_STATUS.md":
        heading = "# Reviewer-run recovery — latest update October 23, 2026"
    path.write_text(heading + "\n\n" + block.strip() + "\n" + rest, encoding="utf-8")


def record(commit=None):
    report = read(HERE / "final_assessment.json")
    archive = report["archive"]
    assert not report["issues"] and report["frozen_original_verified"]
    storage = (f"With Amir's explicit approval, 910 obsolete native-checkpoint folders "
               f"({archive['verified_bytes']:,} bytes; 162.6 GiB; 3640 files) were moved outside "
               f"OneDrive to `{archive['destination']}`. Every archived file was SHA-256 verified. "
               "All Python checkpoints, current/recent native checkpoints, scientific results and policies "
               "were retained. Original-path mappings and hashes are in the archive's restore_manifest.json; "
               "restore only a missing original path, without overwriting newer checkpoints. This same-drive "
               "move reduces the synced file set rather than freeing local disk. The 27 protected worker, "
               "Aspen and coordinator creation identities are unchanged.")
    status = (f"Current production audit at {report['observed_at_utc']}: 12 jobs running, three failed "
              "and no current queued jobs. Polymer repetitions are 50/50 complete; column repetitions "
              "are 40/50 complete, with all ten remaining combined repetitions running. Markov SG seed 23 "
              "and residual plain seed 23 have completed. E1 horizon and combined remain running. "
              "The original automatic combined coordinator is alive. Each of the twelve Python workers "
              "has a positive CPU increase since the pre-archive audit.")
    failure = ("Column E1 weights, Markov and residual failed overnight after saving cycle 190. "
               "The tracebacks show Aspen openDocument failing with RPC error -2147023170, followed "
               "by an AD application.Quit cleanup AttributeError. The underlying crash mechanism and "
               "any relationship to memory or cloud quota remain unproven. Their cycle-190 Python, "
               "native model and snapshot hashes verify; a successful restart has not been tested in "
               "this task. No failed job was restarted. Failed workers' six native applications are "
               "already gone. No NumPy failure is established by these tracebacks.")
    e1 = ("The five original E1 configurations remain horizon, weights, Markov, residual and combined, "
          "with the approved 640-pair budget and 300-cycle training schedule unchanged. Verified primary "
          "pairs total 200/640: combined 80/320, horizon 60/80, and weights/Markov/residual 20/80 each. "
          "Only horizon (case 13) and combined (case 16) are currently active. Each owns two Aspen "
          "applications; comparison applications can validly be idle between capture windows. All "
          "14 open Aspen applications match the twelve running jobs; no stale instance was found or closed.")
    cloud_path = HERE / "cloud_cleanup_result.json"
    if cloud_path.exists():
        report["cloud_cleanup_result"] = read(cloud_path)
        cloud = report["cloud_cleanup_result"]["summary"]
    else:
        cloud = ("SharePoint Storage Metrics still shows 0.00 GB free of 1024.00 GB; Documents uses "
                 "752.8 GB excluding the recycle bin. Archived native folders are now visibly in the "
                 "cloud recycle bin. Permanent removal of only the verified archived cloud copies is "
                 "prepared and awaiting explicit action-time confirmation. No permanent cloud deletion "
                 "has been performed, and quota savings have not been verified. An earlier recoverable "
                 "delete attempt against 29 stale folder-list entries returned File Not Found for every "
                 "entry; it is not evidence of a successful cloud deletion by that attempt. OneDrive "
                 "has restarted since the prior day's temporary quit; it was not stopped in this task.")
    limits = ("NativeCheckpointManager.save alternates two Python slots but creates a new native folder "
              "for every saved step without pruning older folders. This explains the observed checkpoint "
              "accumulation; checkpoint retention has not been changed in sealed running sources. Future "
              "growth and long-term native stability remain open. File integrity does not prove native "
              "restart fidelity. Previously completed polymer threshold/E1/E2/E3 evidence is retained; "
              "E4/E5 and unseen-reference follow-ups remain separately scoped. No performance, safety, "
              "causal or statistical claim is promoted. All 46 frozen original manuscript hashes verify; "
              "no TeX or PDF was changed.")
    title = "## Storage archive and run audit - October 23, 2026"
    body = "\n\n".join((storage, status, failure, e1, cloud, limits)) + "\n"
    sources = ("Inspected: storage inventory and native-export metadata, archive candidate/receipt/hash "
               "records, current manifests/status/checkpoint indexes and E1 capture records, failed worker "
               "tracebacks, exact process creation identities and native ownership inventories, "
               "column_e1/checkpoints.py, revision_comparisons/progress.py, OneDrive Storage Metrics, "
               "file version history and recycle-bin UI, and the frozen-original manifest.\n\n"
               "Changed: only the explicitly approved obsolete native-folder locations; new storage "
               "report/evidence/helpers; RUN_RECOVERY_STATUS.md; synchronized paper operational memory. "
               "The raw moved model/snapshot files remain outside Git. Unrelated dirty "
               "run_polymer_phase2_repeats.py is preserved.\n\n"
               "Cloud accounting guidance: [Microsoft work/school storage management]"
               "(https://support.microsoft.com/en-gb/onedrive/manage-your-onedrive-for-work-or-school-storage).\n")
    if not commit:
        (HERE / "STATUS.md").write_text("# Storage archive and current run audit\n\n" + body + "\n" + sources, encoding="utf-8")
        prepend(ROOT / "RUN_RECOVERY_STATUS.md", title + "\n\n" + body +
                "\nEvidence: `change-reports/storage_20261023_1306/STATUS.md` and `final_assessment.json`.\n")
        plan = HERE / "PLAN.md"
        plan.write_text("> Historical pre-approval plan. The approved archive is now complete; see STATUS.md.\n\n" +
                        plan.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        report["scientific_operational_commit"] = commit
        destination = "manuscript_revised/STORAGE_ARCHIVE_2026_10_23.md"
        evidence = "manuscript_revised/evidence/storage_20261023/assessment.json"
        reference = (f"Record `{destination}`; evidence `{evidence}`; scientific operational commit "
                     f"`{commit}`; decision D-2026-10-23-01.")
        (PAPER / destination).write_text("# Storage archive and current run audit\n\n" + body + "\n" + reference + "\n", encoding="utf-8")
        write(PAPER / evidence, report)
        blocks = dict(ACTIVE_PAPER_STATE=[storage, status, failure, e1, cloud],
                      FREEZE_STATE=[status, failure, limits], METHOD_FAMILY_MAP=[e1, failure, limits],
                      RESULT_BUNDLE_INDEX=[storage, status, failure, e1],
                      CLAIM_LEDGER=[storage, status, failure, cloud, limits],
                      EXPERIMENT_BACKLOG=[status, failure, e1, cloud, limits],
                      OPEN_QUESTIONS=[failure, cloud, limits])
        for name, paragraphs in blocks.items():
            prepend(PAPER / "paper-memory" / f"{name}.md", title + "\n\n" +
                    "\n\n".join(paragraphs) + "\n\n" + reference + "\n")
        decision = ("**D-2026-10-23-01 - local checkpoint archive.** Amir stated OneDrive is full and "
                    "explicitly selected 'Archive outside OneDrive (Recommended)' for about 163 GiB "
                    "of obsolete Aspen checkpoints at C:\\Users\\hamediaa\\RL_MPC_local_archive, "
                    "with verification and preservation of active/recent checkpoints and all results. "
                    "Archived files remain on this computer and lose cloud backup. Amir also requested "
                    "checking failed jobs and closing only stale, nonworking Aspen instances. No "
                    "administrator access, scientific budget change or interruption of live jobs "
                    "was authorized or needed. Permanent cloud-copy removal is a separately confirmed "
                    "action, recorded in the cloud cleanup result if applied.\n\n")
        prepend(PAPER / "paper-memory/DECISION_LOG.md", title + "\n\n" + decision + storage +
                "\n\n" + cloud + "\n\n" + reference + "\n")
    print("Recorded paper evidence" if commit else "Recorded code status")


if __name__ == "__main__":
    if sys.argv[1:] == ["audit"]:
        audit()
    else:
        record(sys.argv[1] if len(sys.argv) > 1 else None)
