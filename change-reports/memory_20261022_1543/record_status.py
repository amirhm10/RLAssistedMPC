"""Preserve the verified memory cleanup and synchronize operational paper records."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / "Rl_assisted_paper/RL_assisted_paper"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def prepend(path, block):
    heading, rest = path.read_text(encoding="utf-8").split("\n", 1)
    assert "## Memory cleanup - October 22, 15:55 Toronto" not in rest
    path.write_text(heading + "\n\n" + block.strip() + "\n" + rest, encoding="utf-8")


def record(commit=None):
    audit = read(HERE / "audit.json")
    cleanup = read(HERE / "console_cleanup_result.json")
    drive = read(HERE / "onedrive_quit_result.json")
    launch = read(HERE / "pagefile_launch.json")
    assert not audit["issues"] and len(audit["workers"]) == 17
    assert cleanup["closed_viewer_windows"] == 8 and cleanup["all_protected_identities_unchanged"]
    assert drive["original_onedrive_processes_closed"] and drive["protected_processes_unchanged"] == 77
    assert drive["persistent_paging_configuration_unchanged"] and launch["state"] == "not_launched"
    assert not (HERE / "pagefile_apply_result.json").exists()
    m = drive["after"]
    free = m["headroom_bytes"] / 1024 ** 3
    freed = drive["commitment_reduction_bytes"] / 1024 ** 3
    fraction = 100 * m["committed_bytes"] / m["commit_limit_bytes"]
    point = datetime.fromisoformat(audit["observed_at_utc"]).astimezone(ZoneInfo("America/Toronto"))
    title = f"## Memory cleanup - October 22, {point:%H:%M} Toronto"
    memory = (f"Eight redundant read-only progress CMD windows were closed, retaining the newest viewer "
              f"(PID {cleanup['kept_viewer']}) and all five active experiment/queue consoles. No stale Aspen "
              f"application was found: all 22 native applications match current worker ownership. With Amir's "
              f"explicit approval, the two Microsoft-signed OneDrive sync-client processes were gracefully "
              f"closed. This reduced system commitment by {freed:.2f} GiB in the before/after sample, leaving "
              f"{free:.2f} GiB of commitment headroom at {fraction:.1f}% of the current limit. Cloud syncing "
              f"is temporarily stopped; running jobs continue saving local files. Reopen OneDrive from Start "
              f"to resume cloud syncing when memory pressure permits.")
    boundary = ("A guarded runtime-only paging-file increase was prepared and its read-only preflight passed. "
                "The Windows UAC launch was canceled because Amir has no administrator access; the helper "
                "was not launched and the proposed 96 GiB allocation was not applied. Paging configuration "
                "remains system-managed and unchanged. No administrator bypass, reboot, worker restart or "
                "Aspen reset was performed.")
    status = (f"At {point:%Y-%m-%d %H:%M} Toronto, all 17 current production jobs remain running: twelve "
              f"column repetitions and five column E1 configurations. No current job is failed, stopped or "
              f"queued. All 17 worker groups have positive CPU increases in the 52-second sample. All 77 "
              f"queryable protected worker/Aspen/supervisor/broker/coordinator/ancestor identities are "
              f"unchanged. The automatic combined coordinator and four recovery supervisors remain live; "
              f"source, bootstrap, job and protocol guards pass. No new Python/Aspen crash event was "
              f"recorded between the previous 13:49 audit and the 15:54 resource sample.")
    e1 = (f"Column E1 is running horizon, weights, Markov, combined and residual simultaneously on "
          f"cases 13, 14, 15, 16 and 17 respectively. These are five configurations: combined uses the "
          f"four learned horizon/weights/Markov/residual channels together. Each configuration owns "
          f"two separate Aspen applications, one for main training and one for counterfactual comparisons, "
          f"so E1 accounts for ten of the 22 live native applications. {audit['primary_completed']}/640 "
          f"primary pairs are complete ({640-audit['primary_completed']} remaining): combined 75/320, "
          f"each standalone 20/80. Zero failed/duplicate pairs were found. The original 300-cycle schedule "
          f"and approved reduced sampling budget are unchanged.")
    risk = ("This cleanup reduces memory pressure but does not repair or prove the cause of Aspen memory "
            "growth. The active combined E1 comparison application was the largest consumer, about 10.9 GiB "
            "private memory at 15:52. It was preserved. Commitment remains high and multi-day crash-free "
            "operation is not established. Memory counters fluctuate with the jobs. Do not use working-set "
            "trimming, forcibly close a live native application, or modify sealed worker sources as an "
            "in-place remedy. Any future native process recycling requires a validated full-state handoff.")
    scope = ("Polymer repetitions remain 50/50 complete and column repetitions 38/50 complete, with the "
             "remaining twelve running. Threshold final trajectory/policy hashes verify again. All 46 "
             "frozen original manuscript files verify unchanged. Separately scoped E4/E5 and unseen-reference "
             "follow-ups remain outside this active pool. No new scientific experiment, controller change "
             "or performance/safety/causal claim is introduced.")
    body = "\n\n".join((memory, boundary, status, e1, risk, scope)) + "\n"
    assessment = dict(observed_at_utc=audit["observed_at_utc"], memory_cleanup=drive,
                      console_cleanup=cleanup, pagefile_launch=launch,
                      active_workers=len(audit["workers"]), aspen_inventory=audit["aspen_inventory"],
                      primary_completed=audit["primary_completed"], primary_expected=640,
                      primary_pairs=audit["primary_pairs"], workers=audit["workers"],
                      queue=audit["queue"], supervisors=audit["supervisors"],
                      issues=audit["issues"], remaining_risks=[risk],
                      activity_window=audit["activity_window"], crash_events=audit["crash_events"],
                      frozen_original=audit["frozen_original"],
                      historical_studies=audit["historical_studies"],
                      source_audit=read(HERE / "audit_source.json"))
    if not commit:
        new(HERE / "assessment.json", json.dumps(assessment, indent=2) + "\n")
        sources = ("Inspected: live recovery/study/job manifests, immutable source/bootstrap/protocol "
                   "hashes, worker/checkpoint status, E1 capture records, all native ownership files, "
                   "Windows CPU/memory/pagefile/disk/event inventories, exact progress console trees, "
                   "OneDrive ownership/signature/startup command, threshold final hashes and frozen-original "
                   "manifest. The read-only audit implementation is recorded in audit_source.json.\n\n"
                   "Changed: only redundant viewer processes and, with explicit approval, OneDrive runtime "
                   "processes; this report/evidence/helpers and RUN_RECOVERY_STATUS.md. Scientific sources, "
                   "native/Python training state, queue definitions, shared packages and saved Windows paging "
                   "configuration were preserved. Raw local process/resource snapshots remain alongside "
                   "the focused committed evidence.\n")
        references = ("Windows references: [commit accounting](https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information), "
                      "[paging growth and allocation errors](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/slow-page-file-growth-memory-allocation-errors), "
                      "[OneDrive quit/resume](https://support.microsoft.com/en-us/onedrive/how-to-cancel-or-stop-sync-in-onedrive). "
                      "The prepared, unapplied native extension signature was checked against "
                      "[System Informer's declaration](https://github.com/winsiderss/systeminformer/blob/master/phnt/include/ntmmapi.h).\n")
        new(HERE / "STATUS.md", "# Memory cleanup and live-run verification\n\n" + body + "\n" + sources + "\n" + references)
        prepend(ROOT / "RUN_RECOVERY_STATUS.md", title + "\n\n" + memory + "\n\n" + status + "\n\n" + e1 +
                "\n\n" + boundary + "\n\n" + risk + "\n\nEvidence: `change-reports/memory_20261022_1543/STATUS.md` and `assessment.json`.\n")
    else:
        assessment["scientific_audit_commit"] = commit
        record_path = "manuscript_revised/MEMORY_CLEANUP_2026_10_22.md"
        evidence = "manuscript_revised/evidence/memory_20261022/assessment.json"
        reference = f"Record `{record_path}`; evidence `{evidence}`; scientific operational commit `{commit}`; decision D-2026-10-22-02."
        new(PAPER / record_path, "# Memory cleanup and live-run verification\n\n" + body + "\n" + reference + "\n")
        new(PAPER / evidence, json.dumps(assessment, indent=2) + "\n")
        blocks = dict(ACTIVE_PAPER_STATE=[memory, status, e1, risk], FREEZE_STATE=[status, boundary, scope],
                      METHOD_FAMILY_MAP=[e1, scope], RESULT_BUNDLE_INDEX=[status, e1, scope],
                      CLAIM_LEDGER=[memory, status, risk], EXPERIMENT_BACKLOG=[status, e1, boundary, risk],
                      OPEN_QUESTIONS=[memory, boundary, risk])
        for name, paragraphs in blocks.items():
            prepend(PAPER / "paper-memory" / (name + ".md"), title + "\n\n" + "\n\n".join(paragraphs) + "\n\n" + reference + "\n")
        decision = ("**D-2026-10-22-02 - operational memory relief.** Amir requested memory/crash prevention "
                    "without interrupting jobs, identification of stale Aspen applications and closure of "
                    "unnecessary CMD windows. Amir stated no administrator access and explicitly selected "
                    "'Temporarily quit OneDrive (Recommended)' after being told cloud syncing would pause "
                    "while local experiment saving continued. This authorizes the recorded viewer cleanup "
                    "and temporary OneDrive quit. The administrator route was canceled without a paging "
                    "change. No scientific budget, state or method change is authorized or inferred.\n\n")
        prepend(PAPER / "paper-memory/DECISION_LOG.md", title + "\n\n" + decision + memory + "\n\n" + status + "\n\n" + reference + "\n")


if __name__ == "__main__":
    record(sys.argv[1] if len(sys.argv) > 1 else None)
