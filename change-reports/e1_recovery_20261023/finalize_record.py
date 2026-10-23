"""Record verified recovery, frozen partial evidence and archive preservation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT.parent / 'Rl_assisted_paper/RL_assisted_paper'
TITLE = '## E1 recovery, interim evidence and verified compression - October 23, 2026'
EVIDENCE = 'manuscript_revised/evidence/column_e1_partial_20261023'
RECORD = 'manuscript_revised/COLUMN_E1_RECOVERY_AND_COMPRESSION_2026_10_23.md'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepend(path, content):
    text = path.read_text(encoding='utf-8')
    if TITLE in text:
        start = text.index(TITLE)
        end = text.find('\n## ', start + len(TITLE))
        text = text[:start] + text[end + 1:] if end >= 0 else text[:start]
    heading, rest = text.split('\n', 1)
    path.write_text(heading + '\n\n' + TITLE + '\n\n' + content.strip() + '\n' + rest, encoding='utf-8')


def main(commit):
    assessment = read(HERE / 'recovery_assessment.json')
    compression = read(Path(r'C:\Users\hamediaa\RL_MPC_local_archive\compressed_storage_20261023\status.json'))
    preservation = read(HERE / 'local_preservation_verified.json')
    assert preservation['verified'] and preservation['files'] == 3640 and preservation['originals_deleted'] == 0
    cloud_source = ROOT / 'change-reports/storage_20261023_1306'
    cloud = read(cloud_source / 'cloud_cleanup_result.json')
    assert cloud['permanently_deleted_folders'] == 878 and cloud['local_originals_deleted'] == 0
    assert cloud['storage_metrics']['free_GB'] == 151.05
    stage = HERE / 'cloud'
    stage.mkdir(exist_ok=True)
    for source, name in (
        ('cloud_cleanup_result.json', 'result.json'), ('cloud_cleanup_receipts.jsonl', 'first_stage_receipts.jsonl'),
        ('cloud_purge/receipts.jsonl', 'permanent_purge_receipts.jsonl'),
        ('cloud_storage_after.jpg', 'storage_after.jpg'), ('cloud_storage_after_snapshot.txt', 'storage_after_snapshot.txt'),
        ('cloud_purge/second_stage_final.jpg', 'second_stage_empty.jpg'),
        ('cloud_purge/second_stage_final_snapshot.txt', 'second_stage_final_snapshot.txt'),
        ('cloud_purge/first_stage_final_snapshot.txt', 'first_stage_final_snapshot.txt')):
        if not commit:
            shutil.copyfile(cloud_source / source, stage / name)
    validation = HERE / 'validation'
    validation.mkdir(exist_ok=True)
    plan_root = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_process'
    for path in list(plan_root.glob('validation_*.json')) + [plan_root / name for name in
            ('manifest.json', 'protocol.json', 'supervisor_lifecycle.json', 'scheduled.json', 'broker_identity.json', 'progress.json')]:
        if not commit:
            shutil.copyfile(path, validation / path.name)
    if not commit:
        shutil.copyfile(ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_probe_a2/acceptance.json', validation / 'native_acceptance.json')
    states = assessment['current_state_counts']
    repeat_counts = assessment['counts']['column_repeats']
    e1_states = '; '.join(j['category'] + ' ' + j['state'] for j in assessment['recovery_manifest']['jobs'])
    operational = (f"At {assessment['at_utc']}, the current inventory has {states.get('running', 0)} running jobs, "
        f"{states.get('pending', 0)} pending jobs and {states.get('failed', 0)} failed jobs. "
        "The original live workers retain their process identities and positive CPU deltas; natural completion is allowed only with final completion records. "
        f"The original automatic combined coordinator remains protected. Polymer repetitions are 50/50 complete and column repetitions {repeat_counts.get('completed', 0)}/50 complete. "
        f"The three recovery states are: {e1_states}. Cases are weights 14, Markov 15 and residual 17. Pending entries start automatically when memory permits. "
        "Do not describe pending jobs as active workers. The supervisor requires 6 GiB commitment headroom, staggers launches and waits for each full restore before another start. "
        "Its independently owned Windows session has no execution time limit or recurring trigger and requires no administrator access.")
    recovery = ("Three failed cycle-190 full checkpoints were preserved and validated. Each native checkpoint loaded twice in fresh owned processes and advanced eight native steps with matching traces and unchanged physical-fidelity bounds. "
        "Full learner/optimizer/replay/controller restoration passed for weights, Markov and residual; weights additionally resumed its newly saved checkpoint in a second process. "
        "The new adapter checks copied model/snapshot hashes before opening, retires only the current worker's recorded PID/creation identity, and records equal full Python-state hashes across native attachment. "
        "Existing scientific/queue/runtime sources remain unchanged. Ten guard tests pass, including reused/unowned PID refusal and verified natural coordinator completion. "
        "The production weights run completed a new paired replay after recovery. Windows recorded four native application faults in disposable validation applications during final cleanup, less than 30 seconds after their verified full checkpoint saves. "
        "Exact PID and creation-time matching excludes all 27 protected identities; the 11 Windows records include duplicate error-reporting records, not eleven distinct crashes. "
        "COM/RPC diagnostic notifications also remain in the production log. Native shutdown remains unresolved; successful state/trace checks and a new replay do not certify clean native shutdown or multi-day stability.")
    evidence = (f"The immutable interim package contains 200 audited primary pairs: combined 80/320, horizon 60/80, and weights/Markov/residual 20/80 each. "
        f"The current live capture count is {assessment['primary_pairs']}/640. "
        "All 200 original trace hashes are unchanged. The audit checks prospective sampling, uniqueness, both completed native branches, live-state restoration, finite core traces and discounted returns recomputed from raw rewards. "
        "The approximately 7 MiB ZIP retains all 200 raw NPZ traces and canonical decisions plus sampling, numerical summaries and provenance. "
        "The findings are mixed: some accepted policy actions yield lower returns, and some beneficial proposals are rejected. Report channels/windows separately; one training seed (7), dependent selected states and incomplete phase coverage do not support confidence, safety, statistical robustness or universal performance claims. "
        "The 200-step CSV uses original discounted reward units, zero sign tolerance, and a separate accepted-delta-below-minus-one sensitivity column. Optional Markov diagnostic NaNs are distinguished from finite core reward/output/input/reference arrays, and null solver counts are not zeros.")
    decision = ("Amir explicitly selected 'Recover E1 and continue toward 640 pairs' after considering a 200-pair stopping option. "
        "No E1 run is intentionally stopped and the 300-cycle schedule, five configurations, four sampling windows, primary 200-step continuations and 10% 800-step subset remain unchanged. "
        "He also selected full lossless archive compression and freshly confirmed permanent deletion of the verified archived cloud copies, conditional on retaining local originals and compressed copies. "
        "His subsequent instruction emphasized avoiding any data loss or rerun cost. Local originals are retained. No E4/E5, unseen-reference or additional seed campaign is introduced.")
    complete = compression['state'] == 'verified_complete'
    if complete:
        independent = read(HERE / 'compressed_archive_verified.json')
        restored = read(HERE / 'restoration_test.json')
        assert independent['verified'] and independent['checked_zips'] == 910
        assert restored['verified'] and len(restored['restored_files']) == 4
        assert restored['original_files_modified'] == restored['original_files_deleted'] == 0
    storage = (f"All 910 original archived folders (3640 files; 174,583,311,764 bytes = 162.6 GiB) were read locally again and SHA-256 matched at {preservation['at_utc']}. "
        "They remain at C:\\Users\\hamediaa\\RL_MPC_local_archive\\storage_20261023. "
        "Complete cycle-190 Python and native-state backups for weights/Markov/residual, plus a 151-file scientific source/protocol ZIP and the 200-pair reviewer ZIP, are hash-verified outside OneDrive under e1_recovery_20261023. "
        "Additional healthy-E1 full checkpoint copies (combined cycle 180; horizon cycle 265) are hash-verified under healthy_e1_checkpoints_20261023, copied exclusively from saved files with zero live-simulator calls. "
        f"Lossless compression is {'verified complete' if complete else 'still running independently'}: {compression['verified_folders']}/910 folders, "
        f"{compression['original_bytes'] / 1024**3:.2f} GiB originals represented by {compression['compressed_bytes'] / 1024**3:.2f} GiB ZIPs. "
        "Each ZIP is decompressed and every restored file SHA-256 is compared with the original manifest; a separate verifier rehashes completed ZIPs and checks the complete restoration mapping. Originals_deleted is zero. "
        "An actual staged extraction of one complete historical native checkpoint passed all four file hashes, with no existing data replaced. The original archive contains zero offline-flag or reparse-point files. "
        "No local disk savings are claimed while originals are retained. Both forms are on C:, so an independent external-drive copy would add protection against drive failure. "
        "Cloud cleanup permanently purged 878 exact whitelisted folders through both recycle-bin stages. The second stage is empty; the 32 remaining local archive folders were not found in either cloud bin. "
        "No unrelated cloud item was deleted. SharePoint Storage Metrics now shows 151.05 GB free of 1024.00 GB, previously 0.00 GB. "
        "The earlier action-time approval rejection was resolved by fresh explicit author confirmation. OneDrive was gracefully quit again using the prior authorization to recover memory; local jobs continue saving normally.")
    limitations = ("Keep active/recent checkpoints and all raw scientific results intact. Do not edit sealed running sources, force-close an owned paused comparison application, or overwrite a newer checkpoint during restoration. "
        "Use the original-path/hash manifests for archive restoration and each failed-run offline_preservation.json for native/Python dependencies. "
        "Checkpoint accumulation in active legacy workers remains possible; retention was not changed in place. No long-term crash-free guarantee is established. "
        "All 46 frozen original manuscript hashes verify; no TeX/PDF or reviewer quotation was edited, no comment was marked resolved and nothing was sent to reviewers.")
    link = f"Record `{RECORD}`; evidence `{EVIDENCE}`; code change report `change-reports/e1_recovery_20261023/STATUS.md`; decision D-2026-10-23-02."
    if commit:
        link += f" Scientific operational commit `{commit}`."
    report = '# Column E1 recovery, partial evidence and data preservation - 2026-10-23\n\n' + '\n\n'.join(
        (decision, operational, recovery, evidence, storage, limitations)) + '\n\n'
    report += ('## Files and provenance\n\nInspected: original failure/status/Windows event evidence; sealed source and budget manifests; native and full Python checkpoints; '
               '200 raw paired traces and canonical decision/restore/sampling records; current process ownership and CPU counters; archive/ZIP SHA-256 receipts; '
               'both cloud recycle-bin stages and Storage Metrics; paper memory and the 46-file original-manuscript manifest.\n\n'
               'Changed: two new isolated root recovery adapters, two focused guard-test files, this report folder and RUN_RECOVERY_STATUS.md; '
               'the dated revised-paper record, partial-evidence copies, seven affected paper-memory records, decision log and figure/table registry. '
               'The unrelated dirty run_polymer_phase2_repeats.py is preserved and excluded from commits. No pre-existing scientific source was edited.\n\n'
               'Generators: audit_partial_results.py, build_partial_analysis.py, audit_live_pairs.py, audit_recovery.py, verify_local_preservation.py, '
               'compress_verified_archive.py, verify_compressed_archive.py, assess_native_shutdown.py and this finalize_record.py in the same change-report folder. '
               'Scientific inputs/units/metrics and trace hashes are in partial_analysis/analysis_manifest.json, summary_200_steps.csv and partial_results_audit.json.\n')
    report += '\n' + link + '\n'
    if not commit:
        (HERE / 'STATUS.md').write_text(report, encoding='utf-8')
        save(HERE / 'final_record_snapshot.json', dict(at_utc=datetime.now(timezone.utc).isoformat(),
            assessment_at=assessment['at_utc'], compression=compression, cloud=cloud, local_originals_deleted=0))
        prepend(ROOT / 'RUN_RECOVERY_STATUS.md', '\n\n'.join((operational, evidence, storage,
                'Evidence: `change-reports/e1_recovery_20261023/STATUS.md` and `recovery_assessment.json`.')))
        return
    destination = PAPER / EVIDENCE
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('ASSESSMENT.md', 'summary_200_steps.csv', 'coverage.png', 'analysis_manifest.json',
                 'decisions.jsonl', 'reviewer_data_200_pairs.zip'):
        shutil.copyfile(HERE / 'partial_analysis' / name, destination / name)
    for name in ('partial_results_audit.json', 'recovery_assessment.json', 'local_preservation_verified.json',
                 'evidence_preservation.json', 'final_record_snapshot.json', 'native_shutdown_assessment.json', 'healthy_e1_preservation.json'):
        shutil.copyfile(HERE / name, destination / name)
    if (HERE / 'compressed_archive_verified.json').exists():
        shutil.copyfile(HERE / 'compressed_archive_verified.json', destination / 'compressed_archive_verified.json')
        shutil.copyfile(HERE / 'restoration_test.json', destination / 'restoration_test.json')
        shutil.copyfile(HERE / 'archive_local_attributes.json', destination / 'archive_local_attributes.json')
    (PAPER / RECORD).write_text(report, encoding='utf-8')
    for name, paragraphs in {
        'ACTIVE_PAPER_STATE': (operational, evidence, storage),
        'FREEZE_STATE': (decision, evidence, limitations),
        'METHOD_FAMILY_MAP': (decision, recovery, limitations),
        'RESULT_BUNDLE_INDEX': (evidence, operational, storage),
        'CLAIM_LEDGER': (evidence, recovery, limitations),
        'EXPERIMENT_BACKLOG': (decision, operational, limitations),
        'OPEN_QUESTIONS': (operational, limitations),
    }.items():
        prepend(PAPER / 'paper-memory' / (name + '.md'), '\n\n'.join((*paragraphs, link)))
    prepend(PAPER / 'paper-memory/DECISION_LOG.md', '**D-2026-10-23-02 - continue E1 and preserve local data.** ' + decision + '\n\n' + storage + '\n\n' + link)
    registry = ("FIG-E1-COLUMN-PARTIAL-COVERAGE-20261023: coverage.png. TAB-E1-COLUMN-PARTIAL-200-20261023: summary_200_steps.csv. "
        "Status: internal/interim reviewer assessment; neither inserted into manuscript/SI. "
        "Source bundle: immutable reviewer_data_200_pairs.zip and decisions.jsonl; raw traces and SHA-256 mapping in analysis_manifest.json and partial_results_audit.json. "
        "Generator: code change-reports/e1_recovery_20261023/build_partial_analysis.py at the operational commit below. "
        "Plant: distillation column; SG mode; seed 7; standalone horizon/weights/Markov/residual and four-channel combined. "
        "Coverage figure counts completed matched pairs against 640 planned (unit: pairs). CSV metrics: 200-step discounted branch returns in original reward units, "
        "selected initial policy/supervisor regret against the better paired initial choice under a common frozen continuation, raw critic sign ranking and gate acceptance counts; "
        "zero sign tolerance plus a minus-one sensitivity count. Claim supported: partial coverage and observed action-choice diagnostics, including lower-return acceptances. "
        "Limitations: one training seed, dependent prospectively selected states, unequal available windows, incomplete later coverage; no safety/confidence/causal or statistically robust superiority inference. "
        "Do not pool horizon's additional windows into an unfair family comparison. Optional Markov diagnostic NaNs and null solver counts are documented explicitly.\n\n" + link)
    prepend(PAPER / 'paper-memory/FIGURE_TABLE_REGISTRY.md', registry)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paper-commit')
    args = parser.parse_args()
    main(args.paper_commit)
