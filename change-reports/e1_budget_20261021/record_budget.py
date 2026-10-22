"""Record the verified E1 budget amendment without editing frozen run sources."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT.parent / 'Rl_assisted_paper/RL_assisted_paper'
REPORT = Path(__file__).resolve().parent


def write_new(path, text):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def prepend(path, block):
    text = path.read_text(encoding='utf-8')
    heading, rest = text.split('\n', 1)
    if '## Reduced column E1 budget - 2026-10-22' in rest:
        raise RuntimeError(f'Already recorded: {path}')
    path.write_text(heading+'\n\n'+block.strip()+'\n'+rest, encoding='utf-8')


def record(audit_path, code_commit=None):
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    assert not audit['issues'] and len(audit['jobs']) == 5
    assert len(audit['protected_jobs']) == 13
    assert audit['original_coordinator_preserved'] and audit['frozen_original']['verified']
    assert len(audit['native_tests']) >= 2 and all(t['passed'] for t in audit['native_tests'])
    now = datetime.fromisoformat(audit['observed_at_utc']).astimezone(ZoneInfo('America/Toronto'))
    supplemental = sum(j['supplemental_pairs'] for j in audit['jobs'])
    compact = dict(observed_at_utc=audit['observed_at_utc'], protocol=audit['protocol'],
                   protocol_sha256=audit['protocol_sha256'], issues=audit['issues'],
                   primary_pairs=640, supplemental_pairs=supplemental,
                   protected_jobs=audit['protected_jobs'], coordinator=audit['coordinator'],
                   supervisor=audit['supervisor'], broker=audit['broker'],
                   broker_identity=audit['broker_identity'], progress_viewers=audit['progress_viewers'],
                   original_budget_supervisor=audit['original_budget_supervisor'],
                   lazy_protected_jobs=audit['lazy_protected_jobs'],
                   lazy_recovery_protocol=audit['lazy_recovery_protocol'],
                   lazy_recovery_protocol_sha256=audit['lazy_recovery_protocol_sha256'],
                   original_coordinator_preserved=True, frozen_original=audit['frozen_original'],
                   native_tests=audit['native_tests'], progress_counts=audit['progress']['counts'],
                   jobs=[dict(category=j['category'], case=j['case'], identity=j['identity'],
                              state=j['state'], expected_pairs=j['expected_pairs'],
                              supplemental_pairs=j['supplemental_pairs'], migration=j['migration'],
                              checkpoint_steps=j['checkpoint']['completed_steps'],
                              pre_capture_timing=j['pre_capture_timing'],
                              restored_in_current_process=j['restored_in_current_process'],
                              restore={key:j['restore'].get(key) for key in
                                       ('completed_steps','native_checksum_verified','python_checksum_verified','full_python_state_sha256')},
                              numpy=j['numeric']['numpy'], options=j['sampling']['options'],
                              selected_steps=j['sampling']['selected_steps']) for j in audit['jobs']])
    status = (f'At {now:%B %d, %H:%M Toronto} ({audit["observed_at_utc"]}), all five reduced column E1 jobs '
              f'are running on cases 13-17. All 13 unrelated production workers retain their original identities '
              f'and the original automatic combined coordinator is preserved. There are 18 active remaining jobs, '
              f'with no current failed or queued job in this verified inventory. Historical failures and replaced '
              f'E1 attempts remain recorded. The 50 polymer and 38 column completed repetitions are unchanged.')
    design = ('Column E1 now uses 20 decisions per channel in each of four original training windows, '
              'giving 640 primary paired tests instead of 3200 (80% fewer). Combined contributes 320; '
              'horizon, weights, Markov and residual contribute 80 each. All five configurations retain seed 7, '
              'the 300-cycle temperature_flip_200_100 main training schedule, 200-step primary continuations '
              'and a 10% 800-step subset. Polymer E1 remains the completed 3200-pair study.')
    sampling = ('Within each channel/window, the new schedule is a deterministic nested subset of the original '
                '100 prospective points: target 10 early, 5 middle and 5 late clock positions in each 200-step '
                'reference block, with one point per time-ordered bin. Scarce strata have explicit quota '
                'reallocation. Independent RNGs avoid perturbing the learner. Selection never uses observed '
                'returns. These clock groups do not establish measured settling or statistical independence. '
                'Report conditional diagnostics and clock-stratum summaries; an unweighted aggregate does '
                'not estimate overall operating-time failure frequency. Precision and coverage are reduced; '
                'there is still only one training seed, with no causal/safety/general-robustness claim.')
    integrity = (f'All five handoffs restore complete pre-capture native/Python checkpoints. Every migration '
                 f'has identical full Python learner/controller state hashes before and after the sampling '
                 f'change. Original checkpoints and {supplemental} completed pre-handoff paired captures '
                 f'are retained under the budget manifest\'s originals directory as supplemental evidence, '
                 f'separate from the replayed primary trajectory. The combined native test advanced '
                 f'4000 -> 4008, then a fresh process restored and advanced 4008 -> 4016 with the full saved '
                 f'Python-state hash verified. Both tests closed their owned Aspen applications. The original '
                 f'physical-fidelity bounds remain in force. New checkpoints include the immutable budget '
                 f'protocol hash and save every cycle. The 37 focused budget/progress/E1 tests passed.')
    initialization_fix = ('The first reduced handoff exposed a migration bug: weights, Markov and residual '
                          'checkpoints at step 4000 predate their first gate, so their sampling tables were '
                          'empty. The adapter originally assumed those tables existed and raised KeyError. '
                          'A separately sealed process-local companion rebuilds each original 100-point '
                          'prospective schedule, verifies exact equality with the preserved sampling.json, '
                          'then applies the unchanged 20-point reduction. The complete controller-state '
                          'hash still matches before/after migration. A bounded real weights continuation '
                          '4000 -> 4008 passed before the three failed jobs restarted. New checkpoint '
                          'provenance also seals this repair. The healthy combined/horizon E1 workers and '
                          'all 13 unrelated workers were preserved. Failed status/log records remain saved; '
                          'the companion aggregates all five configurations when they finish.')
    timing = ('Counterfactual Aspen advances fall from 1664000 to 332800. Including main training, the '
              'campaign falls from 2264000 to 932800 advances. The longest configuration, combined, falls '
              'from 952000 to 286400, about 70% less. At one second per advance that is 3.3 days before '
              'startup, snapshot, checkpoint and contention overhead; roughly 4-6 days is a provisional '
              'planning allowance if that throughput holds. This is not a promised completion date. '
              'Representative full-load branch throughput and multi-day native stability remain open.')
    observed_rates=[j['pre_capture_timing']['seconds_per_advance'] for j in audit['jobs'] if j['pre_capture_timing']]
    if observed_rates:
        timing += (f' Current pre-capture continuations measured {min(observed_rates):.2f}-'
                   f'{max(observed_rates):.2f} seconds per advance; this includes checkpoint overhead '
                   'but does not measure the later counterfactual branches.')
    execution = ('Only identity-verified E1 workers and their recorded owned Aspen trees were stopped. '
                 'Concrete stop plans, creation identities and before/after protected-worker proofs are '
                 'retained. The old supervisors remain available for their unrelated jobs. The new five-job '
                 'supervisor and live progress viewer run independently through the Windows scheduler broker '
                 'with no execution time limit; current primary targets are 320/80, totaling 640. The original '
                 'scientific sources, sealed recovery bootstraps, source models and private NumPy 2.5.3 '
                 'runtime pass their guards. No shared package, scientific controller, TeX/PDF or frozen '
                 'original manuscript file was edited. All 46 frozen original hashes verify.')
    decision = ('Amir explicitly requested: "I just don\'t have enought time or budget for several weeks of run? '
                'Let\'s make it less". D-2026-10-22-01 authorizes this E1-only sampling-budget reduction and '
                'the necessary checkpoint handoff. It supersedes the unchanged 100-decision/3200-pair '
                'column E1 budget retained in D-2026-10-21-01; all other settings and accepted E4/E5/E6/E7 '
                'scopes remain as previously selected.')
    body = '\n\n'.join([decision, status, design, sampling, integrity, initialization_fix, execution, timing])+'\n'
    if code_commit is None:
        write_new(REPORT/'recovery_assessment.json', json.dumps(compact, indent=2)+'\n')
        write_new(REPORT/'RECOVERY.md', '# Reduced column E1 budget - October 22, 2026\n\n'+body+
                  '\nInspected: run_column_e1.py; column_e1/runtime.py and checkpoints.py; '
                  'review_experiments/e1/study.py and analysis.py; existing recovery entrypoints and manifests; '
                  'all five full E1 checkpoint indexes, owned-application records and capture schedules; '
                  'revision_comparisons/progress.py; live protected-worker/queue identities; paper memory '
                  'and the frozen-original manifest.\n\n'
                  'Changed: new column_e1_budget.py, run_column_e1_budget.py, '
                  'column_e1_budget_lazy_recovery.py and run_column_e1_budget_lazy_recovery.py; '
                  'tests/test_column_e1_budget.py and test_column_e1_budget_lazy_recovery.py; '
                  'revision_comparisons/progress.py; RUN_RECOVERY_STATUS.md; '
                  'this scoped change report and audit/record utilities. Operational manifests, new job '
                  'files, checkpoints and archived captures remain local in ignored result directories. '
                  'The pre-existing run_polymer_phase2_repeats.py edit is excluded.\n\n'
                  'Validation: 37 tests passed using the private Python runtime; two real combined native '
                  'checkpoint continuations and the real lazy weights continuation passed; '
                  'recovery_assessment.json records all five migration '
                  'hashes, actual workers, schedules and the protected identities. Audit utility: audit_budget.py.\n')
        prepend(ROOT/'RUN_RECOVERY_STATUS.md', '## Reduced column E1 budget - 2026-10-22\n\n'+status+'\n\n'+design+'\n\n'+integrity+'\n\n'+timing+
                '\n\nDetails and live activation evidence: `change-reports/e1_budget_20261021/RECOVERY.md` '
                'and `recovery_assessment.json`. The prior unchanged-budget statements below are historical.\n')
        status_path = ROOT/'RUN_RECOVERY_STATUS.md'
        status_path.write_text(status_path.read_text(encoding='utf-8').replace(
            'latest update October 21, 2026', 'latest update October 22, 2026', 1), encoding='utf-8')
    else:
        compact['scientific_code_commit'] = code_commit
        evidence = 'manuscript_revised/evidence/column_e1_budget_20261022/recovery_assessment.json'
        record_path = 'manuscript_revised/COLUMN_E1_REDUCED_BUDGET_2026_10_22.md'
        reference = (f'Record `{record_path}`; evidence `{evidence}`; scientific code commit `{code_commit}`; '
                     f'decision D-2026-10-22-01; observation {audit["observed_at_utc"]}.')
        write_new(PAPER/evidence, json.dumps(compact, indent=2)+'\n')
        write_new(PAPER/record_path, '# Reduced column E1 budget - October 22, 2026\n\n'+body+'\n'+reference+
                  '\n\nPaper changes: this protocol/activation record, its compact evidence and the '
                  'seven affected paper-memory files plus DECISION_LOG. No manuscript prose, figure or PDF '
                  'is revised or promoted to completed evidence. The detailed inspected/changed file list '
                  'and technical verification are in the scientific code change report.\n')
        blocks = {
            'ACTIVE_PAPER_STATE': [status, design, integrity, timing],
            'FREEZE_STATE': [decision, status, execution, 'Execution remains active; completed column E1 evidence and final production validity are pending.'],
            'METHOD_FAMILY_MAP': [decision, design, sampling],
            'RESULT_BUNDLE_INDEX': [status, design, integrity, 'Classification: implementation checked and active, not paper-ready completed scientific evidence. Raw primary/supplemental separation and numerical provenance are retained.'],
            'CLAIM_LEDGER': [design, sampling, integrity, 'Allowed: verified implementation, reduced work budget, successful bounded native/Python resume and current activation. Blocked: completed E1, causal contribution, general safety, seed-replicated robustness or a guaranteed completion date.'],
            'EXPERIMENT_BACKLOG': [decision, status, design, timing, 'Finish and validate these 18 existing jobs. Analyze the 640 primary pairs by context, phase and clock stratum, and keep supplemental captures separate. No additional campaign is launched.'],
            'OPEN_QUESTIONS': [decision, integrity, initialization_fix, timing, 'Still open: representativeness and precision under the smaller sample; full-load branch throughput; production completion and long-duration numerical stability. Clock-stratified selection does not prove distinct physical states or measured settling.'],
            'DECISION_LOG': [decision, design, sampling, status],
        }
        for name, paragraphs in blocks.items():
            prepend(PAPER/'paper-memory'/f'{name}.md', '## Reduced column E1 budget - 2026-10-22\n\n'+'\n\n'.join(paragraphs)+'\n\n'+reference+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('audit', type=Path)
    parser.add_argument('--code-commit')
    args = parser.parse_args()
    record(args.audit, args.code_commit)
