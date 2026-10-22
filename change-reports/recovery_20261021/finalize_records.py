"""Synchronize execution records only after a clean final launch assessment."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
OUT=Path(__file__).resolve().parent
HEADING='## NumPy and parallel E1 recovery — 2026-10-21'


def prepend(path,text):
    original=path.read_text(encoding='utf-8')
    if HEADING in original:raise RuntimeError('Record already synchronized: '+str(path))
    title,rest=original.split('\n',1)
    path.write_text(title+'\n\n'+HEADING+'\n\n'+text.strip()+'\n\n'+rest.lstrip(),encoding='utf-8')


def main(paper,commit):
    data=json.loads((OUT/'recovery_assessment.json').read_text())
    assert not data['issues']
    assert all(job['state']=='running' and job['actual_identity']['alive'] for job in data['recovery_jobs'])
    assert len(data['recovery_jobs'])==8
    assert len(data['queue']['jobs'])==10 and all(job['state']=='running' and job['actual_identity']['alive'] for job in data['queue']['jobs'])
    assert data['queue']['coordinator_same_identity'] and not data['recovery_case_collisions']
    observed=data['observed_at_utc']
    local=datetime.fromisoformat(observed).astimezone(ZoneInfo('America/Toronto')).strftime('%B %d, %H:%M Toronto')
    counts=data['progress']['counts']
    status=(f"At {local} ({observed}), **18 remaining production jobs are active, with no current failed or queued job**: "
            "ten combined repetitions, Markov SG seed 23, residual plain seed 23, five column E1 configurations and polymer threshold recovery. "
            f"Polymer repetitions remain 50/50 complete; column repetitions are {counts['column_repeats'].get('completed',0)}/50 complete. "
            "The four residual workers healthy at the preceding audit finished naturally. The original coordinator retains its process creation identity and started the combined cohort automatically.")
    fixes=("New/recovered workers use a verified private NumPy 2.5.3 build with conservative single-thread numerical dispatch; the shared environment is unchanged. "
           "The E1 adapter copies all snapshot variable classes back after zero-time initialization and rechecks the original fidelity bounds. "
           "The failed step-5202 reproduction, four-channel native/learner integration and fresh-process E1, residual and Markov full-state restart checks pass. "
           "All 38 focused tests pass. Threshold saves four nonempty finite policies and complete training state. "
           "The Markov try/finally/context checkpoint incompatibilities and bounded-test stop error were corrected; historical failures and test-stop identity evidence remain saved.")
    parallel=("E1 runs five unchanged configurations concurrently on cases 13–17 (horizon 13, weights 14, Markov 15, combined 16, residual 17). "
              "Case 17 is a private byte-identical source copy. Recovery repetitions use 3 and 11; combined retains 1,2,12,4–10. "
              "There is no case overlap. Independent visible Windows sessions and a 30-second read-only progress viewer are active. "
              "The original 300-cycle/3200-pair E1 budget and all seeds, schedules and controller settings are unchanged.")
    limits=("Multi-day numerical stability, the underlying native-crash mechanism and full production E1 validity remain unestablished. "
            "Bounded numerical agreement is not bitwise full-trajectory equivalence across runtime versions. "
            "Preserve per-bundle numerical provenance when interpreting the mixed-version campaign. "
            "Allow roughly 36–60 hours for remaining column repetitions/combined, provisionally 8–12 hours for threshold, and one to several weeks for E1 depending on branch throughput and full-load overhead. "
            "The largest E1 configuration still needs about 952000 Aspen advances (about 11 days at one second per advance before overhead). "
            "These are conditional planning estimates. No completed E1 production, causal, safety or general performance claim is promoted.")
    if not paper:
        prepend(ROOT/'RUN_RECOVERY_STATUS.md','\n\n'.join((status,fixes,parallel,limits,
                 'Evidence and details: `change-reports/recovery_20261021/RECOVERY.md` and `recovery_assessment.json`. All 46 frozen original manuscript files verify unchanged.')))
        return
    assert commit and len(commit)==40
    frozen=json.loads((PAPER/'manuscript_revised/frozen_original_manifest.json').read_text())
    assert all(hashlib.sha256((PAPER/'manuscript'/item['path']).read_bytes()).hexdigest()==item['sha256'] for item in frozen['files'])
    evidence=PAPER/'manuscript_revised/evidence/numpy_e1_recovery_20261021'
    evidence.mkdir(exist_ok=False)
    compact=dict(data,scientific_recovery_commit=commit,decision='D-2026-10-21-01',
                 evidence_scope='Operational recovery/technical acceptance; no completed production E1 result claim')
    (evidence/'recovery_assessment.json').write_text(json.dumps(compact,indent=2)+'\n')
    record='manuscript_revised/NUMPY_E1_PARALLEL_RECOVERY_2026_10_21.md'
    reference=(f"Record `{record}`; evidence `manuscript_revised/evidence/numpy_e1_recovery_20261021/recovery_assessment.json`; "
               f"scientific recovery commit `{commit}`; decision D-2026-10-21-01; observation {observed}.")
    body='\n\n'.join((status,fixes,parallel,limits,
          "Amir explicitly authorized repairing the NumPy/E1 failures, restarting all unfinished existing runs without interrupting the others or blocking combined, and using free Aspen cases to speed E1. This is execution/runtime recovery and parallel scheduling of the existing approved configurations, not a new scientific campaign or a reduced sampling budget. E4/E5 and accepted E6/E7 response scope remain unchanged.",
          "Three original crash dumps identify the old NumPy extension at the same offset; this is not a proven OpenBLAS-kernel cause. The official new wheel and 945-file inventory are pinned locally. The 50-second production-kernel stress check completed 2911 Markov fits and 8733 linear-algebra batches with SLSQP/PyTorch coexistence and the reference fit unchanged. At the old E1 failure snapshot, three four-step repeats pass with zero post-copy diagnostic differences. Technical integration produced four completed matched decisions and full-state fresh-process restarts, without establishing completed production E1 evidence.",
          "Combined E1 resumes its original cycle-10/step-4000 native/Python checkpoint. The failed repetitions and threshold lacked complete checkpoints from their old trajectories, so their unchanged experiments restart from original initial conditions, with complete checkpoints added. Old data and partial inference weights remain separate. The corrected Markov bounded test restores step 8 and advances to 16 with the complete Python-state hash verified. Only its earlier misbounded technical process and verified case-3 Aspen tree were stopped; production jobs and the queue were preserved.",
          "The code report `change-reports/recovery_20261021/RECOVERY.md` lists inspected and changed files and all technical evidence. The frozen 46-file original manuscript snapshot verifies. No TeX/PDF, original results, notebooks, canonical source models or scientific controller source is edited. The unrelated pre-existing code edit is excluded from the recovery commit. Paper changes are this record, compact evidence and synchronized memory/decision entries; commits remain local.",reference))
    (PAPER/record).write_text('# NumPy, E1 and parallel remaining-run recovery — October 21, 2026\n\n'+body+'\n')
    additions={
      'ACTIVE_PAPER_STATE.md':'\n\n'.join((status,fixes,parallel,limits,reference)),
      'FREEZE_STATE.md':'\n\n'.join((status,"Execution is active; final production E1 evidence and final trained threshold weights remain pending. Original manuscript: 46/46 hashes unchanged. No TeX/PDF edits. Frozen original, accepted method/response scope and claim limits remain in force.",limits,reference)),
      'METHOD_FAMILY_MAP.md':'\n\n'.join((parallel,"D-2026-10-21-01 authorizes runtime recovery and concurrent execution of the existing five E1 configurations. It introduces no new learner, family, scenario, seed inventory, rollout budget or manuscript performance claim. Active DQN/TD3 SG/plain scope and archived-method exclusions are unchanged.",reference)),
      'RESULT_BUNDLE_INDEX.md':'\n\n'.join((status,fixes,"Current recovery outputs: scientific repository `Distillation/Results/state_recovery/20261021_remaining_active/`, Markov companion `20261021_markov_checkpoint_a2/`, original combined E1 attempt `Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1`, automatic combined cohort `Distillation/Results/review_recovery/20261020_queue_case12/combined/`. Recovery checkpoint/policy artifacts are partial operational evidence; new threshold weights belong to the new trajectory. Previously completed polymer E1/E2/E3 and column E2/E3 evidence is retained.",reference)),
      'CLAIM_LEDGER.md':'\n\n'.join(("Implementation/technical evidence: isolated runtime replacement; original-bound CopyValues restoration; full learner/controller/native checkpoint restart; unchanged-protocol parallel E1 and successful automatic combined handoff. "+fixes,limits,reference)),
      'EXPERIMENT_BACKLOG.md':'\n\n'.join((status,parallel,"Finish and validate the 18 active jobs and their final exports; inspect any actionable native/ownership/checkpoint failure. Reassess E1 timing from production branch throughput. Preserve failed attempts and numerical-version metadata. No additional method, E4/E5 change, network-size campaign, recent-method reproduction or NMPC benchmark is inferred.",limits,reference)),
      'OPEN_QUESTIONS.md':'\n\n'.join(("Resolved at the technical level: reproducible E1 priming drift within the declared bounds, Markov checkpoint try/finally/context support, nonempty threshold exports/full-state checkpointing, and automatic combined launch. "+fixes,"Still open: multi-day NumPy/native stability and exact root cause; representative full-load E1 branch throughput/completion time; completion/validation of all production E1 pairs and trained threshold policies. Technical tests do not establish scientific performance or causal robustness.",reference)),
    }
    for name,text in additions.items():prepend(PAPER/'paper-memory'/name,text)
    prepend(PAPER/'paper-memory/DECISION_LOG.md','\n\n'.join((
          "**D-2026-10-21-01 — explicit author execution decision.** Amir requested repair of NumPy crashes and E1 snapshot restoration, recovery and activation of every remaining existing run without interrupting others or blocking combined, and faster E1 using free Aspen cases.",
          parallel,"This authorizes private numerical/runtime and checkpoint fixes, guarded restarts and parallel execution at the existing budget. It does not authorize a new scientific method/campaign or reduced evidence budget. The complete step-5202/native/learner/restart tests pass; long production stability remains pending.",status,reference)))
    print('Paper status and eight affected memory/decision records synchronized; 46 frozen hashes verified.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper',action='store_true');parser.add_argument('--code-commit')
    args=parser.parse_args();main(args.paper,args.code_commit)
