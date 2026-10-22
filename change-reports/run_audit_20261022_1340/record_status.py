"""Synchronize a verified operational audit into local status and paper memory."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
PAPER=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def new(path,text):
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')
def prepend(path,block):
    heading,rest=path.read_text(encoding='utf-8').split('\n',1)
    path.write_text(heading+'\n\n'+block.strip()+'\n'+rest,encoding='utf-8')


def record(code_commit=None):
    data=read(HERE/'final_audit.json');assert not data['issues'] and len(data['workers'])==17
    memory=read(HERE/'memory_latest.json');data['memory_latest']=memory
    observed=datetime.fromisoformat(data['observed_at_utc']).astimezone(timezone(timedelta(hours=-4)))
    phase={w['job']:w['cycle'] for w in data['workers'] if w['study']=='E1_column'}
    sg=[w['cycle'] for w in data['workers'] if w['study']=='column_repeats' and w['job'].startswith('combined/sg')]
    plain=[w['cycle'] for w in data['workers'] if w['study']=='column_repeats' and w['job'].startswith('combined/plain')]
    matched=data['primary_completed'];remaining=640-matched
    headroom=memory['memory']['CommitLimit']-memory['memory']['CommittedBytes']
    fraction=100*memory['memory']['CommittedBytes']/memory['memory']['CommitLimit']
    status=(f'At October 22, {observed:%H:%M} Toronto ({data["observed_at_utc"]}), **17 current jobs are running; '
            f'the polymer threshold recovery has completed; zero current jobs are failed, stopped or queued.** '
            f'The current pool is twelve column repetitions and five reduced column E1 jobs. Polymer repetitions '
            f'remain 50/50 complete; column repetitions remain 38/50 complete, with twelve running. Combined SG '
            f'repetitions are at cycles {min(sg)}-{max(sg)}/300 and plain at {min(plain)}-{max(plain)}/300. '
            f'Markov SG seed 23 is at cycle {next(w["cycle"] for w in data["workers"] if w["job"]=="markov/sg/23")}/300; '
            f'residual plain seed 23 is at {next(w["cycle"] for w in data["workers"] if w["job"]=="residual/plain/23")}/300.')
    e1=(f'Column E1 has {matched}/640 completed primary pairs ({remaining} remaining), with zero failed or '
        f'duplicate pairs and unchanged protocol/source guards. Combined is at cycle {phase["combined"]}, '
        f'with {next(p["recorded"] for p in data["primary_pairs"] if p["category"]=="combined")}/320 pairs. '
        f'The four standalone jobs each have 20/80 release-window pairs; current cycles are horizon '
        f'{phase["horizon"]}, weights {phase["weights"]}, Markov {phase["markov"]}, residual {phase["residual"]}. '
        f'Their next original sampling window starts at cycle 191; they are actively training between '
        f'windows. All 17 worker groups show CPU increases across their Python/Aspen processes in the '
        f'sampled interval; all 22 assigned Aspen applications are live and accounted for. '
        f'The coordinator and all four independent recovery supervisors are live.')
    threshold=('Threshold completed 200/200 cycles at October 22, 06:44 Toronto, with 160000 control steps. '
               'The final trajectory, final inference export and cycle-200 export match their recorded '
               'SHA-256 hashes; four policies contain positive parameter counts and their saved round-trip '
               'validation passes. The final inference export is not a full training-resume checkpoint. '
               'The saved status records one nonfatal optimizer failure among 639973 optimizer calls; '
               'this did not prevent completion. This is export/completion evidence, not a new performance '
               'or unseen-reference evaluation result.')
    ownership=('The residual plain seed-23 ownership JSON still referred to its earlier technical Aspen '
               'process. Its live case-11 worker was progressing normally. The sole unmatched Aspen '
               'application was verified by exact executable, case-11 title, creation time immediately '
               'after the original worker and exclusion of the other 21 recorded owners. Only that '
               'ownership JSON was atomically corrected, with the original file and identity proof saved. '
               'All 17 worker creation identities are unchanged; no process was signaled and no scientific '
               'source or simulator state was changed. Future recovery should refresh ownership metadata '
               'rather than assume a technical-test record identifies the current production application.')
    risk=(f'Current Windows memory commitment is {fraction:.1f}% of its current limit, with '
          f'{headroom/1024**3:.2f} GiB of commitment headroom. The pagefile is system-managed; system '
          f'settings were not changed. Windows reported RADAR_PRE_LEAK_64 for Aspen at 02:44; this is '
          f'a resource warning, not proof of the underlying leak mechanism. No new Python/Aspen '
          f'APPCRASH/Application Error event appears after the preceding 00:23 verified audit. '
          f'All jobs are progressing, but memory headroom remains a reliability concern. No multi-day '
          f'stability guarantee is established.')
    scope=('The active batch has no pending queue entries. Frozen-policy unseen-reference follow-ups and '
           'the separately scoped E4/E5 work have not been launched by this pool. E6/E7 retain their '
           'accepted justification/limitations route. Previously completed E1/E2/E3 evidence is retained. '
           'The 46 frozen original manuscript hashes verify; no TeX/PDF was edited. This status audit '
           'does not commission a new scientific run or alter the approved 640-pair E1 protocol.')
    timing=('Remaining repetitions provisionally require about 1-2 more days at current progress. Column '
            'E1 remains the longest workstream, with several days left; the earlier 4-6-day planning '
            'allowance remains conditional on later branch throughput and memory/native stability. '
            'This audit does not establish a firm completion deadline.')
    body='\n\n'.join((status,e1,threshold,ownership,risk,scope,timing))+'\n'
    if not code_commit:
        new(HERE/'STATUS.md','# Current run status - October 22, 2026\n\n'+body+
            '\nInspected: current study/recovery manifests and source/bootstrap guards; worker status '
            'and checkpoint indexes; sampled Python/Aspen CPU and creation identities; E1 sampling/records; '
            'threshold trajectory/policy hashes; Windows resources/events; frozen-original manifest.\n\n'
            'Changed: the residual case-11 ownership JSON only in operational run metadata, its preserved '
            'backup/proof in this audit folder, RUN_RECOVERY_STATUS.md, and this audit/report utility/evidence. '
            'No scientific source, method, model, checkpoint, raw results or process was modified.\n')
        new(HERE/'assessment.json',json.dumps(data,indent=2)+'\n')
        prepend(ROOT/'RUN_RECOVERY_STATUS.md','## Current run audit - October 22, '+observed.strftime('%H:%M')+' Toronto\n\n'+status+'\n\n'+e1+'\n\n'+risk+
                '\n\nDetails and preserved ownership evidence: `change-reports/run_audit_20261022_1340/STATUS.md` '
                'and `assessment.json`.\n')
    else:
        data['scientific_audit_commit']=code_commit
        record_path='manuscript_revised/ALL_RUNS_STATUS_2026_10_22.md'
        evidence='manuscript_revised/evidence/all_runs_20261022/status_assessment.json'
        reference=f'Record `{record_path}`; evidence `{evidence}`; scientific audit commit `{code_commit}`; observation {data["observed_at_utc"]}.'
        new(PAPER/record_path,'# Current run status - October 22, 2026\n\n'+body+'\n'+reference+'\n')
        new(PAPER/evidence,json.dumps(data,indent=2)+'\n')
        blocks={
            'ACTIVE_PAPER_STATE':[status,e1,threshold,risk,scope],
            'FREEZE_STATE':[status,scope,risk],
            'METHOD_FAMILY_MAP':[e1,'No active family, algorithm, mode, seed, controller, scenario or approved E1 sampling budget changed.'],
            'RESULT_BUNDLE_INDEX':[status,threshold,e1],
            'CLAIM_LEDGER':[threshold,e1,risk,'Current activity/completion and export integrity are verified; general robustness, causal contribution, safety guarantees and unseen-reference performance are not promoted.'],
            'EXPERIMENT_BACKLOG':[status,e1,scope,timing],
            'OPEN_QUESTIONS':[ownership,risk,scope,timing],
        }
        for name,paragraphs in blocks.items():
            prepend(PAPER/'paper-memory'/f'{name}.md','## Current run audit - 2026-10-22 '+observed.strftime('%H:%M')+' Toronto\n\n'+'\n\n'.join(paragraphs)+'\n\n'+reference+'\n')


if __name__=='__main__':record(sys.argv[1] if len(sys.argv)>1 else None)
