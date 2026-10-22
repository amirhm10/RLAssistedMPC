"""Bounded real Markov native/learner checkpoint acceptance before launch."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))

def main(stop):
    import run_markov_checkpoint_recovery as companion
    import run_remaining_recovery as pool
    from column_review.common import read,write,owned_apps_remaining,process_identity,record_owned_apps
    from review_training_recovery import repeat_checkpoints
    import review_markov_checkpoint_recovery as compiler
    import run_distillation_phase2_repeats as repeat
    plan=read(companion.DEFAULT/'manifest.json')
    assert plan['state']=='prepared' and not (companion.DEFAULT/'scheduled.json').exists()
    assert companion.dependencies()==plan['bootstrap_sha256']
    job=next(j for j in plan['jobs'] if j['kind']=='repeat')
    output=Path(job['directory'])
    old=read(output/'worker_status.json')
    assert process_identity(old['pid'])['alive'] is False
    assert not owned_apps_remaining(output/'owned_aspen_apps.json')
    index=read(output/'training_checkpoints/latest.json',{})
    assert index.get('completed_steps',0)==(8 if stop==16 else 0)
    pool.install_atomic_writes();compiler.install()
    with patch.object(repeat,'MAX_ASPEN_FILES',17),record_owned_apps(output),repeat_checkpoints(job['job'],stop_after=stop) as manager:
        manager.technical_resume_steps=8 if stop==16 else 0
        code=repeat.run_worker(Path(job['job']))
    index=read(output/'training_checkpoints/latest.json')
    assert code==1 and index['completed_steps']==stop
    assert not owned_apps_remaining(output/'owned_aspen_apps.json')
    restored=read(output/'training_checkpoints/last_restore.json',{})
    if stop==16:assert restored['completed_steps']==8 and restored['full_python_state_sha256']
    write(output/f'technical_native_{stop}_passed.json',dict(passed=True,completed_steps=stop,restore=restored,index_sha256=index['sha256']))
    print('PASS real Markov native/full-state checkpoint',stop,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stop',type=int,choices=(8,16));main(parser.parse_args().stop)
