"""Validate saved native/learner integration and the independent production launch."""
import argparse
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import review_blas_runtime
review_blas_runtime.activate()
import numpy as np
from column_review.common import read,write,digest,now,process_identity,owned_apps_remaining,check_job
from run_column_e1 import sources,guard_queue
from revision_comparisons.progress import snapshot,log_tail

def technical(directory):
    status=read(directory/'status.json')
    assert status['state']=='technical_checkpoint' and status['completed_steps']==88
    assert status['resumed_from_step']==80 and status['checkpoint_has_four_learners']
    assert status['matched_pairs']==4 and set(status['channels'])=={'horizon','weights','markov','residual'}
    assert not owned_apps_remaining(directory/'owned_aspen_apps.json')
    restore=read(directory/'training_checkpoints/last_restore.json')
    assert restore['completed_steps']==80 and restore['full_python_state_sha256']
    index=read(directory/'training_checkpoints/latest.json')
    assert index['completed_steps']==88 and index['training_resume']
    assert len(index['agents'])==4 and all(a['train_steps']>0 and a['replay_size']>0 for a in index['agents'].values())
    native=index['native']
    for key in ('file','project_file'): assert digest(native[key])==native[key+'_sha256']
    assert digest(directory/'training_checkpoints'/index['file'])==index['sha256']
    checks=[json.loads(line) for line in (directory/'capture/native_restore_checks.jsonl').read_text().splitlines()]
    assert len(checks)==4
    reinit=[]
    for row in checks:
        assert row['live_native_unchanged'] and row['live_learner_unchanged']
        assert row['full_state_before']==row['full_state_after'] and row['full_state_before']
        assert np.array_equal(row['live_native_before'],row['live_native_after'])
        assert len(row['repeat_checks'])==2 and all(c['steps']==4 for c in row['repeat_checks'])
        for prime in row['zero_time_initializations']:
            difference=np.abs(np.asarray(prime['after'])-prime['before'])
            assert difference[0]<=1e-9 and difference[9]<=1e-6 and difference[14]<=2e-3
            reinit.append(difference[[9,14]])
    decisions=[json.loads(line) for line in (directory/'capture/decisions.jsonl').read_text().splitlines()]
    assert len(decisions)==4 and all(r['status']=='completed' for r in decisions)
    traces=directory/'capture/integration_traces'
    repeats=[]
    for path in traces.glob('*_first.npz'):
        other=path.with_name(path.name.replace('_first.npz','_repeat.npz'))
        with np.load(path) as first,np.load(other) as second:
            for field,atol in (('y',[1e-8,1e-4]),('u',1e-3),('reward',1e-5),('reference',1e-10)):
                assert np.allclose(first[field],second[field],rtol=1e-7,atol=atol), (path,field)
            repeats.append({'file':str(path),'max_output_difference':np.max(np.abs(first['y']-second['y']),axis=0).tolist()})
    assert len(repeats)==8
    portable=read(ROOT/'Distillation/Results/column_e1/20261020_portable_replay_a1/result.json')
    assert portable['passed']
    return {'passed':True,'directory':str(directory),'status':status,'fresh_resume':restore,'portable_replay':portable,
            'checkpoint':index,'matched_pairs':4,'repeated_branch_checks':repeats,
            'live_state_hash_checks':[{'channel':r['channel'],'before':r['full_state_before'],'after':r['full_state_after']} for r in checks],
            'maximum_reinitialization_output_difference':np.max(reinit,axis=0).tolist(),
            'native_fidelity_tolerance':[1e-6,2e-3], 'full_production_performance_evidence':False}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--technical',type=Path,required=True)
    parser.add_argument('--production',type=Path)
    args=parser.parse_args()
    tech=technical(args.technical.resolve())
    if not args.production:
        write(Path(__file__).with_name('technical_assessment.json'),tech)
        print(json.dumps({'technical_passed':True,'pairs':4,'resume_step':80,'advanced_to':88},indent=2))
        return
    directory=args.production.resolve()
    manifest=read(directory/'manifest.json')
    assert manifest['state']=='running' and len(manifest['jobs'])==5
    running=[j for j in manifest['jobs'] if j['state']=='running']
    assert len(running)==1 and running[0]['category']=='combined'
    job=running[0]
    worker_dir=Path(job['directory'])
    status=read(worker_dir/'status.json')
    assert status['state']=='running' and status['case']==16 and not status['technical']
    assert process_identity(job['pid'])==job['identity'] and job['identity']['alive'] is True
    for item in manifest['jobs']:
        metadata=read(item['job'])
        assert metadata['sources']==sources() and metadata['profile']=='temperature_flip_200_100'
        assert digest(metadata['source_case']['aspen_path'])==metadata['source_case']['aspen_sha256']
    apps=read(worker_dir/'owned_aspen_apps.json')
    assert len(apps)==2 and all(process_identity(a['pid'])['created']==a['created'] and process_identity(a['pid'])['alive'] for a in apps)
    checkpoint=read(worker_dir/'training_checkpoints/latest.json')
    assert checkpoint['completed_cycles']>=1 and checkpoint['training_resume'] and len(checkpoint['agents'])==4
    assert digest(worker_dir/'training_checkpoints'/checkpoint['file'])==checkpoint['sha256']
    guard_queue()
    check_job(read(ROOT/'Distillation/Results/column_review/20261018_recovered/bo/job.json'))
    baseline=read(Path(__file__).with_name('preflight.json'))
    preserved=[]
    for row in baseline['protected_workers']:
        assert process_identity(row['identity']['pid'])==row['identity']
        preserved.append(dict(row,unchanged=True))
    assert len(preserved)==13
    current=snapshot()
    coordinator=current['repeat_queue_coordinator']
    assert coordinator['pid']==baseline['coordinator']['pid'] and coordinator['alive']
    broker=read(directory/'broker_identity.json')
    assert broker['parent_is_scheduler_service'] and broker['session_id']>0
    assert process_identity(broker['pid'])['alive'] is True
    scheduled=read(directory/'scheduled.json')
    assert scheduled['triggers']==0 and scheduled['execution_time_limit']=='PT0S'
    modules=json.loads(Path(__file__).with_name('production_numerical_modules.json').read_text(encoding='utf-8-sig'))
    assert len(modules)==2 and all('run_recovery_20261020' in row['FileName'] and 'site-packages' in row['FileName'] for row in modules)
    viewer=read(ROOT/'Distillation/Results/column_e1/20261020_live_progress/identity.json')
    assert viewer['parent_is_scheduler_service'] and process_identity(viewer['pid'])['alive']
    cycles=re.findall(r'Sub_Episode:\s*(\d+)',log_tail(worker_dir/'worker.log'))
    tests=ET.parse(Path(__file__).with_name('tests.xml')).getroot()
    suites=tests.findall('testsuite') if tests.tag=='testsuites' else [tests]
    assert all(int(s.get('errors','0'))==int(s.get('failures','0'))==0 for s in suites)
    count=sum(int(s.get('tests','0')) for s in suites)
    paper=ROOT.parent/'Rl_assisted_paper/RL_assisted_paper'
    freeze=read(paper/'manuscript_revised/frozen_original_manifest.json')
    assert len(freeze['files'])==46 and all(digest(paper/'manuscript'/r['path'])==r['sha256'] for r in freeze['files'])
    production={'state':'running','case':16,'pid':job['pid'],'cycle':int(cycles[-1]) if cycles else 0,
                'seed':7,'directory':str(directory),'worker_directory':str(worker_dir),'native_apps':apps,
                'checkpoint':checkpoint,'configurations':5,'pending_configurations':4,'expected_matched_pairs':3200,
                'broker':broker,'scheduled':scheduled}
    result={'observed_at_utc':now(),'technical':tech,'production':production,'combined_queue_guards_pass':True,
            'combined_reserved_cases':baseline['combined_reserved_cases'],'coordinator':coordinator,
            'protected_workers_preserved':preserved,'current':current,
            'production_numerical_modules':modules,'read_only_progress_viewer':viewer,
            'verification':{'focused_tests_passed':count,'frozen_manuscript_files_unchanged':46},
            'limitations':['Production E1 is in progress; the technical test is not performance evidence.',
                           'One seed; combined channel differences are conditional diagnostics.',
                           'Native restart/branch reinitialization uses explicit physical tolerances, not a bitwise integrator-state guarantee.',
                           'The private BLAS runtime mitigates the observed numerical path; multi-day completion remains pending.']}
    write(Path(__file__).with_name('launch_assessment.json'),result)
    print(json.dumps({'production':{k:production[k] for k in ('state','case','pid','cycle','pending_configurations')},
                     'preserved_workers':13,'combined_guards_pass':True,'tests':count},indent=2))

if __name__=='__main__': main()
