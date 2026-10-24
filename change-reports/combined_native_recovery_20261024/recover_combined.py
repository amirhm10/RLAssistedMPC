"""Recover only the retired combined failure on case 16, with four learners."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT/'change-reports/weights_pid_reuse_recovery_20261024'))
import recover_weights as engine

engine.HERE = HERE
engine.CATEGORY = 'combined'
engine.CASE = 16
engine.SCRIPT = Path(__file__).resolve()
engine.OUTPUT = ROOT/'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
engine.STATE = ROOT/'Distillation/Results/state_recovery/20261024_combined_native_retry'
engine.ORIGINAL = ROOT/'Distillation/Results/state_recovery/20261021_e1_budget20'


def scope(plan):
    if (plan['category'],plan['case'],plan['seed'],plan['expected_matched_pairs'],
            plan['decisions_per_window'],plan['main_training_cycles']) != ('combined',16,7,320,20,300):
        raise RuntimeError('Combined recovery scope or scientific budget differs')
    if plan['authorization'] != 'confirmed_failed_run_recovery':
        raise RuntimeError('Only the actually failed combined worker is authorized')


engine.scope = scope


def checked_import(importer, plant, metadata, state_digest, record):
    """Require full learner/controller equality across native import."""
    before = state_digest()
    value = importer(plant,metadata)
    after = state_digest()
    if not before or before != after:
        raise RuntimeError('Native import mutated complete combined Python state')
    record(before,after)
    return value


def source_preserving_manager(original, output):
    """Keep the existing scientific/budget source map byte-for-byte equal."""
    def factory(protocol_sha,job,*,validation_steps=0):
        base = original(protocol_sha,job,validation_steps=validation_steps)
        class CombinedManager(base):
            def restore(self,environment,session):
                import column_e1.checkpoints as native
                import column_e1_process_recovery as fresh
                from column_e1.runtime import digest_state,integrity_payload
                index = engine.read(self.index)
                sources = dict(self.sources)
                if index['source_hashes'] != sources or len(index['agents']) != 4:
                    raise RuntimeError('The original four-learner checkpoint provenance differs')
                importer = native.import_native
                def importing(plant,metadata):
                    return checked_import(importer,plant,metadata,
                        lambda:digest_state(integrity_payload(environment,self.compiled)),
                        lambda before,after:fresh.append(output,dict(event='full_python_restore_verified',
                            before=before,after=after,completed_steps=index['completed_steps'])))
                with patch.object(native,'import_native',importing):
                    super().restore(environment,session)
                if self.sources != sources or session.source_hashes != sources:
                    raise RuntimeError('Combined scientific checkpoint markers changed')
        return CombinedManager
    return factory


def worker(validation_steps):
    if validation_steps not in (0,8):
        raise ValueError('Only eight validation steps or full continuation are allowed')
    plan = engine.verify()
    engine.start_guard(plan)
    engine.protected_check(plan)
    index = engine.read(engine.OUTPUT/'training_checkpoints/latest.json')
    expected = plan['initial_steps']+(0 if validation_steps else 8)
    if index['completed_steps'] != expected or len(index['agents']) != 4:
        raise RuntimeError('The prepared four-learner checkpoint differs')
    if validation_steps and index['sha256'] != plan['initial_checkpoint_sha256']:
        raise RuntimeError('The original full checkpoint changed')
    if not validation_steps:
        validated = engine.read(engine.STATE/'validation.json')
        if not validated['passed'] or validated['index']['sha256'] != index['sha256']:
            raise RuntimeError('Production requires the verified bounded checkpoint')
    engine.preserved_traces(plan,after_validation=not validation_steps)
    # Every refusal above occurs before numerical imports or simulator access.
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_budget as original
    import column_e1_budget as budget
    import column_e1_process_recovery as fresh
    import column_review.common as common
    previous_verify = original.verify
    from importlib.util import spec_from_file_location,module_from_spec
    source = ROOT/'change-reports/combined_memory_restart_20261024/recover_combined.py'
    spec = spec_from_file_location('reviewed_combined_selection',source)
    selection = module_from_spec(spec)
    spec.loader.exec_module(selection)
    def selected(directory):
        return selection.selected_only(previous_verify(directory),plan['job'])
    def redirected_write(path,value):
        path = Path(path)
        if path.parent == original.DEFAULT and path.name.startswith('validation_'):
            path = engine.STATE/'validation.json'
        engine.write(path,value)
    engine.write(engine.STATE/f'worker_{validation_steps}_identity.json',
        dict(identity=engine.process_identity(__import__('os').getpid()),at_utc=engine.now(),
             source_sha256=plan['adapter_sha256'],existing_scientific_protocols_unchanged=True))
    manager = source_preserving_manager(budget.manager_class,engine.OUTPUT)
    with patch.object(original,'read',engine.read),patch.object(original,'verify',selected), \
         patch.object(original,'write',redirected_write),patch.object(common,'read',engine.read), \
         patch.object(budget,'manager_class',manager), \
         engine.ownership_fix.installed(fresh,engine.read,engine.process_identity),fresh.installed(engine.OUTPUT):
        return original.worker(Path(plan['job']),validation_steps)


def supervise():
    from run_aspen_retention import single_instance
    from run_polymer_e3_recovery import environment
    from review_restart_recovery import lifetime,isolated_interactive_context
    plan = engine.verify()
    engine.protected_check(plan)
    identity = lifetime()
    if not isolated_interactive_context(identity):
        raise RuntimeError('An independent current-user supervisor is required')
    engine.STATE.mkdir(parents=True,exist_ok=True)
    with single_instance(engine.STATE):
        if (engine.STATE/'manifest.json').exists():
            raise RuntimeError('Inspect existing supervisor before a duplicate launch')
        engine.write(engine.STATE/'supervisor_identity.json',identity)
        entry = dict(id='e1/combined',kind='e1',category='combined',seed=7,aspen_number=16,
            directory=str(engine.OUTPUT),job=plan['job'],expected_matched_pairs=320,state='pending',attempts=[],error=None)
        manifest = dict(state='prepared',job=entry,jobs=[entry],recovery_plan=str(HERE/'plan.json'))
        pointer_path = ROOT/'Distillation/Results/live_recovery/latest.json'
        stamp = engine.version(pointer_path)
        pointer = engine.read(pointer_path)
        before = deepcopy(pointer)
        registered = pointer.setdefault('additional_manifests',[])
        if str(engine.STATE/'manifest.json') not in registered:
            registered.append(str(engine.STATE/'manifest.json'))
        if engine.version(pointer_path) != stamp:
            raise RuntimeError('Progress pointer changed during registration')
        engine.write(engine.STATE/'progress_pointer_before.json',before)
        engine.write(pointer_path,pointer)
        (engine.STATE/'logs').mkdir()
        for stage,steps in (('validation',8),('production',0)):
            while engine.memory()['headroom_bytes'] < engine.MINIMUM_HEADROOM:
                engine.protected_check(plan)
                manifest.update(state='waiting_for_memory',stage=stage,memory=engine.memory(),updated_at=engine.now())
                engine.write(engine.STATE/'manifest.json',manifest)
                time.sleep(20)
            engine.verify()
            engine.protected_check(plan)
            engine.start_guard(plan)
            log = engine.STATE/'logs'/f'{stage}.log'
            started = engine.now()
            with log.open('xb') as stream:
                child = subprocess.Popen([str(engine.PYTHON),'-B','-u',str(engine.SCRIPT),'--worker',
                    '--validation-steps',str(steps)],cwd=ROOT,
                    env=dict(environment(),PYTHONHASHSEED='7',MPLBACKEND='Agg'),stdout=stream,
                    stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                entry.update(state='running',pid=child.pid,identity=engine.process_identity(child.pid),started_at=started,log=str(log),error=None)
                entry['attempts'].append(dict(pid=child.pid,identity=entry['identity'],directory=str(engine.OUTPUT),stage=stage,started_at=started))
                manifest.update(state='running',stage=stage)
                while True:
                    status = engine.read(engine.OUTPUT/'status.json',{})
                    if status.get('started_at','') >= started:
                        entry.update(pid=status['pid'],identity=engine.process_identity(status['pid']))
                        entry['attempts'][-1].update(pid=entry['pid'],identity=entry['identity'])
                    manifest.update(updated_at=engine.now(),memory=engine.memory())
                    engine.write(engine.STATE/'manifest.json',manifest)
                    if child.poll() is not None:
                        break
                    time.sleep(10)
            status = engine.read(engine.OUTPUT/'status.json',{})
            if child.returncode != 0 or status.get('state') != ('technical_checkpoint' if steps else 'completed'):
                current = status.get('started_at','') >= started and status.get('pid') == entry['pid']
                entry.update(state='failed',exit_code=child.returncode,
                    error=status.get('error') if current else 'Control worker exited before publishing current status; inspect current log',
                    inherited_original_failure=None if current else status.get('error'))
                manifest.update(state='failed',finished_at=engine.now())
                engine.write(engine.STATE/'manifest.json',manifest)
                return
            if steps:
                result = engine.read(engine.STATE/'validation.json')
                if (not result['passed'] or result['index']['completed_steps'] != plan['initial_steps']+8
                        or result['restore']['completed_steps'] != plan['initial_steps']
                        or not result['restore']['native_checksum_verified']
                        or not result['restore']['python_checksum_verified'] or len(result['index']['agents']) != 4
                        or result['index']['source_hashes'] != plan['initial_sources']):
                    raise RuntimeError('Eight-step four-learner continuation did not verify')
                manifest.update(validation_passed=True,replay_archive=engine.preserved_traces(plan,after_validation=True))
                entry.update(state='pending',exit_code=0)
            else:
                if status.get('matched_pairs') != 320 or status.get('failed_pairs') != 0:
                    raise RuntimeError('The original combined comparison budget did not complete')
                entry.update(state='completed',exit_code=0,finished_at=engine.now())
                manifest.update(state='completed',finished_at=engine.now())
                engine.write(engine.STATE/'manifest.json',manifest)
        engine.protected_check(plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ('launch','supervise','worker'):
        group.add_argument('--'+name,action='store_true')
    parser.add_argument('--validation-steps',type=int,default=0)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(worker(args.validation_steps))
    elif args.supervise:
        try:
            supervise()
        except BaseException:
            engine.write(engine.STATE/'supervisor_error.json',dict(at_utc=engine.now(),traceback=traceback.format_exc()))
            raise
    else:
        engine.launch()
