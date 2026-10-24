"""Conditional, verified combined-only checkpoint migration to paired workers."""
import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(ROOT/'change-reports/combined_native_recovery_20261024'))
import recover_combined as serial
r=serial.engine
r.HERE=HERE
r.SCRIPT=Path(__file__).resolve()
r.STATE=ROOT/'Distillation/Results/state_recovery/20261024_combined_parallel'


def scope(plan):
    if (plan['category'],plan['case'],plan['seed'],plan['expected_matched_pairs'],
            plan['decisions_per_window'],plan['main_training_cycles'])!=('combined',16,7,320,20,300):
        raise RuntimeError('The original combined scientific budget/case must remain unchanged')
    if plan.get('authorization')!='user_approved_verified_parallel_migration':
        raise RuntimeError('Explicit conditional parallel migration authorization is required')
    for label in ('pilot_acceptance','transport_acceptance'):
        item=plan[label]
        if r.digest(Path(item['path']))!=item['sha256']:raise RuntimeError('A native acceptance receipt changed')
        receipt=__import__('json').loads(Path(item['path']).read_text(encoding='utf-8'))
        if not receipt.get('passed'):raise RuntimeError('Native adapter fidelity was not accepted')
        if label=='pilot_acceptance' and not receipt.get('enough_time_saved'):
            raise RuntimeError('Measured comparison/time saving is insufficient')
        if label=='transport_acceptance' and not receipt.get('ready_for_conditional_migration'):
            raise RuntimeError('Durable native transport/time/disk gates are incomplete')
    if plan['net_saving_after_development_and_replay_hours']<3.:
        raise RuntimeError('At least three net hours must remain after every migration allowance')


r.scope=scope


def worker(validation_steps):
    if validation_steps not in (0,8):raise ValueError('Use eight-step validation or full continuation only')
    plan=r.verify();r.start_guard(plan);r.protected_check(plan)
    index=r.read(r.OUTPUT/'training_checkpoints/latest.json')
    expected=plan['initial_steps']+(0 if validation_steps else 8)
    if index['completed_steps']!=expected or len(index['agents'])!=4 or index['source_hashes']!=plan['initial_sources']:
        raise RuntimeError('The verified four-learner checkpoint/source map changed')
    if validation_steps and index['sha256']!=plan['initial_checkpoint_sha256']:
        raise RuntimeError('The original complete checkpoint checksum changed')
    if not validation_steps:
        validation=r.read(r.STATE/'validation.json')
        if not validation['passed'] or validation['index']['sha256']!=index['sha256']:
            raise RuntimeError('Production requires the accepted bounded checkpoint')
    r.preserved_traces(plan,after_validation=not validation_steps)
    import review_numeric_recovery
    review_numeric_recovery.install()
    import run_column_e1_budget as original
    import run_column_e1 as runner
    import column_e1_budget as budget
    import column_e1_process_recovery as fresh
    import column_review.common as common
    import session_adapter
    from runtime_integration import ManagedBranchPool,closing_stack
    import parallel_branches as p
    previous_verify=original.verify
    from importlib.util import spec_from_file_location,module_from_spec
    source=ROOT/'change-reports/combined_memory_restart_20261024/recover_combined.py'
    spec=spec_from_file_location('parallel_combined_selection',source);selection=module_from_spec(spec);spec.loader.exec_module(selection)
    def selected(directory):return selection.selected_only(previous_verify(directory),plan['job'])
    def redirected_write(path,value):
        path=Path(path)
        if path.parent==original.DEFAULT and path.name.startswith('validation_'):path=r.STATE/'validation.json'
        r.write(path,value)
    identity=r.process_identity(__import__('os').getpid())
    template=deepcopy(plan['branch_template'])
    template.update(protected_main=identity,aggregate_journal=str(r.OUTPUT/'owned_aspen_apps.json'))
    pool=ManagedBranchPool(r.STATE/('branches_'+str(validation_steps)+'_'+uuid.uuid4().hex[:8]),template)
    # Start lazily at the first diagnostic, after all original native/main
    # initialization and checkpoint restoration. Retain the normal source map.
    base_pair=pool.pair
    def paired(*args,**kwargs):
        if not pool.workers:pool.start()
        return base_pair(*args,**kwargs)
    pool.pair=paired
    r.write(r.STATE/('worker_'+str(validation_steps)+'_identity.json'),dict(identity=identity,at_utc=r.now(),
        source_sha256=plan['adapter_sha256'],existing_scientific_protocols_unchanged=True))
    r.write(r.STATE/('parallel_pool_'+str(validation_steps)+'.json'),dict(directory=str(pool.directory),parent=identity,cases=[15,17]))
    manager=serial.source_preserving_manager(budget.manager_class,r.OUTPUT)
    try:
        with patch.object(original,'read',r.read),patch.object(original,'verify',selected), \
             patch.object(original,'write',redirected_write),patch.object(common,'read',r.read), \
             patch.object(budget,'manager_class',manager),r.ownership_fix.installed(fresh,r.read,r.process_identity), \
             fresh.installed(r.OUTPUT),session_adapter.installed(pool,maximum_pairs=plan['maximum_parallel_decisions']), \
             patch.object(runner,'ExitStack',closing_stack(pool)):
            return original.worker(Path(plan['job']),validation_steps)
    finally:
        pool.close()


def supervise():
    # The accepted two-stage serial supervisor already guards the same
    # four-learner continuation, preserves replay suffixes and registers the
    # existing viewer without changing the shared scientific plan.
    serial.HERE=HERE
    return serial.supervise()


def launch():
    plan=r.verify();r.protected_check(plan);r.start_guard(plan)
    if (r.STATE/'scheduled.json').exists() or (r.STATE/'manifest.json').exists():
        raise RuntimeError('A live or memory-gated parallel continuation already exists')
    import win32api
    import win32con
    import win32security
    import win32com.client
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    try:user=win32security.ConvertSidToStringSid(win32security.GetTokenInformation(token,win32security.TokenUser)[0])
    finally:token.Close()
    service=win32com.client.Dispatch('Schedule.Service');service.Connect()
    definition=service.NewTask(0)
    definition.Principal.UserId=user;definition.Principal.LogonType=3;definition.Principal.RunLevel=0
    settings=definition.Settings
    settings.Enabled=settings.AllowDemandStart=True;settings.ExecutionTimeLimit='PT0S';settings.MultipleInstances=2
    settings.DisallowStartIfOnBatteries=settings.StopIfGoingOnBatteries=False
    action=definition.Actions.Create(0);action.Path=str(r.BASE_PYTHON)
    action.Arguments=subprocess.list2cmdline(['-B',str(Path(__file__).resolve()),'--supervise'])
    action.WorkingDirectory=str(ROOT)
    name='RLAssistedMPC-CombinedParallel-20261024'
    service.GetFolder('\\').RegisterTaskDefinition(name,definition,2,user,None,3)
    r.write(r.STATE/'scheduled.json',dict(task_name=name,administrator_required=False,
        execution_time_limit='PT0S',triggers=0,launch_at_utc=r.now(),cases=[15,17]))
    service.GetFolder('\\').GetTask(name).Run(None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    for name in ('worker','supervise','launch'):group.add_argument('--'+name,action='store_true')
    parser.add_argument('--validation-steps',type=int,choices=(0,8),default=0)
    args=parser.parse_args()
    if args.worker:raise SystemExit(worker(args.validation_steps))
    elif args.supervise:
        try:supervise()
        except BaseException:
            r.write(r.STATE/'supervisor_error.json',dict(at_utc=r.now(),traceback=__import__('traceback').format_exc()))
            raise
    else:launch()
