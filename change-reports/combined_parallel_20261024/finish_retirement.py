"""Finish an already authorized retirement after an ownership guard refusal.

The original helper refused a known Aspen server's conhost child. Inspect all
descendants, leave console hosts untouched, and signal only the journal-owned
native roots and their generation-verified Aspen servers. No Python signal.
"""
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
import parallel_branches as p
import recover_parallel as activation
import preserve


def server_rows(rows):
    result=[];ignored=[]
    for row in rows:
        name=row['name'].lower()
        if name=='conhost.exe':ignored.append(row)
        elif name in ('sim_server.exe','am_task_server.exe'):result.append(row)
        else:raise RuntimeError('Unknown descendant must remain untouched: '+name)
    return result,ignored


def run():
    if (HERE/'retirement.json').exists():raise FileExistsError('Retirement is already complete')
    plan=activation.r.verify();activation.r.protected_check(plan)
    intent=p.closed_json(HERE/'retirement_intent.json')
    if p.h.process(intent['original']['pid'])['alive'] is not False:
        raise RuntimeError('Only an already retired original combined generation is eligible')
    import win32com.client
    from recover_markov_seed23 import terminate_owned_identity
    service=win32com.client.GetObject('winmgmts:');descendants=[]
    def inspect(parent):
        for child in service.ExecQuery('SELECT ProcessId,Name FROM Win32_Process WHERE ParentProcessId='+str(parent['pid'])):
            identity=p.h.process(int(child.ProcessId))
            if not identity.get('alive') or identity.get('created',0)<parent['created']:
                raise RuntimeError('Descendant generation is ambiguous')
            row=dict(identity,name=str(child.Name),parent_identity=parent)
            descendants.append(row);inspect(identity)
    for row in intent['native_owned']:
        if not p.same_generation(row,p.h.process(row['pid'])):raise RuntimeError('Owned native root changed')
        inspect(row)
    servers,ignored=server_rows(descendants)
    p.immutable(HERE/'native_cleanup_plan.json',dict(at_utc=p.now(),native_roots=intent['native_owned'],
        verified_aspen_servers=servers,console_hosts_to_leave_untouched=ignored,
        original_python_already_retired=True,repair_source_sha256=p.h.digest(Path(__file__))))
    actions=[]
    for row in reversed(servers):actions.append(dict(identity=row,action=terminate_owned_identity(row)))
    for row in intent['native_owned']:actions.append(dict(identity=row,action=terminate_owned_identity(row)))
    saved=p.closed_json(HERE/'preservation.json');full=p.closed_json(saved['manifest'])
    roots={Path(x) for point in full['pair']['checkpoints'] for x in point['private_roots']}
    folder=Path(saved['backup'])/'retired_working';copies=[]
    files={Path(row['source']) for row in full['deferred_owned_working_files']}
    files.update(source for root in roots for source in root.rglob('*') if source.is_file())
    for source in sorted(files):copies.append(preserve.copied_generation(source,folder/source.relative_to(ROOT).parent))
    p.immutable(folder/'preservation.json',dict(verified=True,at_utc=p.now(),copies=copies,
        formerly_locked_working_files_copied=True,all_originals_retained=True))
    p.immutable(HERE/'retirement.json',dict(at_utc=p.now(),actions=actions,only_combined_was_signaled=True,
        original_python_signal_record=str(HERE/'retirement_intent.json'),
        former_working_preservation=str(folder/'preservation.json'),former_working_sha256=p.h.digest(folder/'preservation.json'),
        console_host_signals=0,unrelated_process_signals=0))
    activation.r.protected_check(plan);activation.r.start_guard(plan)
    print('Owned combined native cleanup complete; every formerly locked working file is backed up; ready for guarded continuation.',flush=True)


if __name__=='__main__':run()
