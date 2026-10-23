"""Resume only the existing, stopped current-user maintenance task after repair."""
from pathlib import Path
import sys

import psutil
import win32com.client

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from aspen_retention import SESSION, atomic_json, attributes, digest, now, read_json, safe_child
from run_aspen_retention import verify

REPORT = Path(__file__).resolve().parent
CONFIG = REPORT / 'config.json'
STATE = Path(r'C:\Users\hamediaa\RL_MPC_local_archive\aspen_retention_20261023')

assert '\nOK\n' in (REPORT / 'tests_desktop_final.log').read_text(encoding='utf-8-sig').replace('\r', '')
assert '\nOK\n' in (REPORT / 'concurrency_stress_final.log').read_text(encoding='utf-8')
assert (STATE / 'stop.request').is_file()
old_identity = read_json(STATE / 'worker_identity.json')
assert not psutil.pid_exists(old_identity['pid']), 'Inspect the old maintenance PID before another start'

launch = read_json(STATE / 'launch.json')
service = win32com.client.Dispatch('Schedule.Service')
service.Connect()
task = service.GetFolder('\\').GetTask(launch['task_name'])
assert task.State != 4, 'The maintenance task is already running'
assert task.Definition.Principal.RunLevel == 0
assert task.Definition.Triggers.Count == 0
assert task.Definition.Settings.ExecutionTimeLimit == 'PT0S'
assert Path(task.Definition.Actions.Item(1).Path).name.lower() == 'pythonw.exe'

config = read_json(CONFIG)
for row in config['jobs']:
    job = Path(row['directory'])
    sessions = []
    for path in job.iterdir():
        if SESSION.fullmatch(path.name) and attributes(path)['flags'] & 0x10:
            safe_child(path, Path(config['results_root']))
            sessions.append(str(path))
    row['private_sessions'] = sorted(sessions)
config['checkpoint_quiet_seconds'] = 2
config['bootstrap_sha256'] = {name: digest(ROOT / name) for name in config['bootstrap_sha256']}
config['repaired_at_utc'] = now()
config['validation'] = dict(tests_passed=33, repeated_concurrency_checks_passed=15,
                            tests_sha256=digest(REPORT / 'tests_desktop_final.log'),
                            stress_sha256=digest(REPORT / 'concurrency_stress_final.log'))
verify(config)
atomic_json(REPORT / 'first_launch_before_readonly_repair.json', launch)
atomic_json(CONFIG, config)
launch.update(config_sha256=digest(CONFIG), bootstrap_sha256=config['bootstrap_sha256'],
              resumed_at_utc=now(), repair='Owned retired-directory READONLY cleanup; handle-free quiescence; known private-session roots')
atomic_json(STATE / 'launch.json', launch)
# Remove only this worker's own graceful stop flag. No experiment/queue flags.
safe_child(STATE / 'stop.request', STATE).unlink()
task.Run(None)
atomic_json(REPORT / 'resumed.json', dict(at_utc=now(), task_name=launch['task_name'],
            old_maintenance_pid=old_identity['pid'], experiment_process_actions=0,
            scientific_sources_edited=0, source_files_verified=len(config['scientific_sha256']),
            config_sha256=launch['config_sha256']))
print('Resumed existing current-user retention task:', launch['task_name'], flush=True)
