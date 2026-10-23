"""Reopen the real failed E1 checkpoints in disposable owned Aspen processes."""
from pathlib import Path
import json
import sys
import faulthandler
faulthandler.dump_traceback_later(180, repeat=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import review_numeric_recovery
review_numeric_recovery.install()
import numpy as np
import win32com.client
from column_review.common import read, write, record_owned_apps, owned_apps_remaining
from column_review.pilot import profile
from systems.distillation.plant import DistillationColumnAspen
from run_distillation_phase2_repeats import aspen_file_lock, aspen_com_compatibility
import column_e1.runtime as runtime
import column_e1_recovery
import column_e1_process_recovery as recovery

column_e1_recovery.install()
OUT = ROOT / 'Distillation/Results/state_recovery/20261023_e1_native_probe_a2'
OUT.mkdir(parents=True, exist_ok=True)
baseline = read(ROOT / 'change-reports/e1_recovery_20261023/before.json')
for category in ('weights', 'markov', 'residual'):
    output = OUT / category
    output.mkdir(exist_ok=False)
    row = next(r for r in baseline['jobs'] if r['study'] == 'E1_column' and r['job'] == category)
    job = read(Path(row['directory']) / 'budget20_job.json')
    metadata = row['latest_checkpoint']['native']
    p = object.__new__(DistillationColumnAspen)
    p.path = str(output / Path(metadata['project_file']).name)
    p.delta_t = 1 / 6
    p.current_input = np.asarray(metadata['python_state']['current_input'])
    p.current_output = np.asarray(metadata['python_state']['current_output'])
    traces = []
    with aspen_file_lock(Path(job['source_case']['aspen_path']), ROOT / 'Distillation/Results/_aspen_file_locks'):
        with aspen_com_compatibility(), record_owned_apps(output), recovery.installed(output):
            print('Starting fresh native restore:', category, flush=True)
            p.ad = win32com.client.DispatchEx('AD application')
            p.ad.NewDocument()
            try:
                for repeat in range(2):
                    print('Loading', category, repeat, flush=True)
                    priming = runtime.import_native(p, metadata)
                    start = profile(p)[1]
                    trace = []
                    for step in range(8):
                        before = float(p.sim.Time)
                        p.step()
                        assert abs(float(p.sim.Time) - before - p.delta_t) < 1.1e-6
                        trace.append(profile(p)[1].tolist())
                    traces.append(trace)
                    write(output / 'progress.json', dict(repeat=repeat, start=start.tolist(), priming=priming, traces=traces))
                assert np.allclose(traces[0], traces[1], rtol=1e-10, atol=1e-8)
                write(output / 'acceptance.json', dict(passed=True, category=category,
                    checkpoint_cycle=190, repeated_steps=8, repeated_native_traces_equal=True))
                print('PASS fresh-process native restore and repeated continuation:', category, flush=True)
            finally:
                recovery.retire(p, output)
    assert not owned_apps_remaining(output / 'owned_aspen_apps.json')
write(OUT / 'acceptance.json', dict(passed=True, categories=['weights', 'markov', 'residual'], live_workers_touched=0))
