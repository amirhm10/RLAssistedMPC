"""Compare every closed replay array with the preserved original, without simulation."""
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p


def audit():
    plan = p.closed_json(HERE / 'plan.json')
    output = ROOT / 'Distillation/Results/column_e1/20261020_seed7_native_v2/combined/a1'
    records = p.h.completed_records(output / 'capture/decisions.jsonl')
    rows = []
    for old in (row for row in plan['traces'] if not row['committed']):
        if not any(row['step'] == old['step'] for row in records):
            continue
        original, current = Path(old['backup']), Path(old['source'])
        if p.h.digest(original) != old['sha256']:
            raise RuntimeError('A preserved original changed')
        with zipfile.ZipFile(current) as zipped:
            if zipped.testzip() is not None:
                raise RuntimeError('The closed replay trace failed CRC')
        arrays = {}
        with np.load(original, allow_pickle=False) as a, np.load(current, allow_pickle=False) as b:
            if set(a.files) != set(b.files):
                raise RuntimeError('The replay array schema changed')
            for name in a.files:
                aa, bb = a[name], b[name]
                if aa.shape != bb.shape or aa.dtype != bb.dtype:
                    raise RuntimeError('The replay array layout changed: ' + name)
                delta = np.abs(aa.astype(float) - bb.astype(float))
                finite = delta[np.isfinite(delta)]
                arrays[name] = dict(shape=list(aa.shape), exact_equal=bool(np.array_equal(aa, bb, equal_nan=True)),
                    first_row_exact=bool(np.array_equal(aa[:1], bb[:1], equal_nan=True)),
                    max_abs_difference=float(np.max(finite)) if finite.size else 0.,
                    max_per_channel=np.max(delta, axis=0).tolist(),
                    original_mean=np.mean(aa, axis=0).tolist(), replay_mean=np.mean(bb, axis=0).tolist())
        rows.append(dict(step=old['step'], original=str(original), original_sha256=old['sha256'],
            replayed=str(current), replayed_sha256=p.h.digest(current), arrays=arrays,
            all_arrays_exact=all(row['exact_equal'] for row in arrays.values())))
    receipt = dict(at_utc=p.now(), comparisons=rows, originals_retained=True,
        scientific_source_changes=0, simulator_calls=0,
        interpretation='Every closed technical replay array is compared; subsequent native integration is not claimed bitwise identical. No post-hoc tolerance or result-based selection is used.')
    name = 'replay_equivalence_' + p.now().replace(':', '').replace('.', '_') + '.json'
    p.immutable(HERE / name, receipt)
    print(__import__('json').dumps(dict(receipt=str(HERE / name), replayed_steps=[r['step'] for r in rows],
        all_arrays_exact=[r['all_arrays_exact'] for r in rows], originals_retained=True), indent=2))


if __name__ == '__main__':
    audit()
