"""Audit canonical partial E1 evidence without importing NumPy or learners."""
import array
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def npy(content):
    stream = io.BytesIO(content)
    assert stream.read(6) == b'\x93NUMPY'
    major, minor = stream.read(2)
    length = struct.unpack('<H' if major == 1 else '<I', stream.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(stream.read(length).decode('latin1'))
    descriptor = header['descr']; data = stream.read()
    if descriptor[-2:] in ('f4', 'f8'):
        values = array.array('f' if descriptor[-1] == '4' else 'd')
        values.frombytes(data)
        if descriptor[0] == '>': values.byteswap()
        assert len(values) == math.prod(header['shape'])
        return header['shape'], list(values)
    assert 'O' not in descriptor, 'Object array requires explicit inspection'
    return header['shape'], None


def main():
    baseline = read(HERE / 'before.json')
    groups = defaultdict(lambda: dict(pairs=0, accepted=0, harmful_accepted=0,
                                      beneficial_rejected=0, delta_return_sum=0.0, regret_sum=0.0))
    issues, audited, restores, artifacts = [], [], [], []
    diagnostic_nonfinite = Counter()
    expected_sha = read(ROOT / 'Distillation/Results/state_recovery/20261021_e1_budget20/manifest.json')['protocol_sha256']
    for job in baseline['jobs']:
        if job['study'] != 'E1_column': continue
        folder = Path(job['directory']) / 'capture'
        sampling = read(folder / 'sampling.json')
        records = lines(folder / 'decisions.jsonl')
        identities = [(r['category'], r['channel'], r['step']) for r in records]
        assert len(identities) == len(set(identities))
        checks = lines(folder / 'native_restore_checks.jsonl')
        checks_by_key = defaultdict(list)
        for row in checks: checks_by_key[row['channel'], row['step']].append(row)
        for record in records:
            key = (record['category'], record['channel'], record['step'])
            assert record['status'] == 'completed' and not record['smoke']
            assert record['budget_protocol_sha256'] == expected_sha
            assert record['step'] in sampling['selected_steps'][next(k for k in sampling['selected_steps']
                                                    if k.startswith(record['channel'] + ':'))]
            matched = checks_by_key[record['channel'], record['step']]
            assert matched and all(c['live_native_unchanged'] and c['live_learner_unchanged']
                                  and c['live_native_before'] == c['live_native_after'] for c in matched)
            assert all(p.get('full_variable_copy') is True and max(map(abs, p['difference'])) <= 1e-9
                       for c in matched for p in c['zero_time_initializations'])
            path = folder / record['trace']
            assert path.resolve().is_relative_to(folder.resolve()) and path.is_file()
            arrays = {}
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist(): arrays[name[:-4]] = npy(archive.read(name))
            for name, (shape, values) in arrays.items():
                if values is not None:
                    bad = sum(not math.isfinite(v) for v in values)
                    if bad: diagnostic_nonfinite[record['category'] + ':' + name] += bad
            for branch in ('policy', 'supervisor'):
                assert record['branches'][branch]['state'] == 'completed'
                assert record['branches'][branch]['completed_steps'] == record['rollout_steps']
                assert arrays[branch + '__reward'][0] == (record['rollout_steps'],)
                assert all(math.isfinite(v) for v in arrays[branch + '__reward'][1])
                for field in ('y', 'u', 'reference'):
                    assert arrays[branch + '__' + field][0] == (record['rollout_steps'], 2)
                    assert all(math.isfinite(v) for v in arrays[branch + '__' + field][1]), (key, branch, field)
            for horizon, values in record['metrics'].items():
                n = int(horizon); gamma = record['gamma_per_control_step']
                returns = {}
                for branch in ('policy', 'supervisor'):
                    returns[branch] = sum((gamma ** t) * reward for t, reward in
                                          enumerate(arrays[branch + '__reward'][1][:n]))
                    assert math.isclose(returns[branch], values[branch]['return'], rel_tol=1e-10, abs_tol=1e-7)
                assert math.isclose(returns['policy'] - returns['supervisor'], values['delta_return'], rel_tol=1e-9, abs_tol=1e-7)
            metrics = record['metrics']['200']; delta = metrics['delta_return']
            accepted = int(record['gate']['selected_source']) == 2
            group = groups[(record['category'], record['channel'], record['window'])]
            group['pairs'] += 1; group['accepted'] += accepted
            group['harmful_accepted'] += accepted and delta < 0
            group['beneficial_rejected'] += (not accepted) and delta > 0
            group['delta_return_sum'] += delta
            group['regret_sum'] += max(0.0, -delta if accepted else delta)
            artifacts.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            audited.append(key)
        restores.append(dict(category=job['job'], audited_pairs=len(records),
                             retained_restore_records=len(checks), full_state_hashes_enabled=False))
    result = dict(at_utc=datetime.now(timezone.utc).isoformat(), pairs=len(audited), issues=issues,
                  checks=['unique canonical decisions', 'prospective sampling membership', 'budget hash',
                          'completed native branches', 'unchanged live native/learner counters',
                          'original native restoration bounds', 'trace CRC/shape/finite values',
                          'discounted returns recomputed from raw rewards'],
                  groups=[dict(category=k[0], channel=k[1], window=k[2], **v) for k, v in groups.items()],
                  restores=restores, trace_artifacts=artifacts,
                  optional_diagnostic_nonfinite_counts=dict(diagnostic_nonfinite),
                  limitations=['Partial one-seed selected-state diagnostics; 440/640 pairs are still missing at baseline',
                               'Production checks preserve learner counters/tensor versions, not a full per-pair learner hash',
                               'Markov saved solver failure counts are null, not evidence of zero optimizer failures',
                               'Optional trace diagnostics can contain unset NaNs; core reward/output/input/reference traces must be finite',
                               'Native checkpoint file hashes alone do not demonstrate restart fidelity',
                               'No general SG, safety, causal combined-agent or statistical robustness conclusion'])
    (HERE / 'partial_results_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(pairs=result['pairs'], groups=result['groups'], issues=issues), indent=2))


if __name__ == '__main__': main()
