"""Read-only intake of the completed polymer runs; write derived JSON here."""
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from comparison_studies.common import reference_physical

OUTPUT = Path(__file__).resolve().parent
REPEATS = ROOT / 'Polymer/Results/phase2_repeats/20261013_010021_35e44c'
WINDOWS = {'release_14_20': (14, 20), 'prechange_191_199': (191, 199),
           'change_201_210': (201, 210), 'adapted_291_299': (291, 299),
           'evaluation_300': (300, 300)}


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def intake_repeats():
    manifest = json.loads((REPEATS / 'manifest.json').read_text())
    rows, checks, signatures = [], [], {}
    for i, job in enumerate(manifest['jobs'], 1):
        path = Path(job['result_file'])
        with path.open('rb') as stream:
            result = pickle.load(stream)
        metadata = json.loads((path.parent / 'replication_metadata.json').read_text())
        n, size = int(result['nFE']), int(result['time_in_sub_episodes'])
        y, u = np.asarray(result['y']), np.asarray(result['u'])
        reward = np.asarray(result['rewards_step']).reshape(-1)
        averages = np.asarray(result['avg_rewards']).reshape(-1)
        reference = reference_physical(result)
        valid = {
            'shape': n == 240000 and size == 800 and y.shape == (n+1, 2)
                     and u.shape == (n, 2) and reward.size == n and averages.size == 300,
            'finite_primary_signals': all(np.isfinite(a).all() for a in (y, u, reward, reference)),
            'profile': result['training_profile_name'] == 'robustness_200_100',
            'changed_condition_episode': int(result['phase_switch_episode']) == 201,
            'evaluation_episode_enabled': bool(result['phase2_final_episode_evaluation_only']),
            'replication_seed': int(metadata['seed']) == job['seed'],
            'replication_category_mode': metadata['category'] == job['category'] and metadata['mode'] == job['mode'],
            'reward_average_alignment': bool(np.allclose(averages, reward.reshape(300, size).mean(axis=1),
                                                        rtol=1e-10, atol=1e-10)),
        }
        signatures[(job['category'], job['mode'], job['seed'])] = {
            'references': digest(reference), 'scaling_min': digest(result['data_min']),
            'scaling_max': digest(result['data_max']), 'sampling_time': float(result['delta_t']),
            'disturbance_profile': str(result['disturbance_profile']),
            'phase1_setpoints': str(result['phase1_setpoints_phys']),
            'phase2_setpoints': str(result['phase2_setpoints_phys']),
        }
        checks.append({'category': job['category'], 'mode': job['mode'], 'seed': job['seed'],
                       'embedded_seed_field': result.get('seed'), 'result_file': str(path), 'checks': valid})
        for label, (first, last) in WINDOWS.items():
            start, stop = (first-1)*size, last*size
            error = y[start+1:stop+1] - reference[start:stop]
            movement = np.diff(u[max(0, start-1):stop], axis=0)
            rows.append({'category': job['category'], 'mode': job['mode'], 'seed': job['seed'],
                         'window': label, 'mean_reward': float(reward[start:stop].mean()),
                         'eta_rmse_L_per_g': float(np.sqrt(np.mean(error[:, 0]**2))),
                         'T_rmse_K': float(np.sqrt(np.mean(error[:, 1]**2))),
                         'input_total_abs_movement': np.abs(movement).sum(axis=0).tolist()})
        del result, y, u, reward, averages, reference
        gc.collect()
        if i % 10 == 0:
            print(f'Validated primary signals in {i}/50 repeat bundles.', flush=True)
    aggregate = []
    for category in ('horizon', 'weights', 'markov', 'residual', 'combined'):
        for window in WINDOWS:
            groups = {mode: sorted((r for r in rows if r['category'] == category
                                  and r['window'] == window and r['mode'] == mode), key=lambda r:r['seed'])
                      for mode in ('sg', 'plain')}
            assert [r['seed'] for r in groups['sg']] == [r['seed'] for r in groups['plain']]
            sg, plain = (np.array([r['mean_reward'] for r in groups[mode]]) for mode in ('sg', 'plain'))
            delta = sg - plain
            aggregate.append({'category': category, 'window': window, 'seeds': len(delta),
                              'sg_reward_mean': float(sg.mean()), 'sg_reward_sample_sd': float(sg.std(ddof=1)),
                              'plain_reward_mean': float(plain.mean()), 'plain_reward_sample_sd': float(plain.std(ddof=1)),
                              'paired_reward_difference_mean': float(delta.mean()),
                              'paired_reward_difference_sample_sd': float(delta.std(ddof=1)),
                              'sg_reward_higher_seed_count': int(np.sum(delta > 0))})
    paired_signatures = {category: all(signatures[(category, 'sg', seed)] == signatures[(category, 'plain', seed)]
                                    for seed in (7, 11, 23, 37, 53))
                         for category in ('horizon', 'weights', 'markov', 'residual', 'combined')}
    audit = {'batch': str(REPEATS), 'manifest_state': manifest['state'], 'bundle_count': len(checks),
             'failed_integrity_checks': [r for r in checks if not all(r['checks'].values())],
             'paired_saved_profile_metadata_matches': paired_signatures,
             'checks': checks, 'seed_window_metrics': rows, 'aggregate': aggregate,
             'definition': 'Unsmooothed reward mean; physical after-action y[k+1] versus reference[k]; sample SD across five seeds.',
             'limitations': 'Saved-profile metadata match does not reconstruct any unlogged disturbance history. Five seeds and the prescribed references do not establish unseen-reference generalization.'}
    (OUTPUT / 'repeat_intake.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print('REPEAT_INTEGRITY', len(audit['failed_integrity_checks']), paired_signatures)
    print('FINAL_EPISODE_REWARD', json.dumps([r for r in aggregate if r['window'] == 'evaluation_300']))


if __name__ == '__main__':
    intake_repeats()
