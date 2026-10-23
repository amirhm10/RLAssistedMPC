"""Freeze the audited 200-pair data and produce a compact descriptive package."""
from collections import defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import review_numeric_recovery
review_numeric_recovery.install()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = HERE / 'partial_analysis'
OUT.mkdir(exist_ok=True)
baseline = json.loads((HERE / 'before.json').read_text())
audit = json.loads((HERE / 'partial_results_audit.json').read_text())
assert audit['pairs'] == 200 and not audit['issues']
records = []
sources = []
frozen_traces = {r['path'] for r in audit['trace_artifacts']}
for job in baseline['jobs']:
    if job['study'] != 'E1_column': continue
    folder = Path(job['directory']) / 'capture'
    observed = [json.loads(line) for line in (folder / 'decisions.jsonl').read_text().splitlines() if line.strip()]
    subset = [r for r in observed if str(folder / r['trace']) in frozen_traces]
    records.extend(subset)
    sources.append(dict(category=job['job'], directory=str(folder), canonical_pairs=len(subset)))
assert len(records) == 200
(OUT / 'decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
grouped = defaultdict(list)
for r in records: grouped[r['category'], r['channel'], r['window']].append(r)
rows = []
for key, values in sorted(grouped.items()):
    deltas = [r['metrics']['200']['delta_return'] for r in values]
    take = [int(r['gate']['selected_source']) == 2 for r in values]
    ranking = []
    for r, delta in zip(values, deltas):
        g = r['gate']
        if g.get('q_policy') is not None and g.get('q_supervisor') is not None:
            q = g['q_policy'] - g['q_supervisor']
        elif all(g.get(k) is not None for k in ('q1_policy', 'q2_policy', 'q1_supervisor', 'q2_supervisor')):
            q = min(g['q1_policy'], g['q2_policy']) - min(g['q1_supervisor'], g['q2_supervisor'])
        else: continue
        if q != 0 and delta != 0: ranking.append((q > 0) == (delta > 0))
    rows.append(dict(category=key[0], channel=key[1], window=key[2], pairs=len(values),
                     accepted=sum(take), lower_return_acceptances=sum(t and d < 0 for t, d in zip(take, deltas)),
                     beneficial_rejections=sum(not t and d > 0 for t, d in zip(take, deltas)),
                     raw_critic_rank_cases=len(ranking), raw_critic_rank_correct=sum(ranking),
                     mean_gate_regret=sum(max(0, -d if t else d) for t, d in zip(take, deltas)) / len(values),
                     mean_always_supervisor_initial_regret=sum(max(0, d) for d in deltas) / len(values),
                     mean_always_policy_initial_regret=sum(max(0, -d) for d in deltas) / len(values),
                     accepted_delta_below_minus_1=sum(t and d < -1 for t, d in zip(take, deltas))))
with (OUT / 'summary_200_steps.csv').open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
categories = ['horizon', 'weights', 'markov', 'residual', 'combined']
windows = ['release', 'prechange_mature', 'plant_change', 'adapted']
coverage = [[sum(r['category'] == cat and r['window'] == window for r in records) for window in windows] for cat in categories]
fig, ax = plt.subplots(figsize=(9.6, 3.8), constrained_layout=True)
ax.imshow([[min(n, 20) for n in row] for row in coverage], cmap='Blues', vmin=0, vmax=30, aspect='auto')
for i, row in enumerate(coverage):
    for j, n in enumerate(row): ax.text(j, i, str(n) if n else 'Missing', ha='center', va='center', fontsize=11)
ax.set_xticks(range(4), ['Release\ncycles 14–20', 'Mature pre-change\n191–199', 'Plant change\n201–210', 'Adapted\n291–299'])
ax.set_yticks(range(5), ['Horizon', 'Weights', 'Dynamic matrix', 'Residual', 'Combined (4 channels)'])
ax.set_title('Distillation E1: 200 completed paired decisions, one training seed', fontsize=12, pad=12)
ax.tick_params(length=0, pad=8)
for spine in ax.spines.values(): spine.set_visible(False)
fig.savefig(OUT / 'coverage.png', dpi=180)
plt.close(fig)
table = '\n'.join('| ' + ' | '.join((r['category'], r['channel'], r['window'], str(r['pairs']),
                   f"{r['lower_return_acceptances']}/{r['accepted']}",
                   f"{r['raw_critic_rank_correct']}/{r['raw_critic_rank_cases']}",
                   f"{r['mean_gate_regret']:.3f}", f"{r['mean_always_supervisor_initial_regret']:.3f}",
                   f"{r['mean_always_policy_initial_regret']:.3f}")) + ' |' for r in rows)
report = f'''# Partial column E1: reviewer-use assessment

The current 200 paired decisions are usable as explicitly partial, exploratory diagnostics for R2.1, which asks whether critic advantages reflect actual action consequences, particularly at release. All five configurations cover release; horizon additionally covers mature pre-change and plant change. These data can accompany the previously completed polymer E1 and E2/comparator evidence. They do not constitute the planned 640-pair column study or close every reviewer concern. A narrower E1 response is a defensible budget choice if Amir accepts the missing phases and the negative findings. Reviewer acceptance cannot be predicted.

## Evidence that can be retained now

The independent audit verifies 200 unique canonical decisions, original prospective sampling membership and protocol hash, completed policy/supervisor branches, unchanged live native state and learner counters, original native restoration checks, ZIP integrity, finite core traces and trace shapes. Discounted returns were recomputed from every raw reward trace and match the saved metrics. No original result or active checkpoint was modified. Optional Markov prediction-error logs contain unset NaNs; core output/input/reference/reward traces are finite. Markov solver-failure counts are missing, not established as zero.

![Available training windows](coverage.png)

## What these results say

For each sampled state, compare the actual exploratory policy proposal with the supervisor proposal and then use the same frozen gated continuation. The primary difference is policy return minus supervisor return over 200 steps, with the original per-step discount. Actual-gate regret is the larger of zero and the better branch's return minus the initially selected branch's return. Always-supervisor/policy alternatives change only the first choice; they are not separately trained controllers. Raw critic ranking uses the DQN Q difference or minimum twin-Q difference and does not represent the complete penalized TD3 selection score.

| Configuration | Channel | Window | Pairs | Accepted with lower return / accepted | Raw rank correct / cases | Mean gate regret | Mean supervisor-initial regret | Mean policy-initial regret |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

All sign counts use the original zero return tolerance, so small differences count. The CSV also reports accepted differences below minus one reward unit as a labeled sensitivity check. Results are mixed and include unfavorable gate decisions. They support treating the critic gate as an empirical action selector with observed failure modes; they do not establish calibrated confidence, uniformly better action ranking, a safety safeguard or a stability guarantee.

## Current budget decision and interim interpretation

Retain all 200 observed pairs and every unfavorable result. Do not pool the extra horizon windows with the other configurations as a fair cross-configuration comparison; compare their common release window separately. Explicitly report 440 missing planned pairs, missing adapted-window evidence, the three technical failures and a subsequent resource-driven stopping decision if approved. Decisions share one training seed and are correlated; do not add seed-level confidence intervals or treat the row counts as independent training replications. This is a post hoc partial evidence package, with no claim of an outcome-independent complete campaign.

Amir explicitly chose to recover E1 and continue toward the approved 640 pairs after reviewing the partial-evidence option. Accordingly, neither active E1 worker is stopped, the three failed configurations are being recovered through separate owned native processes, and the ten combined repetitions and their coordinator remain protected. This 200-pair package is an immutable interim analysis, not an early termination or a declaration that the planned study is complete.

## Draft response direction

We agree that learned critic differences are not calibrated confidence measures. We therefore report matched action-choice diagnostics at release for all tested column configurations, together with additional pre-change and change-response observations available for horizon selection. Each pair starts from the same saved plant/controller state and uses a common frozen continuation. The observed choices include lower-return policy acceptances and rejected beneficial proposals, which we report rather than interpreting the gate as a guarantee. This partial column diagnostic has one training seed and incomplete training-phase coverage because of native simulation failures and computational limits. It complements the separate polymer diagnostics and controller comparisons; it does not establish universal selector reliability.

This is working response material. It has not been inserted into the manuscript or response letter, and no reviewer comment is marked resolved by this report.

## Preservation and compression

The compact reviewer package retains the canonical decision records, all 200 paired raw NPZ traces, numerical summary, sampling definitions, job/configuration descriptors, source hashes, native-restoration receipts and this audit. Native simulator exports and replay/optimizer checkpoints remain in their existing recovery archives; they are not necessary inside the reviewer-sharing package. Excluding them from that package does not delete them. Preserve recent full checkpoints and frozen policies for restart/reproduction. Use lossless, hash-verified compression for inactive historical checkpoints; avoid changing active checkpoint filenames or simulator files.

Sources: canonical folders and trace SHA-256s are enumerated in analysis_manifest.json and ../partial_results_audit.json. Generator: ../build_partial_analysis.py. Units: discounted original reward units; one column seed (7), SG mode; channels are reported separately. No original manuscript, TeX, PDF, existing scientific runner or training protocol was changed. Separate recovery adapters were added without modifying any live worker.
'''
(OUT / 'ASSESSMENT.md').write_text(report, encoding='utf-8')
manifest = dict(at_utc=datetime.now(timezone.utc).isoformat(), primary_pairs=200, planned_pairs=640,
                sources=sources, trace_artifacts=audit['trace_artifacts'], generator=str(Path(__file__)),
                scope='Partial one-seed selected-state diagnostics; not a complete production study',
                metrics='200-step discounted branch return; original zero tolerance', source_audit=str(HERE / 'partial_results_audit.json'))
(OUT / 'analysis_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
package = OUT / 'reviewer_data_200_pairs.zip'
with zipfile.ZipFile(package, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
    for path in OUT.iterdir():
        if path.is_file() and path != package: archive.write(path, path.name)
    archive.write(HERE / 'partial_results_audit.json', 'provenance/partial_results_audit.json')
    for source in sources:
        folder = Path(source['directory'])
        for name in ('sampling.json', 'native_restore_checks.jsonl'):
            archive.write(folder / name, f"{source['category']}/{name}")
        for name in ('job.json', 'budget20_job.json', 'budget_runtime.json'):
            path = folder.parent / name
            if path.exists(): archive.write(path, f"{source['category']}/{name}")
    for artifact in audit['trace_artifacts']:
        path = Path(artifact['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
        category = next(s['category'] for s in sources if path.is_relative_to(Path(s['directory'])))
        archive.write(path, f'{category}/traces/{path.name}')
with zipfile.ZipFile(package) as archive: assert archive.testzip() is None
print(json.dumps(dict(analysis=str(OUT), pairs=200, package_bytes=package.stat().st_size)))
