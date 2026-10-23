"""Audit current captures separately from the immutable 200-pair package."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
OUT = HERE / 'live_replay_audit'
OUT.mkdir(exist_ok=True)
shutil.copyfile(HERE / 'before.json', OUT / 'before.json')
spec = importlib.util.spec_from_file_location('partial_e1_auditor', HERE / 'audit_partial_results.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE = OUT
module.main()
path = OUT / 'partial_results_audit.json'
result = json.loads(path.read_text(encoding='utf-8'))
result['limitations'][0] = f'Current one-seed selected-state diagnostics; {640 - result["pairs"]}/640 pairs remain at this audit'
result['immutable_interim_package_pairs'] = 200
result['verified_at_utc'] = datetime.now(timezone.utc).isoformat()
path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(f'Current captures audited: {result["pairs"]}; immutable reviewer package remains 200 pairs.')
