"""Run the unchanged accepted branch with scoped ownership-write retries."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'change-reports/combined_parallel_20261024'))
import parallel_branches as p
import metadata_io


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', required=True, type=Path)
    config = parser.parse_args().worker
    worker = Path(p.closed_json(config)['worker_directory']).resolve()
    allowed = ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
    if not worker.is_relative_to(allowed.resolve()):
        raise RuntimeError('Only the fresh failed-startup retry is in scope')
    with metadata_io.installed([worker]):
        p.worker(config)
