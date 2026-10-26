"""Read the current retry, retaining the preceding failed manifest as history."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inspect_progress as progress

progress.STATE = progress.ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'

if __name__ == '__main__':
    progress.inspect()
