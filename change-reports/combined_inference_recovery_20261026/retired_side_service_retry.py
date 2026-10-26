"""Use the unchanged archive algorithm on the accepted metadata-retry pool."""
from pathlib import Path
import argparse
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import retired_side_service as wrapper

wrapper.archive.STATE = wrapper.ROOT / 'Distillation/Results/state_recovery/20261026_combined_metadata_retry'
wrapper.__file__ = str(Path(__file__).resolve())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--launch', action='store_true')
    group.add_argument('--service', action='store_true')
    if parser.parse_args().launch:
        wrapper.launch()
    else:
        wrapper.archive.service()
