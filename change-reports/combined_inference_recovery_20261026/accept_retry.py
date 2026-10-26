"""Apply the unchanged strong acceptance checks to the fresh metadata retry."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recover_production as activation
import accept_recovery as accepted

accepted.activation = activation

if __name__ == '__main__':
    accepted.accept()
