"""Display current E1--E3 progress without launching or controlling simulations."""
import argparse
import json
from pathlib import Path
import time

from revision_comparisons.progress import display, snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.interval < 5:
        parser.error("Use an interval of at least five seconds.")
    while True:
        data = snapshot()
        display(data)
        if args.output:
            if args.output.exists():
                raise FileExistsError("Choose a new audit snapshot filename.")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
