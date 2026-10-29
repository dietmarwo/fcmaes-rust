"""Run the X3 tutorial using the shared model and retain every result."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "structured-support"))
from truss import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--preset", choices=["smoke", "study"], default="smoke")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        start = time.time()
        result = run(args.preset)
        sources = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(sources / "SQP-craft/tutorials/_support"))
        from numerics import write

        write(
            args.output / "summary.json",
            dict(
                chapter="X3",
                preset=args.preset,
                seconds=time.time() - start,
                result=result,
            ),
        )
        print('{"chapter":"X3","status":"passed"}')
    except Exception as e:
        (args.output / "failure.txt").write_text(repr(e))
        raise


if __name__ == "__main__":
    main()
