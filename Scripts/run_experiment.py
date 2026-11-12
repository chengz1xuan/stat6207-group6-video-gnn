"""Prepare, retrain and publish one isolated full experiment; default is plan-only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh isolated run directory.")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--acknowledge-exploratory", action="store_true")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({
            "mode": "plan_only", "training_started": False,
            "steps": ["full offline preparation", "current main model-suite evaluation and brand transfer",
                      "export predictions, cases and App bundle"],
            "output": str(args.output),
        }))
        return 0
    if not args.acknowledge_exploratory:
        parser.error("--run requires --acknowledge-exploratory; manual reviews remain unresolved.")
    output = args.output.resolve()
    for command in (
        [sys.executable, str(ROOT / "Scripts" / "isolate_revision.py"), "--full", "--output", str(output)],
        [sys.executable, str(ROOT / "Scripts" / "evaluate_revision.py"), "--input", str(output),
         "--run", "--acknowledge-exploratory"],
    ):
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            print(f"Experiment failed in {Path(command[1]).name}; no completed App bundle was published.",
                  file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
