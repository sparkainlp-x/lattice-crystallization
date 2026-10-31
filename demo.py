"""Run the held-out experiment and print the J tables as JSON."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from lattice_crystallization.experiment import run_experiment

CAL_FAMILIES = [1000, 1010, 1020, 1030]
EVAL_FAMILIES = [2000, 2010, 2020, 2030]
STEPS = 400
ALPHA = 0.05


def main():
    results = run_experiment(CAL_FAMILIES, EVAL_FAMILIES, STEPS, ALPHA)
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
