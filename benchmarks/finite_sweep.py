"""Replay unchanged ForestColl tree weights in the frozen MCNF finite-round LP."""
import argparse
import json
from pathlib import Path
from clos_benchmark import HERE, case, create_topology, finite_solve, forest_problem

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("case_dir", type=Path)
parser.add_argument("--horizons", type=int, nargs="+", default=[32, 64, 128])
args = parser.parse_args()
data = json.loads((args.case_dir / "result.json").read_text())
config, _, _ = case(data["n"], data["failure_count"])
topology = create_topology(config["network"]["type"], config["network"]["params"])
trees = json.loads((args.case_dir / "trees.json").read_text())
problem, weights = forest_problem(topology, trees, data["forest"]["k"])
out = args.case_dir / "finite_sweep"
out.mkdir(exist_ok=False)
results = []
for horizon in args.horizons:
    row = finite_solve(problem, horizon, out / f"t{horizon}", weights)
    results.append(row)
    (out / "result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(row), flush=True)
