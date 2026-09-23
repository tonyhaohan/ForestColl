"""Apply MCNF's actual depth filter, then reoptimize ForestColl candidates."""
import json
from clos_benchmark import HERE, case, create_topology, forest_problem, finite_solve

for label in ("java-smoke-v1", "java-standard-v1"):
    for path in sorted((HERE / "results" / label).glob("*/result.json")):
        data = json.loads(path.read_text())
        out = path.parent / "pruned_t12"
        out.mkdir(exist_ok=False)
        config, _, _ = case(data["n"], data["failure_count"])
        topology = create_topology(config["network"]["type"], config["network"]["params"])
        trees = json.loads((path.parent / "trees.json").read_text())
        problem, _ = forest_problem(topology, trees, data["forest"]["k"])
        retained = [tree for tree in trees if tree["depth"] <= 12]
        # The count refers to one phase; the model contains reduce + broadcast.
        row = {"retained_out_trees": len(retained), "original_out_trees": len(trees),
               "method": "ForestColl candidates + MCNF depth filter + free LP weights"}
        try:
            row.update(finite_solve(problem, 12, out / "solution", limit=180))
        except ValueError as exc:
            if "没有可用 tree" not in str(exc):
                raise
            row.update(status="no_tree_after_depth_filter", horizon=12)
        (out / "result.json").write_text(json.dumps(row, indent=2) + "\n")
        print(path.parent.name, json.dumps(row), flush=True)
