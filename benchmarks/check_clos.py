"""Runnable contract check; no external MCNF checkout required."""
import hashlib
import json
import networkx as nx
from pathlib import Path
from clos_benchmark import HERE, case, create_topology, graph_from_topology, validate_trees, forest_problem

manifest = json.loads((HERE / "vendor/manifest.json").read_text())
for name, digest in manifest["files"].items():
    path = HERE / "vendor/mcnf_snapshot" / name.removeprefix("src/mcnf_tree/")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
for n, failed in [(16, [3, 0]), (64, [13, 2]), (128, [26, 5]), (256, [52, 11])]:
    for f in range(3):
        config, failure, reference = case(n, f)
        topology = create_topology(config["network"]["type"], config["network"]["params"])
        graph = graph_from_topology(topology)
        assert len(topology.gpu_nodes) == n
        assert set(topology.gpu_nodes) == set(range(n)) - set(failed[:f]) | set(range(n, n+f))
        assert all(v not in graph for v in failed[:f])
        assert failure["replacements"] == [{"failed_gpu": v, "replacement_gpu": n+i} for i, v in enumerate(failed[:f])]
        assert graph.number_of_edges() == 2*n + 2*(n//8 + (f > 0))
        assert all(d["capacity"] == 1 for _, _, d in graph.edges(data=True))
        if reference:
            assert config == reference["config"]
checked = 0
for n in (64, 128, 256):
    for f in range(3):
        config, failure, reference = case(n, f, "ubd")
        topology = create_topology("ubd", config["network"]["params"])
        graph = graph_from_topology(topology)
        replacements = failure["replacements"]
        assert len(topology.gpu_nodes) == n
        assert set(topology.gpu_nodes) == ((set(range(n)) - {r["failed_gpu"] for r in replacements})
                                          | {r["replacement_gpu"] for r in replacements})
        assert all(r["failed_gpu"] not in graph for r in replacements)
        assert config["network"]["params"]["extra_gpus"] == 0
        assert config == reference["config"]
# Different phase bottlenecks must not force sequential RS+AG accounting.
star = nx.DiGraph()
star.add_edges_from([(0, 1), (1, 0), (0, 2), (2, 0)], capacity=1)
toy_trees = {(0, 0): [[(0, 1)], [(0, 2)]], (1, 0): [[(1, 0)], [(0, 2)]],
             (2, 0): [[(2, 0)], [(0, 1)]]}
toy, _ = validate_trees(star, {0, 1, 2}, toy_trees, {key: 1 for key in toy_trees}, 2, 1)
assert toy["rs_ag_congestion_time"] == 4
assert toy["allreduce_fixed_forest_bandwidth_time"] == 3
assert toy["allreduce_dependency_depth"] == 4
for path in sorted((HERE / "results").glob("*/*/trees.json")):
    result = json.loads((path.parent / "result.json").read_text())
    forest = result["forest"]
    if not forest.get("validated"):
        continue
    network = result.get("network", "closd")
    config, _, _ = case(result["n"], result["failure_count"], network)
    topology = create_topology(config["network"]["type"], config["network"]["params"])
    graph = graph_from_topology(topology)
    serialized = json.loads(path.read_text())
    trees = {(t["root"], t["index"]): [[tuple(e) for e in p] for p in t["paths"]] for t in serialized}
    counts = {(t["root"], t["index"]): t["chunks"] for t in serialized}
    validation, _ = validate_trees(graph, set(topology.gpu_nodes), trees, counts, forest["U"], forest["k"])
    assert abs(validation["allgather_congestion_time"] - forest["allgather_optimum_time"]) < 1e-8
    assert validation["allreduce_fixed_forest_bandwidth_time"] <= validation["rs_ag_congestion_time"] + 1e-8
    if network == "closd":
        assert validation["rs_ag_congestion_time"] == 2*(result["n"]-1)
        forest_problem(topology, serialized, forest["k"])
    if checked == 0:
        bad_counts = dict(counts)
        bad_counts[next(iter(bad_counts))] += 1
        try:
            validate_trees(graph, set(topology.gpu_nodes), trees, bad_counts, forest["U"], forest["k"])
        except AssertionError:
            pass
        else:
            raise AssertionError("Corrupt root chunk totals accepted")
    checked += 1
assert checked > 0, "Run at least one complete benchmark before checking trees"
print(f"PASS: {len(manifest['files'])} pristine source files, 21 topology contracts, {checked} forests, overlapping-phase accounting and corrupt-output rejection")
