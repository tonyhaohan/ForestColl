"""ForestColl on frozen/current MCNF physical graphs. Run in conda math."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import random
import resource
import socket
import subprocess
import sys
import time

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE / "vendor")]
from pipeline_allgather import (OptimalBranchingsAlgo, BranchingGenerator,
                               mapBackSwitchNodes, scale_to_integer_capacities)
from mcnf_snapshot.generation import build_problem_from_config, create_topology
from mcnf_snapshot.problem import (AllReduceParticipantSpec, AllReduceTaskSpec,
                                   MCNFProblem, TreeSpec)
from mcnf_snapshot.linear_model import NodeArcLinearTemplate
from mcnf_snapshot.solver_factory.gurobi_native import solve_with_gurobi_native


def case(n, failure, network="closd", inputs=None):
    if n < 16 or n % 8 or failure not in (0, 1, 2):
        raise ValueError("Require N >= 16, N divisible by 8, failure in 0/1/2")
    if inputs:
        archive = json.loads(Path(inputs).read_text())
        for row in archive["rows"]:
            if (row["network"], row["num_gpus"], row["failure_count"]) == (network, n, failure):
                return row["config"], row["failure_plan"], row
        raise ValueError("Case missing from exported canonical inputs")
    if network not in ("closd", "ubd"):
        raise ValueError("HPN requires --inputs from export_mcnf_inputs.py")
    archive = json.loads((HERE / ("ubd_reference.json" if network == "ubd" else "reference.json")).read_text())
    for row in archive["rows"]:
        if (row["num_gpus"], row["failure_count"]) == (n, failure):
            return row["config"], row["failure_plan"], row
    if network == "ubd":
        raise ValueError("UBD uses the exact 64/128/256 standard archive inputs")
    # Small smoke cases follow the standard one-spare-node replacement policy.
    failed = random.Random(20260626).sample(range(n), 2)[:failure]
    replacements = dict(zip(failed, range(n, n + failure)))
    physical, logical = [], []
    for start in range(0, n, 8):
        live = [v for v in range(start, start + 8) if v not in replacements]
        physical.append(live)
        logical.append(live + sorted(replacements[v] for v in range(start, start + 8) if v in replacements))
    if failure:
        physical.append(list(range(n, n + failure)))
    config = {
        "network": {"type": "closd", "params": {
            "gpu_groups": physical, "logical_gpu_groups": logical,
            "gpus_per_node": 8, "n_planes": 1, "high_switches_per_plane": 1,
            "gpu_low_bandwidth": 1.0, "low_high_bandwidth": 1.0,
            "scale_low_high_by_group_size": False}},
        "task": {"type": "allreduce", "params": {"group_size": n, "data_volume": 1.0}},
        "comm": {"name": "hwc"},
        "modeling": {"family": "node_arc", "params": {"time_steps": 12, "beta": 1.0}}}
    return config, {"num_gpus": n, "replacements": [
        {"failed_gpu": v, "replacement_gpu": w} for v, w in replacements.items()]}, None


def graph_from_topology(topology):
    return graph_from_data({"compute_nodes": topology.gpu_nodes, "edges": [
        [u, v, c] for (u, v), c in zip(topology.edges, topology.bandwidths)]})


def graph_from_data(data):
    graph = nx.DiGraph()
    # num_nodes includes failed GPU ID holes; never resurrect them as switches.
    graph.add_nodes_from(data["compute_nodes"])
    for u, v, capacity in data["edges"]:
        assert np.isfinite(capacity) and capacity > 0
        graph.add_edge(int(u), int(v), capacity=float(capacity))
    assert nx.is_strongly_connected(graph)
    assert all(graph[v][u]["capacity"] == d["capacity"] for u, v, d in graph.edges(data=True))
    return graph


def validate_trees(graph, gpus, trees, counts, u_scale, k):
    loads, chunks, depths = Counter(), Counter(), []
    serialized = []
    for (root, index), paths in sorted(trees.items()):
        weight = int(counts[root, index])
        assert weight > 0
        logical = nx.DiGraph()
        logical.add_nodes_from(gpus)
        for path in paths:
            assert path and all(b == c for (_, b), (c, _) in zip(path, path[1:]))
            assert path[0][0] in gpus and path[-1][1] in gpus
            assert all(b not in gpus for _, b in path[:-1])
            logical.add_edge(path[0][0], path[-1][1])
            for edge in path:
                assert graph.has_edge(*edge)
                loads[edge] += weight
        assert nx.is_arborescence(logical)
        assert logical.in_degree(root) == 0
        assert len(nx.descendants(logical, root)) == len(gpus) - 1
        assert len(paths) == len(gpus) - 1
        depth = max(nx.single_source_shortest_path_length(logical, root).values())
        depths.append(depth)
        chunks[root] += weight
        serialized.append({"root": root, "index": index, "chunks": weight,
                           "depth": depth, "paths": paths})
    assert dict(chunks) == {gpu: k for gpu in gpus}
    congestion = max(load / graph[u][v]["capacity"] for (u, v), load in loads.items())
    assert congestion <= u_scale + 1e-8
    combined = max((loads[u, v] + loads[v, u]) / d["capacity"] for u, v, d in graph.edges(data=True)) / k
    return {"tree_count": len(trees), "max_depth": max(depths), "min_depth": min(depths),
            "allreduce_dependency_depth": 2 * max(depths),
            "allgather_congestion_time": congestion / k,
            "rs_ag_congestion_time": 2 * congestion / k,
            "allreduce_fixed_forest_bandwidth_time": combined,
            "capacity_excess": max(0.0, congestion - u_scale),
            "validated": True}, serialized


def barrier_schedule(graph, serialized, k):
    """Executable fluid schedule: global depth barriers, reduce then broadcast."""
    rows = []
    for reverse in (True, False):
        levels = defaultdict(Counter)
        for tree in serialized:
            logical = nx.DiGraph((p[0][0], p[-1][1]) for p in tree["paths"])
            depths = nx.single_source_shortest_path_length(logical, tree["root"])
            for path in tree["paths"]:
                level = tree["depth"] - depths[path[-1][1]] if reverse else depths[path[0][0]]
                edges = [(b, a) for a, b in path] if reverse else path
                for edge in edges:
                    levels[level][edge] += tree["chunks"] / k
        for level in sorted(levels):
            duration = max(load / graph[u][v]["capacity"] for (u, v), load in levels[level].items())
            rows.append({"phase": "reduce" if reverse else "broadcast", "level": level,
                         "duration": duration})
    return {"rounds": len(rows), "time": sum(r["duration"] for r in rows), "steps": rows}


def java_generate(graph, gpus, out, timeout):
    source = out / "graph.txt"
    source.write_text("".join(f"V {v}\n" for v in gpus) + "".join(
        f"E {u} {v} {int(d['capacity'])}\n" for u, v, d in graph.edges(data=True)))
    assert all(d["capacity"] == int(d["capacity"]) for _, _, d in graph.edges(data=True))
    command = ["java", "-ea", "-Xmx4g", "-Djava.util.concurrent.ForkJoinPool.common.parallelism=4",
               "-cp", f"{HERE / 'build'}:{HERE / 'lib'}/*", "ClosBenchmark", str(source), str(out / "forest.txt")]
    started = time.perf_counter()
    with (out / "java.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    metrics = {"backend": "java-parallel", "java_timeout_sec": timeout, "java_wall_sec": time.perf_counter() - started,
               "java_peak_rss_mib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)}
    trees, counts, edge_map = {}, {}, {}
    for line in (out / "forest.txt").read_text().splitlines():
        p = line.split()
        if p[0] == "P":
            metrics.update(U=float(p[1]), k=int(p[2]), search_sec=float(p[3]))
        elif p[0] in ("S", "B"):
            metrics["split_sec" if p[0] == "S" else "branch_sec"] = float(p[1])
        elif p[0] == "E":
            u, v, w, count = map(int, p[1:])
            edge_map.setdefault((u, v), {})[w] = count
        elif p[0] == "T":
            root, index, count = map(int, p[1:4])
            counts[root, index] = count
            trees[root, index] = list(zip(map(int, p[4::2]), map(int, p[5::2])))
    return metrics, trees, counts, edge_map


def forest_problem(topology, serialized, k):
    grouped = defaultdict(dict)
    weights = {}
    for tree in serialized:
        root = tree["root"]
        parent = {path[-1][1]: path[0][0] for path in tree["paths"]}
        signature = tuple(sorted(parent.items()))
        # ClosD is a physical tree, so these are the same exact physical routes.
        for path in tree["paths"]:
            expected = tuple(topology.edge_index[tuple(edge)] for edge in path)
            assert topology.shortest_paths[path[0][0], path[-1][1]] == (expected,)
        if signature in grouped[root]:
            grouped[root][signature][1] += tree["chunks"] / k
        else:
            grouped[root][signature] = [parent, tree["chunks"] / k]
    participants = []
    for root in sorted(grouped):
        phases = []
        for direction in ("reverse", "forward"):
            phase = []
            for index, (parent, weight) in enumerate(grouped[root].values()):
                name = f"forest_{root}_{index}_{direction}"
                phase.append(TreeSpec(root=root, parent=parent, topology=topology,
                                      direction=direction, name=name))
                weights[name] = weight
            phases.append(tuple(phase))
        participants.append(AllReduceParticipantSpec(root, *phases))
    return MCNFProblem(topology, (AllReduceTaskSpec(0, float(len(grouped)), tuple(participants), "allreduce"),)), weights


def finite_solve(problem, horizon, out, weights=None, limit=180):
    start = time.perf_counter()
    template = NodeArcLinearTemplate.from_problem(problem)
    if weights and any(tree.depth > horizon for tree in problem.tasks[0].trees()):
        return {"status": "infeasible_fixed_forest_depth", "horizon": horizon}
    model = template.build(time_steps=horizon)
    if weights:
        lower, upper = model.lower_bounds.copy(), model.upper_bounds.copy()
        for ref in model.layout.trees:
            idx = model.layout.l_offset + model.layout.lambda_index[ref.task_index, ref.tree_index]
            lower[idx] = upper[idx] = weights[ref.tree.name]
        model = replace(model, lower_bounds=lower, upper_bounds=upper)
    build = time.perf_counter() - start
    result = solve_with_gurobi_native(model, {"name": "gurobi_native", "solver_args": {
        "Method": 2, "Crossover": 0, "Threads": 4, "TimeLimit": limit,
        "LogToConsole": 0, "LogFile": str(out.with_suffix(".log")),
        "_mcnf_record_presolved_norms": False, "_mcnf_record_gpu_memory": False}})
    data = {"status": result.status, "objective": result.objective, "horizon": horizon,
            "build_sec": build, "solve_sec": result.solve_time,
            "wall_sec": time.perf_counter() - start,
            "variables": model.layout.total_variables}
    if result.objective is not None:
        data["violations"] = model.constraint_violations(result)
        data["validated"] = model.is_feasible(result, tol=1e-4)
        np.savez_compressed(out.with_suffix(".npz"), z=result.z, A=result.A)
    return data


def run(args):
    start = time.perf_counter()
    config, failure, reference = case(args.n, args.failure, args.network, args.inputs)
    if args.network != "closd" and args.finite:
        raise ValueError("Finite LP replay currently requires unique ClosD physical paths")
    if args.network == "hpn" and args.baseline:
        raise ValueError("HPN MCNF comparison uses exported evidence; the old model snapshot has no HPN")
    out = HERE / "results" / args.label / f"{args.network}{args.n}_f{args.failure}"
    out.mkdir(parents=True, exist_ok=False)
    if args.inputs:
        graph = graph_from_data(reference["graph"])
        gpus = tuple(reference["graph"]["compute_nodes"])
    else:
        topology = create_topology(config["network"]["type"], config["network"]["params"])
        graph = graph_from_topology(topology)
        gpus = topology.gpu_nodes
    graph_data = {"compute_nodes": list(gpus),
                  "edges": [[u, v, d["capacity"]] for u, v, d in graph.edges(data=True)]}
    result = {"network": args.network, "n": args.n, "failure_count": args.failure, "failure_plan": failure,
              "config": config, "environment": {"python": sys.version, "platform": platform.platform(),
              "networkx": nx.__version__, "hostname": socket.gethostname()},
              "inputs_sha256": hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest() if args.inputs else None,
              "input_sha256": hashlib.sha256(json.dumps(graph_data, sort_keys=True).encode()).hexdigest(),
              "reference": reference}
    result["input_build_sec"] = time.perf_counter() - start
    (out / "input.json").write_text(json.dumps(graph_data, indent=2) + "\n")
    def save():
        result["wall_sec"] = time.perf_counter() - start
        result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    scaled, scale = scale_to_integer_capacities(graph)
    algo = OptimalBranchingsAlgo(scaled, capacitated=True, compute_nodes=gpus)
    t = time.perf_counter()
    U, k = algo.binary_search()
    assert algo.test(U, k, False)
    result["forest"] = {"U": U * scale, "k": k, "search_sec": time.perf_counter() - t,
                        "allgather_optimum_time": U * scale / k,
                        "rs_ag_sequential_bandwidth_time": 2 * U * scale / k,
                        "algorithmic_bandwidth": args.n * k / (U * scale), "backend": "python-networkx"}
    result["python_bound_check_sec"] = time.perf_counter() - t
    save()
    print(json.dumps({"network": args.network, "n": args.n, "failure": args.failure, "forest": result["forest"]}), flush=True)
    if not args.bound_only:
        if args.backend == "java":
            t = time.perf_counter()
            try:
                metrics, trees, counts, edge_map = java_generate(graph, gpus, out, args.java_timeout)
            except (subprocess.SubprocessError, OSError) as exc:
                result["forest"]["generation_error"] = repr(exc)
                result["forest"]["java_wall_sec"] = time.perf_counter() - t
                save()
                raise
            assert metrics["U"] == U * scale and metrics["k"] == k
            result["forest"].update(metrics)
        else:
            t = time.perf_counter()
            flow_graph, edge_map = algo.removeSwitchNodes(U, k)
            result["forest"]["split_sec"] = time.perf_counter() - t
            save()
            t = time.perf_counter()
            trees, counts = BranchingGenerator(flow_graph, k).generate()
            result["forest"]["branch_sec"] = time.perf_counter() - t
        t = time.perf_counter()
        trees, counts = mapBackSwitchNodes(trees, counts, edge_map)
        result["forest"]["mapback_sec"] = time.perf_counter() - t
        t = time.perf_counter()
        validation, serialized = validate_trees(graph, set(gpus), trees, counts, U * scale, k)
        result["validation_sec"] = time.perf_counter() - t
        result["forest"].update(validation)
        (out / "trees.json").write_text(json.dumps(serialized) + "\n")
        t = time.perf_counter()
        result["barrier_schedule"] = barrier_schedule(graph, serialized, k)
        result["barrier_schedule_sec"] = time.perf_counter() - t
        save()
        if args.finite:
            topology = create_topology(config["network"]["type"], config["network"]["params"])
            problem, weights = forest_problem(topology, serialized, k)
            result["forest_t12"] = finite_solve(problem, 12, out / "forest_t12", weights)
            save()
    if args.baseline:
        problem = build_problem_from_config(config)
        result["mcnf_t12"] = finite_solve(problem, 12, out / "mcnf_t12")
    save()
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "reference", "barrier_schedule")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--network", choices=("closd", "ubd", "hpn"), default="closd")
    parser.add_argument("--inputs", type=Path, help="Physical graphs exported from the current MCNF checkout")
    parser.add_argument("--failure", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--bound-only", action="store_true")
    parser.add_argument("--finite", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--backend", choices=("python", "java"), default="python")
    parser.add_argument("--java-timeout", type=int, default=5400)
    run(parser.parse_args())
