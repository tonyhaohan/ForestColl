"""Schedule saved ForestColl shards without changing trees, weights or routes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import platform
import resource
import socket
import sys
import time

import networkx as nx
import numpy as np

from clos_benchmark import HERE, graph_from_data, validate_trees
from mcnf_snapshot.problem import (Topology, TreeSpec, MCNFProblem,
                                    AllReduceParticipantSpec, AllReduceTaskSpec)
from mcnf_snapshot.linear_model import NodeArcLinearTemplate
from mcnf_snapshot.solver_factory.gurobi_native import solve_with_gurobi_native


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operations(graph, trees, k):
    """One operation per logical tree edge and phase; switches only forward."""
    offsets, shards, ops = defaultdict(int), [], []
    ranks = sorted(graph["compute_nodes"])
    rank = {gpu: i for i, gpu in enumerate(ranks)}
    for tree in trees:
        root, weight = tree["root"], tree["chunks"] / k
        shard = len(shards)
        start = rank[root] + offsets[root] / k
        offsets[root] += tree["chunks"]
        logical = nx.DiGraph((p[0][0], p[-1][1]) for p in tree["paths"])
        depths = nx.single_source_shortest_path_length(logical, root)
        shards.append({"root": root, "tree_index": tree["index"], "chunks": tree["chunks"],
                       "data_start": start, "amount": weight, "depth": tree["depth"]})
        ids = {}
        for phase in ("reduce", "broadcast"):
            for path in tree["paths"]:
                a, b = path[0][0], path[-1][1]
                ids[phase, b] = len(ops)
                ops.append({"id": len(ops), "shard": shard, "phase": phase,
                            "src": b if phase == "reduce" else a,
                            "dst": a if phase == "reduce" else b,
                            "path": [[v, u] for u, v in reversed(path)] if phase == "reduce" else path,
                            "base_round": tree["depth"] - depths[b] if phase == "reduce"
                                          else tree["depth"] + depths[a]})
        for phase in ("reduce", "broadcast"):
            for a, b in logical.edges():
                if phase == "reduce":
                    deps = [ids["reduce", c] for c in logical.successors(b)]
                elif a == root:
                    deps = [ids["reduce", c] for c in logical.successors(root)]
                else:
                    deps = [ids["broadcast", a]]
                ops[ids[phase, b]]["depends_on"] = sorted(deps)
    if dict(offsets) != {gpu: k for gpu in ranks}:
        raise ValueError("Saved shard counts do not cover one unit per root")
    return shards, ops


def wave_plan(graph, trees, k, horizon):
    shards, ops = operations(graph, trees, k)
    minimum = 2 * max(t["depth"] for t in trees)
    if horizon < minimum:
        raise ValueError(f"Fixed paired-shard causality requires T >= {minimum}")
    packets = horizon - minimum + 1
    edges = {(u, v): i for i, (u, v, _) in enumerate(graph["edges"])}
    delta = np.zeros((len(edges), horizon + 1))
    for op in ops:
        op["repeat"] = packets
        op["amount_per_packet"] = shards[op["shard"]]["amount"] / packets
        start = op.pop("base_round")
        op["first_round"] = start
        for edge in op["path"]:
            i = edges[tuple(edge)]
            delta[i, start] += op["amount_per_packet"]
            delta[i, start + packets] -= op["amount_per_packet"]
    loads = np.maximum(0, np.cumsum(delta[:, :-1], axis=1))
    durations = np.max(loads / np.asarray(graph["edges"])[:, 2, None], axis=0)
    return {"schema": "forestcoll-fixed-schedule-v1", "method": "packet_wave",
            "contract": {"data_volume": len(graph["compute_nodes"]), "beta": 1.0,
                         "alpha": 0.0, "round_boundary": "receive at end; reuse next round",
                         "physical_path": "fluid cut-through; every directed edge is charged",
                         "chunk_identity": "paired reduction/broadcast, FIFO within each shard"},
            "graph": graph, "shards": shards, "operations": ops,
            "time_steps": horizon, "packets_per_shard": packets,
            "durations": durations.tolist(), "objective": float(durations.sum())}


def cumulative(op, amount, horizon):
    flows = np.zeros(horizon)
    if "segments" in op:
        cursor, previous = 0.0, -1
        for t, lo, hi in op["segments"]:
            if not (isinstance(t, int) and previous < t < horizon and
                    abs(lo - cursor) <= 1e-7 and hi > lo and hi <= amount + 1e-7):
                raise ValueError("Invalid FIFO transfer interval")
            flows[t] += hi - lo
            cursor = hi
            previous = t
    else:
        start, repeat, value = op["first_round"], op["repeat"], op["amount_per_packet"]
        if (not isinstance(start, int) or not isinstance(repeat, int) or
                start < 0 or repeat < 1 or start + repeat > horizon or
                not np.isfinite(value) or value <= 0):
            raise ValueError("Invalid packet loop")
        flows[start:start + repeat] = value
    if abs(flows.sum() - amount) > 1e-7:
        raise ValueError("Transfer does not preserve the fixed shard amount")
    return flows, np.concatenate(([0.0], np.cumsum(flows)))


def validate_plan(plan, trees, k, tolerance=1e-6):
    """Replay exported transfers, including FIFO data availability and capacities."""
    graph, horizon = plan["graph"], plan["time_steps"]
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("Invalid horizon")
    expected_shards, expected_ops = operations(graph, trees, k)
    if plan["shards"] != expected_shards or len(plan["operations"]) != len(expected_ops):
        raise ValueError("Shard allocation or forest changed")
    durations = np.asarray(plan["durations"], dtype=float)
    if durations.shape != (horizon,) or not np.all(np.isfinite(durations)) or np.min(durations) < 0:
        raise ValueError("Invalid round durations")
    edge_ids = {(u, v): i for i, (u, v, _) in enumerate(graph["edges"])}
    loads = np.zeros((len(edge_ids), horizon))
    incoming = defaultdict(list)
    for op in expected_ops:
        incoming[op["shard"], op["phase"], op["dst"]].append(op["id"])
    # Only progress at receivers is needed; vectorized minima avoid an E*T cache.
    available = {}
    for op, expected in zip(plan["operations"], expected_ops):
        for field in ("id", "shard", "phase", "src", "dst", "path", "depends_on"):
            if op.get(field) != expected[field]:
                raise ValueError(f"Changed fixed operation: {field}")
        amount = plan["shards"][op["shard"]]["amount"]
        flows, progress = cumulative(op, amount, horizon)
        key = op["shard"], op["phase"], op["dst"]
        if key in available:
            np.minimum(available[key], progress, out=available[key])
        else:
            available[key] = progress
        for edge in op["path"]:
            if tuple(edge) not in edge_ids:
                raise ValueError("Unknown physical edge")
            loads[edge_ids[tuple(edge)]] += flows
    causality = 0.0
    for op in plan["operations"]:
        if not op["depends_on"]:
            continue
        phase = "reduce" if op["phase"] == "reduce" or op["src"] == plan["shards"][op["shard"]]["root"] else "broadcast"
        ready = available[op["shard"], phase, op["src"]]
        _, progress = cumulative(op, plan["shards"][op["shard"]]["amount"], horizon)
        causality = max(causality, float(np.max(progress[1:] - ready[:-1])))
    capacities = np.asarray(graph["edges"], dtype=float)[:, 2]
    excess = max(0.0, float(np.max(loads - capacities[:, None] * durations)))
    objective_error = abs(float(durations.sum()) - plan["objective"])
    if max(causality, excess, objective_error) > tolerance:
        raise ValueError(f"Schedule replay failed: dependency={causality}, capacity={excess}, objective={objective_error}")
    return {"validated": True, "causality_violation": causality,
            "capacity_violation": excess, "objective_error": objective_error,
            "operations": len(plan["operations"]),
            "expanded_transfers": sum(op.get("repeat", len(op.get("segments", []))) for op in plan["operations"])}


def model_size(graph, trees, horizon):
    per_root = defaultdict(list)
    for i, tree in enumerate(trees):
        per_root[tree["root"]].append(i)
    slots = max(map(len, per_root.values()))
    count = slots * len(per_root) * 2
    n = len(graph["compute_nodes"])
    variables = horizon + count + slots * n + count * (n * (horizon + 1) + (n - 1) * horizon)
    return variables, slots


def canonical_model(graph, trees, k, horizon):
    """Pack one paired tree per root/task to preserve identity through the root gate."""
    edges = tuple((u, v) for u, v, _ in graph["edges"])
    capacities = tuple(c for _, _, c in graph["edges"])
    edge_id = {edge: i for i, edge in enumerate(edges)}
    base = Topology(max(max(e) for e in edges) + 1, edges, capacities,
                    shortest_paths={}, gpu_nodes=tuple(graph["compute_nodes"]))
    per_root = defaultdict(list)
    for i, tree in enumerate(trees):
        per_root[tree["root"]].append(i)
    tasks, weights, source = [], {}, {}
    for slot in range(max(map(len, per_root.values()))):
        participants = []
        volume = 0.0
        for root in sorted(per_root):
            active = slot < len(per_root[root])
            i = per_root[root][slot if active else 0]
            tree = trees[i]
            weight = tree["chunks"] / k if active else 0.0
            paths, parent = {}, {}
            for path in tree["paths"]:
                a, b = path[0][0], path[-1][1]
                parent[b] = a
                paths[a, b] = (tuple(edge_id[tuple(e)] for e in path),)
                paths[b, a] = (tuple(edge_id[v, u] for u, v in reversed(path)),)
            routed = replace(base, shortest_paths=paths)
            pair = []
            for direction in ("reverse", "forward"):
                name = f"slot{slot}_root{root}_{direction}"
                pair.append(TreeSpec(root, parent, routed, direction=direction, name=name))
                weights[name] = weight
                if active:
                    source[name] = i
            participants.append(AllReduceParticipantSpec(root, (pair[0],), (pair[1],)))
            volume += weight
        tasks.append(AllReduceTaskSpec(slot, volume, tuple(participants), "allreduce"))
    template = NodeArcLinearTemplate.from_problem(MCNFProblem(base, tuple(tasks)))
    model = template.build(time_steps=horizon)
    lower, upper = model.lower_bounds.copy(), model.upper_bounds.copy()
    for ref in model.layout.trees:
        col = model.layout.l_offset + model.layout.lambda_index[ref.task_index, ref.tree_index]
        lower[col] = upper[col] = weights[ref.tree.name]
    return template, replace(model, lower_bounds=lower, upper_bounds=upper), source


def vector_from_plan(model, source, plan):
    z = np.zeros(model.layout.total_variables)
    z[:plan["time_steps"]] = plan["durations"]
    lookup = {(op["shard"], op["phase"], op["src"] if op["phase"] == "reduce" else op["dst"]): op for op in plan["operations"]}
    for ref in model.layout.trees:
        shard = source.get(ref.tree.name)
        weight = plan["shards"][shard]["amount"] if shard is not None else 0.0
        z[model.layout.l_offset + model.layout.lambda_index[ref.task_index, ref.tree_index]] = weight
        z[model.layout.chunk_offset + model.layout.chunk_index[ref.task_index, ref.tree.root]] = weight
        phase = "reduce" if ref.tree.direction == "reverse" else "broadcast"
        for node in ref.tree.nodes():
            progress = np.zeros(plan["time_steps"] + 1)
            if shard is not None:
                if node != ref.tree.root:
                    _, progress = cumulative(lookup[shard, phase, node], weight, plan["time_steps"])
                else:
                    relevant = [op for op in plan["operations"] if op["shard"] == shard and op["phase"] == phase and
                                (op["dst"] == node if phase == "reduce" else op["src"] == node)]
                    values = [cumulative(op, weight, plan["time_steps"])[1] for op in relevant]
                    if phase == "reduce":
                        progress = np.concatenate(([0.0], np.minimum.reduce(values)[:-1]))
                    else:
                        progress = np.concatenate((np.maximum.reduce(values)[1:], [weight]))
            z[model.layout.node_index(ref.task_index, ref.tree_index, node, np.arange(plan["time_steps"] + 1))] = progress
        if shard is not None:
            for child in ref.tree.children:
                flows, _ = cumulative(lookup[shard, phase, child], weight, plan["time_steps"])
                z[model.layout.flow_index(ref.task_index, ref.tree_index, child, 0, np.arange(plan["time_steps"]))] = flows
    return z


def optimized_plan(model, source, result, base):
    plan = json.loads(json.dumps(base))
    plan.update(method="canonical_fixed_forest_lp", durations=result.A.tolist(), objective=result.objective)
    plan.pop("packets_per_shard", None)
    lookup = {(op["shard"], op["phase"], op["src"] if op["phase"] == "reduce" else op["dst"]): op for op in plan["operations"]}
    for ref in model.layout.trees:
        if ref.tree.name not in source:
            continue
        shard = source[ref.tree.name]
        phase = "reduce" if ref.tree.direction == "reverse" else "broadcast"
        for child in ref.tree.children:
            op = lookup[shard, phase, child]
            for key in ("first_round", "repeat", "amount_per_packet"):
                op.pop(key, None)
            flows = result.z[model.layout.flow_index(ref.task_index, ref.tree_index, child, 0, np.arange(model.time_steps))]
            if np.min(flows) < -1e-8:
                raise ValueError("Negative LP flow cannot be exported")
            cursor, segments = 0.0, []
            for t, amount in enumerate(flows):
                if amount > 0:
                    segments.append([t, cursor, cursor + float(amount)])
                    cursor += float(amount)
            op["segments"] = segments
    return plan


def save_plan(path, plan):
    with gzip.open(path, "wt", encoding="utf-8") as out:
        json.dump(plan, out, separators=(",", ":"))


def run_case(case_dir, out, horizon, max_variables=2_000_000, solve_limit=600):
    start = time.perf_counter()
    case_dir, out = Path(case_dir), Path(out)
    result = json.loads((case_dir / "result.json").read_text())
    graph = json.loads((case_dir / "input.json").read_text())
    tree_file = case_dir / "trees.json"
    trees, k = json.loads(tree_file.read_text()), result["forest"]["k"]
    if not result["forest"].get("validated"):
        raise ValueError("Source forest is unfinished or invalid")
    expected = result["reference"]["graph"]
    if (graph["compute_nodes"] != expected["compute_nodes"] or
            sorted(graph["edges"]) != sorted(expected["edges"])):
        raise ValueError("Physical graph differs from canonical input")
    metrics, checked = validate_trees(graph_from_data(graph), set(graph["compute_nodes"]),
        {(t["root"], t["index"]): [[tuple(e) for e in p] for p in t["paths"]] for t in trees},
        {(t["root"], t["index"]): t["chunks"] for t in trees}, result["forest"]["U"], k)
    if json.loads(json.dumps(checked)) != trees:
        raise ValueError("Saved tree metadata differs from replay")
    out.mkdir(parents=True, exist_ok=False)
    timings = {"input_validation_sec": time.perf_counter() - start}
    row = {"case": case_dir.name, "hostname": socket.gethostname(), "python": sys.version,
           "platform": platform.platform(), "time_steps": horizon,
           "source_trees_sha256": digest(tree_file), "source_input_sha256": digest(case_dir / "input.json"),
           "source_result_sha256": digest(case_dir / "result.json"), "forest": metrics,
           "reference": result["reference"], "timings": timings,
           "minimum_paired_horizon": 2 * metrics["max_depth"], "status": "started"}
    def save():
        timings["case_wall_sec"] = time.perf_counter() - start
        row["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2 if sys.platform == "darwin" else 1024)
        (out / "result.json").write_text(json.dumps(row, indent=2) + "\n")
    save()
    if horizon < row["minimum_paired_horizon"]:
        row.update(status="infeasible_fixed_paired_depth", validated=False,
                   reason="Positive fixed shard on deepest paired tree needs reduce depth + broadcast depth; no pruning or reweighting")
        save()
        return row
    t = time.perf_counter()
    plan = wave_plan(graph, trees, k, horizon)
    timings["wave_schedule_sec"] = time.perf_counter() - t
    t = time.perf_counter()
    row["wave_validation"] = validate_plan(plan, trees, k)
    timings["wave_validation_sec"] = time.perf_counter() - t
    row["wave_objective"] = plan["objective"]
    t = time.perf_counter()
    save_plan(out / "wave_plan.json.gz", plan)
    with gzip.open(out / "wave_plan.json.gz", "rt") as handle:
        exported = json.load(handle)
    if exported["graph"] != graph:
        raise ValueError("Exported physical graph changed")
    row["wave_export_validation"] = validate_plan(exported, trees, k)
    row["wave_plan_sha256"] = digest(out / "wave_plan.json.gz")
    timings["wave_export_sec"] = time.perf_counter() - t
    estimate, slots = model_size(graph, trees, horizon)
    row["lp"] = {"estimated_variables": estimate, "root_slots": slots,
                 "max_variables": max_variables, "solve_limit_sec": solve_limit}
    row.update(status="wave_feasible", validated=True, objective=plan["objective"], plan="wave_plan.json.gz")
    save()
    if estimate > max_variables:
        row["lp"]["status"] = "skipped_size_guard"
        save()
        return row
    t = time.perf_counter()
    template, model, source = canonical_model(graph, trees, k, horizon)
    timings["model_build_sec"] = time.perf_counter() - t
    if model.layout.total_variables != estimate:
        raise ValueError("Preflight variable estimate disagrees with canonical model")
    t = time.perf_counter()
    z = vector_from_plan(model, source, plan)
    wave_result = template.result_from_z(horizon, z, status="candidate", objective=plan["objective"], solve_time=0, backend="packet_wave")
    row["canonical_wave_violations"] = model.constraint_violations(wave_result)
    if not model.is_feasible(wave_result, tol=1e-6):
        raise ValueError("Constructive wave plan failed canonical model validation")
    timings["canonical_wave_validation_sec"] = time.perf_counter() - t
    np.savez_compressed(out / "wave_solution.npz", z=z, A=wave_result.A)
    t = time.perf_counter()
    try:
        solved = solve_with_gurobi_native(model, {"name": "gurobi_native", "solver_args": {
            "Method": 2, "Crossover": 0, "Threads": 4, "TimeLimit": solve_limit,
            "SoftMemLimit": 16, "LogToConsole": 0, "LogFile": str(out / "solver.log"),
            "FeasibilityTol": 1e-8, "OptimalityTol": 1e-8, "BarConvTol": 1e-9,
            "_mcnf_record_presolved_norms": False, "_mcnf_record_gpu_memory": False}})
    except Exception as exc:
        timings["solver_call_sec"] = time.perf_counter() - t
        row["lp"].update(status="solver_error", error=repr(exc))
        save()
        return row
    timings["solver_call_sec"] = time.perf_counter() - t
    row["lp"].update(status=solved.status, objective=solved.objective,
        solve_time_sec=solved.solve_time, stats=solved.solver_stats)
    if solved.objective is not None:
        t = time.perf_counter()
        np.savez_compressed(out / "lp_solution.npz", z=solved.z, A=solved.A)
        row["lp"]["violations"] = model.constraint_violations(solved)
        row["lp"]["accepted_1e4"] = bool(model.is_feasible(solved, tol=1e-4))
        timings["lp_artifact_validation_sec"] = time.perf_counter() - t
        if row["lp"]["accepted_1e4"]:
            t = time.perf_counter()
            try:
                candidate = optimized_plan(model, source, solved, plan)
                replay = validate_plan(candidate, trees, k)
            except ValueError as exc:
                row["lp"]["export_error"] = str(exc)
            else:
                save_plan(out / "lp_plan.json.gz", candidate)
                with gzip.open(out / "lp_plan.json.gz", "rt") as handle:
                    decoded = json.load(handle)
                if decoded["graph"] != graph:
                    raise ValueError("Exported LP graph changed")
                validate_plan(decoded, trees, k)
                row["lp"]["plan_sha256"] = digest(out / "lp_plan.json.gz")
                row["lp"]["export_validation"] = replay
                if candidate["objective"] <= row["wave_objective"]:
                    row.update(status="lp_feasible", objective=candidate["objective"], plan="lp_plan.json.gz")
            timings["lp_export_validation_sec"] = time.perf_counter() - t
    save()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-steps", type=int, required=True)
    parser.add_argument("--max-variables", type=int, default=2_000_000)
    parser.add_argument("--solve-limit", type=int, default=600)
    args = parser.parse_args()
    if min(args.time_steps, args.max_variables, args.solve_limit) < 1:
        parser.error("Time steps, variable limit and solve limit must be positive")
    row = run_case(args.case_dir, args.out, args.time_steps, args.max_variables, args.solve_limit)
    print(json.dumps({key: row.get(key) for key in ("case", "time_steps", "status", "objective", "timings")}), flush=True)


if __name__ == "__main__":
    main()
