"""Freeze standard physical graphs and comparison evidence from an MCNF checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def export(root, output):
    sys.path[:0] = [str(root / "src"), str(root / "scripts")]
    from mcnf_tree.generation import create_topology
    from build_hpn_failure_case import build_failure_case

    evidence = root / "docs/experiment_results"
    comparison = evidence / "issue252_constant_flow_cg_standard_2h/comparison.json"
    references = json.loads(comparison.read_text())["rows"]
    rows = []
    for ref in references:
        network, n, f = ref["network"], ref["num_gpus"], ref["failures"]
        tag = f"{network}{n}_f{f}"
        folder = "hpn" if network == "hpn" else "standard"
        path = evidence / f"issue252_constant_flow/{folder}/configs/{tag}_constant_flow.json"
        original = json.loads(path.read_text())
        config = {key: original[key] for key in ("network", "task", "comm")}
        config["modeling"] = {"family": "node_arc", "params": {"time_steps": 12, "beta": 1.0}}
        failed = {64: [13, 2], 128: [26, 5], 256: [52, 11]}[n][:f]
        if network == "hpn":
            built = build_failure_case(n, failed)
            assert all(config["network"]["params"][k] == v for k, v in built["network"]["params"].items())
            plan = built["failure_plan"]
            assert config["network"]["params"]["bandwidth_intra_host"] == 4
            assert config["network"]["params"]["bandwidth_inter_host"] == 1
        else:
            archive = json.loads((Path(__file__).parent / ("reference.json" if network == "closd" else "ubd_reference.json")).read_text())
            old = next(r for r in archive["rows"] if (r["num_gpus"], r["failure_count"]) == (n, f))
            plan = old["failure_plan"]
        topology = create_topology(network, config["network"]["params"])
        graph = {"compute_nodes": list(topology.gpu_nodes), "edges": [
            [u, v, c] for (u, v), c in zip(topology.edges, topology.bandwidths)]}
        assert len(topology.gpu_nodes) == n
        assert not set(failed) & set(topology.gpu_nodes)
        source = root / ref["t12_source"]
        dynamic = json.loads(source.read_text())
        if "rows" in dynamic:
            dynamic = next(r for r in dynamic["rows"] if r["name"] == ref["case"])
        timings = dict(dynamic.get("timings", {}))
        if "worker_wall_time_sec" in dynamic:
            timings["worker_wall_time_sec"] = dynamic["worker_wall_time_sec"]
        rows.append({"network": network, "num_gpus": n, "failure_count": f,
                     "config": config, "failure_plan": plan, "graph": graph,
                     "objective": ref["t12_objective"], "raw_status": ref["t12_status"],
                     "max_violation": ref["t12_max_violation"], "standard_feasible": ref["t12_feasible_1e4"],
                     "mcnf": ref, "mcnf_t12_timings": timings,
                     "config_source": str(path.relative_to(root)),
                     "config_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "t12_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert len(rows) == 27
    paths = ["src/mcnf_tree/generation/topology.py", "src/mcnf_tree/problem.py", "scripts/build_hpn_failure_case.py"]
    result = {"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
              "source_files": {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
              "comparison_source": str(comparison.relative_to(root)),
              "comparison_sha256": hashlib.sha256(comparison.read_bytes()).hexdigest(), "rows": rows}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exported {len(rows)} current canonical physical graphs to {output.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcnf-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("current_inputs.json"))
    args = parser.parse_args()
    export(args.mcnf_root.resolve(), args.out.resolve())
