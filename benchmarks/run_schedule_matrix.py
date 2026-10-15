"""Queue fixed-forest schedules, consuming completed synthesis cases only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def horizons(tokens, depth):
    return list(dict.fromkeys({"2D": 2 * depth, "4D": 4 * depth}.get(token, int(token) if token.isdecimal() else 0) for token in tokens))


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--horizons", nargs="+", default=["12", "2D", "4D"])
    parser.add_argument("--max-variables", type=int, default=2_000_000)
    parser.add_argument("--solve-limit", type=int, default=600)
    parser.add_argument("--source-wait-limit", type=int, default=21600)
    args = parser.parse_args()
    if (any(token not in ("2D", "4D") and (not token.isdecimal() or int(token) < 1) for token in args.horizons)
            or min(args.max_variables, args.solve_limit, args.source_wait_limit) < 1
            or Path(args.label).name != args.label):
        parser.error("Use positive limits and horizons (integers, 2D or 4D), and a simple new label")
    inputs = HERE / "current_inputs.json"
    source_queue = json.loads((args.source / "queue.json").read_text())
    if source_queue["source_sha256"]["benchmarks/current_inputs.json"] != hashlib.sha256(inputs.read_bytes()).hexdigest():
        parser.error("Synthesis inputs do not match this scheduling workspace")
    refs = json.loads(inputs.read_text())["rows"]
    names = [f"{network}{n}_f{f}" for n in (64, 128, 256) for network in ("hpn", "closd", "ubd") for f in range(3)]
    root = HERE / "results" / args.label
    root.mkdir(parents=True, exist_ok=False)
    files = [*HERE.glob("*.py"), inputs, *sorted((HERE / "vendor").rglob("*.py"))]
    state = {"hostname": socket.gethostname(), "python": sys.version,
             "started_at": datetime.now(timezone.utc).isoformat(), "source_label": args.source.name,
             "horizon_policy": args.horizons, "max_variables": args.max_variables,
             "solve_limit_sec": args.solve_limit, "source_wait_limit_sec": args.source_wait_limit,
             "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             "complete": False, "status": "queued", "pending_cases": names.copy(), "rows": []}
    start = time.monotonic()
    def save():
        atomic_json(root / "queue.json", state)
    def report():
        from build_schedule_report import build_report
        t = time.perf_counter()
        build_report(root)
        state["last_report_sec"] = time.perf_counter() - t
        save()
    save()
    report()
    while state["pending_cases"]:
        progressed = False
        for name in state["pending_cases"].copy():
            folder = args.source / name
            path = folder / "result.json"
            try:
                result = json.loads(path.read_text()) if path.exists() else {}
            except json.JSONDecodeError:
                continue  # The synthesis worker may be replacing its checkpoint.
            if not result.get("forest", {}).get("validated") or not (folder / "trees.json").exists():
                continue
            expected = next(r for r in refs if f"{r['network']}{r['num_gpus']}_f{r['failure_count']}" == name)
            if result["reference"] != expected:
                raise ValueError(f"Source reference changed: {name}")
            depth = result["forest"]["max_depth"]
            for horizon in horizons(args.horizons, depth):
                output = root / name / f"t{horizon}"
                output.parent.mkdir(exist_ok=True)
                command = [sys.executable, "-u", "-B", str(HERE / "fixed_schedule.py"), str(folder),
                           "--out", str(output), "--time-steps", str(horizon),
                           "--max-variables", str(args.max_variables), "--solve-limit", str(args.solve_limit)]
                state.update(status="running", current_case=name, current_horizon=horizon)
                save()
                t = time.perf_counter()
                error = None
                with (output.parent / f"t{horizon}.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        code = process.wait(timeout=args.solve_limit + 600)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        code, error = None, "worker_timeout"
                row = {"case": name, "time_steps": horizon, "exit_code": code, "error": error,
                       "process_wall_sec": time.perf_counter() - t,
                       "result": f"{name}/t{horizon}/result.json"}
                state["rows"].append(row)
                save()
                report()
                print(json.dumps(row), flush=True)
            state["pending_cases"].remove(name)
            progressed = True
            save()
        if not state["pending_cases"]:
            break
        try:
            upstream_done = json.loads((args.source / "queue.json").read_text())["complete"]
        except json.JSONDecodeError:
            upstream_done = False
        if upstream_done or time.monotonic() - start > args.source_wait_limit:
            for name in state["pending_cases"]:
                state["rows"].append({"case": name, "time_steps": None, "exit_code": None,
                                      "error": "source_missing_or_failed" if upstream_done else "source_wait_timeout",
                                      "process_wall_sec": None, "result": None})
            state["pending_cases"] = []
            break
        if not progressed:
            state.update(status="waiting_for_saved_forests", current_case=None, current_horizon=None)
            save()
            time.sleep(30)
    state.update(complete=True, status="finished", current_case=None, current_horizon=None,
                 finished_at=datetime.now(timezone.utc).isoformat())
    save()
    report()


if __name__ == "__main__":
    main()
