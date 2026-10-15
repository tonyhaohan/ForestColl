"""Run one sequential WH1234 ForestColl matrix, retaining failed attempts and times."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--java-timeout", type=int, default=7200)
    args = parser.parse_args()
    root = HERE / "results" / args.label
    root.mkdir(parents=True, exist_ok=False)
    inputs = HERE / "current_inputs.json"
    files = [*HERE.glob("*.py"), HERE / "ClosBenchmark.java", inputs,
             *sorted((HERE.parent / "parallel/src").glob("*.java"))]
    state = {"started_at": datetime.now(timezone.utc).isoformat(),
             "hostname": socket.gethostname(), "python": sys.version, "platform": platform.platform(),
             "source_sha256": {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             "java_version": subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True),
             "complete": False, "rows": []}
    def save():
        (root / "queue.json").write_text(json.dumps(state, indent=2) + "\n")
    for n in (64, 128, 256):
        for network in ("hpn", "closd", "ubd"):
            for f in range(3):
                name = f"{network}{n}_f{f}"
                command = [sys.executable, "-B", str(HERE / "clos_benchmark.py"),
                           "--inputs", str(inputs), "--network", network, "--n", str(n),
                           "--failure", str(f), "--label", args.label, "--backend", "java",
                           "--java-timeout", str(args.java_timeout)]
                state["current_case"] = name
                save()
                start = time.perf_counter()
                error = None
                with (root / f"{name}.log").open("w") as log:
                    try:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                        code = process.wait(timeout=args.java_timeout + 300)
                    except subprocess.TimeoutExpired as exc:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        code, error = None, str(exc)
                elapsed = time.perf_counter() - start
                path = root / name / "result.json"
                result = json.loads(path.read_text()) if path.exists() else {}
                forest = result.get("forest", {})
                row = {"case": name, "exit_code": code, "process_wall_sec": elapsed,
                       "validated": code == 0 and forest.get("validated", False),
                       "error": error or forest.get("generation_error"),
                       "command": [part.replace(str(HERE), "benchmarks") for part in command[1:]]}
                state["rows"].append(row)
                save()
                print(json.dumps(row), flush=True)
    state.update(complete=True, current_case=None, finished_at=datetime.now(timezone.utc).isoformat())
    save()


if __name__ == "__main__":
    main()
