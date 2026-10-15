"""Check timeout cleanup and failed-row reporting without launching any solvers."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import build_wh1234_report as report
import run_matrix as matrix


with TemporaryDirectory() as temporary:
    here = Path(temporary) / "benchmarks"
    here.mkdir()
    shutil.copyfile(matrix.HERE / "current_inputs.json", here / "current_inputs.json")
    (here / "ClosBenchmark.java").write_text("test fixture\n")
    (here.parent / "parallel/src").mkdir(parents=True)
    processes = [Mock(pid=12345) for _ in range(27)]
    processes[0].wait.side_effect = [subprocess.TimeoutExpired("fixture", 1), -9]
    for process in processes[1:]:
        process.wait.return_value = 1
    with patch.object(matrix, "HERE", here), patch.object(matrix.sys, "argv", ["run_matrix.py", "--label", "check"]), \
         patch.object(matrix.platform, "platform", return_value="test platform"), \
         patch.object(matrix.subprocess, "check_output", return_value="test JDK"), \
         patch.object(matrix.subprocess, "Popen", side_effect=processes), \
         patch.object(matrix.os, "killpg") as kill, redirect_stdout(io.StringIO()):
        matrix.main()
        kill.assert_called_once_with(12345, matrix.signal.SIGKILL)
        try:
            matrix.main()
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing results would be overwritten")
    root = here / "results/check"
    queue = json.loads((root / "queue.json").read_text())
    assert queue["complete"] and len(queue["rows"]) == 27
    assert queue["rows"][0]["error"] and queue["rows"][0]["exit_code"] is None
    assert all(not row["validated"] and row["process_wall_sec"] >= 0 for row in queue["rows"])
    with patch.object(report, "HERE", here):
        _, rows = report.load_rows(root)
    assert len(rows) == 27 and len({row["case"] for row in rows}) == 27
    assert all(row["network"] and row["mcnf_t12_objective"] is not None for row in rows)
    assert all(row["forest_B_inf"] is None for row in rows)
print("PASS: process-group timeout cleanup, 27 retained failures, missing-result evidence and overwrite rejection")
