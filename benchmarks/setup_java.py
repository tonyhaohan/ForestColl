"""Fetch the pinned upstream Java dependencies and compile the Clos adapter."""
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

HERE = Path(__file__).resolve().parent
(HERE / "lib").mkdir(exist_ok=True)
(HERE / "build").mkdir(exist_ok=True)
for dependency in json.loads((HERE / "java_dependencies.json").read_text()):
    path = HERE / "lib" / dependency["url"].rsplit("/", 1)[-1]
    data = path.read_bytes() if path.exists() else urllib.request.urlopen(dependency["url"], timeout=60).read()
    assert hashlib.sha256(data).hexdigest() == dependency["sha256"], path
    if not path.exists():
        path.write_bytes(data)
subprocess.run(["javac", "-cp", str(HERE / "lib/*"), "-d", str(HERE / "build"),
    *[str(HERE.parent / "parallel/src" / (name + ".java")) for name in
      ("BinarySearchParallel", "EdgeSplittingParallel", "BranchingGenerator")],
    str(HERE / "ClosBenchmark.java")], check=True)
print("Java adapter ready")
