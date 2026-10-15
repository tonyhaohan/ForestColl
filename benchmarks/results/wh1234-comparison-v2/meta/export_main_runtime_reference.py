"""Extract small round-duration references; keep the original z on WH1234."""
import hashlib
import json
from pathlib import Path
import sys

mcnf, inputs, closd, output = map(Path, sys.argv[1:])
exported = json.loads(inputs.read_text())
artifacts = {r["case"]: r for r in json.loads(closd.read_text())}
rows = []
for ref in exported["rows"]:
    old = ref["mcnf"]
    case = old["case"].removesuffix("_t12")
    source = mcnf / old["t12_source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == ref["t12_source_sha256"]
    data = json.loads(source.read_text())
    if ref["network"] == "closd":
        raw_a = artifacts[case]["A"]
        artifact = {k: v for k, v in artifacts[case].items() if k not in ("case", "A")}
        artifact["remote_run"] = data["remote_root"]
    else:
        solution = data["main_solution"]
        raw_a = solution.get("A", data["solution"]["A"])
        artifact = {}
    multiplier = old["t12_objective_multiplier"]
    a = [float(v) * multiplier for v in raw_a]
    assert len(a) == 12 and all(v >= 0 for v in a)
    assert abs(sum(a) - old["t12_objective"]) < 1e-8 * max(1, old["t12_objective"])
    rows.append({"case": case, "source": old["t12_source"],
                 "source_sha256": ref["t12_source_sha256"], "raw_A": raw_a,
                 "objective_multiplier": multiplier, "A": a, **artifact})
assert len(rows) == 27
output.write_text(json.dumps({"source_revision": exported["source_revision"],
    "current_inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
    "rows": rows}, indent=2) + "\n")
print(f"Exported {len(rows)} archived T12 round-duration vectors")
