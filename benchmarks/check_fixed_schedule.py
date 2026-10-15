"""Small independent schedule checks; --solve is reserved for WH1234."""
import argparse
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fixed_schedule import (wave_plan, validate_plan, canonical_model, model_size,
                            vector_from_plan, optimized_plan, save_plan, run_case)
from run_schedule_matrix import horizons
from mcnf_snapshot.solver_factory.gurobi_native import solve_with_gurobi_native


def fixture():
    gpus = [1, 3, 7]
    edges = [[u, v, 2.0 if 9 in (u, v) else 1.0]
             for u in [*gpus, 9] for v in [*gpus, 9] if u != v]
    trees = []
    for root in gpus:
        a, b = [gpu for gpu in gpus if gpu != root]
        for i, count in enumerate((1, 2)):
            # Same logical parent map, different physical routes and weights.
            route = [[root, a]] if i == 0 else [[root, 9], [9, a]]
            trees.append({"root": root, "index": i, "chunks": count, "depth": 2,
                          "paths": [route, [[a, b]]]})
    return {"compute_nodes": gpus, "edges": edges}, trees, 3


class ScheduleChecks(unittest.TestCase):
    def test_pipeline_and_canonical_equivalence(self):
        graph, trees, k = fixture()
        for horizon in (4, 8):
            plan = wave_plan(graph, trees, k, horizon)
            self.assertTrue(validate_plan(plan, trees, k)["validated"])
            template, model, source = canonical_model(graph, trees, k, horizon)
            self.assertEqual(model_size(graph, trees, horizon)[0], model.layout.total_variables)
            self.assertEqual(len(source), 12)
            vector = vector_from_plan(model, source, plan)
            result = template.result_from_z(horizon, vector, status="candidate", objective=plan["objective"], solve_time=0, backend="test")
            self.assertTrue(model.is_feasible(result, tol=1e-8), model.constraint_violations(result))
            exported = optimized_plan(model, source, result, plan)
            self.assertTrue(validate_plan(exported, trees, k)["validated"])
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "plan.json.gz"
                save_plan(path, exported)
                with gzip.open(path, "rt") as handle:
                    self.assertEqual(json.load(handle), exported)

    def test_real_overlap(self):
        graph, trees, k = fixture()
        plan = wave_plan(graph, trees, k, 8)
        phases = {}
        for op in plan["operations"]:
            for t in range(op["first_round"], op["first_round"] + op["repeat"]):
                phases.setdefault(t, set()).add(op["phase"])
        self.assertTrue(any(len(value) == 2 for value in phases.values()))
        self.assertLess(plan["objective"], wave_plan(graph, trees, k, 4)["objective"])

    def test_infeasible_horizon(self):
        with self.assertRaisesRegex(ValueError, "T >= 4"):
            wave_plan(*fixture(), 3)

    def test_corrupt_exports_rejected(self):
        graph, trees, k = fixture()
        plan = wave_plan(graph, trees, k, 8)
        mutations = [
            lambda p: p["operations"][0].update(path=[[1, 7]]),
            lambda p: p["operations"][0].update(amount_per_packet=5),
            lambda p: p["operations"][-1].update(first_round=0),
            lambda p: p["operations"][0].update(depends_on=[999]),
            lambda p: p["durations"].__setitem__(0, 0),
            lambda p: p["shards"][0].update(chunks=2),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(plan)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_plan(bad, trees, k)

    def test_zero_weight_padding_slots(self):
        graph, trees, k = fixture()
        trees = [tree for tree in trees if tree["root"] != 1 or tree["index"] == 0]
        trees[0]["chunks"] = 3
        plan = wave_plan(graph, trees, k, 8)
        template, model, source = canonical_model(graph, trees, k, 8)
        z = vector_from_plan(model, source, plan)
        result = template.result_from_z(8, z, status="candidate", objective=plan["objective"], solve_time=0, backend="test")
        self.assertTrue(model.is_feasible(result, tol=1e-8), model.constraint_violations(result))

    def test_checkpoints_size_guard_and_no_overwrite(self):
        graph, trees, k = fixture()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            case = folder / "source"
            case.mkdir()
            (case / "input.json").write_text(json.dumps(graph))
            (case / "trees.json").write_text(json.dumps(trees))
            expected = {**graph, "edges": list(reversed(graph["edges"]))}
            (case / "result.json").write_text(json.dumps({"forest": {"validated": True, "U": 100, "k": k},
                                                           "reference": {"graph": expected}}))
            impossible = run_case(case, folder / "t3", 3, 1)
            self.assertEqual(impossible["status"], "infeasible_fixed_paired_depth")
            feasible = run_case(case, folder / "t8", 8, 1)
            self.assertTrue(feasible["wave_export_validation"]["validated"])
            self.assertEqual(feasible["lp"]["status"], "skipped_size_guard")
            with self.assertRaises(FileExistsError):
                run_case(case, folder / "t8", 8, 1)
            self.assertEqual(horizons(["12", "2D", "4D", "12"], 6), [12, 24])


def solve_check():
    graph, trees, k = fixture()
    plan = wave_plan(graph, trees, k, 8)
    _, model, source = canonical_model(graph, trees, k, 8)
    with tempfile.TemporaryDirectory() as folder:
        result = solve_with_gurobi_native(model, {"solver_args": {
            "Threads": 2, "Method": 2, "Crossover": 0, "LogToConsole": 0,
            "LogFile": str(Path(folder) / "solver.log"), "FeasibilityTol": 1e-9,
            "BarConvTol": 1e-10, "_mcnf_record_presolved_norms": False,
            "_mcnf_record_gpu_memory": False}})
        assert model.is_feasible(result, tol=1e-6), model.constraint_violations(result)
        exported = optimized_plan(model, source, result, plan)
        assert validate_plan(exported, trees, k)["validated"]
        assert result.objective <= plan["objective"] + 1e-6
        print(f"PASS: WH1234 native solve, objective={result.objective}, wave={plan['objective']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScheduleChecks)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.solve:
        solve_check()
