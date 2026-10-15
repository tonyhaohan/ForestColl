"""Check missing/guarded/invalid results and historical timing accounting."""
import hashlib
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import build_comparison_report as report


class ReportChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.schedule = self.root / "schedule"
        (self.schedule / "hpn64_f0/t4").mkdir(parents=True)
        self.ref = {"network": "hpn", "num_gpus": 64, "failure_count": 0,
                    "mcnf": {"t12_objective": 2, "t12_feasible_1e4": True, "compact_seconds": 7,
                             "compact_model_build_seconds": 3},
                    "mcnf_t12_timings": {"worker_wall_time_sec": 10}}
        (self.root / "current_inputs.json").write_text(json.dumps({"rows": [self.ref]}))
        self.queue = {"complete": False, "source_sha256": {"benchmarks/current_inputs.json":
                      hashlib.sha256((self.root / "current_inputs.json").read_bytes()).hexdigest()},
                      "rows": [{"case": "hpn64_f0", "time_steps": 4,
                                "process_wall_sec": 5, "result": "hpn64_f0/t4/result.json"}]}
        self.forest = {"case": "hpn64_f0", "depth_D": 2, "forest_B_inf": 2,
                       "trees_sha256": "same trees", "validated": True, "process_wall_sec": 3}
        self.result = {"time_steps": 4, "reference": self.ref, "source_trees_sha256": "same trees",
                       "forest": {"allreduce_fixed_forest_bandwidth_time": 2},
                       "validated": True, "status": "wave_feasible", "objective": 4,
                       "lp": {"status": "skipped_size_guard"}}

    def collect(self, forests=None):
        (self.schedule / "queue.json").write_text(json.dumps(self.queue))
        (self.schedule / "hpn64_f0/t4/result.json").write_text(json.dumps(self.result))
        with patch.object(report, "HERE", self.root), patch.object(report, "load_rows", return_value=(
                {"complete": False}, [self.forest] if forests is None else forests)):
            return report.collect_rows(self.root / "source", self.schedule)[2][0]

    def test_upper_bound_and_total_count_synthesis_once(self):
        row = self.collect()
        self.assertEqual(row["objective_kind_2D"], "wave_upper_bound")
        self.assertIsNone(row["lp_objective_2D"])
        self.assertEqual(row["total_process_sec_2D"], 8)
        self.assertEqual(row["total_to_main_worker_ratio_2D"], .8)
        self.assertEqual(row["main_compact_call_sec"], 7)
        self.assertIsNone(row["main_T12_model_build_sec"])
        self.assertIsNone(row["objective_4D"])
        self.assertEqual(row["status_4D"], "pending")

    def test_lp_requires_primal_and_export_acceptance(self):
        self.result.update(status="lp_feasible")
        self.result["lp"] = {"status": "optimal", "objective": 4, "accepted_1e4": True,
                             "export_validation": {"validated": False}}
        self.assertEqual(self.collect()["objective_kind_2D"], "missing")
        self.assertIsNone(self.collect()["objective_2D"])
        self.result["lp"]["export_validation"]["validated"] = True
        self.assertEqual(self.collect()["objective_kind_2D"], "lp_optimal")
        self.result["validated"] = False
        self.assertIsNone(self.collect()["objective_2D"])

    def test_failed_and_missing_forest_rows_retained(self):
        self.queue["rows"].append({"case": "hpn64_f0", "time_steps": None, "error": "source_wait_timeout"})
        row = self.collect(forests=[])
        self.assertIsNone(row["B_inf"])
        self.assertIsNone(row["T_2D"])
        self.assertEqual(row["status_2D"], "source_wait_timeout")
        self.assertEqual(row["main_T12_worker_sec"], 10)

    def test_mismatched_forests_and_inputs_rejected(self):
        self.result["source_trees_sha256"] = "other run"
        with self.assertRaises(AssertionError):
            self.collect()
        self.result["source_trees_sha256"] = "same trees"
        self.queue["source_sha256"]["benchmarks/current_inputs.json"] = "wrong inputs"
        with self.assertRaises(AssertionError):
            self.collect()


class RuntimeChecks(unittest.TestCase):
    def test_runtime_figures_use_log_time_axes(self):
        import matplotlib.pyplot as plt
        rows = [{"network": network, "num_gpus": 8, "failures": 0,
                 "forest_B_inf_bandwidth_us": 1, "forest_4D_runtime_us": 3,
                 "main_T12_runtime_us": 2, "main_T12_accepted_1e4": True}
                for network in ("hpn", "closd", "ubd")]
        figure, axes = plt.subplots(1, 3)
        try:
            report.plot_runtime(axes, rows)
            self.assertEqual([ax.get_yscale() for ax in axes], ["log"] * 3)
        finally:
            plt.close(figure)

    def test_defaults_drive_bandwidth_and_startup_together(self):
        with patch.dict(report.REPORT_DEFAULTS, data_per_gpu_MB=.5, unit_link_Gbps=200,
                        alpha_us_per_active_step=.25):
            self.assertEqual(report.runtime_us(8, [1, 0, 7], 8), (20.5, 2))
            self.assertEqual(report.bandwidth_us(2, 8), 5)

    def test_decimal_units_and_only_active_steps_pay_startup(self):
        value, active = report.runtime_us(8, [1, 0, 7], 8)
        self.assertEqual(active, 2)
        self.assertAlmostEqual(value, 5.2)
        self.assertEqual(report.runtime_us(None, None, 8), (None, None))
        for bad in ([1, 2], [9, -1], [float("nan")]):
            with self.assertRaises(AssertionError):
                report.runtime_us(8, bad, 8)

    def test_saved_plan_identity_and_missing_plan_stay_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = root / "current_inputs.json"
            inputs.write_text("{}")
            (root / "main_runtime_reference.json").write_text(json.dumps({
                "current_inputs_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
                "rows": [{"case": "test", "source_sha256": "original", "A": [1, 0, 7] + [0] * 9}]}))
            plan_path = root / "wave_plan.json.gz"
            with gzip.open(plan_path, "wt") as handle:
                json.dump({"time_steps": 4, "durations": [1, 0, 2, 1]}, handle)
            result = {"plan": plan_path.name, "status": "wave_feasible",
                      "wave_plan_sha256": hashlib.sha256(plan_path.read_bytes()).hexdigest()}
            (root / "result.json").write_text(json.dumps(result))
            row = {"case": "test", "num_gpus": 8, "B_inf": 2,
                   "main_T12_source_sha256": "original", "main_T12_objective": 8,
                   "result_4D": "result.json", "validated_4D": True,
                   "T_4D": 4, "objective_4D": 4}
            with patch.object(report, "HERE", root):
                report.add_runtime_values([row], root)
                self.assertAlmostEqual(row["forest_4D_runtime_us"], 2.8)
                self.assertEqual(row["forest_4D_active_steps"], 3)
                self.assertEqual(row["forest_B_inf_bandwidth_us"], 1.25)
                result["wave_plan_sha256"] = "changed"
                (root / "result.json").write_text(json.dumps(result))
                with self.assertRaises(AssertionError):
                    report.add_runtime_values([row], root)
                plan_path.unlink()
                report.add_runtime_values([row], root)
                self.assertIsNone(row["forest_4D_runtime_us"])
                row["main_T12_source_sha256"] = "other run"
                with self.assertRaises(AssertionError):
                    report.add_runtime_values([row], root)


if __name__ == "__main__":
    unittest.main()
