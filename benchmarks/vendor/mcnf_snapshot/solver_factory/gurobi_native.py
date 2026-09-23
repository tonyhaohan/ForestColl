from __future__ import annotations

import time
from typing import Any, cast

import cvxpy as cp
import numpy as np

from ..linear_model import NodeArcLinearModel, NodeArcResult, _result_from_z
from .gpu_memory import _INTERNAL_RECORD_GPU_MEMORY, PeakGpuMemoryMonitor
from .presolve_stats import presolved_model_stats, save_presolved_model_stats

_INTERNAL_RECORD_PRESOLVED_NORMS = "_mcnf_record_presolved_norms"


def is_gurobi_native_backend(config: dict[str, Any]) -> bool:
    return str(config.get("name", "")).lower() == "gurobi_native"


def solve_with_gurobi_native(
    solve_model: NodeArcLinearModel,
    solver_backend: dict[str, Any],
    *,
    result_model: NodeArcLinearModel | None = None,
) -> NodeArcResult:
    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ValueError("solver_backend=gurobi_native 需要安装 gurobipy") from exc

    result_model = solve_model if result_model is None else result_model
    if solve_model.layout.total_variables != result_model.layout.total_variables:
        raise ValueError("gurobi_native 后端要求 solve_model 与 result_model 使用同一变量布局")

    solver_args = dict(solver_backend.get("solver_args", {}))
    solver_args.pop("verbose", None)
    record_presolved_norms = bool(solver_args.pop(_INTERNAL_RECORD_PRESOLVED_NORMS, True))
    record_gpu_memory = bool(solver_args.pop(_INTERNAL_RECORD_GPU_MEMORY, True))
    log_file = solver_args.get("LogFile")
    rhs_l2_norm, objective_l2_norm = _pdhg_relative_scales(solve_model)
    start_time = time.perf_counter()
    problem = gp.Model("mcnf_tree_node_arc")
    try:
        for key, value in solver_args.items():
            problem.setParam(str(key), value)

        lower = _gurobi_bounds(np.asarray(solve_model.lower_bounds, dtype=float), gp)
        upper = _gurobi_bounds(np.asarray(solve_model.upper_bounds, dtype=float), gp)
        z = problem.addMVar(
            shape=int(solve_model.layout.total_variables),
            lb=lower,
            ub=upper,
            obj=np.asarray(solve_model.c, dtype=float),
            name="z",
        )
        if solve_model.eq_matrix.get_shape()[0]:
            cast(Any, problem).addMConstr(solve_model.eq_matrix, z, gp.GRB.EQUAL, solve_model.eq_rhs, name="eq")
        if solve_model.ineq_matrix.get_shape()[0]:
            cast(Any, problem).addMConstr(solve_model.ineq_matrix, z, gp.GRB.LESS_EQUAL, solve_model.ineq_rhs, name="ineq")
        problem.ModelSense = gp.GRB.MINIMIZE
        setup_time = time.perf_counter() - start_time
        probe_stats: dict[str, Any] = {}
        probe_time = 0.0
        if record_presolved_norms:
            probe_started = time.perf_counter()
            output_flag = int(problem.Params.OutputFlag)
            problem.setParam("OutputFlag", 0)
            try:
                reduced = problem.presolve()
                try:
                    reduced_rhs = np.asarray(reduced.getAttr(gp.GRB.Attr.RHS, reduced.getConstrs()), dtype=float)
                    reduced_objective = np.asarray(reduced.getAttr(gp.GRB.Attr.Obj, reduced.getVars()), dtype=float)
                    probe_stats.update(
                        presolved_model_stats(
                            reduced_rhs,
                            reduced_objective,
                            rows=reduced.NumConstrs,
                            columns=reduced.NumVars,
                            nnz=reduced.NumNZs,
                        )
                    )
                finally:
                    reduced.dispose()
            except gp.GurobiError as exc:
                probe_stats["presolved_norm_probe_error"] = str(exc)
            finally:
                problem.setParam("OutputFlag", output_flag)
            probe_time = time.perf_counter() - probe_started
            probe_stats["presolved_norm_probe_time_sec"] = probe_time
            if "presolved_rhs_l2_norm" in probe_stats:
                save_presolved_model_stats(log_file, solve_model, probe_stats)
        with PeakGpuMemoryMonitor(record_gpu_memory) as gpu_memory:
            problem.optimize()
        elapsed = time.perf_counter() - start_time - probe_time

        has_primal_solution = int(getattr(problem, "SolCount", 0)) > 0
        z_value = np.asarray(z.X if has_primal_solution else np.zeros(solve_model.layout.total_variables), dtype=float)
        status = _gurobi_status(gp, int(problem.Status))
        objective = float(np.asarray(result_model.c @ z_value, dtype=float).item()) if has_primal_solution else None
        backend_time = float(getattr(problem, "Runtime", 0.0))
        solver_stats = {
            "setup_time": float(setup_time),
            "solve_time": backend_time,
            "build_time": float(result_model.build_time),
            "gurobi_native_wall_time": float(elapsed),
            "gurobi_native_overhead_time": max(float(elapsed) - backend_time, 0.0),
            "gurobi_status_code": int(problem.Status),
            "gurobi_sol_count": int(getattr(problem, "SolCount", 0)),
            "gurobi_num_vars": int(problem.NumVars),
            "gurobi_num_constrs": int(problem.NumConstrs),
            "gurobi_num_nz": int(problem.NumNZs),
            "gurobi_iter_count": float(getattr(problem, "IterCount", 0.0)),
            "gurobi_bar_iter_count": int(getattr(problem, "BarIterCount", 0)),
            "gurobi_pdhg_iter_count": float(getattr(problem, "PDHGIterCount", 0.0)),
            "rhs_l2_norm": rhs_l2_norm,
            "objective_l2_norm": objective_l2_norm,
            "gurobi_work": float(getattr(problem, "Work", 0.0)),
            "gurobi_max_mem_used_gb": float(getattr(problem, "MaxMemUsed", 0.0)),
        }
        solver_stats.update(probe_stats)
        solver_stats.update(gpu_memory.stats())
        return _result_from_z(
            result_model,
            z_value,
            status=status,
            objective=objective,
            solve_time=float(elapsed),
            backend="gurobi_native",
            solver_stats=solver_stats,
        )
    finally:
        problem.dispose()


def solve_postprocess_with_gurobi_native(base_model: NodeArcLinearModel, solve_model: NodeArcLinearModel, solver_backend: dict[str, Any]) -> NodeArcResult:
    return solve_with_gurobi_native(solve_model, solver_backend, result_model=base_model)


def _gurobi_bounds(values: np.ndarray, gp: Any) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    out[np.isneginf(out)] = -float(gp.GRB.INFINITY)
    out[np.isposinf(out)] = float(gp.GRB.INFINITY)
    return out


def _pdhg_relative_scales(model: NodeArcLinearModel) -> tuple[float, float]:
    rhs = np.concatenate((model.eq_rhs, model.ineq_rhs))
    return float(np.linalg.norm(rhs)), float(np.linalg.norm(model.c))


def _gurobi_status(gp: Any, status: int) -> str:
    mapping = {
        int(gp.GRB.OPTIMAL): cp.OPTIMAL,
        int(gp.GRB.INFEASIBLE): cp.INFEASIBLE,
        int(gp.GRB.UNBOUNDED): cp.UNBOUNDED,
        int(gp.GRB.INF_OR_UNBD): "infeasible_or_unbounded",
        int(gp.GRB.TIME_LIMIT): "user_limit",
        int(gp.GRB.ITERATION_LIMIT): "user_limit",
        int(gp.GRB.NODE_LIMIT): "user_limit",
        int(gp.GRB.SUBOPTIMAL): "user_limit",
        int(gp.GRB.INTERRUPTED): "interrupted",
    }
    user_obj_limit = getattr(gp.GRB, "USER_OBJ_LIMIT", None)
    if user_obj_limit is not None:
        mapping[int(user_obj_limit)] = "user_limit"
    return mapping.get(int(status), "solver_error")
