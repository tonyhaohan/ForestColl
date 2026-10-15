from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .defaults import DEFAULT_SINK_THRESHOLD

if TYPE_CHECKING:
    from .linear_model import VariableLayout


def transfer_result_for_template(
    template: Any,
    result: Any,
    time_steps: int | None = None,
    sink_threshold: float = DEFAULT_SINK_THRESHOLD,
) -> Any | None:
    if not np.isfinite(float(sink_threshold)) or float(sink_threshold) <= 0.0:
        raise ValueError(f"sink_threshold 必须是正有限数，收到 {sink_threshold}")
    active = _active_timesteps(result.A, float(sink_threshold))
    T = len(active) if time_steps is None else int(time_steps)
    if T <= 0 or T < len(active):
        return None
    start = time.perf_counter()
    source = template.build(result.time_steps)
    if result.z.shape != (source.layout.total_variables,):
        raise ValueError(f"result.z 长度必须为 {source.layout.total_variables}，收到 {result.z.shape}")
    target = template.build(T)
    keep = _transfer_kept_timesteps(active, result.time_steps, T)
    if keep is None or not _dropped_trees_are_zero(source, target, result.z, float(sink_threshold)):
        return None
    dropped = np.setdiff1d(np.arange(source.time_steps), keep)
    if dropped.size:
        flows = result.z[source.layout.f_offset :].reshape(-1, source.time_steps)
        states = result.z[source.layout.x_offset : source.layout.f_offset].reshape(-1, source.time_steps + 1)
        # 时间占比小不代表流量为零；累计检查被删步骤，避免破坏守恒和边界状态。
        if np.any(np.sum(np.abs(flows[:, dropped]), axis=1) > sink_threshold) or np.any(
            np.sum(np.abs(np.diff(states, axis=1)[:, dropped]), axis=1) > sink_threshold
        ):
            return None
    if source.layout_signature == target.layout_signature:
        stats = dict(result.solver_stats)
        stats["transfer_result"] = _transfer_stats(source, target, keep, start)
        return replace(result, solver_stats=stats)
    z = _transfer_z(source, target, result.z, keep)
    stats = dict(result.solver_stats)
    stats["transfer_result"] = _transfer_stats(source, target, keep, start)
    return _transfer_result_from_existing(target, result, z, keep, stats)


def _transfer_kept_timesteps(active: list[int], source_t: int, target_t: int) -> list[int] | None:
    if int(target_t) < len(active):
        return None
    if int(target_t) >= int(source_t):
        return list(range(int(source_t)))
    keep = list(active)
    for t in range(int(source_t)):
        if len(keep) >= int(target_t):
            break
        if t not in keep:
            keep.append(t)
    return sorted(keep)


def _active_timesteps(A: np.ndarray, threshold: float) -> list[int]:
    values = np.maximum(np.asarray(A, dtype=float), 0.0)
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        return []
    return [int(idx) for idx in np.flatnonzero(values / total > float(threshold))]


def _dropped_trees_are_zero(source: Any, target: Any, z: np.ndarray, threshold: float) -> bool:
    source_keys, target_keys = {(ref.task_index, ref.tree_index) for ref in source.layout.trees}, {(ref.task_index, ref.tree_index) for ref in target.layout.trees}
    dropped_keys = source_keys - target_keys
    if not dropped_keys:
        return True
    lambda_idx = [source.layout.l_offset + source.layout.lambda_index[key] for key in dropped_keys]
    if lambda_idx and bool(np.any(np.abs(z[np.asarray(lambda_idx, dtype=np.int64)]) > float(threshold))):
        return False
    for key in dropped_keys:
        flow_slice = source.layout.tree_flow_slice.get(key)
        if flow_slice is not None and bool(np.any(np.abs(z[flow_slice]) > float(threshold))):
            return False
    return True


def _transfer_z(source: Any, target: Any, z: np.ndarray, keep: list[int]) -> np.ndarray:
    if source.tree_signature == target.tree_signature:
        return _transfer_same_tree_layout(source.layout, target.layout, z, keep)
    out = np.zeros(target.layout.total_variables, dtype=float)
    keep_array = np.asarray(keep, dtype=np.int64)
    state_map = np.asarray([_source_state_time(t, keep, source.time_steps) for t in range(target.time_steps + 1)], dtype=np.int64)
    out[target.layout.a_offset : target.layout.a_offset + len(keep)] = z[source.layout.a_offset + keep_array]
    source_tree_keys = {(ref.task_index, ref.tree_index) for ref in source.layout.trees}
    for ref in target.layout.trees:
        tree_key = (ref.task_index, ref.tree_index)
        if tree_key not in source_tree_keys:
            continue
        out[target.layout.l_offset + target.layout.lambda_index[tree_key]] = z[source.layout.l_offset + source.layout.lambda_index[tree_key]]
        source_nodes = z[source.layout.tree_node_slice[tree_key]].reshape(-1, source.time_steps + 1)
        target_nodes = out[target.layout.tree_node_slice[tree_key]].reshape(-1, target.time_steps + 1)
        target_nodes[:] = source_nodes[:, state_map]
        source_flows = z[source.layout.tree_flow_slice[tree_key]].reshape(-1, source.time_steps)
        target_flows = out[target.layout.tree_flow_slice[tree_key]].reshape(-1, target.time_steps)
        target_flows[:, :len(keep)] = source_flows[:, keep_array]
    for key, target_idx in target.layout.chunk_index.items():
        source_idx = source.layout.chunk_index.get(key)
        if source_idx is not None:
            out[target.layout.chunk_offset + target_idx] = z[source.layout.chunk_offset + source_idx]
    return out


def _transfer_same_tree_layout(source: VariableLayout, target: VariableLayout, z: np.ndarray, keep: list[int]) -> np.ndarray:
    out = np.zeros(target.total_variables, dtype=float)
    out[target.a_offset : target.a_offset + len(keep)] = z[source.a_offset + np.asarray(keep, dtype=np.int64)]
    tree_count = len(source.trees)
    out[target.l_offset : target.l_offset + tree_count] = z[source.l_offset : source.l_offset + tree_count]
    chunk_count = len(source.chunk_index)
    out[target.chunk_offset : target.chunk_offset + chunk_count] = z[source.chunk_offset : source.chunk_offset + chunk_count]
    node_count = (source.f_offset - source.x_offset) // (source.time_steps + 1)
    if node_count:
        source_nodes = z[source.x_offset : source.f_offset].reshape(node_count, source.time_steps + 1)
        target_nodes = out[target.x_offset : target.f_offset].reshape(node_count, target.time_steps + 1)
        state_map = [_source_state_time(t, keep, source.time_steps) for t in range(target.time_steps + 1)]
        target_nodes[:, :] = source_nodes[:, np.asarray(state_map, dtype=np.int64)]
    flow_count = (source.total_variables - source.f_offset) // source.time_steps
    if flow_count:
        source_flows = z[source.f_offset : source.total_variables].reshape(flow_count, source.time_steps)
        target_flows = out[target.f_offset : target.total_variables].reshape(flow_count, target.time_steps)
        if keep:
            target_flows[:, : len(keep)] = source_flows[:, np.asarray(keep, dtype=np.int64)]
    return out


def _source_state_time(target_state_t: int, keep: list[int], source_t: int) -> int:
    if not keep:
        return min(int(target_state_t), int(source_t))
    if int(target_state_t) == 0:
        return int(keep[0]) if len(keep) < int(source_t) else 0
    if int(target_state_t) <= len(keep):
        return int(keep[int(target_state_t) - 1]) + 1 if len(keep) < int(source_t) else int(target_state_t)
    return int(source_t)


def _transfer_result_from_existing(
    target: Any,
    result: Any,
    z: np.ndarray,
    keep: list[int],
    solver_stats: dict[str, Any],
) -> Any:
    A, edge_flows = np.zeros(target.time_steps, dtype=float), np.zeros((target.num_edges, target.time_steps), dtype=float)
    task_ids = {int(ref.task.task_id) for ref in target.layout.trees}
    per_task = {task_id: np.zeros((target.num_edges, target.time_steps), dtype=float) for task_id in task_ids}
    n = min(len(keep), target.time_steps)
    if n:
        keep_arr = np.asarray(keep[:n], dtype=np.int64)
        A[:n] = result.A[keep_arr]
        edge_flows[:, :n] = result.edge_flows[:, keep_arr]
        for task_id in task_ids:
            if task_id in result.per_task_edge_flows:
                per_task[task_id][:, :n] = result.per_task_edge_flows[task_id][:, keep_arr]
    return replace(
        result,
        time_steps=target.time_steps,
        objective=float(np.asarray(target.c @ z, dtype=float).item()),
        z=z,
        A=A,
        edge_flows=edge_flows,
        per_task_edge_flows=per_task,
        tree_flows={(int(ref.task.task_id), int(ref.tree_index)): float(result.tree_flows.get((int(ref.task.task_id), int(ref.tree_index)), 0.0)) for ref in target.layout.trees},
        solver_stats=solver_stats,
    )


def _transfer_stats(source: Any, target: Any, keep: list[int], start: float) -> dict[str, Any]:
    source_keys = {(ref.task_index, ref.tree_index) for ref in source.layout.trees}
    target_keys = {(ref.task_index, ref.tree_index) for ref in target.layout.trees}
    return {
        "source_time_steps": int(source.time_steps),
        "target_time_steps": int(target.time_steps),
        "source_tree_signature": list(source.tree_signature),
        "target_tree_signature": list(target.tree_signature),
        "source_layout_signature": [source.time_steps, list(source.tree_signature)],
        "target_layout_signature": [target.time_steps, list(target.tree_signature)],
        "kept_timesteps": [int(t) for t in keep],
        "dropped_timesteps": [int(t) for t in range(source.time_steps) if t not in set(keep)],
        "dropped_trees": [list(key) for key in sorted(source_keys - target_keys)],
        "padded_trees": [list(key) for key in sorted(target_keys - source_keys)],
        "mapping_time": float(time.perf_counter() - start),
    }
