from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, field
from functools import cached_property
from itertools import groupby
from typing import Any, Sequence, cast, overload

import numpy as np
import scipy.sparse as sp

from .defaults import DEFAULT_SINK_THRESHOLD
from .problem import AllReduceTaskSpec, BaseTaskSpec, MCNFProblem, TaskSpec, TreeSpec
from .result_transfer import _transfer_same_tree_layout, transfer_result_for_template


EQ_ROOT, EQ_BOUNDARY, EQ_FLOW, EQ_VOLUME = range(4)
INEQ_DEPENDENCY = 0
ScalarIndex = int | np.integer
IndexArray = Sequence[int] | np.ndarray
IndexInput = ScalarIndex | IndexArray


def _index_array(value: IndexInput) -> np.ndarray:
    values = np.asarray(value)
    if values.size and values.dtype.kind not in "iu":
        raise TypeError("索引必须是整数")
    return values.astype(np.int64, copy=False)


@dataclass(frozen=True)
class _TreeRef:
    task_index: int
    tree_index: int
    task: BaseTaskSpec
    tree: TreeSpec


@dataclass(frozen=True)
class _AllReduceParticipantRef:
    task_index: int
    participant: int
    reduce_tree_indices: tuple[int, ...]
    broadcast_tree_indices: tuple[int, ...]


@dataclass(frozen=True)
class VariableLayout:
    time_steps: int
    a_offset: int
    l_offset: int
    chunk_offset: int
    x_offset: int
    f_offset: int
    total_variables: int
    lambda_index: dict[tuple[int, int], int]
    chunk_index: dict[tuple[int, int], int]
    tree_node_slice: dict[tuple[int, int], slice]
    tree_flow_slice: dict[tuple[int, int], slice]
    trees: tuple[_TreeRef, ...]
    _children: tuple[np.ndarray, ...]
    _path_offsets: tuple[np.ndarray, ...]
    _node_starts: np.ndarray
    _flow_starts: np.ndarray

    @overload
    def node_index(self, task_i: int, tree_i: int, node: ScalarIndex, t: ScalarIndex) -> int: ...

    @overload
    def node_index(self, task_i: int, tree_i: int, node: IndexArray, t: IndexInput) -> np.ndarray: ...

    @overload
    def node_index(self, task_i: int, tree_i: int, node: ScalarIndex, t: IndexArray) -> np.ndarray: ...

    def node_index(self, task_i: int, tree_i: int, node: IndexInput, t: IndexInput) -> int | np.ndarray:
        slot = self.lambda_index[(task_i, tree_i)]
        root = self.trees[slot].tree.root
        children = self.trees[slot].tree.children
        start = int(self._node_starts[slot])
        if isinstance(node, (int, np.integer)) and isinstance(t, (int, np.integer)):
            position = bisect_left(children, node)
            if node != root and (position == len(children) or children[position] != node):
                raise IndexError("节点不属于该树")
            if not 0 <= t <= self.time_steps:
                raise IndexError("节点时间索引越界")
            return start + (position + int(node > root)) * (self.time_steps + 1) + int(t)
        nodes, times = _index_array(node), _index_array(t)
        positions = np.searchsorted(self._children[slot], nodes)
        matched = (positions < len(children)) & (self._children[slot][np.minimum(positions, len(children) - 1)] == nodes)
        if np.any((nodes != root) & ~matched):
            raise IndexError("节点不属于该树")
        if np.any((times < 0) | (times > self.time_steps)):
            raise IndexError("节点时间索引越界")
        return np.asarray(start + (positions + (nodes > root)) * (self.time_steps + 1) + times)

    @overload
    def flow_index(self, task_i: int, tree_i: int, child: ScalarIndex, path_i: ScalarIndex, t: ScalarIndex) -> int: ...

    @overload
    def flow_index(self, task_i: int, tree_i: int, child: IndexArray, path_i: IndexInput, t: IndexInput) -> np.ndarray: ...

    @overload
    def flow_index(self, task_i: int, tree_i: int, child: ScalarIndex, path_i: IndexArray, t: IndexInput) -> np.ndarray: ...

    @overload
    def flow_index(self, task_i: int, tree_i: int, child: ScalarIndex, path_i: ScalarIndex, t: IndexArray) -> np.ndarray: ...

    def flow_index(self, task_i: int, tree_i: int, child: IndexInput, path_i: IndexInput, t: IndexInput) -> int | np.ndarray:
        slot = self.lambda_index[(task_i, tree_i)]
        children = self.trees[slot].tree.children
        offsets = self._path_offsets[slot]
        start = int(self._flow_starts[slot])
        if (
            isinstance(child, (int, np.integer))
            and isinstance(path_i, (int, np.integer))
            and isinstance(t, (int, np.integer))
        ):
            position = bisect_left(children, child)
            if position == len(children) or children[position] != child:
                raise IndexError("child 不属于该树")
            if not 0 <= path_i < offsets[position + 1] - offsets[position] or not 0 <= t < self.time_steps:
                raise IndexError("路径或流时间索引越界")
            return start + (int(offsets[position]) + int(path_i)) * self.time_steps + int(t)
        child_ids, paths, times = _index_array(child), _index_array(path_i), _index_array(t)
        positions = np.searchsorted(self._children[slot], child_ids)
        if np.any((positions >= len(children)) | (self._children[slot][np.minimum(positions, len(children) - 1)] != child_ids)):
            raise IndexError("child 不属于该树")
        if np.any((paths < 0) | (paths >= offsets[positions + 1] - offsets[positions])) or np.any(
            (times < 0) | (times >= self.time_steps)
        ):
            raise IndexError("路径或流时间索引越界")
        return np.asarray(start + (offsets[positions] + paths) * self.time_steps + times)


@dataclass(frozen=True)
class _ActiveTemplateView:
    problem: MCNFProblem
    tree_refs: tuple[_TreeRef, ...]
    lambda_index: dict[tuple[int, int], int]
    task_tree_indices: dict[int, tuple[int, ...]]
    allreduce_participants: tuple[_AllReduceParticipantRef, ...]
    chunk_keys: tuple[tuple[int, int], ...]
    flow_edge_ids: np.ndarray
    flow_edge_path_slots: np.ndarray
    tree_signature: tuple[int, ...]


@dataclass
class _CsrRows:
    """按 CSR 三数组直接追加约束行，避免 COO 三元组再转换。"""

    n_cols: int
    indices: list[np.ndarray] = field(default_factory=list)
    data: list[np.ndarray] = field(default_factory=list)
    row_lengths: list[int] = field(default_factory=list)
    rhs: list[float] = field(default_factory=list)

    def add(self, terms: list[tuple[int, float]], rhs: float = 0.0) -> None:
        self.indices.append(np.asarray([col for col, _value in terms], dtype=np.int32))
        self.data.append(np.asarray([value for _col, value in terms], dtype=float))
        self.row_lengths.append(len(terms))
        self.rhs.append(float(rhs))

    def add_rows_np(self, indices: np.ndarray, data: np.ndarray, row_lengths: np.ndarray, rhs: float = 0.0) -> None:
        lengths = np.asarray(row_lengths, dtype=np.int32).ravel()
        self.indices.append(np.asarray(indices, dtype=np.int32).ravel())
        self.data.append(np.asarray(data, dtype=float).ravel())
        self.row_lengths.extend(lengths.tolist())
        self.rhs.extend([float(rhs)] * int(lengths.size))

    def matrix(self) -> sp.csr_matrix:
        row_lengths = np.asarray(self.row_lengths, dtype=np.int32)
        indptr = np.zeros(row_lengths.size + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(row_lengths, dtype=np.int32)
        return sp.csr_matrix(
            (
                np.concatenate(self.data) if self.data else np.asarray([], dtype=float),
                np.concatenate(self.indices) if self.indices else np.asarray([], dtype=np.int32),
                indptr,
            ),
            shape=(len(self.row_lengths), int(self.n_cols)),
        )


def _make_variable_layout(tree_refs: tuple[_TreeRef, ...], time_steps: int, chunk_keys: tuple[tuple[int, int], ...] = ()) -> VariableLayout:
    T = int(time_steps)
    if T <= 0:
        raise ValueError("time_steps 必须是正整数")
    children = []
    path_offsets = []
    for ref in tree_refs:
        child_ids = np.asarray(ref.tree.children, dtype=np.int64)
        offsets = np.zeros(len(child_ids) + 1, dtype=np.int64)
        offsets[1:] = np.cumsum([len(paths) for paths in ref.tree.child_paths], dtype=np.int64)
        child_ids.setflags(write=False)
        offsets.setflags(write=False)
        children.append(child_ids)
        path_offsets.append(offsets)
    a_offset = 0
    l_offset = T
    chunk_offset = l_offset + len(tree_refs)
    x_offset = chunk_offset + len(chunk_keys)
    node_counts = np.asarray([len(ref.tree.children) + 1 for ref in tree_refs], dtype=np.int64)
    flow_counts = np.asarray([offsets[-1] for offsets in path_offsets], dtype=np.int64)
    node_starts = x_offset + (np.cumsum(node_counts) - node_counts) * (T + 1)
    f_offset = x_offset + int(np.sum(node_counts)) * (T + 1)
    flow_starts = f_offset + (np.cumsum(flow_counts) - flow_counts) * T
    return VariableLayout(
        time_steps=T,
        a_offset=a_offset,
        l_offset=l_offset,
        chunk_offset=chunk_offset,
        x_offset=x_offset,
        f_offset=f_offset,
        total_variables=f_offset + int(np.sum(flow_counts)) * T,
        lambda_index={(ref.task_index, ref.tree_index): idx for idx, ref in enumerate(tree_refs)},
        chunk_index={(int(task_i), int(participant)): idx for idx, (task_i, participant) in enumerate(chunk_keys)},
        tree_node_slice={
            (ref.task_index, ref.tree_index): slice(int(start), int(start + count * (T + 1)))
            for ref, start, count in zip(tree_refs, node_starts, node_counts)
        },
        tree_flow_slice={
            (ref.task_index, ref.tree_index): slice(int(start), int(start + count * T))
            for ref, start, count in zip(tree_refs, flow_starts, flow_counts)
        },
        trees=tree_refs,
        _children=tuple(children),
        _path_offsets=tuple(path_offsets),
        _node_starts=node_starts,
        _flow_starts=flow_starts,
    )


class NodeArcLinearBuilder:
    """node-arc primal 的连续线性模型构建器。"""

    def __init__(self, view: _ActiveTemplateView, time_steps: int) -> None:
        self.view = view
        self.time_steps = int(time_steps)
        if self.time_steps <= 0:
            raise ValueError(f"time_steps 必须是正整数，收到 {time_steps}")
        self.layout = _make_variable_layout(view.tree_refs, self.time_steps, view.chunk_keys)
        self._ref_by_task_tree = {(ref.task_index, ref.tree_index): ref for ref in view.tree_refs}

    def build(self) -> "NodeArcLinearModel":
        """构建标准 LP 矩阵，不触发求解。"""
        start = time.perf_counter()
        self._reset_constraint_blocks()
        problem = self.view.problem
        for task_i, task in enumerate(problem.tasks):
            if isinstance(task, AllReduceTaskSpec):
                self._add_allreduce_volume_constraints(task_i, task)
            else:
                self._add_task_volume_constraint(task_i, task)
        for ref in self.view.tree_refs:
            if isinstance(ref.task, AllReduceTaskSpec):
                self._add_allreduce_tree_boundary_constraint(ref)
                self._add_tree_dependency_constraint(ref)
            else:
                self._add_root_constraint(ref)
                self._add_boundary_constraint(ref)
                self._add_dependency_constraint(ref)
            self._add_flow_conservation_constraint(ref)
        for task_i, task in enumerate(problem.tasks):
            if isinstance(task, AllReduceTaskSpec):
                self._add_allreduce_stage_constraints(task_i, task)
        eq = cast(sp.csr_matrix, sp.vstack([block.matrix() for block in self._eq_blocks], format="csr"))
        capacity = self._edge_capacity_matrix()
        ineq = cast(sp.csr_matrix, sp.vstack([self._ineq_blocks[INEQ_DEPENDENCY].matrix(), capacity], format="csr"))
        eq_rhs = np.concatenate([np.asarray(block.rhs, dtype=float) for block in self._eq_blocks])
        capacity_rows = int(problem.topology.num_edges) * int(self.time_steps)
        ineq_rhs = np.concatenate([np.asarray(self._ineq_blocks[INEQ_DEPENDENCY].rhs, dtype=float), np.zeros(capacity_rows, dtype=float)])
        build_time = time.perf_counter() - start
        c = np.zeros(self.layout.total_variables, dtype=float)
        c[self.layout.a_offset : self.layout.l_offset] = float(problem.beta)
        return NodeArcLinearModel(
            model_type="main",
            time_steps=self.time_steps,
            num_edges=problem.topology.num_edges,
            c=c,
            eq_matrix=eq,
            eq_rhs=eq_rhs,
            ineq_matrix=ineq,
            ineq_rhs=ineq_rhs,
            lower_bounds=np.zeros(self.layout.total_variables, dtype=float),
            upper_bounds=np.full(self.layout.total_variables, np.inf, dtype=float),
            layout=self.layout,
            build_time=build_time,
            edge_capacity_row_start=ineq.get_shape()[0] - capacity_rows,
        )

    def _reset_constraint_blocks(self) -> None:
        self._eq_blocks = [_CsrRows(self.layout.total_variables) for _ in range(EQ_VOLUME + 1)]
        self._ineq_blocks = [_CsrRows(self.layout.total_variables)]

    def _add_root_constraint(self, ref: _TreeRef) -> None:
        T = self.time_steps
        root_start = self._x(ref, ref.tree.root, 0)
        if ref.tree.direction == "reverse":
            self._eq_blocks[EQ_ROOT].add_rows_np(np.arange(root_start, root_start + T + 1), np.ones(T + 1), np.ones(T + 1, dtype=np.int32))
            return
        li = self._l(ref)
        cols = np.column_stack((np.arange(root_start, root_start + T + 1), np.full(T + 1, li))).ravel()
        self._eq_blocks[EQ_ROOT].add_rows_np(cols, np.tile([1.0, -1.0], T + 1), np.full(T + 1, 2, dtype=np.int32))

    def _add_boundary_constraint(self, ref: _TreeRef) -> None:
        if ref.task.kind in {"deepep", "deepep-reverse"}:
            self._add_deepep_boundary_constraint(ref)
            return
        self._add_tree_boundary_constraint(ref)

    def _add_tree_boundary_constraint(self, ref: _TreeRef) -> None:
        child_starts = np.asarray(self.layout.node_index(ref.task_index, ref.tree_index, ref.tree.children, 0), dtype=np.int32)
        cols = np.column_stack((child_starts, child_starts + self.time_steps, np.full(len(child_starts), self._l(ref)))).ravel()
        self._eq_blocks[EQ_BOUNDARY].add_rows_np(cols, np.tile([1.0, 1.0, -1.0], len(child_starts)), np.tile([1, 2], len(child_starts)).astype(np.int32))

    def _add_deepep_boundary_constraint(self, ref: _TreeRef) -> None:
        starts = np.asarray(self.layout.node_index(ref.task_index, ref.tree_index, ref.tree.children, 0), dtype=np.int32)
        quotas = np.asarray([ref.tree.node_quota(child) for child in ref.tree.children])
        cols = np.column_stack((starts, starts + self.time_steps, np.full(starts.size, self._l(ref)))).ravel()
        data = np.column_stack((np.ones(starts.size), np.ones(starts.size), -quotas)).ravel()
        self._eq_blocks[EQ_BOUNDARY].add_rows_np(cols, data, np.tile([1, 2], starts.size).astype(np.int32))

    def _add_dependency_constraint(self, ref: _TreeRef) -> None:
        if ref.task.kind in {"deepep", "deepep-reverse"}:
            self._add_deepep_dependency_constraint(ref)
            return
        self._add_tree_dependency_constraint(ref)

    def _add_tree_dependency_constraint(self, ref: _TreeRef) -> None:
        T = self.time_steps
        reverse = ref.tree.direction == "reverse"
        lhs, rhs = (ref.tree.child_parents, ref.tree.children) if reverse else (ref.tree.children, ref.tree.child_parents)
        dep = np.column_stack((self.layout.node_index(ref.task_index, ref.tree_index, lhs, 1), self.layout.node_index(ref.task_index, ref.tree_index, rhs, 0))).astype(np.int32)
        cols = np.stack([dep[:, 0, None] + np.arange(T), dep[:, 1, None] + np.arange(T)], axis=2).ravel()
        self._ineq_blocks[INEQ_DEPENDENCY].add_rows_np(cols, np.tile([1.0, -1.0], T * len(lhs)), np.full(T * len(lhs), 2, dtype=np.int32))

    def _add_deepep_dependency_constraint(self, ref: _TreeRef) -> None:
        T = self.time_steps
        reverse = ref.tree.direction == "reverse"
        lhs, rhs = (ref.tree.child_parents, ref.tree.children) if reverse else (ref.tree.children, ref.tree.child_parents)
        dep = np.column_stack((self.layout.node_index(ref.task_index, ref.tree_index, lhs, 1), self.layout.node_index(ref.task_index, ref.tree_index, rhs, 0))).astype(np.int32)
        scales = np.asarray([(ref.tree.node_quota(right), -ref.tree.node_quota(left)) for left, right in zip(lhs, rhs)], dtype=float)
        cols = np.stack([dep[:, 0, None] + np.arange(T), dep[:, 1, None] + np.arange(T)], axis=2).ravel()
        data = np.tile(scales[:, None, :], (1, T, 1)).ravel()
        self._ineq_blocks[INEQ_DEPENDENCY].add_rows_np(cols, data, np.full(T * len(lhs), 2, dtype=np.int32))

    def _add_allreduce_tree_boundary_constraint(self, ref: _TreeRef) -> None:
        nodes = ref.tree.nodes()
        starts = np.asarray(self.layout.node_index(ref.task_index, ref.tree_index, nodes, 0), dtype=np.int32)
        cols = np.column_stack((starts, starts + self.time_steps, np.full(len(nodes), self._l(ref)))).ravel()
        self._eq_blocks[EQ_BOUNDARY].add_rows_np(cols, np.tile([1., 1., -1.], len(nodes)), np.tile([1, 2], len(nodes)))

    def _add_allreduce_volume_constraints(self, task_i: int, task: AllReduceTaskSpec) -> None:
        chunk_terms = [(self._chunk(int(task_i), int(item.participant)), 1.0) for item in task.participants]
        self._eq_blocks[EQ_VOLUME].add(chunk_terms, float(task.data_volume))
        participant_refs = [item for item in self.view.allreduce_participants if int(item.task_index) == int(task_i)]
        for participant_ref in participant_refs:
            chunk_col = self._chunk(int(task_i), int(participant_ref.participant))
            reduce_terms = [(self._l_by_task_tree(int(task_i), tree_i), 1.0) for tree_i in participant_ref.reduce_tree_indices]
            broadcast_terms = [(self._l_by_task_tree(int(task_i), tree_i), 1.0) for tree_i in participant_ref.broadcast_tree_indices]
            # 某阶段没有可用树时，空和约束会将该 root 的 chunk 固定为零。
            self._eq_blocks[EQ_VOLUME].add([*reduce_terms, (chunk_col, -1.0)], 0.0)
            self._eq_blocks[EQ_VOLUME].add([*broadcast_terms, (chunk_col, -1.0)], 0.0)

    def _add_allreduce_stage_constraints(self, task_i: int, task: AllReduceTaskSpec) -> None:
        T = self.time_steps
        for participant_ref in (item for item in self.view.allreduce_participants if int(item.task_index) == int(task_i)):
            participant = int(participant_ref.participant)
            tree_indices = participant_ref.reduce_tree_indices + participant_ref.broadcast_tree_indices
            for tree_i in tree_indices:
                ref = self._tree_ref(int(task_i), int(tree_i))
                x_start = self._x(ref, participant, 0)
                cols = np.stack([x_start + np.arange(T), x_start + np.arange(1, T + 1)], axis=1).ravel()
                self._ineq_blocks[INEQ_DEPENDENCY].add_rows_np(cols, np.tile([1.0, -1.0], T), np.full(T, 2, dtype=np.int32))
            for t in range(T):
                terms: list[tuple[int, float]] = []
                for tree_i in participant_ref.reduce_tree_indices:
                    ref = self._tree_ref(int(task_i), int(tree_i))
                    terms.append((self._x(ref, participant, t + 1), -1.0))
                for tree_i in participant_ref.broadcast_tree_indices:
                    ref = self._tree_ref(int(task_i), int(tree_i))
                    terms.append((self._x(ref, participant, t), 1.0))
                self._ineq_blocks[INEQ_DEPENDENCY].add(terms, 0.0)

    def _add_flow_conservation_constraint(self, ref: _TreeRef) -> None:
        T = self.time_steps
        counts = np.asarray([len(paths) for paths in ref.tree.child_paths], dtype=np.int32)
        if not counts.size:
            return
        lengths = np.repeat(counts + 2, T)
        starts = np.cumsum(lengths, dtype=np.int64) - lengths
        cols = np.empty(int(np.sum(lengths)), dtype=np.int32)
        data = np.full(cols.size, -1.0)
        time_ids = np.arange(T, dtype=np.int32)
        nodes = np.asarray(self.layout.node_index(ref.task_index, ref.tree_index, ref.tree.children, 0), dtype=np.int32)
        cols[starts] = (nodes[:, None] + time_ids + 1).ravel()
        cols[starts + 1] = (nodes[:, None] + time_ids).ravel()
        data[starts] = 1.0
        first_f = self._f(ref, ref.tree.children[0], 0, 0)
        path_starts = first_f + (np.cumsum(counts, dtype=np.int64) - counts) * T
        # 同路径数的 child 批量填充，保留原 child/time/path 次序及 [1,-1,-1,...] 系数。
        for count in np.unique(counts):
            children = np.flatnonzero(counts == count)
            positions = starts.reshape(-1, T)[children, :, None] + np.arange(2, 2 + count)
            values = path_starts[children, None, None] + time_ids[None, :, None] + np.arange(count)[None, None, :] * T
            cols[positions.ravel()] = values.ravel()
        self._eq_blocks[EQ_FLOW].add_rows_np(cols, data, lengths)

    def _add_task_volume_constraint(self, task_i: int, task: TaskSpec) -> None:
        terms = [(self._l_by_task_tree(int(task_i), tree_i), 1.0) for tree_i in self.view.task_tree_indices[int(task_i)]]
        self._eq_blocks[EQ_VOLUME].add(terms, float(task.data_volume))

    def _edge_capacity_matrix(self) -> sp.csr_matrix:
        topo = self.view.problem.topology
        T = self.time_steps
        rows = np.arange(topo.num_edges * T, dtype=np.int64)
        cols = np.tile(np.arange(self.layout.a_offset, self.layout.l_offset, dtype=np.int64), topo.num_edges)
        data = -np.repeat(np.asarray(topo.bandwidths, dtype=float), T)
        edge_ids = self.view.flow_edge_ids.astype(np.int64, copy=False)
        path_slots = self.view.flow_edge_path_slots.astype(np.int64, copy=False)
        if edge_ids.size:
            time_ids = np.tile(np.arange(T, dtype=np.int64), edge_ids.size)
            flow_cols = np.repeat(self.layout.f_offset + path_slots * T, T) + time_ids
            rows = np.concatenate([rows, np.repeat(edge_ids * T, T) + time_ids])
            cols = np.concatenate([cols, flow_cols])
            data = np.concatenate([data, np.ones(flow_cols.size, dtype=float)])
        return cast(sp.csr_matrix, sp.coo_matrix((data, (rows, cols)), shape=(topo.num_edges * T, self.layout.total_variables)).tocsr())

    def _l(self, ref: _TreeRef) -> int:
        return self._l_by_task_tree(ref.task_index, ref.tree_index)

    def _l_by_task_tree(self, task_i: int, tree_i: int) -> int:
        return self.layout.l_offset + self.view.lambda_index[(int(task_i), int(tree_i))]

    def _chunk(self, task_i: int, participant: int) -> int:
        return self.layout.chunk_offset + self.layout.chunk_index[(int(task_i), int(participant))]

    def _tree_ref(self, task_i: int, tree_i: int) -> _TreeRef:
        return self._ref_by_task_tree[(int(task_i), int(tree_i))]

    def _x(self, ref: _TreeRef, node: int, t: int) -> int:
        return self.layout.node_index(ref.task_index, ref.tree_index, int(node), int(t))

    def _f(self, ref: _TreeRef, child: int, path_i: int, t: int) -> int:
        return self.layout.flow_index(ref.task_index, ref.tree_index, int(child), int(path_i), int(t))


@dataclass(frozen=True)
class NodeArcLinearModel:
    model_type: str
    time_steps: int
    num_edges: int
    c: np.ndarray
    eq_matrix: sp.csr_matrix
    eq_rhs: np.ndarray
    ineq_matrix: sp.csr_matrix
    ineq_rhs: np.ndarray
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    layout: VariableLayout
    build_time: float
    edge_capacity_row_start: int

    @cached_property
    def edge_flow_matrix(self) -> sp.csr_matrix:
        rows = self.num_edges * self.time_steps
        return self.ineq_matrix[self.edge_capacity_row_start : self.edge_capacity_row_start + rows, self.layout.f_offset :].tocsr()

    @property
    def minimum_time_steps(self) -> int:
        return max(ref.tree.depth for ref in self.layout.trees)

    @property
    def tree_signature(self) -> tuple[int, ...]:
        counts = [0] * (max(ref.task_index for ref in self.layout.trees) + 1)
        for ref in self.layout.trees:
            counts[int(ref.task_index)] += 1
        return tuple(counts)

    @property
    def layout_signature(self) -> tuple[int, tuple[int, ...]]:
        return (int(self.time_steps), self.tree_signature)

    def constraint_violations(self, result: "NodeArcResult", *, allow_smaller_t: bool = False) -> dict[str, float]:
        z = self._result_z(result, allow_smaller_t=allow_smaller_t)
        if not np.all(np.isfinite(z)):
            return {key: float("inf") for key in ("eq", "ineq", "lower_bounds", "upper_bounds")}
        eq = self.eq_matrix @ z - self.eq_rhs
        ineq = self.ineq_matrix @ z - self.ineq_rhs
        lower = self.lower_bounds - z
        finite_ub = np.isfinite(self.upper_bounds)
        upper = z[finite_ub] - self.upper_bounds[finite_ub]
        return {
            "eq": _max_abs(eq),
            "ineq": _max_positive(ineq),
            "lower_bounds": _max_positive(lower),
            "upper_bounds": _max_positive(upper),
        }

    def is_feasible(self, result: "NodeArcResult", tol: float = 1e-6, *, allow_smaller_t: bool = False) -> bool:
        return max(self.constraint_violations(result, allow_smaller_t=allow_smaller_t).values()) <= float(tol)

    def validate_result(self, result: "NodeArcResult", tol: float = 1e-6, *, allow_smaller_t: bool = False) -> None:
        violations = self.constraint_violations(result, allow_smaller_t=allow_smaller_t)
        worst = max(violations.values())
        if worst > float(tol):
            raise ValueError(f"NodeArcResult 不满足线性模型约束: {violations}")

    def _result_z(self, result: "NodeArcResult", *, allow_smaller_t: bool) -> np.ndarray:
        if result.time_steps == self.time_steps:
            if result.z.shape != (self.layout.total_variables,):
                raise ValueError(f"z 长度必须为 {self.layout.total_variables}，收到 {result.z.shape}")
            return result.z
        if not allow_smaller_t or result.time_steps > self.time_steps:
            raise ValueError(f"result.time_steps={result.time_steps} 与模型 T={self.time_steps} 不匹配")
        return self._embed_smaller_z(result.z, int(result.time_steps))

    def _embed_smaller_z(self, z: np.ndarray, small_t: int) -> np.ndarray:
        small = _make_variable_layout(self.layout.trees, small_t, tuple(self.layout.chunk_index.keys()))
        if z.shape != (small.total_variables,):
            raise ValueError(
                f"小 T z 长度必须为 {small.total_variables}，收到 {z.shape}；"
                "allow_smaller_t=True 只支持同一 active tree set 的小 T result"
            )
        return _transfer_same_tree_layout(small, self.layout, z, list(range(small_t)))


@dataclass(frozen=True)
class NodeArcResult:
    time_steps: int
    status: str
    objective: float | None
    z: np.ndarray
    A: np.ndarray
    edge_flows: np.ndarray
    per_task_edge_flows: dict[int, np.ndarray]
    tree_flows: dict[tuple[int, int], float]
    solve_time: float
    backend: str
    solver_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "z", np.asarray(self.z, dtype=float))
        object.__setattr__(self, "A", np.asarray(self.A, dtype=float))
        object.__setattr__(self, "edge_flows", np.asarray(self.edge_flows, dtype=float))
        object.__setattr__(
            self,
            "per_task_edge_flows",
            {int(task_id): np.asarray(value, dtype=float) for task_id, value in self.per_task_edge_flows.items()},
        )

    def to_dict(self, *, include_z: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "time_steps": self.time_steps,
            "status": self.status,
            "objective": self.objective,
            "A": self.A.tolist(),
            "edge_flows": self.edge_flows.tolist(),
            "per_task_edge_flows": {int(key): value.tolist() for key, value in self.per_task_edge_flows.items()},
            "tree_flows": {f"{task_id}:{tree_i}": value for (task_id, tree_i), value in self.tree_flows.items()},
            "solve_time": self.solve_time,
            "backend": self.backend,
            "solver_stats": self.solver_stats,
        }
        if include_z:
            out["z"] = self.z.tolist()
        return out


def _max_abs(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.abs(values)))


def _max_positive(values: np.ndarray) -> float:
    return 0.0 if values.size == 0 else float(np.max(np.maximum(values, 0.0)))


@dataclass(frozen=True)
class NodeArcLinearTemplate:
    problem: MCNFProblem
    tree_refs: tuple[_TreeRef, ...]
    lambda_index: dict[tuple[int, int], int]
    minimum_time_steps: int
    _model_cache: dict[int, NodeArcLinearModel] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def automatic_time_steps(self) -> int:
        required = []
        for task in self.problem.tasks:
            if isinstance(task, AllReduceTaskSpec):
                required.append(min(
                    min(tree.depth for tree in item.reduce_trees)
                    + min(tree.depth for tree in item.broadcast_trees)
                    for item in task.participants
                ))
            else:
                required.append(min(tree.depth for tree in cast(TaskSpec, task).trees))
        return max(required)

    @classmethod
    def from_problem(cls, problem: MCNFProblem) -> "NodeArcLinearTemplate":
        tree_refs = tuple(
            _TreeRef(task_i, tree_i, task, tree)
            for task_i, task in enumerate(problem.tasks)
            for tree_i, tree in enumerate(_task_trees(task))
        )
        return cls(
            problem=problem,
            tree_refs=tree_refs,
            lambda_index={(ref.task_index, ref.tree_index): i for i, ref in enumerate(tree_refs)},
            minimum_time_steps=max(ref.tree.depth for ref in tree_refs),
        )

    def build(self, time_steps: int) -> NodeArcLinearModel:
        T = int(time_steps)
        cached = self._model_cache.get(T)
        if cached is not None:
            return cached
        model = NodeArcLinearBuilder(self._active_view(T), T).build()
        self._model_cache[T] = model
        return model

    def result_from_z(
        self,
        time_steps: int,
        z: np.ndarray,
        *,
        status: str,
        objective: float | None,
        solve_time: float,
        backend: str,
        solver_stats: dict[str, Any] | None = None,
    ) -> NodeArcResult:
        return _result_from_z(
            self.build(time_steps),
            z,
            status=status,
            objective=objective,
            solve_time=solve_time,
            backend=backend,
            solver_stats=solver_stats,
        )

    def transfer_result(self, result: NodeArcResult, time_steps: int | None = None, sink_threshold: float = DEFAULT_SINK_THRESHOLD) -> NodeArcResult | None:
        return transfer_result_for_template(self, result, time_steps, sink_threshold)

    def _active_view(self, time_steps: int) -> _ActiveTemplateView:
        T = int(time_steps)
        if T <= 0:
            raise ValueError(f"time_steps 必须是正整数，收到 {time_steps}")
        active_refs: list[_TreeRef] = []
        signature: list[int] = []
        for task_i, task in enumerate(self.problem.tasks):
            trees = _task_trees(task)
            active_tree_indices = tuple(tree_i for tree_i, tree in enumerate(trees) if int(tree.depth) <= T)
            if not active_tree_indices:
                raise ValueError(f"time_steps={T} 下 task {task_i} 没有可用 tree")
            signature.append(len(active_tree_indices))
            active_refs.extend(self.tree_refs[self.lambda_index[(task_i, tree_i)]] for tree_i in active_tree_indices)
        return _make_active_view(self.problem, tuple(active_refs), tuple(signature))


def _make_active_view(problem: MCNFProblem, tree_refs: tuple[_TreeRef, ...], signature: tuple[int, ...]) -> _ActiveTemplateView:
    task_tree_indices: dict[int, list[int]] = {task_i: [] for task_i in range(len(problem.tasks))}
    active_tree_indices = {(int(ref.task_index), int(ref.tree_index)) for ref in tree_refs}
    allreduce_participants: list[_AllReduceParticipantRef] = []
    chunk_keys: list[tuple[int, int]] = []
    for task_i, task in enumerate(problem.tasks):
        if not isinstance(task, AllReduceTaskSpec):
            continue
        cursor = 0
        for participant in task.participants:
            reduce_indices = tuple(idx for idx in range(cursor, cursor + len(participant.reduce_trees)) if (task_i, idx) in active_tree_indices)
            cursor += len(participant.reduce_trees)
            broadcast_indices = tuple(idx for idx in range(cursor, cursor + len(participant.broadcast_trees)) if (task_i, idx) in active_tree_indices)
            cursor += len(participant.broadcast_trees)
            chunk_keys.append((int(task_i), int(participant.participant)))
            allreduce_participants.append(
                _AllReduceParticipantRef(
                    task_index=int(task_i),
                    participant=int(participant.participant),
                    reduce_tree_indices=reduce_indices,
                    broadcast_tree_indices=broadcast_indices,
                )
            )
    slot = 0
    flow_edge_ids: list[int] = []
    flow_edge_path_slots: list[int] = []
    for ref in tree_refs:
        task_tree_indices[int(ref.task_index)].append(int(ref.tree_index))
        for paths in ref.tree.child_paths:
            for path in paths:
                flow_edge_ids.extend(int(edge) for edge in path)
                flow_edge_path_slots.extend([slot] * len(path))
                slot += 1
    return _ActiveTemplateView(
        problem=problem,
        tree_refs=tree_refs,
        lambda_index={(ref.task_index, ref.tree_index): i for i, ref in enumerate(tree_refs)},
        task_tree_indices={task_i: tuple(indices) for task_i, indices in task_tree_indices.items()},
        allreduce_participants=tuple(allreduce_participants),
        chunk_keys=tuple(chunk_keys),
        flow_edge_ids=np.asarray(flow_edge_ids, dtype=np.int32),
        flow_edge_path_slots=np.asarray(flow_edge_path_slots, dtype=np.int32),
        tree_signature=signature,
    )


def _result_from_z(
    model: NodeArcLinearModel,
    z: np.ndarray,
    *,
    status: str,
    objective: float | None,
    solve_time: float,
    backend: str,
    solver_stats: dict[str, Any] | None = None,
) -> NodeArcResult:
    layout = model.layout
    z = np.asarray(z, dtype=float)
    if z.shape != (layout.total_variables,):
        raise ValueError(f"z 长度必须为 {layout.total_variables}，收到 {z.shape}")
    A = np.array(z[layout.a_offset : layout.l_offset], copy=True)
    lambda_value = z[layout.l_offset : layout.chunk_offset]
    T = model.time_steps
    # 容量块按时间重复；提取 t=0 的行和流列，复用现有路径—边矩阵。
    row_start = model.edge_capacity_row_start
    incidence = model.ineq_matrix[row_start : row_start + model.num_edges * T : T, layout.f_offset :: T].tocsc()
    flows = z[layout.f_offset :].reshape(-1, T)
    edge_flows = np.asarray(incidence @ flows)
    per_task: dict[int, np.ndarray] = {}
    tree_flows: dict[tuple[int, int], float] = {}
    for task_id, task_refs in groupby(layout.trees, key=lambda ref: int(ref.task.task_id)):
        refs = tuple(task_refs)
        first, last = refs[0], refs[-1]
        start = (layout.tree_flow_slice[(first.task_index, first.tree_index)].start - layout.f_offset) // T
        stop = (layout.tree_flow_slice[(last.task_index, last.tree_index)].stop - layout.f_offset) // T
        per_task[task_id] = edge_flows.copy() if start == 0 and stop == flows.shape[0] else np.asarray(incidence[:, start:stop] @ flows[start:stop])
        for ref in refs:
            tree_key = (ref.task_index, ref.tree_index)
            tree_flows[(task_id, int(ref.tree_index))] = float(lambda_value[layout.lambda_index[tree_key]])
    return NodeArcResult(
        time_steps=model.time_steps,
        status=status,
        objective=objective,
        z=z,
        A=A,
        edge_flows=edge_flows,
        per_task_edge_flows=per_task,
        tree_flows=tree_flows,
        solve_time=float(solve_time),
        backend=backend,
        solver_stats={} if solver_stats is None else dict(solver_stats),
    )


def build_linear_model(problem: MCNFProblem, time_steps: int) -> NodeArcLinearModel:
    return NodeArcLinearTemplate.from_problem(problem).build(time_steps)


def _task_trees(task: BaseTaskSpec) -> tuple[TreeSpec, ...]:
    if isinstance(task, TaskSpec):
        return task.trees
    if isinstance(task, AllReduceTaskSpec):
        return task.trees()
    raise TypeError(f"不支持的 task spec 类型: {type(task).__name__}")
