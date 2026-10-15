from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import networkx as nx
import numpy as np


Path = tuple[int, ...]
PathMap = Mapping[tuple[int, int], tuple[Path, ...]]
TASK_KINDS = frozenset({"p2p", "broadcast", "reduce", "deepep", "deepep-reverse", "allreduce"})


def _validate_physical_paths(num_nodes: int, edges: tuple[tuple[int, int], ...], src: int, dst: int, paths: tuple[Path, ...]) -> None:
    if not (0 <= src < num_nodes and 0 <= dst < num_nodes):
        raise ValueError(f"物理路径端点越界: {(src, dst)}")
    for path in paths:
        node = src
        for edge in path:
            if not 0 <= edge < len(edges):
                raise ValueError(f"路径边索引越界: {edge}")
            start, end = edges[edge]
            if start != node:
                raise ValueError(f"物理路径不连续或起点不匹配: {(src, dst)}")
            node = end
        if node != dst:
            raise ValueError(f"物理路径终点不匹配: {(src, dst)}")


def _as_positive_float(name: str, value: float) -> float:
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f"{name} 必须是正有限数，收到 {value}")
    return out


class LazyShortestPathMap(dict[tuple[int, int], tuple[Path, ...]]):
    """按需计算无权最短路径，避免大拓扑初始化时展开所有节点对。"""

    def __init__(self, num_nodes: int, edges: tuple[tuple[int, int], ...]) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.edges = tuple((int(src), int(dst)) for src, dst in edges)
        self.edge_index = {edge: idx for idx, edge in enumerate(self.edges)}
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.graph.add_edges_from(self.edges)

    def __missing__(self, key: tuple[int, int]) -> tuple[Path, ...]:
        src, dst = int(key[0]), int(key[1])
        if not (0 <= src < self.num_nodes and 0 <= dst < self.num_nodes):
            raise ValueError(f"物理路径端点越界: {(src, dst)}")
        if src == dst:
            paths: tuple[Path, ...] = ((),)
        else:
            try:
                node_paths = tuple(nx.all_shortest_paths(self.graph, src, dst))
            except nx.NetworkXNoPath:
                node_paths = ()
            paths = tuple(
                tuple(self.edge_index[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1))
                for nodes in node_paths
            )
        _validate_physical_paths(self.num_nodes, self.edges, src, dst, paths)
        self[(src, dst)] = paths
        return paths

    def get(self, key: tuple[int, int], default: Any = None) -> tuple[Path, ...]:
        src, dst = int(key[0]), int(key[1])
        return self[(src, dst)]


class PooledShortestPathMap(Mapping[tuple[int, int], tuple[Path, ...]]):
    """共享物理路径对象的最短路表，避免同一路径在大量 GPU pair 中重复存储。"""

    def __init__(
        self,
        pair_to_path_ids: Mapping[tuple[int, int], tuple[int, ...]],
        path_edges: tuple[Path, ...],
    ) -> None:
        self.pair_to_path_ids = {
            (int(src), int(dst)): tuple(int(path_id) for path_id in path_ids)
            for (src, dst), path_ids in pair_to_path_ids.items()
        }
        self.path_edges = tuple(tuple(int(edge) for edge in path) for path in path_edges)

    @classmethod
    def from_pair_paths(cls, pair_paths: Mapping[tuple[int, int], tuple[Path, ...]]) -> "PooledShortestPathMap":
        path_to_id: dict[Path, int] = {}
        path_edges: list[Path] = []
        pair_to_path_ids: dict[tuple[int, int], tuple[int, ...]] = {}
        for raw_key, paths in pair_paths.items():
            key = (int(raw_key[0]), int(raw_key[1]))
            ids: list[int] = []
            for raw_path in paths:
                path = tuple(int(edge) for edge in raw_path)
                path_id = path_to_id.get(path)
                if path_id is None:
                    path_id = len(path_edges)
                    path_to_id[path] = path_id
                    path_edges.append(path)
                ids.append(path_id)
            pair_to_path_ids[key] = tuple(ids)
        return cls(pair_to_path_ids, tuple(path_edges))

    def __len__(self) -> int:
        return len(self.pair_to_path_ids)

    def __bool__(self) -> bool:
        return bool(self.pair_to_path_ids)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, tuple) or len(key) != 2:
            return False
        try:
            normalized = (int(key[0]), int(key[1]))
        except (TypeError, ValueError):
            return False
        return normalized in self.pair_to_path_ids

    def __getitem__(self, key: tuple[int, int]) -> tuple[Path, ...]:
        src, dst = int(key[0]), int(key[1])
        return tuple(self.path_edges[path_id] for path_id in self.pair_to_path_ids[(src, dst)])

    def __iter__(self) -> Any:
        return iter(self.pair_to_path_ids)

    def get(self, key: tuple[int, int], default: Any = None) -> tuple[Path, ...]:
        src, dst = int(key[0]), int(key[1])
        path_ids = self.pair_to_path_ids.get((src, dst))
        if path_ids is None:
            return default
        return tuple(self.path_edges[path_id] for path_id in path_ids)

    def items(self) -> Any:
        for key in self.pair_to_path_ids:
            yield key, self[key]


@dataclass(frozen=True)
class Topology:
    num_nodes: int
    edges: tuple[tuple[int, int], ...]
    bandwidths: tuple[float, ...]
    shortest_paths: PathMap | PooledShortestPathMap = field(default_factory=dict)
    gpu_nodes: tuple[int, ...] = ()
    gpu_groups: tuple[tuple[int, ...], ...] = ()
    physical_gpu_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        if int(self.num_nodes) <= 0:
            raise ValueError("num_nodes 必须是正整数")
        normalized_edges = tuple((int(src), int(dst)) for src, dst in self.edges)
        if len(normalized_edges) == 0:
            raise ValueError("edges 不能为空")
        for src, dst in normalized_edges:
            if not (0 <= src < int(self.num_nodes)) or not (0 <= dst < int(self.num_nodes)):
                raise ValueError(f"边节点越界: {(src, dst)}")
            if src == dst:
                raise ValueError(f"不支持自环边: {(src, dst)}")
        if len(set(normalized_edges)) != len(normalized_edges):
            raise ValueError("edges 中存在重复边")
        bandwidths = tuple(_as_positive_float("bandwidth", value) for value in self.bandwidths)
        if len(bandwidths) != len(normalized_edges):
            raise ValueError("bandwidths 长度必须等于 edges 长度")
        paths: PathMap | PooledShortestPathMap
        if isinstance(self.shortest_paths, (LazyShortestPathMap, PooledShortestPathMap)):
            paths = self.shortest_paths
        else:
            paths = self._normalize_paths(self.shortest_paths)
        if not isinstance(paths, LazyShortestPathMap):
            for (src, dst), candidates in paths.items():
                _validate_physical_paths(int(self.num_nodes), normalized_edges, src, dst, candidates)
        gpu_nodes = tuple(int(node) for node in (self.gpu_nodes or tuple(range(int(self.num_nodes)))))
        if not gpu_nodes:
            raise ValueError("gpu_nodes 不能为空")
        for node in gpu_nodes:
            if not (0 <= node < int(self.num_nodes)):
                raise ValueError(f"GPU 节点越界: {node}")
        gpu_node_set = set(gpu_nodes)
        if len(gpu_node_set) != len(gpu_nodes):
            raise ValueError("gpu_nodes 中存在重复节点")
        gpu_groups = tuple(tuple(int(node) for node in group) for group in self.gpu_groups)
        if not gpu_groups:
            gpu_groups = tuple((node,) for node in gpu_nodes)
        for group in gpu_groups:
            if not group:
                raise ValueError("gpu_groups 不能包含空组")
            for node in group:
                if node not in gpu_node_set:
                    raise ValueError(f"gpu_groups 包含非 GPU 节点: {node}")
        physical_gpu_groups = tuple(tuple(int(node) for node in group) for group in (self.physical_gpu_groups or gpu_groups))
        for group in physical_gpu_groups:
            if not group:
                raise ValueError("physical_gpu_groups 不能包含空组")
            for node in group:
                if node not in gpu_node_set:
                    raise ValueError(f"physical_gpu_groups 包含非 GPU 节点: {node}")
        object.__setattr__(self, "edges", normalized_edges)
        object.__setattr__(self, "bandwidths", bandwidths)
        object.__setattr__(self, "shortest_paths", paths or LazyShortestPathMap(int(self.num_nodes), normalized_edges))
        object.__setattr__(self, "gpu_nodes", gpu_nodes)
        object.__setattr__(self, "gpu_groups", gpu_groups)
        object.__setattr__(self, "physical_gpu_groups", physical_gpu_groups)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def edge_index(self) -> dict[tuple[int, int], int]:
        return {edge: idx for idx, edge in enumerate(self.edges)}

    @classmethod
    def from_edges(
        cls,
        num_nodes: int,
        edges: list[tuple[int, int]] | tuple[tuple[int, int], ...],
        bandwidths: list[float] | tuple[float, ...] | None = None,
        gpu_nodes: tuple[int, ...] = (),
        gpu_groups: tuple[tuple[int, ...], ...] = (),
        physical_gpu_groups: tuple[tuple[int, ...], ...] = (),
        shortest_paths: PathMap | PooledShortestPathMap | None = None,
    ) -> "Topology":
        if bandwidths is None:
            bandwidths = tuple(1.0 for _ in edges)
        return cls(
            int(num_nodes),
            tuple(edges),
            tuple(float(v) for v in bandwidths),
            shortest_paths={} if shortest_paths is None else shortest_paths,
            gpu_nodes=gpu_nodes,
            gpu_groups=gpu_groups,
            physical_gpu_groups=physical_gpu_groups,
        )

    def _normalize_paths(self, raw: Mapping[tuple[int, int], tuple[Path, ...]]) -> dict[tuple[int, int], tuple[Path, ...]]:
        normalized: dict[tuple[int, int], tuple[Path, ...]] = {}
        for (src, dst), paths in dict(raw).items():
            key = (int(src), int(dst))
            normalized[key] = tuple(tuple(int(edge) for edge in path) for path in paths)
        return normalized

@dataclass(frozen=True)
class TreeSpec:
    root: int
    parent: dict[int, int]
    topology: Topology = field(repr=False, compare=False)
    paths: dict[int, tuple[Path, ...]] = field(init=False)
    direction: str = "forward"
    name: str = "tree"
    node_quotas: dict[int, float] = field(default_factory=dict)
    children: tuple[int, ...] = ()
    child_parents: tuple[int, ...] = ()
    child_paths: tuple[tuple[Path, ...], ...] = ()
    depth: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.direction not in {"forward", "reverse"}:
            raise ValueError(f"不支持的树方向: {self.direction}")
        children = tuple(sorted(int(child) for child in self.parent))
        parents = {int(child): int(parent) for child, parent in self.parent.items()}
        depth_map = {int(self.root): 0}
        visiting: set[int] = set()

        def get_depth(node: int) -> int:
            node = int(node)
            if node in depth_map:
                return depth_map[node]
            if node in visiting:
                raise ValueError("TreeSpec.parent 存在环")
            if node not in parents:
                raise ValueError(f"节点 {node} 的 parent chain 不连到 root")
            visiting.add(node)
            depth_map[node] = get_depth(parents[node]) + 1
            visiting.remove(node)
            return depth_map[node]

        for child in children:
            get_depth(child)
        paths = {}
        for child, parent in parents.items():
            pair = (child, parent) if self.direction == "reverse" else (parent, child)
            candidates = self.topology.shortest_paths.get(pair, ())
            if not candidates:
                raise ValueError(f"缺少从 {pair[0]} 到 {pair[1]} 的物理路径")
            paths[child] = candidates
        object.__setattr__(self, "paths", paths)
        normalized_quotas = {int(node): float(value) for node, value in self.node_quotas.items()}
        for node, value in normalized_quotas.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"节点 {node} 的 quota 必须是非负有限数，收到 {value}")
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "child_parents", tuple(int(self.parent[child]) for child in children))
        object.__setattr__(
            self,
            "child_paths",
            tuple(tuple(tuple(int(edge) for edge in path) for path in self.paths.get(child, ())) for child in children),
        )
        object.__setattr__(self, "node_quotas", normalized_quotas)
        object.__setattr__(self, "depth", max(depth_map.values()))

    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted({int(self.root), *[int(k) for k in self.parent], *[int(v) for v in self.parent.values()]}))

    def node_quota(self, node: int) -> float:
        return float(self.node_quotas.get(int(node), 1.0))

    def validate(self, topology: Topology) -> None:
        if not (0 <= int(self.root) < topology.num_nodes):
            raise ValueError(f"root 越界: {self.root}")
        if not self.parent:
            raise ValueError("TreeSpec.parent 不能为空")
        for child, parent in self.parent.items():
            child = int(child)
            parent = int(parent)
            if child == int(self.root):
                raise ValueError("root 不能出现在 parent 的 child 位置")
            if not (0 <= child < topology.num_nodes) or not (0 <= parent < topology.num_nodes):
                raise ValueError(f"树节点越界: child={child}, parent={parent}")


@dataclass(frozen=True)
class BaseTaskSpec:
    task_id: int
    data_volume: float


@dataclass(frozen=True)
class TaskSpec(BaseTaskSpec):
    trees: tuple[TreeSpec, ...]
    kind: str
    node_quotas: dict[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind not in TASK_KINDS - {"allreduce"}:
            raise ValueError(f"不支持的 task.kind: {self.kind}")
        normalized_quotas = {int(node): float(value) for node, value in self.node_quotas.items()}
        for node, value in normalized_quotas.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"节点 {node} 的 task quota 必须是非负有限数，收到 {value}")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "node_quotas", normalized_quotas)

    def node_quota(self, node: int) -> float:
        return float(self.node_quotas.get(int(node), 1.0))

    def validate(self, topology: Topology) -> None:
        _as_positive_float("data_volume", self.data_volume)
        if not self.trees:
            raise ValueError("TaskSpec.trees 不能为空")
        if self.kind in {"deepep", "deepep-reverse"}:
            all_nodes = {int(node) for tree in self.trees for node in tree.nodes()}
            missing = sorted(node for node in all_nodes if node not in self.node_quotas)
            if missing:
                raise ValueError(f"DeepEP task 缺少节点 quota: {missing}")
        for tree in self.trees:
            tree.validate(topology)


@dataclass(frozen=True)
class AllReduceParticipantSpec:
    participant: int
    reduce_trees: tuple[TreeSpec, ...]
    broadcast_trees: tuple[TreeSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participant", int(self.participant))
        object.__setattr__(self, "reduce_trees", tuple(self.reduce_trees))
        object.__setattr__(self, "broadcast_trees", tuple(self.broadcast_trees))
        if not self.reduce_trees:
            raise ValueError("AllReduce participant 缺少 reduce trees")
        if not self.broadcast_trees:
            raise ValueError("AllReduce participant 缺少 broadcast trees")
        for tree in self.reduce_trees:
            if int(tree.root) != int(self.participant):
                raise ValueError("AllReduce reduce tree 的 root 必须等于 participant")
            if tree.direction != "reverse":
                raise ValueError("AllReduce reduce tree 必须是 reverse 方向")
        for tree in self.broadcast_trees:
            if int(tree.root) != int(self.participant):
                raise ValueError("AllReduce broadcast tree 的 root 必须等于 participant")
            if tree.direction != "forward":
                raise ValueError("AllReduce broadcast tree 必须是 forward 方向")

    def trees(self) -> tuple[TreeSpec, ...]:
        return self.reduce_trees + self.broadcast_trees


@dataclass(frozen=True)
class AllReduceTaskSpec(BaseTaskSpec):
    participants: tuple[AllReduceParticipantSpec, ...]
    kind: str

    def __post_init__(self) -> None:
        normalized_kind = str(self.kind).strip().lower()
        if normalized_kind != "allreduce":
            raise ValueError(f"AllReduceTaskSpec.kind 必须是 allreduce，收到 {self.kind}")
        participants = tuple(self.participants)
        if not participants:
            raise ValueError("AllReduceTaskSpec.participants 不能为空")
        participant_ids = tuple(int(item.participant) for item in participants)
        if len(set(participant_ids)) != len(participant_ids):
            raise ValueError("AllReduce participant 存在重复")
        object.__setattr__(self, "kind", normalized_kind)
        object.__setattr__(self, "participants", participants)

    def trees(self) -> tuple[TreeSpec, ...]:
        out: list[TreeSpec] = []
        for participant in self.participants:
            out.extend(participant.reduce_trees)
            out.extend(participant.broadcast_trees)
        return tuple(out)

    def validate(self, topology: Topology) -> None:
        _as_positive_float("data_volume", self.data_volume)
        participant_ids = {int(item.participant) for item in self.participants}
        for participant in self.participants:
            all_trees = participant.trees()
            tree_node_sets = {frozenset(tree.nodes()) for tree in all_trees}
            if tree_node_sets != {frozenset(participant_ids)}:
                raise ValueError("AllReduce 每个 participant 的所有树都必须覆盖完整参与节点集合")
            for tree in all_trees:
                tree.validate(topology)


def _tree_topology_key(tree: TreeSpec) -> tuple[object, ...]:
    fingerprint = tuple((int(child), int(tree.parent[child]), tree.node_quota(int(child))) for child in tree.children)
    return (str(tree.direction), fingerprint)


def _dedup_trees_by_topology(trees: tuple[TreeSpec, ...]) -> tuple[TreeSpec, ...]:
    seen: set[tuple[object, ...]] = set()
    out: list[TreeSpec] = []
    for tree in trees:
        key = _tree_topology_key(tree)
        if key in seen:
            continue
        seen.add(key)
        out.append(tree)
    return tuple(out)


@dataclass(frozen=True)
class MCNFProblem:
    topology: Topology
    tasks: tuple[BaseTaskSpec, ...]
    beta: float = 1.0

    def __post_init__(self) -> None:
        _as_positive_float("beta", self.beta)
        if not self.tasks:
            raise ValueError("tasks 不能为空")
        if len({int(task.task_id) for task in self.tasks}) != len(self.tasks):
            raise ValueError("task_id 必须唯一")
        tasks: list[BaseTaskSpec] = []
        for task in self.tasks:
            if isinstance(task, TaskSpec):
                trees = _dedup_trees_by_topology(tuple(task.trees))
                tasks.append(replace(task, trees=tuple(sorted(trees, key=lambda tree: int(tree.depth)))))
                continue
            if isinstance(task, AllReduceTaskSpec):
                participants = tuple(
                    replace(
                        participant,
                        reduce_trees=tuple(sorted(_dedup_trees_by_topology(tuple(participant.reduce_trees)), key=lambda tree: int(tree.depth))),
                        broadcast_trees=tuple(sorted(_dedup_trees_by_topology(tuple(participant.broadcast_trees)), key=lambda tree: int(tree.depth))),
                    )
                    for participant in sorted(task.participants, key=lambda item: int(item.participant))
                )
                tasks.append(replace(task, participants=participants))
                continue
            raise TypeError(f"不支持的 task spec 类型: {type(task).__name__}")
        for task in tasks:
            task.validate(self.topology)
        object.__setattr__(self, "tasks", tuple(tasks))
