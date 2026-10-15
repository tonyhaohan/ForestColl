from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..collectives import PatternSpec, build_collective_trees, normalize_comm_patterns, single_layer_tree
from ..problem import AllReduceParticipantSpec, AllReduceTaskSpec, BaseTaskSpec, TaskSpec, Topology, TreeSpec


def build_tasks(topology: Topology, task_config: dict[str, Any], comm: dict[str, Any] | None = None, *, rng: np.random.Generator | None = None) -> tuple[BaseTaskSpec, ...]:
    forward_patterns = normalize_comm_patterns(comm, direction="forward")
    reverse_patterns = normalize_comm_patterns(comm, direction="reverse")
    task_type = str(task_config.get("type", "broadcast"))
    params = dict(task_config.get("params", {}))
    if task_type == "mixed":
        tasks: list[BaseTaskSpec] = []
        for entry in task_config.get("mix", []):
            tasks.extend(build_tasks(topology, dict(entry), comm, rng=rng))
        return tuple(renumber_tasks(tasks))
    nodes = [int(node) for node in topology.gpu_nodes]
    gpu_node_set = set(nodes)
    if task_type == "hybrid_pp_dp":
        return _build_hybrid_pp_dp_tasks(topology, params, forward_patterns, reverse_patterns)
    if task_type == "allreduce":
        group_size = _positive_group_size(params.get("group_size", len(nodes)), "group_size")
        volume = _data_volume(params)
        return tuple(renumber_tasks(build_dp_tasks(topology, dp_size=group_size, data_volume=volume, patterns=forward_patterns, reverse_patterns=reverse_patterns, mode="allreduce")))
    if task_type == "tp_partitioned_dp_allreduce":
        if "data_volume" in params:
            raise ValueError("tp_partitioned_dp_allreduce 的任务量按 DP group 大小自动生成，不能指定 data_volume")
        return tuple(
            renumber_tasks(
                build_tp_partitioned_dp_tasks(
                    topology,
                    tp_size=_positive_group_size(params.get("tp_size", 1), "tp_size"),
                    dp_groups=params.get("dp_groups"),
                    patterns=forward_patterns,
                    reverse_patterns=reverse_patterns,
                )
            )
        )
    if task_type == "tp_broadcast":
        if "data_volume" in params:
            raise ValueError("tp_broadcast 的任务量固定为每个 TP group broadcast 1，不能指定 data_volume")
        return tuple(build_tp_broadcast_tasks(topology, tp_size=_positive_group_size(params.get("tp_size", 2), "tp_size"), patterns=forward_patterns))
    if task_type == "deepep":
        return _build_deepep_tasks(topology, params, rng=rng)
    if task_type == "broadcast":
        _reject_root_specific_ahc(forward_patterns, "broadcast")
        volume = _data_volume(params)
        root = _require_gpu_node(gpu_node_set, params.get("root", nodes[0]), "root")
        receivers = [node for node in nodes if node != root]
        trees = build_dp_collective_trees(topology, root=root, receivers=receivers, prefix="broadcast", patterns=forward_patterns)
        return (TaskSpec(0, volume, trees, "broadcast"),)
    if task_type == "p2p":
        volume = _data_volume(params)
        count = int(params.get("task_count", 1))
        if count <= 0:
            raise ValueError(f"task_count 必须是正整数，收到 {count}")
        tasks = []
        for task_id in range(count):
            src = _require_gpu_node(gpu_node_set, params["source"], "source") if "source" in params else nodes[task_id % len(nodes)]
            dst = _require_gpu_node(gpu_node_set, params["target"], "target") if "target" in params else _next_gpu_node(nodes, src)
            if src == dst:
                raise ValueError("p2p 的 source 和 target 必须是不同 GPU 节点")
            trees = build_collective_trees(
                topology,
                root=src,
                receivers=[dst],
                prefix=f"p2p_{src}_{dst}",
                patterns=normalize_comm_patterns({"name": "single_layer"}),
            )
            tasks.append(TaskSpec(task_id, volume, trees, "p2p"))
        return tuple(tasks)
    if task_type == "reduce":
        _reject_root_specific_ahc(reverse_patterns, "reduce")
        volume = _data_volume(params)
        root = _require_gpu_node(gpu_node_set, params.get("root", nodes[0]), "root")
        receivers = [node for node in nodes if node != root]
        trees = build_dp_collective_trees(topology, root=root, receivers=receivers, prefix="reduce", patterns=reverse_patterns, direction="reverse")
        return (TaskSpec(0, volume, trees, "reduce"),)
    raise ValueError(f"不支持的 task.type: {task_type}")


def renumber_tasks(tasks: Sequence[BaseTaskSpec], start: int = 0) -> list[BaseTaskSpec]:
    out: list[BaseTaskSpec] = []
    for idx, task in enumerate(tasks):
        task_id = int(start) + idx
        if isinstance(task, TaskSpec):
            out.append(TaskSpec(task_id, task.data_volume, task.trees, task.kind, task.node_quotas))
        elif isinstance(task, AllReduceTaskSpec):
            out.append(AllReduceTaskSpec(task_id, task.data_volume, task.participants, task.kind))
        else:
            raise TypeError(f"不支持的 task spec 类型: {type(task).__name__}")
    return out


def build_dp_collective_trees(
    topology: Topology,
    *,
    root: int,
    receivers: list[int],
    prefix: str,
    patterns: tuple[PatternSpec, ...],
    direction: str = "forward",
) -> tuple[TreeSpec, ...]:
    return build_collective_trees(topology, root=int(root), receivers=list(receivers), prefix=prefix, patterns=patterns, direction=direction)


def build_pp_collective_trees(topology: Topology, *, source: int, target: int, prefix: str) -> tuple[TreeSpec, ...]:
    return (single_layer_tree(topology, root=int(source), receivers=[int(target)], name=prefix),)


def build_deepep_collective_trees(
    topology: Topology,
    *,
    source: int,
    source_group: list[int],
    target_group: list[int],
    receivers: list[int],
    direction: str,
    node_quotas: dict[int, float],
    prefix: str,
    mode: str = "narrow",
) -> tuple[TreeSpec, ...]:
    relays = _deepep_relays(source=int(source), source_group=source_group, target_group=target_group, receivers=receivers, mode=mode)
    trees: list[TreeSpec] = []
    for relay in relays:
        tree_quotas = dict(node_quotas)
        tree_quotas[int(source)] = 1.0
        tree_quotas[int(relay)] = 1.0
        trees.append(
            _deepep_tree(
                topology,
                source=int(source),
                relay=int(relay),
                receivers=receivers,
                direction=direction,
                node_quotas=tree_quotas,
                name=f"{prefix}_relay_{int(relay)}",
            )
        )
    return tuple(trees)


def build_dp_tasks(
    topology: Topology,
    *,
    dp_size: int,
    data_volume: float,
    patterns: tuple[PatternSpec, ...],
    mode: str = "broadcast",
    include_reduce: bool = False,
    reduce_data_volume: float | None = None,
    reverse_patterns: tuple[PatternSpec, ...] = (),
) -> list[BaseTaskSpec]:
    gpu_nodes = [int(node) for node in topology.gpu_nodes]
    total = len(gpu_nodes)
    group_size = _positive_group_size(dp_size, "dp_size")
    if total % group_size != 0:
        raise ValueError(f"dp_size 必须整除 GPU 总数，收到 total={total}, dp_size={group_size}")
    groups = [gpu_nodes[start : start + group_size] for start in range(0, total, group_size)]
    return _build_dp_tasks_for_groups(
        topology,
        groups=groups,
        data_volume=data_volume,
        patterns=patterns,
        mode=mode,
        include_reduce=include_reduce,
        reduce_data_volume=reduce_data_volume,
        reverse_patterns=reverse_patterns,
    )


def build_tp_partitioned_dp_tasks(
    topology: Topology,
    *,
    tp_size: int,
    patterns: tuple[PatternSpec, ...],
    reverse_patterns: tuple[PatternSpec, ...] = (),
    dp_groups: Sequence[Sequence[int]] | None = None,
) -> list[BaseTaskSpec]:
    logical_groups, resolved_tp = _tp_logical_groups(topology, tp_size)

    groups = (
        [[node for logical in logical_groups for rank, node in enumerate(logical) if rank % resolved_tp == offset] for offset in range(resolved_tp)]
        if dp_groups is None
        else [[int(node) for node in group] for group in dp_groups]
    )
    _validate_tp_partitioned_dp_groups(topology, logical_groups, groups, resolved_tp)
    return _build_dp_tasks_for_groups(
        topology,
        groups=groups,
        data_volume=1.0,
        patterns=patterns,
        mode="allreduce",
        reverse_patterns=reverse_patterns,
    )


def build_tp_broadcast_tasks(
    topology: Topology,
    *,
    tp_size: int,
    patterns: tuple[PatternSpec, ...],
) -> list[TaskSpec]:
    _reject_root_specific_ahc(patterns, "broadcast")
    logical_groups, resolved_tp = _tp_logical_groups(topology, tp_size)
    if resolved_tp == 1:
        raise ValueError("tp_size=1 没有 TP 通信任务")
    logical_size = len(logical_groups[0])

    tasks: list[TaskSpec] = []
    for logical in logical_groups:
        for start in range(0, logical_size, resolved_tp):
            group = logical[start : start + resolved_tp]
            root = int(group[0])
            trees = build_dp_collective_trees(topology, root=root, receivers=list(group[1:]), prefix=f"tp{resolved_tp}_broadcast_{root}", patterns=patterns)
            tasks.append(TaskSpec(len(tasks), 1.0, trees, "broadcast"))
    return tasks


def _tp_logical_groups(topology: Topology, tp_size: int) -> tuple[tuple[tuple[int, ...], ...], int]:
    resolved_tp = _positive_group_size(tp_size, "tp_size")
    groups = tuple(tuple(int(node) for node in group) for group in topology.gpu_groups)
    sizes = {len(group) for group in groups}
    if len(sizes) != 1:
        raise ValueError(f"TP 切分要求逻辑节点大小一致，收到 {sorted(sizes)}")
    logical_size = next(iter(sizes))
    if logical_size % resolved_tp != 0:
        raise ValueError(f"tp_size 必须整除逻辑节点大小，收到 logical_size={logical_size}, tp_size={resolved_tp}")
    return groups, resolved_tp


def _validate_tp_partitioned_dp_groups(
    topology: Topology,
    logical_groups: Sequence[Sequence[int]],
    dp_groups: Sequence[Sequence[int]],
    tp_size: int,
) -> None:
    if len(dp_groups) != int(tp_size):
        raise ValueError(f"DP 任务数必须等于 tp_size，收到 {len(dp_groups)} vs {tp_size}")
    flattened = [int(node) for group in dp_groups for node in group]
    gpu_nodes = set(int(node) for node in topology.gpu_nodes)
    unknown = sorted(set(flattened) - gpu_nodes)
    if unknown:
        raise ValueError(f"DP groups 包含非 GPU 节点: {unknown}")
    if len(flattened) != len(set(flattened)):
        raise ValueError("DP groups 中存在重复 GPU")
    missing = sorted(gpu_nodes - set(flattened))
    if missing:
        raise ValueError(f"DP groups 未覆盖全部 GPU: {missing}")
    expected = len(logical_groups[0]) // int(tp_size)
    for logical_id, logical in enumerate(logical_groups):
        logical_set = set(int(node) for node in logical)
        counts = [sum(int(node) in logical_set for node in group) for group in dp_groups]
        if any(count != expected for count in counts):
            raise ValueError(f"逻辑节点 {logical_id} 对每个 DP group 必须贡献 {expected} 张卡，收到 {counts}")


def _build_dp_tasks_for_groups(
    topology: Topology,
    *,
    groups: Sequence[Sequence[int]],
    data_volume: float,
    patterns: tuple[PatternSpec, ...],
    mode: str = "broadcast",
    include_reduce: bool = False,
    reduce_data_volume: float | None = None,
    reverse_patterns: tuple[PatternSpec, ...] = (),
) -> list[BaseTaskSpec]:
    normalized_mode = str(mode).strip().lower()
    if include_reduce and normalized_mode == "broadcast":
        normalized_mode = "broadcast_and_reduce"
    if normalized_mode not in {"broadcast", "reduce", "broadcast_and_reduce", "allreduce"}:
        raise ValueError(f"DP mode 只支持 broadcast/reduce/broadcast_and_reduce/allreduce，收到 {mode}")
    if normalized_mode in {"reduce", "broadcast_and_reduce", "allreduce"} and not reverse_patterns:
        reverse_patterns = patterns
    tasks: list[BaseTaskSpec] = []
    if normalized_mode != "allreduce":
        if normalized_mode in {"broadcast", "broadcast_and_reduce"}:
            _reject_root_specific_ahc(patterns, "broadcast")
        if normalized_mode in {"reduce", "broadcast_and_reduce"}:
            _reject_root_specific_ahc(reverse_patterns, "reduce")
    for group_id, raw_group in enumerate(groups):
        group = [int(node) for node in raw_group]
        group_size = len(group)
        if normalized_mode == "allreduce":
            participants: list[AllReduceParticipantSpec] = []
            for participant in group:
                receivers = [node for node in group if node != participant]
                reduce_trees = build_dp_collective_trees(
                    topology,
                    root=participant,
                    receivers=receivers,
                    prefix=f"dp{group_id}_allreduce_reduce_{participant}",
                    patterns=reverse_patterns,
                    direction="reverse",
                )
                broadcast_trees = build_dp_collective_trees(
                    topology,
                    root=participant,
                    receivers=receivers,
                    prefix=f"dp{group_id}_allreduce_broadcast_{participant}",
                    patterns=patterns,
                )
                participants.append(AllReduceParticipantSpec(int(participant), reduce_trees, broadcast_trees))
            tasks.append(AllReduceTaskSpec(len(tasks), float(data_volume) * float(group_size), tuple(participants), "allreduce"))
            continue
        for root in group:
            receivers = [node for node in group if node != root]
            if normalized_mode in {"broadcast", "broadcast_and_reduce"}:
                trees = build_dp_collective_trees(topology, root=root, receivers=receivers, prefix=f"dp{group_id}_broadcast_{root}", patterns=patterns)
                tasks.append(TaskSpec(len(tasks), float(data_volume), trees, "broadcast"))
            if normalized_mode in {"reduce", "broadcast_and_reduce"}:
                reduce_trees = build_dp_collective_trees(
                    topology,
                    root=root,
                    receivers=receivers,
                    prefix=f"dp{group_id}_reduce_{root}",
                    patterns=reverse_patterns,
                    direction="reverse",
                )
                tasks.append(TaskSpec(len(tasks), float(reduce_data_volume if reduce_data_volume is not None else data_volume), reduce_trees, "reduce"))
    return tasks


def _reject_root_specific_ahc(patterns: tuple[PatternSpec, ...], kind: str) -> None:
    if any(pattern.name == "ahc_2level" for pattern in patterns):
        raise ValueError(f"ahc_2level 只支持 allreduce/AHC baseline，不支持普通 {kind} 任务")


def build_pp_tasks(topology: Topology, *, pp_size: int, dp_size: int, data_volume: float) -> list[TaskSpec]:
    gpu_nodes = [int(node) for node in topology.gpu_nodes]
    total = len(gpu_nodes)
    resolved_pp = _positive_group_size(pp_size, "pp_size")
    resolved_dp = _positive_group_size(dp_size, "dp_size")
    if resolved_pp * resolved_dp != total:
        raise ValueError(f"pp_size * dp_size 必须等于 GPU 总数，收到 total={total}, pp_size={resolved_pp}, dp_size={resolved_dp}")
    tasks: list[TaskSpec] = []
    for pp_group_id in range(resolved_dp):
        pp_group = [gpu_nodes[pp_group_id + stage * resolved_dp] for stage in range(resolved_pp)]
        for stage in range(resolved_pp - 1):
            src = pp_group[stage]
            dst = pp_group[stage + 1]
            trees = build_pp_collective_trees(topology, source=src, target=dst, prefix=f"pp{pp_group_id}_p2p_{src}_{dst}")
            tasks.append(TaskSpec(len(tasks), float(data_volume), trees, "p2p"))
    return tasks


def _data_volume(params: dict[str, Any]) -> float:
    if "data_volume" in params:
        return float(params["data_volume"])
    if "min_data_volume" in params:
        return float(params["min_data_volume"])
    raise ValueError("任务参数必须包含 data_volume 或 min_data_volume")


def _hybrid_volume(params: dict[str, Any], key: str) -> float:
    if key in params:
        return float(params[key])
    if "data_volume" in params:
        return float(params["data_volume"])
    raise ValueError(f"hybrid_pp_dp 参数必须包含 {key} 或 data_volume")


def _require_gpu_node(gpu_node_set: set[int], value: Any, name: str) -> int:
    node = int(value)
    if node not in gpu_node_set:
        raise ValueError(f"{name} 必须是 GPU 节点，收到 {node}")
    return node


def _next_gpu_node(gpu_nodes: list[int], src: int) -> int:
    if len(gpu_nodes) < 2:
        raise ValueError("p2p 任务要求至少两个 GPU 节点")
    return gpu_nodes[(gpu_nodes.index(int(src)) + 1) % len(gpu_nodes)]


def _build_hybrid_pp_dp_tasks(
    topology: Topology,
    params: dict[str, Any],
    forward_patterns: tuple[PatternSpec, ...],
    reverse_patterns: tuple[PatternSpec, ...],
) -> tuple[BaseTaskSpec, ...]:
    gpu_nodes = [int(node) for node in topology.gpu_nodes]
    total = len(gpu_nodes)
    pp_size = int(params.get("pp_size", 4))
    if pp_size <= 0 or total % pp_size != 0:
        raise ValueError(f"pp_size 必须为正数并整除 GPU 总数，收到 total={total}, pp_size={pp_size}")
    dp_size = int(params.get("dp_size", total // pp_size))
    if dp_size != total // pp_size:
        raise ValueError(f"极简 hybrid_pp_dp 固定 dp_size=总GPU/pp_size，收到 {dp_size} vs {total // pp_size}")

    include_dp_reduce = bool(params.get("include_dp_reduce", False))
    dp_mode = str(params.get("dp_mode", "broadcast")).strip().lower()
    if include_dp_reduce and dp_mode == "broadcast":
        dp_mode = "broadcast_and_reduce"
    dp_volume = _hybrid_volume(params, "dp_data_volume")
    pp_volume = _hybrid_volume(params, "pp_data_volume")
    dp_reduce_volume = _hybrid_volume(params, "dp_reduce_data_volume") if "dp_reduce_data_volume" in params else dp_volume

    tasks = build_dp_tasks(
        topology,
        dp_size=dp_size,
        data_volume=dp_volume,
        patterns=forward_patterns,
        mode=dp_mode,
        include_reduce=include_dp_reduce,
        reduce_data_volume=dp_reduce_volume,
        reverse_patterns=reverse_patterns,
    ) + build_pp_tasks(topology, pp_size=pp_size, dp_size=dp_size, data_volume=pp_volume)
    return tuple(renumber_tasks(tasks))


def _build_deepep_tasks(topology: Topology, params: dict[str, Any], *, rng: np.random.Generator | None = None) -> tuple[BaseTaskSpec, ...]:
    return tuple(renumber_tasks(build_deepep_ep_tasks(topology, **params, rng=rng)))


def build_deepep_ep_tasks(
    topology: Topology,
    *,
    ep_group_size: int | None = None,
    data_volume: float = 1.0,
    direction: str = "forward",
    min_recv_ratio: float = 0.5,
    max_recv_ratio: float = 1.0,
    recv_ratio_by_local_rank: Any = None,
    mode: str = "narrow",
    rng: np.random.Generator | None = None,
) -> list[TaskSpec]:
    rng = np.random.default_rng() if rng is None else rng
    gpu_nodes = [int(node) for node in topology.gpu_nodes]
    group_size = _positive_group_size(len(gpu_nodes) if ep_group_size is None else ep_group_size, "ep_group_size")
    if len(gpu_nodes) % group_size != 0:
        raise ValueError(f"ep_group_size 必须整除 GPU 总数，收到 total={len(gpu_nodes)}, ep_group_size={group_size}")
    normalized_direction = str(direction).strip().lower()
    if normalized_direction not in {"forward", "reverse"}:
        raise ValueError(f"DeepEP direction 只支持 forward/reverse，收到 {normalized_direction}")
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"narrow", "wide"}:
        raise ValueError(f"DeepEP mode 只支持 narrow/wide，收到 {normalized_mode}")
    tasks: list[TaskSpec] = []
    ep_groups = [gpu_nodes[start : start + group_size] for start in range(0, len(gpu_nodes), group_size)]
    for ep_group in ep_groups:
        physical_groups = _physical_groups_for_nodes(topology, ep_group)
        for source_group in physical_groups:
            for source in source_group:
                for target_group in physical_groups:
                    receivers = _deepep_receivers(source=int(source), target_group=target_group)
                    if not receivers:
                        warnings.warn(
                            f"DeepEP source {int(source)} 的目标物理节点没有剩余 receiver，跳过该任务",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    ratios = _recv_ratio_array(
                        receivers=receivers,
                        raw=recv_ratio_by_local_rank,
                        min_ratio=float(min_recv_ratio),
                        max_ratio=float(max_recv_ratio),
                        rng=rng,
                    )
                    task_quotas = {int(source): 1.0}
                    for receiver, ratio in zip(receivers, ratios):
                        task_quotas[int(receiver)] = float(ratio)
                    trees = build_deepep_collective_trees(
                        topology,
                        source=int(source),
                        source_group=source_group,
                        target_group=target_group,
                        receivers=receivers,
                        direction=normalized_direction,
                        node_quotas=task_quotas,
                        prefix=f"deepep_{len(tasks)}",
                        mode=normalized_mode,
                    )
                    kind = "deepep-reverse" if normalized_direction == "reverse" else "deepep"
                    tasks.append(TaskSpec(len(tasks), float(data_volume), trees, kind, task_quotas))
    return tasks


def _positive_group_size(raw: Any, name: str) -> int:
    if isinstance(raw, bool):
        raise TypeError(f"{name} 必须是正整数，收到 {raw!r}")
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数，收到 {value}")
    return value


def _physical_groups_for_nodes(topology: Topology, nodes: list[int]) -> list[list[int]]:
    node_set = set(int(node) for node in nodes)
    groups = [
        [int(node) for node in group if int(node) in node_set]
        for group in topology.gpu_groups
    ]
    return [group for group in groups if group]


def _deepep_receivers(*, source: int, target_group: list[int]) -> list[int]:
    receivers = [int(node) for node in target_group]
    if int(source) in receivers:
        warnings.warn(
            f"DeepEP source {int(source)} 不能作为 receiver，已从目标物理节点中删除",
            RuntimeWarning,
            stacklevel=2,
        )
        receivers = [node for node in receivers if node != int(source)]
    return receivers


def _deepep_relays(
    *,
    source: int,
    source_group: list[int],
    target_group: list[int],
    receivers: list[int],
    mode: str,
) -> list[int]:
    if int(source) not in [int(node) for node in source_group]:
        raise ValueError(f"DeepEP source {source} 不属于 source_group")
    if not target_group:
        raise ValueError("DeepEP target_group 不能为空")
    local_rank = [int(node) for node in source_group].index(int(source))
    narrow_relay = int(target_group[local_rank % len(target_group)])
    if mode == "narrow":
        return [narrow_relay]
    relays = [int(node) for node in receivers]
    if narrow_relay not in relays:
        relays.append(narrow_relay)
    return relays


def _recv_ratio_array(
    *,
    receivers: list[int],
    raw: Any,
    min_ratio: float,
    max_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if min_ratio < 0.0 or min_ratio > 1.0 or max_ratio < 0.0 or max_ratio > 1.0:
        raise ValueError("DeepEP min_recv_ratio / max_recv_ratio 必须落在 [0, 1] 内")
    if max_ratio < min_ratio:
        raise ValueError("DeepEP max_recv_ratio 不能小于 min_recv_ratio")
    if raw is None:
        if min_ratio == max_ratio:
            return np.full(len(receivers), min_ratio, dtype=float)
        return rng.uniform(min_ratio, max_ratio, size=len(receivers))
    values = np.asarray(raw, dtype=float)
    if values.ndim != 1 or int(values.shape[0]) != len(receivers):
        raise ValueError("DeepEP recv_ratio_by_local_rank 长度必须与 receivers 一致")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("DeepEP recv_ratio_by_local_rank 必须落在 [0, 1] 内")
    return values.astype(float, copy=True)


def _deepep_tree(
    topology: Topology,
    *,
    source: int,
    relay: int,
    receivers: list[int],
    direction: str,
    node_quotas: dict[int, float],
    name: str,
) -> TreeSpec:
    normalized_source = int(source)
    normalized_relay = int(relay)
    receiver_set = {int(node) for node in receivers}
    if normalized_source in receiver_set:
        raise ValueError("DeepEP receivers 不能包含 source")
    nodes = sorted({normalized_source, normalized_relay, *receiver_set})
    parent: dict[int, int] = {}
    if normalized_source != normalized_relay:
        parent[normalized_relay] = normalized_source
    for receiver in sorted(receiver_set):
        if int(receiver) == normalized_relay:
            continue
        parent[int(receiver)] = normalized_relay
    return TreeSpec(
        root=normalized_source,
        parent=parent,
        topology=topology,
        direction=direction,
        name=name,
        node_quotas={int(node): float(node_quotas[int(node)]) for node in nodes},
    )
