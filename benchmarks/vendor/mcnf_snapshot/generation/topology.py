from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ..problem import Path, PooledShortestPathMap, Topology


CM_REAL_PLANES = 7
CM_REAL_HIGH_SWITCHES_PER_PLANE = 16
CM_REAL_GPU_LOW_LANES_PER_PLANE = 2


def create_topology(network_type: str, params: dict[str, Any] | None = None) -> Topology:
    params = dict(params or {})
    kind = str(network_type).lower()
    if kind in {"ub", "ubmesh"}:
        return _create_ubmesh_topology(params)
    if kind in {"ubd", "ubmeshd"}:
        return _create_ubmeshd_topology(params)
    if kind in {"clos"}:
        return _create_clos_topology(params)
    if kind == "closd":
        params.setdefault("low_high_bandwidth", 1.0)
        return _create_cm_topology(params)
    if kind in {"cm", "cloudmatrix"}:
        return _create_cm_topology(params)
    if kind in {"ds", "deepseek"}:
        return _create_ds_topology(params)
    raise ValueError(f"不支持的 network.type: {network_type}")


def _positive_int(params: dict[str, Any], name: str, default: int) -> int:
    value = int(params.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} 必须是正整数，收到 {value}")
    return value


def _positive_float(params: dict[str, Any], name: str, default: float) -> float:
    value = float(params.get(name, default))
    if value <= 0.0:
        raise ValueError(f"{name} 必须大于 0，收到 {value}")
    return value


def _topology_from_edges(
    num_nodes: int,
    edges: list[tuple[int, int]],
    bandwidth_by_edge: dict[tuple[int, int], float],
    gpu_groups: list[list[int]],
    shortest_paths: PooledShortestPathMap | None = None,
    physical_gpu_groups: list[list[int]] | None = None,
) -> Topology:
    bandwidths = [float(bandwidth_by_edge.get(edge, 1.0)) for edge in edges]
    gpu_nodes = tuple(int(node) for group in gpu_groups for node in group)
    return Topology.from_edges(
        int(num_nodes),
        tuple(edges),
        tuple(bandwidths),
        gpu_nodes=gpu_nodes,
        gpu_groups=tuple(tuple(group) for group in gpu_groups),
        physical_gpu_groups=tuple(tuple(group) for group in (physical_gpu_groups or gpu_groups)),
        shortest_paths=shortest_paths,
    )


def _assert_edge_count(edges: list[tuple[int, int]], expected: int) -> None:
    assert len(edges) == int(expected)
    assert len(set(edges)) == len(edges)


def _assert_gpu_groups(groups: list[list[int]], total_gpus: int, group_size: int | None = None, *, contiguous: bool = True) -> None:
    flattened = [int(gpu) for group in groups for gpu in group]
    if contiguous:
        assert sorted(flattened) == list(range(int(total_gpus)))
    else:
        assert all(0 <= gpu < int(total_gpus) for gpu in flattened)
    assert len(set(flattened)) == len(flattened)
    if group_size is not None:
        assert all(len(group) == int(group_size) for group in groups)


def _shortest_path_options(params: dict[str, Any]) -> dict[str, Any]:
    raw = params.get("shortest_paths", False)
    if isinstance(raw, bool):
        return {"mode": "exact_structured_precompute" if raw else "lazy"}
    if raw is None:
        return {"mode": "lazy"}
    if not isinstance(raw, dict):
        raise ValueError("shortest_paths 必须是 bool 或 dict")
    mode = str(raw.get("mode", "lazy"))
    if mode not in {"lazy", "exact_structured_precompute"}:
        raise ValueError(f"不支持的 shortest_paths.mode: {mode}")
    out = dict(raw)
    out["mode"] = mode
    return out


def _maybe_precompute_shortest_paths(
    params: dict[str, Any],
    *,
    num_nodes: int,
    edges: list[tuple[int, int]],
    gpu_nodes: tuple[int, ...],
) -> PooledShortestPathMap | None:
    options = _shortest_path_options(params)
    if options["mode"] == "lazy":
        return None
    scope = str(options.get("scope", "gpu_pairs"))
    if scope != "gpu_pairs":
        raise ValueError(f"不支持的 shortest_paths.scope: {scope}")
    workers = _resolve_workers(options.get("workers", 1), len(gpu_nodes))
    return _precompute_gpu_shortest_paths(int(num_nodes), tuple(edges), tuple(gpu_nodes), workers=workers)


def _resolve_workers(raw: Any, task_count: int) -> int:
    if str(raw).lower() == "auto":
        return max(1, min(int(task_count), os.cpu_count() or 1))
    workers = int(raw)
    if workers <= 0:
        raise ValueError(f"shortest_paths.workers 必须是正整数或 auto，收到 {raw}")
    return min(workers, max(int(task_count), 1))


def _precompute_gpu_shortest_paths(
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
    gpu_nodes: tuple[int, ...],
    *,
    workers: int = 1,
) -> PooledShortestPathMap:
    chunks = _source_chunks(gpu_nodes, int(workers))
    args = [(int(num_nodes), edges, gpu_nodes, chunk) for chunk in chunks]
    pair_paths: dict[tuple[int, int], tuple[Path, ...]] = {}
    if int(workers) == 1:
        for arg in args:
            pair_paths.update(_shortest_paths_for_sources(arg))
    else:
        try:
            with ProcessPoolExecutor(max_workers=int(workers)) as pool:
                for part in pool.map(_shortest_paths_for_sources, args):
                    pair_paths.update(part)
        except PermissionError:
            for arg in args:
                pair_paths.update(_shortest_paths_for_sources(arg))
    return PooledShortestPathMap.from_pair_paths(pair_paths)


def _source_chunks(values: tuple[int, ...], workers: int) -> tuple[tuple[int, ...], ...]:
    count = max(1, min(int(workers), len(values)))
    return tuple(tuple(values[idx::count]) for idx in range(count) if values[idx::count])


def _shortest_paths_for_sources(args: tuple[int, tuple[tuple[int, int], ...], tuple[int, ...], tuple[int, ...]]) -> dict[tuple[int, int], tuple[Path, ...]]:
    num_nodes, edges, gpu_nodes, sources = args
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(int(num_nodes))]
    for edge_i, (src, dst) in enumerate(edges):
        adjacency[int(src)].append((int(dst), int(edge_i)))
    out: dict[tuple[int, int], tuple[Path, ...]] = {}
    for src in sources:
        dist, predecessors = _bfs_predecessors(int(src), adjacency)
        memo: dict[int, tuple[Path, ...]] = {}
        for dst in gpu_nodes:
            out[(int(src), int(dst))] = _edge_paths_from_predecessors(int(src), int(dst), dist, predecessors, memo)
    return out


def _bfs_predecessors(src: int, adjacency: list[list[tuple[int, int]]]) -> tuple[list[int], list[list[tuple[int, int]]]]:
    dist = [-1] * len(adjacency)
    predecessors: list[list[tuple[int, int]]] = [[] for _ in adjacency]
    queue = [int(src)]
    dist[int(src)] = 0
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        next_dist = dist[node] + 1
        for nxt, edge_i in adjacency[node]:
            if dist[nxt] < 0:
                dist[nxt] = next_dist
                predecessors[nxt].append((node, edge_i))
                queue.append(nxt)
            elif dist[nxt] == next_dist:
                predecessors[nxt].append((node, edge_i))
    return dist, predecessors


def _edge_paths_from_predecessors(
    src: int,
    dst: int,
    dist: list[int],
    predecessors: list[list[tuple[int, int]]],
    memo: dict[int, tuple[Path, ...]] | None = None,
) -> tuple[Path, ...]:
    if int(src) == int(dst):
        return ((),)
    if dist[int(dst)] < 0:
        return ()
    if memo is None:
        memo = {}

    def walk(node: int) -> tuple[Path, ...]:
        cached = memo.get(int(node))
        if cached is not None:
            return cached
        if int(node) == int(src):
            memo[int(node)] = ((),)
            return memo[int(node)]
        paths: list[Path] = []
        for prev, edge_i in predecessors[int(node)]:
            paths.extend(prefix + (int(edge_i),) for prefix in walk(int(prev)))
        memo[int(node)] = tuple(paths)
        return memo[int(node)]

    return walk(int(dst))


def _create_clos_topology(params: dict[str, Any]) -> Topology:
    n_gpus = _positive_int(params, "num_gpus", 16)
    n_switches = _positive_int(params, "num_switches", 4)
    bandwidth = _positive_float(params, "bandwidth", 1.0)
    total_nodes = n_gpus + n_switches
    edges = []
    for gpu in range(n_gpus):
        for switch in range(n_gpus, total_nodes):
            edges.append((gpu, switch))
            edges.append((switch, gpu))
    groups = [[gpu] for gpu in range(n_gpus)]
    _assert_gpu_groups(groups, n_gpus, group_size=1)
    _assert_edge_count(edges, 2 * n_gpus * n_switches)
    shortest_paths = _maybe_precompute_shortest_paths(
        params,
        num_nodes=total_nodes,
        edges=edges,
        gpu_nodes=tuple(gpu for group in groups for gpu in group),
    )
    return _topology_from_edges(total_nodes, edges, {edge: bandwidth for edge in edges}, groups, shortest_paths)


def _create_ubmesh_topology(params: dict[str, Any]) -> Topology:
    n_nodes = _positive_int(params, "n_nodes", 8)
    node_x = _positive_int(params, "node_x", 4)
    node_y = _positive_int(params, "node_y", 4)
    n_switches = _positive_int(params, "n_switches", 1)
    intra_bw = _positive_float(params, "bandwidth_intra_pod", 1.0)
    super_bw = _positive_float(params, "bandwidth_superpod", 2.0)
    racks_per_pod = node_x * node_y
    total_gpus = n_nodes * racks_per_pod
    total_nodes = total_gpus + n_switches
    switches = list(range(total_gpus, total_nodes))
    per_switch_super_bw = super_bw / float(n_switches)
    if "gpu_groups" in params:
        raw_groups = params["gpu_groups"]
        if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
            raise ValueError("gpu_groups 必须是非空列表")
        physical_groups = [[int(gpu) for gpu in group] for group in raw_groups]
        if any(not group for group in physical_groups):
            raise ValueError("gpu_groups 不能包含空组")
    else:
        physical_groups = [list(range(pod * racks_per_pod, (pod + 1) * racks_per_pod)) for pod in range(n_nodes)]
    logical_groups = _logical_gpu_groups(params, physical_groups)
    _assert_gpu_groups(physical_groups, total_gpus, contiguous="gpu_groups" not in params)
    _assert_gpu_groups(logical_groups, total_gpus, contiguous="logical_gpu_groups" not in params and "gpu_groups" not in params)
    if {gpu for group in physical_groups for gpu in group} != {gpu for group in logical_groups for gpu in group}:
        raise ValueError("gpu_groups 与 logical_gpu_groups 必须包含相同 GPU")
    active_gpus = {gpu for group in physical_groups for gpu in group}

    def rack_id(pod: int, x_idx: int, y_idx: int) -> int:
        return pod * racks_per_pod + x_idx * node_y + y_idx

    edges: list[tuple[int, int]] = []
    bandwidths: dict[tuple[int, int], float] = {}
    expected_intra = 0
    for pod in range(n_nodes):
        for x_idx in range(node_x):
            row = [gpu for y_idx in range(node_y) if (gpu := rack_id(pod, x_idx, y_idx)) in active_gpus]
            expected_intra += len(row) * (len(row) - 1)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    for edge in ((row[i], row[j]), (row[j], row[i])):
                        edges.append(edge)
                        bandwidths[edge] = intra_bw
        for y_idx in range(node_y):
            col = [gpu for x_idx in range(node_x) if (gpu := rack_id(pod, x_idx, y_idx)) in active_gpus]
            expected_intra += len(col) * (len(col) - 1)
            for i in range(len(col)):
                for j in range(i + 1, len(col)):
                    for edge in ((col[i], col[j]), (col[j], col[i])):
                        edges.append(edge)
                        bandwidths[edge] = intra_bw
    for gpu in sorted(active_gpus):
        for switch in switches:
            for edge in ((gpu, switch), (switch, gpu)):
                edges.append(edge)
                bandwidths[edge] = per_switch_super_bw
    expected_super = 2 * len(active_gpus) * n_switches
    _assert_edge_count(edges, expected_intra + expected_super)
    shortest_paths = _maybe_precompute_shortest_paths(
        params,
        num_nodes=total_nodes,
        edges=edges,
        gpu_nodes=tuple(gpu for group in logical_groups for gpu in group),
    )
    return _topology_from_edges(
        total_nodes,
        edges,
        bandwidths,
        logical_groups,
        shortest_paths,
        physical_gpu_groups=physical_groups,
    )


def _create_ubmeshd_topology(params: dict[str, Any]) -> Topology:
    rack_x = _positive_int(params, "rack_x", 8)
    rack_y = _positive_int(params, "rack_y", 8)
    pod_x = _positive_int(params, "pod_x", 4)
    pod_y = _positive_int(params, "pod_y", 4)
    n_pod = _positive_int(params, "n_pod", 1)
    bandwidth_inrack = _positive_float(params, "bandwidth_inrack", 1.0)
    bandwidth_uprack = _positive_float(params, "bandwidth_uprack", 4.0)
    bandwidth_backup = _positive_float(params, "bandwidth_backup", bandwidth_uprack)
    bandwidth_inpod = _positive_float(params, "bandwidth_inpod", 32.0)
    bandwidth_uppod = _positive_float(params, "bandwidth_uppod", 64.0)
    gpus_per_rack = rack_x * rack_y
    racks_per_pod = pod_x * pod_y
    base_racks = racks_per_pod * n_pod
    extra_gpus = params.get("extra_gpus", 0)
    if isinstance(extra_gpus, bool) or not isinstance(extra_gpus, int) or extra_gpus < 0:
        raise ValueError("extra_gpus 必须是非负整数")
    total_racks = base_racks + (extra_gpus + gpus_per_rack - 1) // gpus_per_rack
    total_gpus = gpus_per_rack * base_racks + extra_gpus
    lrs_offset = gpus_per_rack * total_racks
    hrs = lrs_offset + total_racks
    total_nodes = hrs + 1

    if "failed_racks" in params:
        raise ValueError("failed_racks 已废弃；请分别指定 failed_gpus 和 use_rack_backup")

    def node_ids(name: str, limit: int) -> set[int]:
        raw = params.get(name, [])
        if not isinstance(raw, (list, tuple)) or any(isinstance(node, bool) or not isinstance(node, int) for node in raw):
            raise ValueError(f"{name} 必须是整数编号列表")
        if len(set(raw)) != len(raw):
            raise ValueError(f"{name} 不能包含重复编号")
        if any(node < 0 or node >= limit for node in raw):
            raise ValueError(f"{name} 必须位于 [0, {limit})")
        return set(raw)

    failed_gpus = node_ids("failed_gpus", total_gpus)
    backup_racks = node_ids("use_rack_backup", total_racks)
    # 备用卡编号放在 HRS 后，按 rack 保留槽位，不随启用顺序或坏卡变化。
    backup_offset = hrs + 1
    if backup_racks:
        total_nodes = backup_offset + max(backup_racks) + 1
    groups = [
        [gpu for gpu in range(rack * gpus_per_rack, (rack + 1) * gpus_per_rack) if gpu < total_gpus and gpu not in failed_gpus]
        + ([backup_offset + rack] if rack in backup_racks else [])
        for rack in range(total_racks)
    ]
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("UBD 必须保留至少一张活动 GPU")
    logical_groups = _logical_gpu_groups(params, groups)
    logical_gpus = [gpu for group in logical_groups for gpu in group]
    physical_gpus = {gpu for group in groups for gpu in group}
    if set(logical_gpus) != physical_gpus or len(logical_gpus) != len(physical_gpus):
        raise ValueError("logical_gpu_groups 必须恰好包含全部活动 GPU，且不能重复")
    edges: list[tuple[int, int]] = []
    bandwidths: dict[tuple[int, int], float] = {}

    def connect(left: int, right: int, bandwidth: float) -> None:
        for edge in ((left, right), (right, left)):
            edges.append(edge)
            bandwidths[edge] = bandwidth

    def connect_clique(nodes: list[int], bandwidth: float) -> None:
        for left_i, left in enumerate(nodes):
            for right in nodes[left_i + 1 :]:
                connect(left, right, bandwidth)

    for rack in range(total_racks):
        base = rack * gpus_per_rack
        for rack_y_i in range(rack_y):
            connect_clique(
                [
                    base + rack_x_i + rack_x * rack_y_i
                    for rack_x_i in range(rack_x)
                    if base + rack_x_i + rack_x * rack_y_i < total_gpus
                    and base + rack_x_i + rack_x * rack_y_i not in failed_gpus
                ],
                bandwidth_inrack,
            )
        for rack_x_i in range(rack_x):
            connect_clique(
                [
                    base + rack_x_i + rack_x * rack_y_i
                    for rack_y_i in range(rack_y)
                    if base + rack_x_i + rack_x * rack_y_i < total_gpus
                    and base + rack_x_i + rack_x * rack_y_i not in failed_gpus
                ],
                bandwidth_inrack,
            )
        lrs = lrs_offset + rack
        for gpu in range(base, base + gpus_per_rack):
            if gpu < total_gpus and gpu not in failed_gpus:
                connect(gpu, lrs, bandwidth_uprack)
        if rack in backup_racks:
            connect(backup_offset + rack, lrs, bandwidth_backup)

    def lrs_id(pod: int, pod_x_i: int, pod_y_i: int) -> int:
        rack = pod_x_i + pod_x * (pod_y_i + pod_y * pod)
        return lrs_offset + rack

    for pod in range(n_pod):
        pod_racks = racks_per_pod + (total_racks - base_racks if pod == n_pod - 1 else 0)
        rows = (pod_racks + pod_x - 1) // pod_x
        for pod_y_i in range(rows):
            connect_clique([
                lrs_id(pod, x, pod_y_i) for x in range(pod_x)
                if pod_y_i * pod_x + x < pod_racks
            ], bandwidth_inpod)
        for pod_x_i in range(pod_x):
            connect_clique([
                lrs_id(pod, pod_x_i, y) for y in range(rows)
                if y * pod_x + pod_x_i < pod_racks
            ], bandwidth_inpod)
    for rack in range(total_racks):
        connect(lrs_offset + rack, hrs, bandwidth_uppod)

    active_gpus = [gpu for gpu in range(total_gpus) if gpu not in failed_gpus]
    rack_rows = Counter((gpu // gpus_per_rack, gpu % gpus_per_rack // rack_x) for gpu in active_gpus)
    rack_columns = Counter((gpu // gpus_per_rack, gpu % rack_x) for gpu in active_gpus)
    pod_rows = Counter()
    pod_columns = Counter()
    for rack in range(total_racks):
        pod = min(rack // racks_per_pod, n_pod - 1)
        local = rack - pod * racks_per_pod
        pod_rows[(pod, local // pod_x)] += 1
        pod_columns[(pod, local % pod_x)] += 1
    expected_mesh_edges = sum(
        count * (count - 1)
        for counts in (rack_rows, rack_columns, pod_rows, pod_columns) for count in counts.values()
    )
    _assert_edge_count(edges, expected_mesh_edges + 2 * (len(active_gpus) + len(backup_racks) + total_racks))
    shortest_paths = _maybe_precompute_shortest_paths(
        params,
        num_nodes=total_nodes,
        edges=edges,
        gpu_nodes=tuple(gpu for group in groups for gpu in group),
    )
    return _topology_from_edges(total_nodes, edges, bandwidths, logical_groups, shortest_paths, groups)


def _create_cm_topology(params: dict[str, Any]) -> Topology:
    groups = _cm_gpu_groups(params)
    logical_groups = _logical_gpu_groups(params, groups)
    n_groups = len(groups)
    n_planes = _positive_int(params, "n_planes", 1)
    high_per_plane = _positive_int(params, "high_switches_per_plane", 1)
    if n_planes > CM_REAL_PLANES:
        raise ValueError(f"n_planes 不能超过 {CM_REAL_PLANES}")
    if CM_REAL_HIGH_SWITCHES_PER_PLANE % high_per_plane != 0:
        raise ValueError("high_switches_per_plane 必须整除 16")

    total_gpus = _cm_gpu_slot_count(groups, params, logical_groups)
    active_gpus = sum(len(group) for group in groups)
    total_low = n_groups * n_planes
    total_high = n_planes * high_per_plane
    scale = float(CM_REAL_GPU_LOW_LANES_PER_PLANE * CM_REAL_PLANES)
    gpu_low_bw = float(CM_REAL_GPU_LOW_LANES_PER_PLANE) * float(CM_REAL_PLANES) / float(n_planes) / scale
    low_high_bw = float(CM_REAL_HIGH_SWITCHES_PER_PLANE) / float(high_per_plane) * float(CM_REAL_PLANES) / float(n_planes) / scale
    gpu_low_bw = float(params.get("gpu_low_bandwidth", gpu_low_bw))
    low_high_bw = float(params.get("low_high_bandwidth", low_high_bw))
    if gpu_low_bw <= 0.0 or low_high_bw <= 0.0:
        raise ValueError("CM 带宽必须大于 0")
    low_failures = _cm_low_switch_failures(params, n_groups)
    high_failures = int(params.get("high_switch_failures", 0))
    if high_failures < 0 or high_failures >= CM_REAL_HIGH_SWITCHES_PER_PLANE:
        raise ValueError(f"high_switch_failures 必须在 [0, {CM_REAL_HIGH_SWITCHES_PER_PLANE}) 内")
    high_failure_scale = float(CM_REAL_HIGH_SWITCHES_PER_PLANE - high_failures) / float(CM_REAL_HIGH_SWITCHES_PER_PLANE)
    scale_low_high_by_group_size = bool(params.get("scale_low_high_by_group_size", False))
    nominal_group_size = _positive_int(params, "gpus_per_node", 8)
    edges: list[tuple[int, int]] = []
    bandwidths: dict[tuple[int, int], float] = {}

    def low_id(group: int, plane: int) -> int:
        return total_gpus + plane * n_groups + group

    def high_id(plane: int, high_idx: int) -> int:
        return total_gpus + total_low + plane * high_per_plane + high_idx

    for group in range(n_groups):
        group_gpus = groups[group]
        gpu_low_scale = 6.0 / 7.0 if group in low_failures else 1.0
        for plane in range(n_planes):
            low = low_id(group, plane)
            for gpu in group_gpus:
                for edge in ((gpu, low), (low, gpu)):
                    edges.append(edge)
                    bandwidths[edge] = gpu_low_bw * gpu_low_scale
    for plane in range(n_planes):
        for group in range(n_groups):
            low = low_id(group, plane)
            group_scale = float(len(groups[group])) / float(nominal_group_size) if scale_low_high_by_group_size else 1.0
            low_scale = 6.0 / 7.0 if group in low_failures else 1.0
            for high_idx in range(high_per_plane):
                high = high_id(plane, high_idx)
                for edge in ((low, high), (high, low)):
                    edges.append(edge)
                    bandwidths[edge] = low_high_bw * group_scale * low_scale * high_failure_scale
    group_sizes = [len(group) for group in groups]
    _assert_gpu_groups(
        groups,
        total_gpus,
        group_size=group_sizes[0] if "gpu_groups" not in params and len(set(group_sizes)) == 1 else None,
        contiguous="gpu_groups" not in params,
    )
    _assert_edge_count(edges, 2 * active_gpus * n_planes + 2 * total_low * high_per_plane)
    shortest_paths = _maybe_precompute_shortest_paths(
        params,
        num_nodes=total_gpus + total_low + total_high,
        edges=edges,
        gpu_nodes=tuple(gpu for group in logical_groups for gpu in group),
    )
    return _topology_from_edges(total_gpus + total_low + total_high, edges, bandwidths, logical_groups, shortest_paths, physical_gpu_groups=groups)


def _cm_gpu_groups(params: dict[str, Any]) -> list[list[int]]:
    if "gpu_groups" in params:
        raw_groups = params["gpu_groups"]
        if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
            raise ValueError("gpu_groups 必须是非空列表")
        groups = [[int(gpu) for gpu in group] for group in raw_groups]
        if any(not group for group in groups):
            raise ValueError("gpu_groups 不能包含空组")
        return groups
    if "backup_failures" in params:
        return _cm_backup_groups(params)
    raw = params.get("group_sizes")
    if raw is None:
        size = _positive_int(params, "gpus_per_node", 8)
        return [list(range(group * size, (group + 1) * size)) for group in range(_positive_int(params, "n_groups", 48))]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("group_sizes 必须是非空列表")
    sizes = [int(value) for value in raw]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"group_sizes 必须都是正整数，收到 {raw}")
    if "n_groups" in params and int(params["n_groups"]) != len(sizes):
        raise ValueError(f"n_groups 必须等于 group_sizes 长度，收到 {params['n_groups']} vs {len(sizes)}")
    groups: list[list[int]] = []
    cursor = 0
    for size in sizes:
        groups.append(list(range(cursor, cursor + size)))
        cursor += size
    return groups


def _cm_backup_groups(params: dict[str, Any]) -> list[list[int]]:
    n_groups = _positive_int(params, "n_groups", 48)
    group_size = _positive_int(params, "gpus_per_node", 8)
    raw = params["backup_failures"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("backup_failures 必须是非空列表")
    failures = [int(value) for value in raw]
    if len(failures) > n_groups:
        raise ValueError("backup_failures 长度不能超过 n_groups")
    if any(value < 0 or value >= group_size for value in failures):
        raise ValueError(f"backup_failures 每项必须在 [0, {group_size}) 内，收到 {raw}")
    groups = [list(range(group * group_size, (group + 1) * group_size)) for group in range(n_groups)]
    backup: list[int] = []
    start = n_groups - len(failures)
    for offset, failed in enumerate(failures):
        if failed == 0:
            continue
        group = groups[start + offset]
        backup.extend(group[-failed:])
        del group[-failed:]
    if backup:
        groups.append(backup)
    return groups


def _logical_gpu_groups(params: dict[str, Any], physical_groups: list[list[int]]) -> list[list[int]]:
    if "logical_gpu_groups" not in params:
        return physical_groups
    raw_groups = params["logical_gpu_groups"]
    if not isinstance(raw_groups, (list, tuple)) or not raw_groups:
        raise ValueError("logical_gpu_groups 必须是非空列表")
    groups = [[int(gpu) for gpu in group] for group in raw_groups]
    if any(not group for group in groups):
        raise ValueError("logical_gpu_groups 不能包含空组")
    return groups


def _cm_gpu_slot_count(groups: list[list[int]], params: dict[str, Any], logical_groups: list[list[int]]) -> int:
    if "gpu_groups" not in params and "logical_gpu_groups" not in params:
        return sum(len(group) for group in groups)
    # 自定义 GPU 组保留 GPU 编号作为图节点编号；中间空洞是已失效或未使用槽位，不压缩重映射。
    return max(max(group) for group in groups + logical_groups) + 1


def _cm_low_switch_failures(params: dict[str, Any], n_groups: int) -> set[int]:
    raw = params.get("low_switch_failures", ())
    if raw is None:
        return set()
    if isinstance(raw, int):
        raw = (raw,)
    if not isinstance(raw, (list, tuple, set)):
        raise ValueError("low_switch_failures 必须是节点下标列表")
    out = {int(value) for value in raw}
    if any(value < 0 or value >= int(n_groups) for value in out):
        raise ValueError(f"low_switch_failures 必须落在 [0, {int(n_groups)}) 内")
    return out


def _create_ds_topology(params: dict[str, Any]) -> Topology:
    n_nodes = _positive_int(params, "n_nodes", 16)
    gpus_per_node = _positive_int(params, "gpus_per_node", 8)
    switches_per_plane = _positive_int(params, "switches_per_plane", 2)
    if n_nodes % switches_per_plane != 0:
        raise ValueError(f"switches_per_plane 必须整除 n_nodes，收到 n_nodes={n_nodes}, switches_per_plane={switches_per_plane}")
    total_gpus = n_nodes * gpus_per_node
    n_planes = gpus_per_node
    node_switch_offset = total_gpus
    low_switch_offset = total_gpus + n_nodes
    switches_in_plane = n_nodes // switches_per_plane
    total_low = n_planes * switches_in_plane
    high_switch_offset = total_gpus + n_nodes + total_low
    total_nodes = high_switch_offset + n_planes
    edges: list[tuple[int, int]] = []
    bandwidths: dict[tuple[int, int], float] = {}

    for node_id in range(n_nodes):
        node_switch = node_switch_offset + node_id
        for gpu in range(node_id * gpus_per_node, (node_id + 1) * gpus_per_node):
            for edge in ((gpu, node_switch), (node_switch, gpu)):
                edges.append(edge)
                bandwidths[edge] = 160.0
    for plane in range(n_planes):
        plane_gpus = [node_id * gpus_per_node + plane for node_id in range(n_nodes)]
        for i in range(0, len(plane_gpus), switches_per_plane):
            low = low_switch_offset + plane * switches_in_plane + i // switches_per_plane
            for gpu in plane_gpus[i : i + switches_per_plane]:
                for edge in ((gpu, low), (low, gpu)):
                    edges.append(edge)
                    bandwidths[edge] = 50.0
    for plane in range(n_planes):
        high = high_switch_offset + plane
        for i in range(switches_in_plane):
            low = low_switch_offset + plane * switches_in_plane + i
            for edge in ((low, high), (high, low)):
                edges.append(edge)
                bandwidths[edge] = 50.0
    groups = [list(range(node * gpus_per_node, (node + 1) * gpus_per_node)) for node in range(n_nodes)]
    _assert_gpu_groups(groups, total_gpus, group_size=gpus_per_node)
    _assert_edge_count(edges, 2 * total_gpus + 2 * total_gpus + 2 * total_low)
    shortest_paths = _maybe_precompute_shortest_paths(
        params,
        num_nodes=total_nodes,
        edges=edges,
        gpu_nodes=tuple(gpu for group in groups for gpu in group),
    )
    return _topology_from_edges(total_nodes, edges, bandwidths, groups, shortest_paths)
