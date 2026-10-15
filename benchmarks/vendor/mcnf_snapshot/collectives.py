from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import ceil
from typing import Any, Literal, Sequence, overload

from .problem import Topology, TreeSpec


@dataclass(frozen=True)
class PatternSpec:
    name: str
    params: dict[str, Any]


DEFAULT_PATTERNS = (
    PatternSpec("single_layer", {}),
    PatternSpec("tree_tree", {"method": "node"}),
)

NO_RING_PATTERNS = DEFAULT_PATTERNS + (
    PatternSpec("knomial", {"radix": 4}),
    PatternSpec("hd", {}),
    PatternSpec("tree_ring", {"method": "node"}),
)

HWC_PATTERNS = NO_RING_PATTERNS + (
    PatternSpec("ahc_2level", {"grouping": "physical"}),
    PatternSpec("nhr", {}),
)


def normalize_comm_patterns(comm: dict[str, Any] | None, *, direction: str = "forward") -> tuple[PatternSpec, ...]:
    if comm is None:
        return (PatternSpec("single_layer", {}),)
    if any(key in comm for key in ("patterns", "reduce_patterns", "broadcast_patterns")):
        key = "reduce_patterns" if direction == "reverse" else "broadcast_patterns"
        raw = comm.get(key) or comm.get("patterns") or []
        return tuple(PatternSpec(str(item["name"]), dict(item.get("params", {}))) for item in raw)
    name = str(comm.get("name", "single_layer"))
    aliases: dict[str, tuple[PatternSpec, ...]] = {
        "single_layer": (PatternSpec("single_layer", {}),),
        "standard": DEFAULT_PATTERNS,
        "single_plus_nht": (PatternSpec("single_layer", {}), PatternSpec("tree_tree", {"method": "NHT"})),
        "nht_only": (PatternSpec("tree_tree", {"method": "NHT"}),),
        "tree_tree": (PatternSpec("tree_tree", {"method": "node"}),),
        "hd": (PatternSpec("hd", {}),),
        "nhr": (PatternSpec("nhr", {}),),
        "tree_ring": (PatternSpec("tree_ring", {"method": "node"}),),
        "ring_ring": (PatternSpec("ring_ring", {"method": "node"}),),
        "standard_plus_hd": DEFAULT_PATTERNS + (PatternSpec("hd", {}),),
        "no_ring": NO_RING_PATTERNS,
        "hwc": HWC_PATTERNS,
        "all_tree": DEFAULT_PATTERNS + (PatternSpec("ring_ring", {"method": "node"}),),
        "ring": (PatternSpec("ring", {}),),
        "ed_binary_pack": (PatternSpec("ed_binary_pack", {}),),
        "ed_ring_pack": (PatternSpec("ed_ring_pack", {}),),
        "ahc_2level": (PatternSpec("ahc_2level", {}),),
    }
    if name not in aliases:
        raise ValueError(f"不支持的 comm: {name}")
    patterns = aliases[name]
    if comm.get("params"):
        if len(patterns) != 1:
            raise ValueError("多模式 comm 请在 patterns 中分别设置 params")
        params = dict(patterns[0].params)
        params.update(dict(comm.get("params", {})))
        return (PatternSpec(patterns[0].name, params),)
    return patterns


def single_layer_tree(
    topology: Topology,
    *,
    root: int,
    receivers: list[int],
    name: str,
    direction: str = "forward",
) -> TreeSpec:
    nodes = _nodes(root, receivers)
    edges = tuple((int(root), int(node)) for node in nodes if int(node) != int(root))
    return _tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=name)


@overload
def build_ahc_2level_trees(
    topology: Topology,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    grouping: str = "physical",
    prefix: str = "ahc_2level",
    direction: str = "forward",
    return_weights: Literal[False] = False,
) -> tuple[TreeSpec, ...]: ...


@overload
def build_ahc_2level_trees(
    topology: Topology,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    grouping: str = "physical",
    prefix: str = "ahc_2level",
    direction: str = "forward",
    return_weights: Literal[True],
) -> tuple[tuple[TreeSpec, ...], tuple[float, ...]]: ...


def build_ahc_2level_trees(
    topology: Topology,
    *,
    groups: Sequence[Sequence[int]] | None = None,
    grouping: str = "physical",
    prefix: str = "ahc_2level",
    direction: str = "forward",
    return_weights: bool = False,
) -> tuple[TreeSpec, ...] | tuple[tuple[TreeSpec, ...], tuple[float, ...]]:
    normalized_groups = _normalize_ahc_groups(topology, ahc_2level_groups(topology, grouping=grouping) if groups is None else groups)
    pairs = _ahc_2level_tree_weight_pairs(topology, normalized_groups, root=None, prefix=prefix, direction=direction)
    out = tuple(tree for tree, _weight in pairs)
    if return_weights:
        return out, tuple(weight for _tree, weight in pairs)
    return out


def _ahc_2level_trees_for_root(
    topology: Topology,
    *,
    groups: Sequence[Sequence[int]],
    root: int,
    prefix: str,
    direction: str,
) -> tuple[TreeSpec, ...]:
    normalized_groups = _normalize_ahc_groups(topology, groups)
    pairs = _ahc_2level_tree_weight_pairs(topology, normalized_groups, root=int(root), prefix=prefix, direction=direction)
    return tuple(tree for tree, _weight in pairs)


def _ahc_2level_tree_weight_pairs(
    topology: Topology,
    groups: Sequence[Sequence[int]],
    *,
    root: int | None,
    prefix: str,
    direction: str,
) -> tuple[tuple[TreeSpec, float], ...]:
    nodes = tuple(node for group in groups for node in group)
    selected_root = None if root is None else int(root)
    if selected_root is not None and selected_root not in nodes:
        raise ValueError(f"root {selected_root} 不在 ahc_2level 节点组中")
    bounds = _ahc_bounds(groups)
    pairs: list[tuple[TreeSpec, float]] = []
    for idx, (left, right) in enumerate(zip(bounds, bounds[1:])):
        owners = _ahc_interval_owners(groups, right)
        weight = float(right - left) / float(len(groups))
        if selected_root is None:
            roots = owners
        elif selected_root in owners:
            roots = (selected_root,)
        else:
            continue
        for tree_root in roots:
            edges = _ahc_2level_edges(groups, owners, root=int(tree_root))
            name = f"{prefix}_interval{idx}_root{int(tree_root)}"
            pairs.append((_tree_from_edges(topology, root=int(tree_root), nodes=nodes, direction=direction, edges=edges, name=name), weight))
    return tuple(pairs)


def build_collective_trees(
    topology: Topology,
    *,
    root: int,
    receivers: list[int],
    prefix: str,
    patterns: Sequence[PatternSpec],
    direction: str = "forward",
) -> tuple[TreeSpec, ...]:
    nodes = _nodes(root, receivers)
    trees: list[TreeSpec] = []
    for pattern in patterns:
        name = str(pattern.name)
        params = dict(pattern.params)
        if name == "single_layer":
            trees.append(single_layer_tree(topology, root=root, receivers=receivers, direction=direction, name=f"{prefix}_single_layer"))
        elif name == "tree_tree":
            groups = _tree_tree_groups(topology, nodes=nodes, root=int(root), method=params.get("method"))
            edges = _tree_tree_edges(groups, root=int(root))
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_tree_tree"))
        elif name == "tree_ring":
            groups = _tree_ring_groups(topology, nodes=nodes, root=int(root), method=params.get("method"))
            for local_rev in (False, True):
                edges = _tree_ring_edges(groups, root=int(root), intra_reverse=local_rev)
                trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_tree_ring_local{int(local_rev)}"))
        elif name in {"knomial", "hd"}:
            radix = 2 if name == "hd" else int(params.get("radix", 2))
            edges = _knomial_edges(topology, root=int(root), nodes=nodes, radix=radix)
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_{name}"))
        elif name == "nhr":
            edges = _nhr_edges(root=int(root), nodes=nodes)
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_nhr"))
        elif name == "ring":
            ordered = _ordered_ring_nodes(root=int(root), nodes=nodes)
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=_chain(ordered), name=f"{prefix}_ring"))
        elif name == "ring_ring":
            groups = _ring_groups(topology, nodes=nodes, root=int(root), method=params.get("method"))
            for top_rev, local_rev in ((False, False), (True, False), (False, True), (True, True)):
                edges = _ring_ring_edges(groups, root=int(root), inter_reverse=top_rev, intra_reverse=local_rev)
                variant = f"top{int(top_rev)}_local{int(local_rev)}"
                trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_ring_ring_{variant}"))
        elif name == "ed_binary_pack":
            edges = _root_rank_edges(_difference_binary_edges(len(nodes)), nodes=nodes, root=int(root))
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_ed_binary_pack"))
        elif name == "ed_ring_pack":
            edges = _shift_rank_edges(_difference_path_edges(len(nodes)), nodes=nodes, root=int(root))
            trees.append(_tree_from_edges(topology, root=int(root), nodes=nodes, direction=direction, edges=edges, name=f"{prefix}_ed_ring_pack"))
        elif name == "ahc_2level":
            grouping = str(params.get("grouping", "physical"))
            groups = _groups_in_nodes(ahc_2level_groups(topology, grouping=grouping), nodes)
            trees.extend(
                _ahc_2level_trees_for_root(
                    topology,
                    groups=groups,
                    root=int(root),
                    prefix=f"{prefix}_ahc_2level_{grouping}",
                    direction=direction,
                )
            )
        elif name == "p2p":
            if len(nodes) != 2:
                raise ValueError("p2p 模式要求恰好两个节点")
            trees.append(single_layer_tree(topology, root=root, receivers=receivers, direction=direction, name=f"{prefix}_p2p"))
        else:
            raise ValueError(f"不支持的集合通信模式: {name}")
    return tuple(trees)


def _nodes(root: int, receivers: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted({int(root), *[int(node) for node in receivers]}))


def _tree_from_edges(
    topology: Topology,
    *,
    root: int,
    nodes: Sequence[int],
    direction: str,
    edges: Sequence[tuple[int, int]],
    name: str,
) -> TreeSpec:
    normalized_direction = str(direction)
    if normalized_direction not in {"forward", "reverse"}:
        raise ValueError(f"不支持的树方向: {direction}")
    node_set = set(int(node) for node in nodes)
    parent: dict[int, int] = {}
    for src, dst in edges:
        src = int(src)
        dst = int(dst)
        if src not in node_set or dst not in node_set:
            raise ValueError("集合通信树边包含未声明节点")
        if dst == int(root):
            continue
        if dst in parent and parent[dst] != src:
            raise ValueError(f"节点 {dst} 存在多个父节点")
        parent[dst] = src
    missing = sorted(node for node in node_set if node != int(root) and node not in parent)
    if missing:
        raise ValueError(f"集合通信树未覆盖节点: {missing}")
    return TreeSpec(root=int(root), parent=parent, topology=topology, direction=normalized_direction, name=str(name))


def _knomial_edges(topology: Topology, *, root: int, nodes: Sequence[int], radix: int) -> tuple[tuple[int, int], ...]:
    if isinstance(radix, bool) or int(radix) < 2:
        raise ValueError(f"radix 必须是 >=2 的整数，收到 {radix}")
    ordered = tuple(int(node) for node in nodes)
    if int(root) not in ordered:
        raise ValueError(f"root {root} 不在集合通信节点中")
    if int(radix) == 2 and _has_regular_topology_groups(topology, ordered):
        try:
            return _knomial_algebra_edges(root=int(root), nodes=ordered, radix=int(radix))
        except ValueError:
            pass
    return _partial_knomial_group_edges(topology, root=int(root), nodes=ordered, radix=int(radix))


def _has_regular_topology_groups(topology: Topology, nodes: Sequence[int]) -> bool:
    groups = _topology_groups(topology, nodes)
    sizes = {len(group) for group in groups}
    return len(sizes) == 1 and sum(sizes) * len(groups) == len(nodes)


def _knomial_algebra_edges(*, root: int, nodes: Sequence[int], radix: int) -> tuple[tuple[int, int], ...]:
    ordered = list(int(node) for node in nodes)
    root_idx = ordered.index(int(root))
    return _knomial_algebra_rank_edges(ordered, root_idx, int(radix))


def _knomial_algebra_rank_edges(values: Sequence[int], root_pos: int, radix: int) -> tuple[tuple[int, int], ...]:
    n = len(values)
    radices = _knomial_radices(n, int(radix))
    edges: list[tuple[int, int]] = []
    place = n
    for dim_radix in radices:
        place //= int(dim_radix)
        step = place * int(dim_radix)
        for base in range(0, n, step):
            src = _add_mixed_radix_rank(int(root_pos), base, radices=radices)
            for digit in range(1, int(dim_radix)):
                dst = _add_mixed_radix_rank(int(root_pos), base + digit * place, radices=radices)
                edges.append((int(values[src]), int(values[dst])))
    return tuple(edges)


def _partial_knomial_group_edges(topology: Topology, *, root: int, nodes: Sequence[int], radix: int) -> tuple[tuple[int, int], ...]:
    groups = _topology_groups(topology, nodes)
    root_group = next(idx for idx, group in enumerate(groups) if int(root) in group)
    root_local = tuple(groups[root_group]).index(int(root))
    representatives = tuple(int(group[root_local % len(group)]) for group in groups)
    edges = [(representatives[src], representatives[dst]) for src, dst in _partial_knomial_rank_edges(len(groups), root_group, int(radix))]
    for group_idx, group in enumerate(groups):
        local_root = tuple(group).index(representatives[group_idx])
        edges.extend((int(group[src]), int(group[dst])) for src, dst in _partial_knomial_rank_edges(len(group), local_root, int(radix)))
    return tuple(edges)


def _partial_knomial_rank_edges(n: int, root_rank: int, radix: int) -> tuple[tuple[int, int], ...]:
    if int(n) <= 1:
        return ()
    try:
        return _knomial_algebra_rank_edges(tuple(range(int(n))), int(root_rank), int(radix))
    except ValueError:
        core_size = _largest_knomial_core_size(int(n), int(radix))
        ordered = tuple((int(root_rank) + offset) % int(n) for offset in range(int(n)))
        core = ordered[:core_size]
        edges = list(_knomial_algebra_rank_edges(core, 0, int(radix)))
        edges.extend((core[idx % len(core)], extra) for idx, extra in enumerate(ordered[core_size:]))
        return tuple(edges)


def _largest_knomial_core_size(n: int, radix: int) -> int:
    for size in range(int(n) - 1, 0, -1):
        try:
            _knomial_radices(size, int(radix))
            return size
        except ValueError:
            continue
    return 1


def _knomial_radices(n: int, radix: int) -> tuple[int, ...]:
    value = 1
    radices: list[int] = []
    while value < int(n):
        value *= int(radix)
        radices.append(int(radix))
    if value != int(n):
        if int(radix) & (int(radix) - 1) or int(n) & (int(n) - 1):
            raise ValueError(f"knomial/hd 要求节点数是 radix 的整数幂或二次幂混合 radix，收到 n={n}, radix={radix}")
        n_bits = int(n).bit_length() - 1
        radix_bits = int(radix).bit_length() - 1
        rem = n_bits % radix_bits
        radices = ([1 << rem] if rem else []) + [int(radix)] * (n_bits // radix_bits)
    return tuple(radices)


def _add_mixed_radix_rank(rank: int, delta: int, *, radices: Sequence[int]) -> int:
    out = 0
    place = 1
    lhs = int(rank)
    rhs = int(delta)
    for radix in reversed(tuple(int(value) for value in radices)):
        out += ((lhs % int(radix)) + (rhs % int(radix))) % int(radix) * place
        lhs //= int(radix)
        rhs //= int(radix)
        place *= int(radix)
    return int(out)


def _math_groups(nodes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    import math

    ordered = list(int(node) for node in nodes)
    n = len(ordered)
    m = max(1, int(math.isqrt(n)))
    if m * m <= n <= m * (m + 1):
        count, large = m, n - m * m
    else:
        count, large = m + 1, n - m * (m + 1)
    groups: list[tuple[int, ...]] = []
    cursor = 0
    for idx in range(count):
        size = m + 1 if idx < large else m
        if size > 0:
            groups.append(tuple(ordered[cursor : cursor + size]))
        cursor += size
    return tuple(group for group in groups if group)


def _topology_groups(topology: Topology, nodes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    node_set = set(int(node) for node in nodes)
    groups = [tuple(sorted(int(node) for node in group if int(node) in node_set)) for group in topology.gpu_groups]
    return tuple(group for group in groups if group)


def ahc_2level_groups(topology: Topology, *, grouping: str = "physical") -> tuple[tuple[int, ...], ...]:
    normalized = str(grouping).strip().lower()
    if normalized == "physical":
        return tuple(tuple(int(node) for node in group) for group in topology.physical_gpu_groups)
    if normalized == "logical":
        return tuple(tuple(int(node) for node in group) for group in topology.gpu_groups)
    raise ValueError(f"不支持的 ahc_2level grouping: {grouping}")


def _groups_in_nodes(groups: Sequence[Sequence[int]], nodes: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    node_set = set(int(node) for node in nodes)
    return tuple(tuple(int(node) for node in group if int(node) in node_set) for group in groups if any(int(node) in node_set for node in group))


def _normalize_ahc_groups(topology: Topology, groups: Sequence[Sequence[int]] | None) -> tuple[tuple[int, ...], ...]:
    raw = topology.gpu_groups if groups is None else groups
    normalized = tuple(tuple(int(node) for node in group) for group in raw)
    if len(normalized) < 2:
        raise ValueError("ahc_2level 至少需要两个 group")
    seen: set[int] = set()
    for group in normalized:
        if not group:
            raise ValueError("ahc_2level group 不能为空")
        for node in group:
            if node in seen:
                raise ValueError(f"ahc_2level group 中存在重复节点: {node}")
            if node not in topology.gpu_nodes:
                raise ValueError(f"ahc_2level group 包含非 GPU 节点: {node}")
            seen.add(node)
    return normalized


def _ahc_bounds(groups: Sequence[Sequence[int]]) -> tuple[Fraction, ...]:
    bounds = {Fraction(0), Fraction(1)}
    for group in groups:
        size = len(group)
        bounds.update(Fraction(idx, size) for idx in range(1, size))
    return tuple(sorted(bounds))


def _ahc_interval_owners(groups: Sequence[Sequence[int]], right: Fraction) -> tuple[int, ...]:
    owners: list[int] = []
    for group in groups:
        local = min(ceil(right * len(group)) - 1, len(group) - 1)
        owners.append(int(group[local]))
    return tuple(owners)


def _ahc_2level_edges(groups: Sequence[Sequence[int]], owners: Sequence[int], *, root: int) -> tuple[tuple[int, int], ...]:
    edges = [(int(root), int(owner)) for owner in owners if int(owner) != int(root)]
    for owner, group in zip(owners, groups):
        edges.extend((int(owner), int(node)) for node in group if int(node) != int(owner))
    return tuple(edges)


def _rooted_groups(groups: Sequence[Sequence[int]], root: int, *, rotate: bool = False) -> tuple[tuple[int, ...], ...]:
    raw = [list(int(node) for node in group) for group in groups]
    root_group = next((idx for idx, group in enumerate(raw) if int(root) in group), None)
    if root_group is None:
        raise ValueError(f"root {root} 不在节点组中")
    root_rank = raw[root_group].index(int(root))
    out: list[tuple[int, ...]] = []
    for idx, group in enumerate(raw):
        root_pos = root_rank % len(group)
        if idx == root_group:
            root_pos = group.index(int(root))
        if rotate:
            out.append(tuple(int(group[(root_pos + offset) % len(group)]) for offset in range(len(group))))
        else:
            group_root = int(group[root_pos])
            out.append(tuple([group_root] + [int(node) for node in group if int(node) != group_root]))
    return tuple(out)


def _tree_tree_groups(topology: Topology, *, nodes: Sequence[int], root: int, method: Any) -> tuple[tuple[int, ...], ...]:
    normalized = "node" if method is None else str(method).lower()
    if normalized not in {"nht", "node"}:
        raise ValueError(f"不支持的 tree_tree method: {method}")
    groups = _math_groups(nodes) if normalized == "nht" else _topology_groups(topology, nodes)
    return _rooted_groups(groups, int(root))


def _tree_ring_groups(topology: Topology, *, nodes: Sequence[int], root: int, method: Any) -> tuple[tuple[int, ...], ...]:
    normalized = "node" if method is None else str(method).lower()
    if normalized not in {"nht", "node"}:
        raise ValueError(f"不支持的 tree_ring method: {method}")
    groups = _math_groups(nodes) if normalized == "nht" else _topology_groups(topology, nodes)
    return _rooted_groups(groups, int(root), rotate=True)


def _tree_tree_edges(groups: Sequence[Sequence[int]], *, root: int) -> tuple[tuple[int, int], ...]:
    root_group = next(idx for idx, group in enumerate(groups) if int(root) in group)
    edges: list[tuple[int, int]] = []
    for idx, group in enumerate(groups):
        if idx != root_group:
            edges.append((int(root), int(group[0])))
    for group in groups:
        for node in group[1:]:
            edges.append((int(group[0]), int(node)))
    return tuple(edges)


def _tree_ring_edges(groups: Sequence[Sequence[int]], *, root: int, intra_reverse: bool) -> tuple[tuple[int, int], ...]:
    root_group = next(idx for idx, group in enumerate(groups) if int(root) in group)
    edges: list[tuple[int, int]] = []
    for idx, group in enumerate(groups):
        if idx != root_group:
            edges.append((int(root), int(group[0])))
    for group in groups:
        ordered = [int(group[0])] + [int(node) for node in group[1:]]
        if bool(intra_reverse) and len(ordered) > 1:
            ordered = [ordered[0]] + list(reversed(ordered[1:]))
        edges.extend(_chain(ordered))
    return tuple(edges)


def _ordered_ring_nodes(*, root: int, nodes: Sequence[int]) -> tuple[int, ...]:
    ordered = list(int(node) for node in nodes)
    root_idx = ordered.index(int(root))
    return tuple(ordered[(root_idx + offset) % len(ordered)] for offset in range(len(ordered)))


def _chain(nodes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    return tuple((int(nodes[idx]), int(nodes[idx + 1])) for idx in range(max(len(nodes) - 1, 0)))


def _nhr_edges(*, root: int, nodes: Sequence[int]) -> tuple[tuple[int, int], ...]:
    ordered = tuple(int(node) for node in nodes)
    n = len(ordered)
    if int(root) not in ordered:
        raise ValueError(f"root {root} 不在集合通信节点中")
    root_rank = ordered.index(int(root))
    edges: list[tuple[int, int]] = []
    for step in range((n - 1).bit_length()):
        distance = 1 << ((n - 1).bit_length() - step - 1)
        count = (n - 1 + distance) // (2 * distance)
        ranks = sorted({(root_rank + 2 * distance * idx) % n for idx in range(count)})
        edges.extend((ordered[rank], ordered[(rank + distance) % n]) for rank in ranks)
    return tuple(edges)


def _ring_groups(topology: Topology, *, nodes: Sequence[int], root: int, method: Any) -> tuple[tuple[int, ...], ...]:
    normalized = "node" if method is None else str(method).lower()
    if normalized not in {"nhr", "node"}:
        raise ValueError(f"不支持的 ring_ring method: {method}")
    groups = _math_groups(nodes) if normalized == "nhr" else _topology_groups(topology, nodes)
    return _rooted_groups(groups, int(root), rotate=True)


def _ring_ring_edges(groups: Sequence[Sequence[int]], *, root: int, inter_reverse: bool, intra_reverse: bool) -> tuple[tuple[int, int], ...]:
    roots = [int(group[0]) for group in groups]
    root_group = next(idx for idx, group in enumerate(groups) if int(root) in group)
    ordered_roots = [roots[(root_group + offset) % len(roots)] for offset in range(len(roots))]
    if inter_reverse and len(ordered_roots) > 1:
        ordered_roots = [ordered_roots[0]] + list(reversed(ordered_roots[1:]))
    edges = list(_chain(ordered_roots))
    for group in groups:
        ordered = [int(group[0])] + [int(node) for node in group[1:]]
        if intra_reverse and len(ordered) > 1:
            ordered = [ordered[0]] + list(reversed(ordered[1:]))
        edges.extend(_chain(ordered))
    return tuple(edges)


def _validate_difference_edges(n: int, edges: Sequence[tuple[int, int]], max_out: int) -> None:
    diffs = {(int(dst) - int(src)) % int(n) for src, dst in edges}
    if diffs != set(range(1, int(n))):
        raise ValueError("edge-disjoint pack 未覆盖全部非零差值")
    for node in range(int(n)):
        if sum(1 for src, _dst in edges if int(src) == node) > int(max_out):
            raise ValueError("edge-disjoint pack 超过最大出度")


def _validate_graceful_edges(n: int, edges: Sequence[tuple[int, int]], max_out: int) -> None:
    diffs = {abs(int(dst) - int(src)) for src, dst in edges}
    if diffs != set(range(1, int(n))):
        raise ValueError("graceful binary pack 未覆盖全部无向差值")
    for node in range(int(n)):
        if sum(1 for src, _dst in edges if int(src) == node) > int(max_out):
            raise ValueError("graceful binary pack 超过最大出度")


@lru_cache(maxsize=None)
def _difference_binary_edges(n: int) -> tuple[tuple[int, int], ...]:
    normalized = int(n)
    if normalized < 2 or normalized & (normalized - 1):
        raise ValueError("ed_binary_pack 的 graceful 构造要求节点数是 2 的幂")
    edges = _graceful_binary_edges(normalized)
    _validate_graceful_edges(int(n), edges, 2)
    return tuple(edges)


def _graceful_binary_edges(n: int) -> tuple[tuple[int, int], ...]:
    degrees = [1] + [2] * (int(n).bit_length() - 2)
    q = len(degrees) + 1
    subtree_sizes = [0] * (q + 1)
    subtree_sizes[q] = 1
    for level in range(q - 1, 0, -1):
        subtree_sizes[level] = 1 + degrees[level - 1] * subtree_sizes[level + 1]
    labels = [0]
    parents: list[int | None] = [None]
    current = [(0, ())]
    for level, degree in enumerate(degrees, start=1):
        next_level = []
        for parent_idx, prefix in current:
            for child_rank in range(degree):
                path = prefix + (child_rank,)
                tree_level = level + 1
                if tree_level % 2 == 1:
                    label = sum(path[idx - 1] * subtree_sizes[idx + 1] for idx in range(1, tree_level)) + ((tree_level - 1) // 2)
                else:
                    label = (
                        (degrees[0] - path[0]) * subtree_sizes[2]
                        - sum(path[idx - 1] * subtree_sizes[idx + 1] for idx in range(2, tree_level))
                        - (tree_level // 2 - 1)
                    )
                node_idx = len(labels)
                labels.append(int(label))
                parents.append(int(parent_idx))
                next_level.append((node_idx, path))
        current = next_level
    return tuple((int(labels[parent]), int(labels[child])) for child, parent in enumerate(parents) if parent is not None)


@lru_cache(maxsize=None)
def _difference_path_edges(n: int) -> tuple[tuple[int, int], ...]:
    if int(n) % 2 != 0:
        raise ValueError("ed_ring_pack 需要偶数个节点")
    seq = [0]
    half = int(n) // 2
    for offset in range(1, half):
        seq.extend([int(n) - offset, offset])
    seq.append(half)
    edges = _chain(seq)
    _validate_difference_edges(int(n), edges, 1)
    return edges


def _shift_rank_edges(edges: Sequence[tuple[int, int]], *, nodes: Sequence[int], root: int) -> tuple[tuple[int, int], ...]:
    ordered = tuple(int(node) for node in nodes)
    root_rank = ordered.index(int(root))
    return tuple((ordered[(int(src) + root_rank) % len(ordered)], ordered[(int(dst) + root_rank) % len(ordered)]) for src, dst in edges)


def _root_rank_edges(edges: Sequence[tuple[int, int]], *, nodes: Sequence[int], root: int) -> tuple[tuple[int, int], ...]:
    ordered = tuple(int(node) for node in nodes)
    root_rank = ordered.index(int(root))
    adjacent: dict[int, list[int]] = {idx: [] for idx in range(len(ordered))}
    for src, dst in edges:
        adjacent[int(src)].append(int(dst))
        adjacent[int(dst)].append(int(src))
    out: list[tuple[int, int]] = []
    seen = {root_rank}
    frontier = [root_rank]
    while frontier:
        src = frontier.pop(0)
        for dst in adjacent[src]:
            if dst in seen:
                continue
            seen.add(dst)
            frontier.append(dst)
            out.append((ordered[src], ordered[dst]))
    return tuple(out)
