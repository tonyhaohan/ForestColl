from __future__ import annotations

from typing import Any

import numpy as np

from ..problem import MCNFProblem
from .tasks import build_tasks
from .topology import create_topology


def build_problem_from_config(config: dict[str, Any]) -> MCNFProblem:
    modeling = dict(config.get("modeling", {}))
    _validate_modeling_config(modeling)
    topology = create_topology(config.get("network", {}).get("type", "ub"), config.get("network", {}).get("params", {}))
    rng = np.random.default_rng(int(config.get("run", {}).get("seed", 42)))
    tasks = build_tasks(topology, dict(config.get("task", {})), dict(config.get("comm", {"name": "single_layer"})), rng=rng)
    params = dict(modeling.get("params", {}))
    return MCNFProblem(
        topology=topology,
        tasks=tasks,
        beta=float(params.get("beta", 1.0)),
    )


def _validate_modeling_config(modeling: dict[str, Any]) -> None:
    if str(modeling.get("family", "node_arc")) != "node_arc":
        raise ValueError("极简版本只支持 modeling=node_arc")
