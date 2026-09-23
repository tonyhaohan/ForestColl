from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..linear_model import NodeArcLinearModel


def presolved_model_stats(rhs: np.ndarray, objective: np.ndarray, *, rows: int, columns: int, nnz: int) -> dict[str, Any]:
    return {
        "presolved_rhs_l2_norm": float(np.linalg.norm(rhs)),
        "presolved_objective_l2_norm": float(np.linalg.norm(objective)),
        "presolved_rows": int(rows),
        "presolved_columns": int(columns),
        "presolved_nnz": int(nnz),
    }


def save_presolved_model_stats(log_file: Any, model: NodeArcLinearModel, stats: dict[str, Any]) -> None:
    if not log_file:
        return
    path = Path(f"{log_file}.presolved_norms.jsonl").expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "saved_at_unix_sec": time.time(),
        "input_rows": int(model.eq_matrix.get_shape()[0] + model.ineq_matrix.get_shape()[0]),
        "input_columns": int(model.layout.total_variables),
        "input_nnz": int(model.eq_matrix.nnz + model.ineq_matrix.nnz),
        **stats,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
