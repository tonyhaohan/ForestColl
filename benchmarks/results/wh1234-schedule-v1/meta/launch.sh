#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="$HOME/mcnf-runs/issue272_schedule_20261015_v1"
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate math
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN_ROOT/ForestColl"
python -B - <<'PY' > "$RUN_ROOT/meta/environment.json"
from datetime import datetime, timezone
import json, platform, socket, sys
import gurobipy, networkx, numpy, scipy
print(json.dumps({"hostname":socket.gethostname(),"started_at":datetime.now(timezone.utc).isoformat(),
                  "python":sys.version,"platform":platform.platform(),"conda_env":"math",
                  "gurobi":gurobipy.gurobi.version(),"networkx":networkx.__version__,
                  "numpy":numpy.__version__,"scipy":scipy.__version__,
                  "threads":4,"solver_soft_memory_gb":16},indent=2))
PY
python -u -B benchmarks/run_schedule_matrix.py \
  --source "$HOME/mcnf-runs/issue272_forestcoll_20261015/ForestColl/benchmarks/results/wh1234-issue272-v1" \
  --label wh1234-schedule-v1 --horizons 12 2D 4D \
  --max-variables 2000000 --solve-limit 600 --source-wait-limit 21600 \
  > "$RUN_ROOT/meta/queue.log" 2>&1
