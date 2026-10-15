set -euo pipefail
runroot=$HOME/mcnf-runs/issue272_forestcoll_20261015
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate math
export JAVA_HOME=$(cat "$runroot/meta/java-home.txt")
export PATH="$JAVA_HOME/bin:$PATH"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$runroot/ForestColl"
python -B benchmarks/check_clos.py > "$runroot/meta/contracts.log" 2>&1
python -u -B benchmarks/run_matrix.py --label wh1234-issue272-v1 --java-timeout 7200 > "$runroot/meta/queue.log" 2>&1
python -B benchmarks/check_clos.py > "$runroot/meta/contracts-final.log" 2>&1
python -B benchmarks/build_wh1234_report.py --label wh1234-issue272-v1 > "$runroot/meta/report.log" 2>&1
