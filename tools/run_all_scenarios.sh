#!/usr/bin/env bash
# Runs the 4 scenario test batches (25 tests each) as 2 parallel queues and
# builds the results table. Resumable: finished tests are skipped on a rerun.
#   tools/run_all_scenarios.sh [python] [out dir] [runs]
cd "$(dirname "$0")/.."
PY=${1:-python}
OUT=${2:-results/scenario_tests}
RUNS=${3:-25}
mkdir -p "$OUT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 YOLO_THREADS=2
queue() {
  for s in "$@"; do
    "$PY" tools/scenario_tests.py --scenario "$s" --runs "$RUNS" --video-tests 0 \
        --shot-test 0 --out "$OUT" >> "$OUT/log_cenario$s.txt" 2>&1
  done
}
queue 1 3 &
queue 2 4 &
wait
"$PY" tools/scenario_tests.py --summary "$OUT"
echo "ALL DONE"
