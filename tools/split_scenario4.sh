#!/usr/bin/env bash
# Overnight helper: once scenario 3 is done, run scenario 4 tests 14-25 in a
# second process, stop the first one after test 13, then build the final table.
cd "$(dirname "$0")/.."
PY=$1; OUT=$2
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 YOLO_THREADS=2
running() { ps -C python -o args= | grep -q -- "$1"; }
while running "--scenario 3 "; do sleep 30; done
"$PY" tools/scenario_tests.py --scenario 4 --runs 25 --tests 14-25 --video-tests "" --shot-test -1 \
    --out "$OUT" >> "$OUT/log_cenario4b.txt" 2>&1 &
B=$!
while ! grep -q '"test": 13,' "$OUT/cenario4/runs.jsonl" 2>/dev/null; do
  running "--scenario 4 --runs 25 --video-tests 0" || break
  sleep 20
done
ps -C python -o pid=,args= | grep -- "--scenario 4 --runs 25 --video-tests 0" | awk '{print $1}' | xargs -r kill
wait $B
while running "--scenario "; do sleep 30; done
"$PY" tools/scenario_tests.py --summary "$OUT" > "$OUT/final_summary2.txt" 2>&1
echo "ALL DONE" >> "$OUT/final_summary2.txt"
