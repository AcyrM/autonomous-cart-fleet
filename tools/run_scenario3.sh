#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 YOLO_THREADS=2
"$1" tools/scenario_tests.py --scenario 3 --runs 25 --video-tests 0 --shot-test 0 --out "$2" >> "$2/log_cenario3.txt" 2>&1
while pgrep -f "tools/run_all_scenarios.sh" >/dev/null; do sleep 30; done
"$1" tools/scenario_tests.py --summary "$2" > "$2/final_summary.txt" 2>&1
echo "ALL DONE" >> "$2/final_summary.txt"
