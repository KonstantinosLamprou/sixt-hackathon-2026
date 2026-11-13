#!/usr/bin/env bash
# Single entry point for the whole pipeline.   Usage:  ./run.sh <command> [<command> ...]
#
#   ./run.sh setup        create .venv and install requirements
#   ./run.sh submission   rebuild the submission files from the saved parameters (about 5 min)
#   ./run.sh all          everything, including tuning and all evaluations (about 1 h)
#   ./run.sh <step> ...   single steps, in any order:
#       explore  baselines  coldstart  nowcast  tuning  save_nowcast  backtest  partb  predict  export  slides
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"
[ -d .venv ] && PY=".venv/bin/python"

usage() { sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

step() {
  case "$1" in
    setup)        "${PYTHON:-python3}" -m venv .venv && .venv/bin/pip install -q -r requirements.txt && echo "venv ready" ;;
    explore)      "$PY" -m scripts.run_explore ;;
    baselines)    "$PY" -m scripts.run_baselines ;;
    coldstart)    "$PY" -m scripts.run_coldstart ;;
    nowcast)      "$PY" -m scripts.run_nowcast ;;
    tuning)       "$PY" -m scripts.run_tuning ;;
    save_nowcast) "$PY" -m scripts.run_save_nowcast ;;
    backtest)     "$PY" -m scripts.run_backtest_test ;;
    partb)        "$PY" -m scripts.run_partb ;;
    predict)      "$PY" -m scripts.run_predict_test ;;
    export)       "$PY" -m scripts.run_export ;;
    slides)       "$PY" -m scripts.make_slides ;;
    submission)   for s in partb predict export slides; do step "$s"; done ;;
    all)          for s in baselines coldstart nowcast tuning save_nowcast backtest partb predict export slides; do step "$s"; done ;;
    help|-h|--help) usage ;;
    *) echo "unknown command: $1" >&2; usage; exit 1 ;;
  esac
}

[ $# -eq 0 ] && { usage; exit 0; }
for cmd in "$@"; do
  echo "=== $cmd ==="
  step "$cmd"
done
