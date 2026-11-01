#!/bin/bash
# Full pipeline on one GPU (or run each stage as its own job; every stage is resumable by uid).
# Shard a stage across processes/GPUs with SHARDS=n: stage <i> runs shard i/n, e.g.
#   SHARDS=8 bash reencode/run_reencode.sh probe 3
# Run from efficient_reasoning/. HF_HOME may point at a pre-downloaded cache (offline nodes).
set -euo pipefail
cd "$(dirname "$0")/.."
SHARDS=${SHARDS:-1}
stage=${1:-all}; i=${2:-0}
case $stage in
  probe)   python reencode/s2e_reencode.py probe --shard "$i/$SHARDS" ;;
  extract) python reencode/s2e_reencode.py merge_probe
           python extract_directions.py --model qwen3-4b-thinking ;;
  checks)  python reencode/checks.py ;;
  run)     python reencode/s2e_reencode.py run --arms vanilla add add_re --shard "$i/$SHARDS" ;;
  eval)    python reencode/evaluate_reencode.py ;;
  edit_checks) python reencode/s2e_edit_reencode.py checks ;;          # weight edit: builds the edit packs too
  edit_run)    python reencode/s2e_edit_reencode.py run --shard "$i/$SHARDS" ;;
  edit_eval)   python reencode/evaluate_edit_reencode.py ;;
  curve_packs) python reencode/s2e_edit_reencode.py packs ;;           # trade-off curve at alpha 0.05, after edit_run
  curve_run)   python reencode/s2e_edit_reencode.py run --plan curve --shard "$i/$SHARDS" ;;
  curve_eval)  python reencode/evaluate_edit_reencode.py && python reencode/evaluate_edit_curve.py ;;
  all)     for s in probe extract checks run eval; do bash "$0" "$s" 0; done ;;
  *) echo "usage: $0 {probe|extract|checks|run|eval|all|edit_checks|edit_run|edit_eval|curve_packs|curve_run|curve_eval} [shard]"; exit 1 ;;
esac
