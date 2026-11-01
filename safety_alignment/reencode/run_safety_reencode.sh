#!/bin/bash
# Full pipeline on one GPU (or run each stage as its own job; every stage is resumable by uid).
# Shard a stage across processes/GPUs with SHARDS=n: stage <i> runs shard i/n, e.g.
#   SHARDS=8 bash reencode/run_safety_reencode.sh run 3
# Run from safety_alignment/. HF_HOME may point at a pre-downloaded cache (offline nodes).
set -euo pipefail
cd "$(dirname "$0")/.."
SHARDS=${SHARDS:-1}
stage=${1:-all}; i=${2:-0}
case $stage in
  probe)   python reencode/s2e_safety_reencode.py probe --shard "$i/$SHARDS" ;;
  extract) python reencode/s2e_safety_reencode.py merge_probe
           python extract_directions.py --model mistral-7b ;;
  checks)  python reencode/s2e_safety_reencode.py checks ;;
  run)     python reencode/s2e_safety_reencode.py run --shard "$i/$SHARDS" ;;
  eval)    python reencode/evaluate_safety_reencode.py ;;
  all)     for s in probe extract checks run eval; do bash "$0" "$s" 0; done ;;
  *) echo "usage: $0 {probe|extract|checks|run|eval|all} [shard]"; exit 1 ;;
esac
