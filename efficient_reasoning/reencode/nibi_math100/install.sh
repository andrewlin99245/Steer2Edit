#!/bin/bash
# Put the Nibi MATH-500 runs (vanilla, and edit / edit_re / edit_ent with the full-GSM8K-probe direction) where
# evaluate_edit_reencode.py reads them, so new runs are graded and paired against them directly. gen_ids are dropped;
# graders use text. Existing files are not overwritten.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; R="$H/../../reencode_results"
mkdir -p "$R/edit"
[ -e "$R/math100.nibi_vanilla.jsonl" ] || gunzip -c "$H/vanilla.jsonl.gz" > "$R/math100.nibi_vanilla.jsonl"
[ -e "$R/edit/math100.nibi.jsonl" ] || gunzip -c "$H/edit.jsonl.gz" > "$R/edit/math100.nibi.jsonl"
echo "installed into $R; grade with: bash reencode/run_reencode.sh edit_eval"
