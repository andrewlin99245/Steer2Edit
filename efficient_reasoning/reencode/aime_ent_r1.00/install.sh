#!/bin/bash
# Put the Nibi edit pack, threshold and finished runs where s2e_edit_reencode.py looks, so ent_aime_run uses the same edit
# and threshold and skips the finished problems (resume by uid). Run from anywhere; existing files are not overwritten.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; R="$H/../../reencode_results"
mkdir -p "$R/edit_packs" "$R/edit"
cp -n "$H/r1.00_a0.05.pt" "$R/edit_packs/r1.00_a0.05.pt"
cp -n "$H/edit_ent_tau_r1.00_a0.05.json" "$R/edit_ent_tau_r1.00_a0.05.json"
[ -e "$R/edit/aime.nibi_partial.jsonl" ] || gunzip -c "$H/partial_nibi.jsonl.gz" > "$R/edit/aime.nibi_partial.jsonl"
echo "installed into $R; AIME_FILE=$H/aime22_24.jsonl"
