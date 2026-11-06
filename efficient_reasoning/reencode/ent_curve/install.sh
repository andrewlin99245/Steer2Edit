#!/bin/bash
# Put the edit packs used for edit and edit_re at rho 0.65 / 0.70 / 0.75 (full-GSM8K-probe direction) where
# s2e_edit_reencode.py looks, so edit_ent uses the same edits. Existing files are not overwritten.
set -euo pipefail
H="$(cd "$(dirname "$0")" && pwd)"; R="$H/../../reencode_results"
mkdir -p "$R/edit_packs"
for r in 0.65 0.70 0.75; do cp -n "$H/r${r}_a0.05.pt" "$R/edit_packs/r${r}_a0.05.pt"; done
echo "installed into $R/edit_packs"
