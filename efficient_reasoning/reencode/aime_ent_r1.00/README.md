# edit_ent at ρ_mlp 1.00 on AIME: Nibi state, to finish elsewhere

16 of the 60 problems were run on Nibi (H100 MIG), shards 0 and 1 of 8. `todo.json` lists the done and remaining problem
idx and which of the 8 shards still have work (2 to 7).

- `r1.00_a0.05.pt`: the edit pack Nibi used (their compute_edits, full-GSM8K-probe direction, ρ_attn -1, ρ_mlp 1.00, α 0.05).
- `edit_ent_tau_r1.00_a0.05.json`: the threshold Nibi used (0.164 bits; top 20% of token entropy on 30 GSM8K probes).
- `partial_nibi.jsonl.gz`: the 16 finished runs, in `s2e_edit_reencode.py`'s record format and uids.
- `aime22_24.jsonl`: the AIME 2022-2024 problems (`idx`, `question`, `gt`) for `AIME_FILE`.

```bash
cd efficient_reasoning
bash reencode/aime_ent_r1.00/install.sh
bash reencode/run_reencode.sh ent_aime_checks          # optional but recommended once: checks at ρ_mlp 1.00 (~13 min)
export AIME_FILE=$PWD/reencode/aime_ent_r1.00/aime22_24.jsonl
for i in 2 3 4 5 6 7; do SHARDS=8 bash reencode/run_reencode.sh ent_aime_run $i; done   # or one GPU job per shard
bash reencode/run_reencode.sh aime_eval
```

Shards must stay `SHARDS=8` so they split the problems the same way. Each run peaks at 18.7 GiB with one process per GPU;
a shard took 2 to 3 h on a 3g.40gb slice. Finished problems are skipped. `aime_eval` compares against vanilla, edit and edit_re
only if their AIME runs are in `reencode_results/edit/`; otherwise copy the new `reencode_results/edit/aime.*.jsonl` back for
grading with the reference runs.
