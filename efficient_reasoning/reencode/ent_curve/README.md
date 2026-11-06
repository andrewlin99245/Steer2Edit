# edit_ent curve on MATH-500 (ρ_mlp 0.65, 0.70, 0.75; α 0.05)

The same 100 MATH-500 questions, seed and noise as the curve above, so each run pairs with edit and edit_re at the same ρ.
Each ρ gets its own threshold (top 20% of token entropy under that edit on the first 30 GSM8K probes), built by the checks.
The packs here are the edits that edit and edit_re used.

```bash
cd efficient_reasoning
bash reencode/ent_curve/install.sh
bash reencode/run_reencode.sh ent_curve_checks                                   # thresholds + checks for the 3 settings, ~40 min
for i in 0 1 2 3 4 5 6 7; do SHARDS=8 bash reencode/run_reencode.sh ent_curve_run $i; done   # or one job per shard
```

`math100()` loads `HuggingFaceH4/MATH-500` with `datasets`, so download it on a login node first. Peak memory on MATH-500 was
15.5 GiB, so two processes fit on a 40 GB GPU. 300 runs; edit_re averaged about 190 s per run on an H100 3g.40gb slice.
