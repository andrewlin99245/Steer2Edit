# Step re-encoding on Steer2Edit's activation-steering baseline

Experiment: does Steer2Edit's activation-steering baseline (efficient reasoning, Qwen3-4B-Thinking-2507) work better if
the steered tokens' KV-cache entries are re-encoded without steering once their reasoning step ends?

The baseline here is Steer2Edit's own. `evaluate.py`'s `LinearInterventionHook` on every layer's `self_attn.o_proj`
and `mlp.down_proj` (`output + direction * alpha`, on every position) is reproduced exactly, with directions built by
the unmodified `generate_response.py` → `extract_directions.py` recipe. Grading uses the unmodified
`utils.extract_answer` / `utils.verify_answer`.

## Arms

| arm | what is steered | KV cache |
|---|---|---|
| `vanilla` | nothing | — |
| `add` | every forward (Steer2Edit's baseline) | steered entries kept |
| `add_re` | every forward | at each fed token containing `\n\n`, and at `</think>`, the tokens since the last restore are re-encoded unsteered and replace their steered K/V. The `\n\n` that opens the next step stays steered until that step ends. |

The prompt stays steered in both steering arms. Sampling uses the model's `generation_config` (T 0.6, top-k 20,
top-p 0.95) with Gumbel-max common random numbers keyed by (question, seed, step). The arms see the same noise, so they
differ only through their logits. Generation runs in HF transformers (bf16, sdpa, batch 1), not vLLM, because
re-encoding needs per-token cache surgery.

## Data

`math100.json` lists 100 MATH-500 rows (`HuggingFaceH4/MATH-500` test, `random.Random(20261026).sample(500, 100)`).
The default sweep is α ∈ {0.1, 0.3, 0.5, 0.8, 1.2}, seed 64, and a budget of min(32768, max_pos − 512) tokens, as in
`evaluate.py`. That is 1 + 2·5 = 11 generations per question, 1,100 in total.

## Files

- `s2e_reencode.py`: the engine (`Steer` hook, sampler, `generate` for the three arms), the probe / run drivers, and
  `merge_probe` (writes `responses/qwen3-4b-thinking_gsm8k_responses.json` for `extract_directions.py`).
- `checks.py`: run before the main sweep; it exits non-zero on any failure. It checks:
  - the hook against `LinearInterventionHook`, bitwise;
  - the restore schedule;
  - the restored cache against an independent rebuild, bitwise;
  - closeness to unsteered per-token processing (relative error < 0.05, and lower than `add`'s);
  - that the held opener stays steered;
  - determinism.
- `evaluate_reencode.py`: grades and prints:
  - accuracy, mean reasoning length and cap hits per (arm, α);
  - paired `add_re` vs `add` at each α: sign test on discordant correctness, mean length difference, and
    geometric-mean length ratio with sign-flip permutation p;
  - the baseline check (`add` length relative to `vanilla`).
- `evaluate_edit_aime.py`, `aime_order.json`: grading and the frozen problem order for the AIME curve below.
- `run_reencode.sh`: runs the stages `probe → extract → checks → run → eval`, each resumable and shardable.

## Usage

```sh
cd efficient_reasoning
pip install torch transformers datasets math_verify latex2sympy2_extended   # plus the repo's own requirements
bash reencode/run_reencode.sh all
# or sharded, one stage per job:
SHARDS=8 bash reencode/run_reencode.sh probe $i     # i = 0..7
bash reencode/run_reencode.sh extract
bash reencode/run_reencode.sh checks
SHARDS=8 bash reencode/run_reencode.sh run $i
bash reencode/run_reencode.sh eval
```

Outputs go to `efficient_reasoning/reencode_results/`, which is git-ignored. Memory per process is about 8 GB of
weights plus up to about 4.8 GB of KV cache at 32k tokens, so two processes fit on a 40 GB GPU.

This branch starts from upstream commit `86382c2f44fdfda2bafe80faf09634e6d729562c`.

## Weight edit (Steer2Edit's method) with step re-encoding

The same re-encoding applied to Steer2Edit's actual method, the rank-1 weight edit, instead of the additive baseline.
The edit is theirs, unmodified: `edit.py`'s `compute_edits` builds the factors and `evaluate.py`'s
`apply_rank1_edits_to_hf_model` applies them. Both functions are loaded from source, so vLLM isn't needed. For
Qwen3-4B-Thinking only `mlp.down_proj` columns are edited (ρ_attn = −1).

| arm | weights | KV cache |
|---|---|---|
| `edit` | their edited model on every forward | kept |
| `edit_re` | edited on every forward | each finished step is replayed with the **original** weights and its K/V replaced; the next step's `\n\n` opener stays edited until that step ends |

Switching is a weight swap. Each edited module keeps its original and its edited tensors, and the switch points
`weight.data` at one or the other. With the edit on, the forward is exactly their edited model's.

- Settings: the four corners of their `edit.sh` grid, ρ_mlp ∈ {0.65, 0.80} × α ∈ {0.05, 0.20}. One corner is the
  best setting they report (ρ_mlp 0.8, α 0.05).
- Same 100 questions, seed and noise as above, so the `vanilla` runs are shared.
- `checks.py`'s cache checks are shared as `reencode_checks`. `s2e_edit_reencode.py checks` also verifies the swapped
  weights against their `apply_rank1_edits_to_hf_model` run on CPU, bitwise.
- Run: `bash reencode/run_reencode.sh edit_checks`, then `edit_run` (shardable), then `edit_eval`.

### Direction from the full GSM8K train probe

The paper says it uses the GSM8K training set, while `probe` above draws 200 questions. To probe all 7,473, use their
`generate_response.py` on vLLM with `full_probe/run_part.py` and `full_probe/merge_parts.py`, then their
`extract_directions.py`, which writes the direction file the edit stages read:

1. Write the train split to `data/gsm8k_train_full.jsonl` and split it into `data/part_0.jsonl` ... `data/part_7.jsonl`.
2. From `efficient_reasoning/`, copy the two scripts here and run `python run_part.py $i` for each part, with
   `VLLM_WORKER_MULTIPROC_METHOD=spawn`. Then run `python merge_parts.py` and `extract_directions.py`.

To fit a 40 GB MIG slice, change only these `LLM()` arguments: `max_model_len=min(max_pos, 16384)`, `max_num_seqs=256`,
`gpu_memory_utilization=0.85`. They do not change the sampling settings.

### Results: four corners, direction from the full probe

One seed, 100 MATH-500 questions, graded with their `extract_answer` / `verify_answer`. Vanilla is 99/100 at a mean of
4,850 tokens. Length ratio is the geometric mean of edit_re / edit per question. p values are not corrected for
multiple comparisons.

| setting | edit correct / mean length | edit_re correct / mean length | only edit_re / only edit correct (sign p) | length ratio (p) |
|---|---|---|---|---|
| ρ_mlp 0.80, α 0.05 | 92 / 3,107 | 98 / 3,410 | 6 / 0 (.031) | 1.136 (<.001) |
| ρ_mlp 0.65, α 0.05 | 80 / 2,782 | 91 / 2,993 | 15 / 4 (.019) | 1.134 (.014) |
| ρ_mlp 0.80, α 0.20 | 84 / 3,726 | 81 / 3,528 | 6 / 9 (.607) | 0.954 (.33) |
| ρ_mlp 0.65, α 0.20 | 69 / 2,457 | 59 / 2,213 | 5 / 15 (.041) | 0.867 (.086) |

At α 0.05 edit_re is more accurate and longer, like a weaker edit. At α 0.20 it is less accurate and shorter.

### Trade-off curve

`curve_packs`, `curve_run` (shardable) and `curve_eval` in `run_reencode.sh` map accuracy against length at α 0.05.
They add edit at ρ_mlp 0.55, 0.60, 0.70, 0.75, 1.00 and edit_re at 0.50, 0.55, 0.60, 0.70, 0.75 to the ρ 0.65 and 0.80
runs from `edit_run`. ρ 0.50, 0.55, 0.60 and 1.00 are outside their grid.

Results at α 0.05 (one seed, the same 100 questions, vanilla 99/100 at 4,850 tokens):

| ρ_mlp | edit correct / mean length | edit_re correct / mean length | only edit_re / only edit correct (sign p) |
|---|---|---|---|
| 0.50 | not run | 57 / 4,603 | |
| 0.55 | 34 / 3,384 | 83 / 3,670 | 49 / 0 (<.001) |
| 0.60 | 67 / 3,028 | 91 / 3,349 | 26 / 2 (<.001) |
| 0.65 | 80 / 2,782 | 91 / 2,993 | 15 / 4 (.019) |
| 0.70 | 89 / 2,973 | 92 / 3,069 | 5 / 2 (.453) |
| 0.75 | 91 / 2,918 | 92 / 3,504 | 2 / 1 (1.00) |
| 0.80 | 92 / 3,107 | 98 / 3,410 | 6 / 0 (.031) |
| 1.00 | 96 / 3,439 | not run | |

Below ρ_mlp 0.65 the edit collapses: answers get longer and less accurate. edit_re stays far more accurate there. From
ρ_mlp 0.65 up, the average gap over the shared length range (2,993 to 3,439 tokens) is +1.36 points, 95% CI
[-3.42, +8.32], so a better accuracy-length trade-off is not shown. That range was chosen after seeing the data.
Standard errors of mean length are 300 to 500 tokens.

### AIME curve

`curve_packs` (after `edit_checks`), then `aime_run` (shardable) and `aime_eval` in `run_reencode.sh` run the same edit on
harder problems. This is a separate curve and is not pooled with MATH.

- Problems: the first 60 of AIME 2022-2024 in the frozen random order of `aime_order.json`. The problem file is not in this
  repo. Set `AIME_FILE` to a JSONL with `idx` (row of `AI-MO/aimo-validation-aime`, revision 13f9e12), `question` and `gt`
  (the answer as a plain integer, with zero padding and a trailing `^\circ` removed). That dataset has since been renamed
  on the Hub, so look up its current name.
- Settings: α 0.05, one seed, cap 32,768 tokens, 60 runs per setting. edit_re ρ 0.65 has 58 runs: the run was stopped early.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is recommended. Three of the 16 shards on 40 GB MIG slices at 2
  processes per slice ran out of memory. The likely cause is allocator fragmentation, but it was not confirmed.

| setting | correct / 60 | average length | cap hits | correct vs vanilla, only this / only vanilla (sign p) | length difference from vanilla (p) |
|---|---|---|---|---|---|
| vanilla | 45 | 20,450 | 10 | | |
| edit ρ 1.00 | 40 | 18,557 | 11 | 3 / 8 (.227) | -1,892 (.069) |
| edit_re ρ 1.00 | 42 | 18,987 | 10 | 3 / 6 (.508) | -1,463 (.065) |
| edit ρ 0.80 | 30 | 18,124 | 7 | 2 / 17 (.001) | -2,326 (.022) |
| edit_re ρ 0.80 | 34 | 17,854 | 8 | 2 / 13 (.007) | -2,596 (.014) |
| edit ρ 0.70 | 18 | 14,008 | 2 | 2 / 29 (<.001) | -6,442 (<.001) |
| edit_re ρ 0.65 | 20 of 58 | 15,928 | 4 | 1 / 25 (<.001; 44 of 58 for vanilla) | -4,436 (.001) |

edit_re vs edit on the same questions:

| pair | correct | only edit_re / only edit (sign p) | length difference (p) |
|---|---|---|---|
| edit_re ρ 1.00 vs edit ρ 1.00 | 42 vs 40 | 4 / 2 (.688) | +429 (.593) |
| edit_re ρ 0.80 vs edit ρ 0.80 | 34 vs 30 | 9 / 5 (.424) | -270 (.755) |
| edit_re ρ 0.80 vs edit ρ 1.00 | 34 vs 40 | 2 / 8 (.109) | -704 (.426) |
| edit_re ρ 0.65 vs edit ρ 0.70 | 20 vs 17 (58 questions) | 8 / 5 (.581) | +1,901 (.066) |

Lowering ρ costs far more accuracy here than on MATH, so the MATH working range does not carry over. The edit shortens
answers by about 7 to 9% at ρ 1.00, with accuracy not clearly below vanilla. No difference between edit and edit_re is
shown, but 60 questions can only detect gaps of about 20 points. Cap hits truncate lengths, so length differences are
uncertain. These results are exploratory: one seed, p values uncorrected, and vanilla and edit_re ρ 1.00 were added after
partial results were seen. AIME answers may be in the model's training data.

## Entropy-gated state (edit_ent)

A variation on `edit_re` that controls which edited state stays in the cache. Every forward stays edited and the prompt stays
edited. At each step end a token's K/V are replaced by the clean ones only if the token was sampled with entropy below a
threshold; uncertain tokens keep their edited K/V for the rest of the run. The threshold is the top 20% quantile of token entropy
under the edit on the first 30 GSM8K probes, at ρ_mlp 0.80, α 0.05.

```bash
bash reencode/run_reencode.sh ent_checks   # after edit_checks: builds the threshold, then checks
bash reencode/run_reencode.sh ent_run      # 100 MATH-500 questions; grade with edit_eval
bash reencode/run_reencode.sh ent_aime_checks   # after curve_packs: threshold and checks at ρ_mlp 1.00
AIME_FILE=... bash reencode/run_reencode.sh ent_aime_run   # 60 AIME problems at ρ_mlp 1.00; grade with aime_eval
```

The checks require threshold -inf to equal `edit` and +inf to equal `edit_re`, in tokens and cache, bitwise.

One seed, exploratory, Qwen3-4B, ρ_mlp 0.80, α 0.05, on 100 MATH-500 questions:

| Arm | Correct | Mean tokens | Cap hits |
|---|---:|---:|---:|
| `edit` | 92 | 3,107 | |
| `edit_ent` | 97 | 3,215 | 0 |
| `edit_re` | 98 | 3,410 | |
| vanilla | 99 | 4,850 | |

These settings and questions had already been seen, so this is a flag and not a result. No AIME result yet.
