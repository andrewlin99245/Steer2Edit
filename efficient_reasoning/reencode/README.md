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
