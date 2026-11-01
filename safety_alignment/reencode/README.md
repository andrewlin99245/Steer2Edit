# Line re-encoding on Steer2Edit's safety-alignment steering baseline

Experiment: does Steer2Edit's refusal-steering baseline (Mistral-7B-Instruct-v0.2) keep more utility at the same refusal
rate if the steered tokens' KV-cache entries are re-encoded without steering once their line ends? It is the
safety-alignment counterpart of `efficient_reasoning/reencode/`, and it reuses that engine (`s2e_reencode.generate`,
`Steer`, `Noise`) unchanged. Llama-2-7B-Chat is gated, so we used Mistral.

The baseline is Steer2Edit's own: `evaluate.py`'s `LinearInterventionHook` on every `o_proj` and `down_proj`
(`output + direction * alpha`, every position including the prompt). The direction is the "post" refusal direction
from the unmodified `generate_response.py` → `extract_directions.py --model mistral-7b` recipe. Refusal is any
`REFUSAL_SIGNAL` substring. Utility is graded with the unmodified `utils.extract_answer` / `utils.verify_answer` and
`evaluate_utility.py`'s multiple-choice fallback.

## Arms

| arm | what is steered | KV cache |
|---|---|---|
| `vanilla` | nothing | — |
| `add` | every forward (Steer2Edit's baseline) | steered entries kept |
| `add_re` | every forward | at each fed token containing `\n`, the tokens since the last restore are re-encoded unsteered and replace their steered K/V; the newline that opens the next line stays steered until that line ends |

Chat answers have no `\n\n` reasoning steps, so here a line plays the role of a step.

## Setup

- Safety: GCG (their `data/gcg_results/mistral-7b`, 100) and AdvLLM (their `advbench/greedy/mistral-7b.pkl`, first
  100). Prompts go through their `apply_chat` with the Mistral system prompt, with 128 new tokens.
- Utility: GSM8K, CommonsenseQA and Code-MMLU, 100 fixed random items each (`utility300.json`, seed 20261027), with
  their option formatting and no system prompt, with 512 new tokens.
- α ∈ {0.1, 0.3, 0.5, 0.8, 1.2}, seed 64, and one sample per prompt (their `n_sample` 10 cut to 1). That is
  500 prompts × 11 = 5,500 generations.
- Sampling uses vLLM's defaults (T 1, no top-k / top-p; the generation config sets none), with Gumbel-max common
  random numbers keyed by (part, idx, seed). Prompts carry two BOS tokens, as in their vLLM runs. Generation runs in
  HF transformers (bf16, sdpa, batch 1).

## Files

- `s2e_safety_reencode.py`: `probe`, `merge_probe`, `checks`, `run`. `checks` exits non-zero on any failure. It
  checks the hook bitwise against `LinearInterventionHook`, the restore schedule, the restored cache bitwise against an
  independent rebuild, closeness to unsteered processing (lower error than `add`), the held newline, and determinism.
- `evaluate_safety_reencode.py`: grading and the paired add_re vs add readout (sign tests).
- `run_safety_reencode.sh`: the pipeline (`probe | extract | checks | run | eval | all`).

## Result (Mistral-7B-Instruct-v0.2, seed 64)

Refusal is the mean over GCG and AdvLLM; utility is the mean over the three sets (%).

| α | refusal add / add_re | utility add / add_re | utility, add_re-only / add-only correct (sign p) |
|---|---|---|---|
| 0 (vanilla) | 33.5 | 53.7 | |
| 0.1 | 36.5 / 38.0 | 53.0 / 52.0 | 5/8 (0.58) |
| 0.3 | 48.5 / 50.0 | 50.7 / 51.0 | 9/8 (1.00) |
| 0.5 | 61.5 / 61.0 | 48.7 / 49.3 | 11/9 (0.82) |
| 0.8 | 67.5 / 68.0 | 47.3 / 48.3 | 11/8 (0.65) |
| 1.2 | 66.0 / 68.0 | 28.0 / 30.3 | 10/3 (0.09) |

Refusal stays within 2 points of `add` at every α, and utility is slightly higher at α ≥ 0.5, but no difference is
significant. Re-encoding gives no measurable trade-off gain here. Many answers are short or a single line, which leaves
little to re-encode: with 0 restores in 52% / 62% / 67% of utility answers at α 0.5 / 0.8 / 1.2.
