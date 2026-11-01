"""Merge the 8 part responses into the file extract_directions.py reads; fails unless all 7,473 are present."""
import json

out = []
for i in range(8):
    out += json.load(open(f"responses/qwen3-4b-thinking_part{i}_responses.json"))
assert len(out) == 7473, len(out)
assert len({r["question"] for r in out}) == len({json.loads(l)["question"] for l in open("data/gsm8k_train_full.jsonl")})
json.dump(out, open("responses/qwen3-4b-thinking_gsm8k_responses.json", "w"), indent=4)
print("merged", len(out), "correct", sum(bool(r["correct"]) for r in out))
