"""Steer2Edit's activation-steering baseline with and without step re-encoding.

    cd efficient_reasoning
    python reencode/s2e_reencode.py probe [--shard i/n]                 GSM8K probe responses, vanilla
    python reencode/s2e_reencode.py merge_probe                         -> responses/ JSON for extract_directions.py
    python reencode/s2e_reencode.py run --arms vanilla add add_re [--shard i/n]   100 MATH-500 questions x arms x ALPHAS

Model Qwen/Qwen3-4B-Thinking-2507 (bf16, sdpa). Steering repeats Steer2Edit's efficient_reasoning/evaluate.py
(LinearInterventionHook, components "attnmlp"): every layer's self_attn.o_proj and mlp.down_proj output gets
`output + direction.type_as(output) * alpha`, with direction = directions["post"]["direction"][component] from their
extract_directions.py. As in their vLLM run, the hook acts on every position of every forward, prompt included.

Arms
  vanilla  no steering
  add      Steer2Edit's baseline: every forward steered; the steered cache is kept
  add_re   add + step re-encoding: at each fed token containing '\\n\\n' (anywhere) and at </think>, the tokens fed since
           the last restore are re-encoded unsteered and their K/V replace the steered ones; the new '\\n\\n' (the
           next step's opener) keeps its steered entry until its own step ends (held, as in k1a2). The prompt stays
           steered in both steering arms (a fixed offset, not re-encoded).

Re-encoding uses one cache: earlier entries are already clean, so cropping to the last restore point and replaying
the pending tokens unsteered (REPLAY_PIECE tokens per forward) equals replaying them against a separate clean cache.
A held opener's steered K/V are saved before the crop and appended back unchanged.

Sampling: the model's generation_config (temperature, top_k, top_p, eos ids) with Gumbel-max common random numbers
keyed by (question, seed, step), so arms differ only through their logits. Budget min(32768, max_pos - 512) for MATH-500
(their evaluate.py), 8192 for the probe (their generate_response.py).
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import DynamicCache


MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
HERE = Path(__file__).resolve().parent.parent  # efficient_reasoning/
S2E_COMMIT = "86382c2f44fdfda2bafe80faf09634e6d729562c"  # upstream commit this branch starts from
DIRS = HERE / "directions/qwen3-4b-thinking_efficient_reasoning_dirs.pt"
PROBE = HERE / "data/gsm8k_probe.jsonl"
PROBE_OUT = HERE / "responses/qwen3-4b-thinking_gsm8k_responses.json"
SPLIT = Path(__file__).resolve().parent / "math100.json"
ALPHAS = (0.1, 0.3, 0.5, 0.8, 1.2)
ARMS = ("vanilla", "add", "add_re")
SEED = 64
PROBE_MAX_NEW = 8192
REPLAY_PIECE = 8
OUT = HERE / "reencode_results"


class Noise:
    """Gumbel noise shared across arms, keyed by (question, seed); one draw per decoding step."""

    def __init__(self, key, seed, vocab, device):
        s = int(hashlib.sha256(f"crest|{key}|{seed}".encode()).hexdigest()[:15], 16)
        self.g = torch.Generator(device=device).manual_seed(s)
        self.vocab, self.device = vocab, device

    def next(self):
        u = torch.rand(self.vocab, generator=self.g, device=self.device).clamp_(1e-10, 1 - 1e-7)
        return -torch.log(-torch.log(u))


def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, GenerationConfig.from_pretrained(MODEL_ID)


class Steer:
    """Steer2Edit's LinearInterventionHook on every o_proj and down_proj, active only while `on` is True."""

    def __init__(self, model, alpha, path=DIRS):
        d = torch.load(path, map_location="cpu")["post"]["direction"]
        self.alpha, self.on, self.handles = alpha, False, []
        for i, layer in enumerate(model.model.layers):
            for name, mod in ((f"model.layers[{i}].self_attn.o_proj", layer.self_attn.o_proj),
                              (f"model.layers[{i}].mlp.down_proj", layer.mlp.down_proj)):
                self.handles.append(mod.register_forward_hook(self._mk(d[name].to(model.device))))

    def _mk(self, direction):
        def hook(module, inputs, output):
            if not self.on:
                return output
            return output + direction.type_as(output) * self.alpha  # their hook, op for op
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def sample(logits, noise, temp, top_k, top_p):
    """vLLM order: temperature, top-k, then top-p on the surviving tokens; Gumbel-max with the shared noise."""
    z = logits.float() / temp
    kth = torch.topk(z, top_k).values[-1]
    z = z.masked_fill(z < kth, float("-inf"))
    sz, idx = z.sort(descending=True)
    p = sz.softmax(-1)
    drop = p.cumsum(-1) - p >= top_p
    z = z.scatter(0, idx[drop], float("-inf"))
    return int((z + noise.next()[: z.shape[0]]).argmax())


def split_set(tok):
    return {i for t, i in tok.get_vocab().items() if "ĊĊ" in t}


@torch.no_grad()
def generate(model, steer, prompt_ids, arm, gcfg, nn_set, think_end, noise, max_new, return_cache=False):
    assert arm in ARMS
    dev = model.device
    eos = set(gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id])
    steered = arm != "vanilla"
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    cache = DynamicCache()
    steer.on = steered
    logits = model(input_ids=torch.tensor([prompt_ids], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
    steer.on = False
    P = len(prompt_ids)
    gen, pending, restores = [], [], []
    chunk_start, replay_s, in_think = P, 0.0, True
    while True:
        nxt = sample(logits, noise, gcfg.temperature, gcfg.top_k, gcfg.top_p)
        gen.append(nxt)
        if nxt in eos or len(gen) >= max_new:
            break
        was_think = in_think
        in_think &= nxt != think_end
        steer.on = steered
        logits = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
        steer.on = False
        if arm != "add_re":
            continue
        pending.append(nxt)
        if nxt in nn_set or (was_think and not in_think):
            hold = nxt in nn_set
            replay = pending[:len(pending) - hold]
            if replay:
                torch.cuda.synchronize()
                t0 = time.time()
                kept = [(k[..., -1:, :].clone(), v[..., -1:, :].clone())
                        for k, v in zip(cache.key_cache, cache.value_cache)] if hold else None
                cache.crop(chunk_start)
                for j in range(0, len(replay), REPLAY_PIECE):
                    model(input_ids=torch.tensor([replay[j:j + REPLAY_PIECE]], device=dev), past_key_values=cache,
                          use_cache=True)
                if hold:
                    for l, (k, v) in enumerate(kept):
                        cache.update(k, v, l)
                torch.cuda.synchronize()
                replay_s += time.time() - t0
                restores.append((chunk_start, chunk_start + len(replay)))
                chunk_start += len(replay)
            pending = pending[len(pending) - hold:]
    torch.cuda.synchronize()
    res = {"gen_ids": gen, "n_gen": len(gen), "hit_eos": gen[-1] in eos, "n_restores": len(restores),
           "restored_upto": chunk_start, "seconds": time.time() - t_start, "replay_seconds": replay_s,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    if return_cache:
        res.update(cache=cache, restores=restores)
    return res


def prompt_ids(tok, question):
    """Steer2Edit's apply_chat: the bare question as one user message, generation prompt added (ends '<think>\\n')."""
    text = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"]


def math100():
    """100 MATH-500 questions drawn once with a fixed seed (rows of HuggingFaceH4/MATH-500 test, idx = row)."""
    from datasets import load_dataset
    rows = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if not SPLIT.exists():
        idx = sorted(random.Random(20261026).sample(range(len(rows)), 100))
        SPLIT.write_text(json.dumps({"source": "HuggingFaceH4/MATH-500 test, random.Random(20261026).sample(500, 100)",
                                     "idx": idx}, indent=1))
    idx = json.load(open(SPLIT))["idx"]
    return [{"idx": i, "question": rows[i]["problem"], "gold": rows[i]["answer"], "level": rows[i]["level"]} for i in idx]


def shard_of(args):
    si, sn = map(int, args.shard.split("/"))
    return si, sn


def done(prefix):
    out = set()
    for fn in OUT.glob(f"{prefix}.*.jsonl"):
        for l in open(fn):
            out.add(json.loads(l)["uid"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["probe", "merge_probe", "run"])
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    if args.mode == "merge_probe":
        return write_probe_json()
    si, sn = shard_of(args)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == "probe":
        qs = [json.loads(l) for l in open(PROBE) if l.strip()]
        jobs = [(i, q, "vanilla", 0.0) for i, q in enumerate(qs)][si::sn]
        prefix, max_new = "probe", PROBE_MAX_NEW
    else:
        qs = math100()
        # vanilla and the baseline first, so the baseline check can be read before add_re runs
        order = [("vanilla", 0.0)] + [("add", a) for a in args.alphas] + [("add_re", a) for a in args.alphas]
        order = [(arm, a) for arm, a in order if arm in args.arms]
        jobs = [(q["idx"], q, arm, a) for arm, a in order for q in qs][si::sn]
        prefix, max_new = "math100", None
    uid = lambda i, arm, a: f"{prefix}|{arm}|{i}|a{a}|s{SEED}"
    seen = done(prefix)
    jobs = [j for j in jobs if uid(j[0], j[2], j[3]) not in seen]
    print(f"{len(jobs)} generations", flush=True)
    if not jobs:
        return
    model, tok, gcfg = load_model()
    if max_new is None:
        max_new = min(32768, model.config.max_position_embeddings - 512)
    nn_set, think_end = split_set(tok), tok.convert_tokens_to_ids("</think>")
    steer = Steer(model, 0.0) if DIRS.exists() else None
    path = OUT / f"{prefix}.s{si}of{sn}.{int(time.time())}.jsonl"
    with open(path, "a") as f:
        for i, q, arm, a in jobs:
            if steer is None:
                assert arm == "vanilla", "directions not built yet"
                steer = type("NoSteer", (), {"on": False})()
            steer.alpha = a
            ids = prompt_ids(tok, q["question"])
            g = generate(model, steer, ids, arm, gcfg, nn_set, think_end,
                         Noise(f"{prefix}|{i}", SEED, model.config.vocab_size, model.device), max_new)
            body = g["gen_ids"][:-1] if g["hit_eos"] else g["gen_ids"]
            text = tok.decode(body, skip_special_tokens=True).strip()
            rec = {"uid": uid(i, arm, a), "idx": i, "arm": arm, "alpha": a, "seed": SEED, "question": q["question"],
                   "gold": q.get("gold", q.get("answer")), "text": text,
                   "reasoning_length": len(tok.encode(text, add_special_tokens=False)),  # their evaluate.py metric
                   **{k: g[k] for k in ("n_gen", "hit_eos", "n_restores", "seconds", "replay_seconds", "peak_mem_gb")},
                   "gen_ids": g["gen_ids"], "max_new": max_new, "gpu": torch.cuda.get_device_name()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{prefix}] q{i} {arm} a={a} tok={g['n_gen']} eos={g['hit_eos']} restores={g['n_restores']} "
                  f"{g['seconds']:.0f}s (replay {g['replay_seconds']:.1f}s) mem {g['peak_mem_gb']:.1f}G", flush=True)
            del g
            torch.cuda.empty_cache()


def write_probe_json():
    """Merge probe shards into Steer2Edit's responses JSON (generate_response.py's fields) for extract_directions.py."""
    import re
    import sys
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE))
    from utils import extract_answer, verify_answer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    qs = [json.loads(l) for l in open(PROBE) if l.strip()]
    recs = {}
    for fn in OUT.glob("probe.*.jsonl"):
        for l in open(fn):
            r = json.loads(l)
            recs[r["idx"]] = r
    assert len(recs) == len(qs), f"{len(recs)} of {len(qs)} probe responses"
    out = []
    for i, q in enumerate(qs):
        text = recs[i]["text"]
        n = len(tok(text, return_tensors="np")["input_ids"][0])
        m = re.search(r"(<think>.*?</think>)", text, re.DOTALL)
        think = m.group(1).strip() if m else ""
        pred = extract_answer(text)
        try:
            ok = verify_answer(q["answer"], pred)
        except Exception:
            ok = False
        out.append({"question": q["question"], "response": text, "thinking": think, "response_length": int(n),
                    "thinking_length": int(len(tok(think, return_tensors="np")["input_ids"][0])) if m else -1,
                    "prediction": pred, "gold": q["answer"], "correct": bool(ok)})
    PROBE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PROBE_OUT.write_text(json.dumps(out, indent=4))
    print(f"wrote {PROBE_OUT}: {len(out)} responses, correct {sum(o['correct'] for o in out)}, "
          f"length min/median/max {sorted(o['response_length'] for o in out)[0]}/"
          f"{sorted(o['response_length'] for o in out)[len(out)//2]}/{max(o['response_length'] for o in out)}")


if __name__ == "__main__":
    main()
