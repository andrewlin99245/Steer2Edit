"""Steer2Edit's weight edit (efficient reasoning) with and without step re-encoding.

    python reencode/s2e_edit_reencode.py checks          -> reencode_results/checks_s2e_edit.json (exits non-zero on failure)
    python reencode/s2e_edit_reencode.py run [--cfgs ...] [--shard i/n]
    python reencode/s2e_edit_reencode.py run --aime FILE [--n 60] [--shard i/n]    AIME 2022-2024 (see aime)
    python reencode/s2e_edit_reencode.py ent-checks      -> tau + reencode_results/checks_s2e_edit_ent.json (needs the r0.80_a0.05 pack)
    python reencode/s2e_edit_reencode.py run --plan ent [--shard i/n]    edit_ent at r0.80_a0.05

The edit is theirs, unmodified: edit.py's compute_edits (rank-1 factors from the extract_directions.py directions) applied with
evaluate.py's apply_rank1_edits_to_hf_model. Qwen3-4B-Thinking's settings edit only mlp.down_proj columns
(rho_attn -1). Switching is a weight swap: each edited module keeps its original and its edited weight tensor, and
`on` points module.weight.data at one of them. With the edit on, the forward is their edited model's, bitwise.

Arms (s2e_reencode.generate, unchanged: edit runs as "add", edit_re as "add_re")
  edit     their edited model; every forward edited, cache kept
  edit_re  edit + step re-encoding: at each fed '\\n\\n' and at </think>, the tokens since the last restore are
           replayed with the original weights and their K/V replace the edited ones; the new '\\n\\n' stays edited
           until its step ends (held, as in s2e_reencode). The prompt stays edited (their edited model sees it).
  edit_ent entropy-gated state (generate_ent): edit_re's schedule and clean entries, kept in a second cache, but at each
           restore only entries whose token was sampled with entropy < tau are replaced; uncertain tokens keep their
           edited K/V. tau -inf == edit, tau +inf == edit_re, bitwise (ent-checks).
Questions, seed, noise keys and budget as in s2e_reencode.py, so its vanilla generations are the shared reference.
"""
import argparse
import ast
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import s2e_reencode as S
from s2e_reencode import Noise

ER = S.HERE
# name -> (rho_attn, rho_mlp, alpha): their edit.sh / evaluate_editing.sh sweep for qwen3-4b-thinking,
# rho_attn -1 (no attention edits) x rho_mlp 0.65:0.05:0.80 x alpha 0.05:0.05:0.20 (16 settings)
CFGS = {f"r{rm:.2f}_a{a:.2f}": (-1.0, rm, a) for rm in (0.65, 0.7, 0.75, 0.8) for a in (0.05, 0.1, 0.15, 0.2)}
RUN = ["r0.65_a0.05", "r0.65_a0.20", "r0.80_a0.05", "r0.80_a0.20"]  # the grid's four corners
# Trade-off curve at alpha 0.05 (run --plan curve); rho 0.50, 0.55, 0.60, 1.00 are outside their grid. (arm, cfg), by cfg
CFGS |= {f"r{rm:.2f}_a0.05": (-1.0, rm, 0.05) for rm in (0.5, 0.55, 0.6, 1.0)}
CURVE = sorted([("edit", f"r{rm:.2f}_a0.05") for rm in (0.55, 0.6, 0.7, 0.75, 1.0)] +
               [("edit_re", f"r{rm:.2f}_a0.05") for rm in (0.5, 0.55, 0.6, 0.7, 0.75)], key=lambda x: x[1])
# AIME settings, including vanilla (the original weights, same noise keys); run question by question
AIME = [("edit_re", "r0.65_a0.05"), ("edit", "r0.70_a0.05"), ("edit", "r0.80_a0.05"), ("edit_re", "r0.80_a0.05"),
        ("edit", "r1.00_a0.05"), ("edit_re", "r1.00_a0.05"), ("vanilla", "vanilla")]
# Entropy-gated state: tau = the ENT_KEEP top quantile of token entropy under the edit, on the first ENT_N GSM8K probes
ENT_CFG, ENT_KEEP, ENT_N, ENT_MAX_NEW = "r0.80_a0.05", 0.20, 30, 4096
ENT = [("edit_ent", ENT_CFG)]
OUT = S.OUT / "edit"


def _their():
    """Their compute_edits (edit.py, with its helpers) and apply_rank1_edits_to_hf_model (evaluate.py), source
    unmodified, exec'd without their module-level imports (evaluate.py imports vllm, edit.py imports utils ->
    math_verify; neither is in the GPU env)."""
    ns = {"os": os, "torch": torch, "F": torch.nn.functional, "tqdm": __import__("tqdm").tqdm}
    for fn, names in (("edit.py", ("l2_normalize", "load_directions", "elastic_net_lambda", "compute_edits")),
                      ("evaluate.py", ("apply_rank1_edits_to_hf_model",))):
        src = (ER / fn).read_text()
        for node in ast.parse(src).body:
            if isinstance(node, ast.FunctionDef) and node.name in names:
                exec(compile(ast.get_source_segment(src, node), str(ER / fn), "exec"), ns)
    return SimpleNamespace(compute_edits=ns["compute_edits"], apply=ns["apply_rank1_edits_to_hf_model"])


def pack_path(name):
    return S.OUT / "edit_packs" / f"{name}.pt"


def build_pack(model, name):
    """Their compute_edits on the unedited model; computed once (in checks) and loaded by every run shard."""
    p = pack_path(name)
    if not p.exists():
        ra, rm, a = CFGS[name]
        _their().compute_edits(model, str(S.DIRS), rho_attn=ra, rho_mlp=rm, alpha=a, save_path=str(p))
    return torch.load(p, map_location="cpu")


class EditSwap:
    """Same interface as s2e.Steer (`on`, `remove`): `on` selects the edited or the original weights by pointer."""

    def __init__(self, model, pack):
        mods = [L.self_attn.o_proj for L in model.model.layers] + [L.mlp.down_proj for L in model.model.layers]
        orig = {m: m.weight.data.clone() for m in mods}
        _their().apply(model, pack)  # in place, their code
        self.pairs = [(m, orig[m], m.weight.data) for m in mods if not torch.equal(orig[m], m.weight.data)]
        self._on = True
        self.on = False

    @property
    def on(self):
        return self._on

    @on.setter
    def on(self, v):
        if v != self._on:
            for m, w0, w1 in self.pairs:
                m.weight.data = w1 if v else w0
            self._on = v

    def remove(self):
        self.on = False


def ref_edited_weights(model, pack):
    """Independent reference: their apply_rank1_edits_to_hf_model on CPU copies of the original weights, as their
    evaluate.py does (bf16 model on CPU)."""
    lay = [SimpleNamespace(self_attn=SimpleNamespace(o_proj=SimpleNamespace(weight=torch.nn.Parameter(
               L.self_attn.o_proj.weight.data.cpu().clone(), requires_grad=False))),
           mlp=SimpleNamespace(down_proj=SimpleNamespace(weight=torch.nn.Parameter(
               L.mlp.down_proj.weight.data.cpu().clone(), requires_grad=False)))) for L in model.model.layers]
    fake = SimpleNamespace(model=SimpleNamespace(layers=lay), config=model.config)
    _their().apply(fake, pack)
    return lay


@torch.no_grad()
def checks():
    from checks import reencode_checks
    model, tok, gcfg = S.load_model()
    dev = model.device
    report, ok = [], True

    def chk(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        report.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str)[:300], flush=True)

    q = json.loads(open(S.PROBE).readline())
    x = torch.tensor([S.prompt_ids(tok, q["question"])], device=dev)
    base = model(input_ids=x).logits
    packs = {n: build_pack(model, n) for n in RUN}  # before any swap object edits the model
    for n, pk in packs.items():
        chk(f"{n}: pack built by their compute_edits", True, n_attn=len(pk["attn"]), n_mlp=len(pk["mlp"]),
            cfg=CFGS[n])
    name = max(packs, key=lambda n: len(packs[n]["mlp"]))  # the densest edit for the cache checks
    ref = ref_edited_weights(model, packs[name])
    sw = EditSwap(model, packs[name])
    sw.on = True
    eq = all(torch.equal(L.self_attn.o_proj.weight.data.cpu(), R.self_attn.o_proj.weight.data) and
             torch.equal(L.mlp.down_proj.weight.data.cpu(), R.mlp.down_proj.weight.data)
             for L, R in zip(model.model.layers, ref))
    on_logits = model(input_ids=x).logits
    sw.on = False
    off = model(input_ids=x).logits
    chk(f"{name}: on == their apply_rank1_edits_to_hf_model on CPU, every o_proj / down_proj bitwise; off gives "
        "the unedited logits bitwise; the edit changes the logits", eq and torch.equal(off, base) and
        not torch.equal(on_logits, base), n_swapped=len(sw.pairs),
        logit_rel_change=float((on_logits - base).float().norm() / base.float().norm()))
    sw.on = True
    sw.on = False
    chk("toggling restores the original tensors (same storage)", all(m.weight.data.data_ptr() == w0.data_ptr()
                                                                     for m, w0, _ in sw.pairs))
    reencode_checks(model, tok, gcfg, sw, chk)
    sw.remove()
    S.OUT.mkdir(exist_ok=True)
    (S.OUT / "checks_s2e_edit.json").write_text(json.dumps({"all_pass": ok, "cfg": name, "report": report},
                                                                 indent=1, default=str))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


def token_entropy(logits, temp):
    """Bits; the full-vocabulary distribution at the sampling temperature, before top-k / top-p."""
    lp = (logits.float() / temp).log_softmax(-1)
    return float(-(lp.exp() * lp).sum()) / math.log(2)


@torch.no_grad()
def generate_ent(model, sw, prompt_ids, tau, gcfg, nn_set, think_end, noise, max_new, return_cache=False, clean=True):
    """edit_ent; ent[i] is the entropy of the logits that sampled gen[i]. Boundaries, held opener and replay pieces as in
    s2e_reencode.generate's add_re, but replays go to a second cache that holds the edited prompt and only clean entries, so a
    replaced entry is the one edit_re would have. clean=False: no clean cache or replay (the edit, for calibration)."""
    dev = model.device
    eos = set(gcfg.eos_token_id if isinstance(gcfg.eos_token_id, list) else [gcfg.eos_token_id])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    cache = S.DynamicCache()
    sw.on = True
    logits = model(input_ids=torch.tensor([prompt_ids], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
    sw.on = False
    cache_c = None
    if clean:
        cache_c = S.DynamicCache()
        for l in range(len(cache.key_cache)):
            cache_c.update(cache.key_cache[l].clone(), cache.value_cache[l].clone(), l)
    P = len(prompt_ids)
    gen, ent, pending, restores = [], [], [], []
    chunk_start, replay_s, in_think, n_replaced, n_kept = P, 0.0, True, 0, 0
    while True:
        h = token_entropy(logits, gcfg.temperature)
        nxt = S.sample(logits, noise, gcfg.temperature, gcfg.top_k, gcfg.top_p)
        gen.append(nxt)
        ent.append(h)
        if nxt in eos or len(gen) >= max_new:
            break
        was_think = in_think
        in_think &= nxt != think_end
        sw.on = True
        logits = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache, use_cache=True).logits[0, -1]
        sw.on = False
        if not clean:
            continue
        pending.append(nxt)
        if nxt in nn_set or (was_think and not in_think):
            hold = nxt in nn_set
            replay = pending[:len(pending) - hold]
            if replay:
                torch.cuda.synchronize()
                t0 = time.time()
                for j in range(0, len(replay), S.REPLAY_PIECE):
                    model(input_ids=torch.tensor([replay[j:j + S.REPLAY_PIECE]], device=dev), past_key_values=cache_c,
                          use_cache=True)
                idx = [p for p in range(chunk_start, chunk_start + len(replay)) if ent[p - P] < tau]
                if idx:
                    ix = torch.tensor(idx, device=dev)
                    for l in range(len(cache.key_cache)):
                        cache.key_cache[l][..., ix, :] = cache_c.key_cache[l][..., ix, :]
                        cache.value_cache[l][..., ix, :] = cache_c.value_cache[l][..., ix, :]
                n_replaced, n_kept = n_replaced + len(idx), n_kept + len(replay) - len(idx)
                torch.cuda.synchronize()
                replay_s += time.time() - t0
                restores.append((chunk_start, chunk_start + len(replay)))
                chunk_start += len(replay)
            pending = pending[len(pending) - hold:]
    torch.cuda.synchronize()
    res = {"gen_ids": gen, "ent": ent, "n_gen": len(gen), "hit_eos": gen[-1] in eos, "n_restores": len(restores),
           "n_replaced": n_replaced, "n_kept": n_kept, "restored_upto": chunk_start, "seconds": time.time() - t_start,
           "replay_seconds": replay_s, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    if return_cache:
        res.update(cache=cache, cache_c=cache_c, restores=restores)
    return res


def tau_path():
    return S.OUT / f"edit_ent_tau_{ENT_CFG}.json"


def calibrate(model, tok, gcfg, sw, nn_set, think_end):
    """tau from the first ENT_N GSM8K probes under the edit; not from the test questions. Reused if already built."""
    p = tau_path()
    if p.exists():
        return json.loads(p.read_text())
    hs, lens = [], []
    for k, line in enumerate(open(S.PROBE)):
        if k == ENT_N:
            break
        ids = S.prompt_ids(tok, json.loads(line)["question"])
        g = generate_ent(model, sw, ids, float("-inf"), gcfg, nn_set, think_end,
                         Noise(f"entcal|{k}", S.SEED, model.config.vocab_size, model.device), ENT_MAX_NEW, clean=False)
        hs += g["ent"]
        lens.append(g["n_gen"])
    t = torch.tensor(hs)
    d = {"tau": float(t.quantile(1 - ENT_KEEP)), "keep": ENT_KEEP, "n_questions": ENT_N, "n_tokens": len(hs), "cfg": ENT_CFG,
         "mean_len": sum(lens) / len(lens), "quantiles": {str(q): float(t.quantile(q)) for q in (.1, .25, .5, .75, .9, .95)}}
    S.OUT.mkdir(exist_ok=True)
    p.write_text(json.dumps(d, indent=1))
    return d


@torch.no_grad()
def ent_checks():
    from checks import bit_eq, rel_err
    model, tok, gcfg = S.load_model()
    dev = model.device
    report, ok = [], True

    def chk(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        report.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str)[:300], flush=True)

    sw = EditSwap(model, torch.load(pack_path(ENT_CFG), map_location="cpu"))
    nn_set, te = S.split_set(tok), tok.convert_tokens_to_ids("</think>")
    v = model.config.vocab_size
    flat, peak = token_entropy(torch.zeros(1024), 1.0), torch.zeros(1024)
    peak[3] = 100.0
    chk("entropy: uniform over 1024 == 10 bits, a point mass ~= 0", abs(flat - 10) < 1e-3 and token_entropy(peak, 1.0) < 1e-3)
    cal = calibrate(model, tok, gcfg, sw, nn_set, te)
    tau = cal["tau"]
    chk(f"tau from {ENT_N} GSM8K probes under the edit, keeping the top {ENT_KEEP:.0%}", tau > 0, **cal)
    q = json.loads(open(S.PROBE).readline())
    ids = S.prompt_ids(tok, q["question"])
    P = len(ids)
    noise = lambda: Noise("probe|0", S.SEED, v, dev)
    kw = (gcfg, nn_set, te)

    def same(a, ca, b, cb):
        n = cb.get_seq_length()
        return a["gen_ids"] == b["gen_ids"] and ca.get_seq_length() == n and bit_eq(ca, cb, 0, n)

    lo = generate_ent(model, sw, ids, float("-inf"), *kw, noise(), ENT_MAX_NEW, return_cache=True)
    add = S.generate(model, sw, ids, "add", *kw, noise(), ENT_MAX_NEW, return_cache=True)
    chk("tau -inf == edit (s2e_reencode.generate add), tokens and whole cache bitwise", same(lo, lo["cache"], add, add["cache"]),
        n_gen=add["n_gen"], n_replaced=lo["n_replaced"])
    hi = generate_ent(model, sw, ids, float("inf"), *kw, noise(), ENT_MAX_NEW, return_cache=True)
    re = S.generate(model, sw, ids, "add_re", *kw, noise(), ENT_MAX_NEW, return_cache=True)
    chk("tau +inf == edit_re (s2e_reencode.generate add_re), tokens, restores and whole cache bitwise",
        same(hi, hi["cache"], re, re["cache"]) and hi["restores"] == re["restores"], n_gen=re["n_gen"], n_kept=hi["n_kept"])
    del lo, add, hi, re
    g = generate_ent(model, sw, ids, tau, *kw, noise(), ENT_MAX_NEW, return_cache=True)
    fed = g["gen_ids"][:-1]
    chk("generation closed </think> within the budget", te in fed, n_gen=g["n_gen"])
    if te in fed:
        t_end, exp, pend, start = fed.index(te), [], 0, P
        for i, t in enumerate(fed):
            pend += 1
            if t in nn_set or i == t_end:
                hold = 1 if t in nn_set else 0
                if pend - hold:
                    exp.append((start, start + pend - hold))
                    start += pend - hold
                pend = hold
        chk("restores exactly at each fed '\\n\\n' (opener held) and at </think>", g["restores"] == exp, n_restores=len(exp))
        ct, cc, upto = g["cache"], g["cache_c"], g["restored_upto"]
        L = range(len(ct.key_cache))
        eq = [all(torch.equal(ct.key_cache[l][..., p, :], cc.key_cache[l][..., p, :]) and
                  torch.equal(ct.value_cache[l][..., p, :], cc.value_cache[l][..., p, :]) for l in L) for p in range(P, upto)]
        below = [g["ent"][p - P] < tau for p in range(P, upto)]
        chk("restored range: entropy < tau == clean entry bitwise, entropy >= tau == entry differs from clean; both kinds occur",
            eq == below and any(below) and not all(below), n_replaced=g["n_replaced"], n_kept=g["n_kept"],
            kept_share=g["n_kept"] / max(1, g["n_replaced"] + g["n_kept"]))
        ref = S.DynamicCache()
        sw.on = True
        out = model(input_ids=torch.tensor([ids], device=dev), past_key_values=ref, use_cache=True)
        sw.on = False
        h0 = token_entropy(out.logits[0, -1], gcfg.temperature)
        for t in fed[:upto - P]:
            model(input_ids=torch.tensor([[t]], device=dev), past_key_values=ref, use_cache=True)
        e_clean = rel_err(cc, ref, P, upto)
        chk("clean cache == edited prompt bitwise and ~= per-token unedited processing (rel < 0.05)",
            bit_eq(cc, ct, 0, P) and e_clean < 0.05, clean_err=e_clean)
        chk("ent[0] == entropy of the edited prompt forward's logits", abs(g["ent"][0] - h0) < 1e-4, ent0=g["ent"][0], h0=h0)
        sp = [i for i, t in enumerate(fed[:t_end]) if t in nn_set]
        j = next(a for a, b in zip(sp, sp[1:] + [t_end]) if b - a > 4 and a >= 4)
        g2 = generate_ent(model, sw, ids, tau, *kw, noise(), j + 5, return_cache=True)
        held = rel_err(g2["cache"], ref, P + j, P + j + 1)
        chk("mid-step: same tokens; restored range ends at the held opener; opener and later tokens still edited, clean cache "
            "ends there", g2["gen_ids"][:-1] == fed[:j + 4] and g2["restored_upto"] == P + j and
            g2["cache_c"].get_seq_length() == P + j and held > 0.05 and g2["ent"] == g["ent"][:j + 5], split_pos=j, held_err=held)
        g3 = generate_ent(model, sw, ids, tau, *kw, noise(), j + 5)
        chk("rerun gives identical tokens and entropies", g3["gen_ids"] == g2["gen_ids"] and g3["ent"] == g2["ent"])
        chk("peak GPU memory reported", True, peak_mem_gb=g["peak_mem_gb"])
    sw.remove()
    S.OUT.mkdir(exist_ok=True)
    (S.OUT / "checks_s2e_edit_ent.json").write_text(json.dumps({"all_pass": ok, "cfg": ENT_CFG, "report": report}, indent=1, default=str))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


def aime(path, n):
    """The first n of AIME 2022-2024 in the frozen order of reencode/aime_order.json. path is a JSONL you build (not in this
    repo) with idx (row of AI-MO/aimo-validation-aime), question and gt (the answer as a plain integer); see README."""
    by = {r["idx"]: r for r in map(json.loads, open(path))}
    order = json.load(open(Path(__file__).resolve().parent / "aime_order.json"))["order"]
    return [{"idx": i, "question": by[i]["question"], "gold": by[i]["gt"]} for i in order[:n]]


def run(cfgs, shard, plan=None, aime_file=None, aime_n=60):
    si, sn = map(int, shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    qs = aime(aime_file, aime_n) if aime_file else S.math100()
    name = "aime" if aime_file else "math100"
    order = plan or [(arm, n) for n in cfgs for arm in ("edit", "edit_re")]
    if aime_file:  # every setting of a question before the next question, so stopping early keeps results paired
        jobs = [(q["idx"], q, arm, n) for q in qs for arm, n in order][si::sn]
    else:
        jobs = sorted([(q["idx"], q, arm, n) for arm, n in order for q in qs][si::sn], key=lambda j: j[3])
    uid = lambda i, arm, n: f"{name}|{arm}|{i}|{n}|s{S.SEED}"
    seen = set()
    for fn in OUT.glob(f"{name}.*.jsonl"):
        seen |= {json.loads(l)["uid"] for l in open(fn)}
    jobs = [j for j in jobs if uid(j[0], j[2], j[3]) not in seen]
    print(f"{len(jobs)} generations", flush=True)
    if not jobs:
        return
    model, tok, gcfg = S.load_model()
    max_new = min(32768, model.config.max_position_embeddings - 512)
    nn_set, think_end = S.split_set(tok), tok.convert_tokens_to_ids("</think>")
    path = OUT / f"{name}.s{si}of{sn}.{int(time.time())}.jsonl"
    sw, cur = None, None
    with open(path, "a") as f:
        for i, q, arm, n in jobs:
            if n != cur:  # remove() leaves the originals in place
                if sw is not None:
                    sw.remove()
                sw = SimpleNamespace(on=False, remove=lambda: None) if n == "vanilla" else \
                    EditSwap(model, torch.load(pack_path(n), map_location="cpu"))
                cur = n
            ids = S.prompt_ids(tok, q["question"])
            noise = Noise(f"{name}|{i}", S.SEED, model.config.vocab_size, model.device)
            if arm == "edit_ent":
                g = generate_ent(model, sw, ids, json.loads(tau_path().read_text())["tau"], gcfg, nn_set, think_end, noise, max_new)
            else:
                g = S.generate(model, sw, ids, {"edit": "add", "edit_re": "add_re", "vanilla": "vanilla"}[arm], gcfg, nn_set,
                               think_end, noise, max_new)
            body = g["gen_ids"][:-1] if g["hit_eos"] else g["gen_ids"]
            text = tok.decode(body, skip_special_tokens=True).strip()
            rec = {"uid": uid(i, arm, n), "idx": i, "arm": arm, "cfg": n, "seed": S.SEED,
                   "question": q["question"], "gold": q["gold"], "text": text,
                   "reasoning_length": len(tok.encode(text, add_special_tokens=False)),
                   **{k: g[k] for k in ("n_gen", "hit_eos", "n_restores", "seconds", "replay_seconds", "peak_mem_gb")},
                   **{k: g[k] for k in ("n_replaced", "n_kept", "ent") if k in g},
                   "gen_ids": g["gen_ids"], "max_new": max_new, "gpu": torch.cuda.get_device_name()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[edit] q{i} {arm} {n} tok={g['n_gen']} eos={g['hit_eos']} restores={g['n_restores']} "
                  f"{g['seconds']:.0f}s (replay {g['replay_seconds']:.1f}s) mem {g['peak_mem_gb']:.1f}G", flush=True)
            del g
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "run", "packs", "ent-checks"])
    ap.add_argument("--plan", choices=["curve", "ent"])
    ap.add_argument("--aime", metavar="FILE")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--cfgs", nargs="+", default=RUN, choices=list(CFGS))
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.mode == "packs":  # only their compute_edits, for settings the checks stage did not build
        model = S.load_model()[0]
        for n in sorted({c for _, c in CURVE}):
            build_pack(model, n)
    elif a.mode == "checks":
        checks()
    elif a.mode == "ent-checks":
        ent_checks()
    else:
        run(a.cfgs, a.shard, AIME if a.aime else CURVE if a.plan == "curve" else ENT if a.plan == "ent" else None, a.aime, a.n)


if __name__ == "__main__":
    main()
