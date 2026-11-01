"""Steer2Edit's weight edit (efficient reasoning) with and without step re-encoding.

    python reencode/s2e_edit_reencode.py checks          -> reencode_results/checks_s2e_edit.json (exits non-zero on failure)
    python reencode/s2e_edit_reencode.py run [--cfgs ...] [--shard i/n]

The edit is theirs, unmodified: edit.py's compute_edits (rank-1 factors from the extract_directions.py directions) applied with
evaluate.py's apply_rank1_edits_to_hf_model. Qwen3-4B-Thinking's settings edit only mlp.down_proj columns
(rho_attn -1). Switching is a weight swap: each edited module keeps its original and its edited weight tensor, and
`on` points module.weight.data at one of them. With the edit on, the forward is their edited model's, bitwise.

Arms (s2e_reencode.generate, unchanged: edit runs as "add", edit_re as "add_re")
  edit     their edited model; every forward edited, cache kept
  edit_re  edit + step re-encoding: at each fed '\\n\\n' and at </think>, the tokens since the last restore are
           replayed with the original weights and their K/V replace the edited ones; the new '\\n\\n' stays edited
           until its step ends (held, as in s2e_reencode). The prompt stays edited (their edited model sees it).
Questions, seed, noise keys and budget as in s2e_reencode.py, so its vanilla generations are the shared reference.
"""
import argparse
import ast
import json
import os
import time
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


def run(cfgs, shard, plan=None):
    si, sn = map(int, shard.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    qs = S.math100()
    order = plan or [(arm, n) for n in cfgs for arm in ("edit", "edit_re")]
    jobs = sorted([(q["idx"], q, arm, n) for arm, n in order for q in qs][si::sn], key=lambda j: j[3])
    uid = lambda i, arm, n: f"math100|{arm}|{i}|{n}|s{S.SEED}"
    seen = set()
    for fn in OUT.glob("math100.*.jsonl"):
        seen |= {json.loads(l)["uid"] for l in open(fn)}
    jobs = [j for j in jobs if uid(j[0], j[2], j[3]) not in seen]
    print(f"{len(jobs)} generations", flush=True)
    if not jobs:
        return
    model, tok, gcfg = S.load_model()
    max_new = min(32768, model.config.max_position_embeddings - 512)
    nn_set, think_end = S.split_set(tok), tok.convert_tokens_to_ids("</think>")
    path = OUT / f"math100.s{si}of{sn}.{int(time.time())}.jsonl"
    sw, cur = None, None
    with open(path, "a") as f:
        for i, q, arm, n in jobs:
            if n != cur:  # jobs are ordered by cfg, so this happens once per cfg; remove() leaves the originals in place
                if sw is not None:
                    sw.remove()
                sw, cur = EditSwap(model, torch.load(pack_path(n), map_location="cpu")), n
            ids = S.prompt_ids(tok, q["question"])
            g = S.generate(model, sw, ids, {"edit": "add", "edit_re": "add_re"}[arm], gcfg, nn_set, think_end,
                           Noise(f"math100|{i}", S.SEED, model.config.vocab_size, model.device), max_new)
            body = g["gen_ids"][:-1] if g["hit_eos"] else g["gen_ids"]
            text = tok.decode(body, skip_special_tokens=True).strip()
            rec = {"uid": uid(i, arm, n), "idx": i, "arm": arm, "cfg": n, "seed": S.SEED,
                   "question": q["question"], "gold": q["gold"], "text": text,
                   "reasoning_length": len(tok.encode(text, add_special_tokens=False)),
                   **{k: g[k] for k in ("n_gen", "hit_eos", "n_restores", "seconds", "replay_seconds", "peak_mem_gb")},
                   "gen_ids": g["gen_ids"], "max_new": max_new, "gpu": torch.cuda.get_device_name()}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[edit] q{i} {arm} {n} tok={g['n_gen']} eos={g['hit_eos']} restores={g['n_restores']} "
                  f"{g['seconds']:.0f}s (replay {g['replay_seconds']:.1f}s) mem {g['peak_mem_gb']:.1f}G", flush=True)
            del g
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["checks", "run", "packs"])
    ap.add_argument("--plan", choices=["curve"])
    ap.add_argument("--cfgs", nargs="+", default=RUN, choices=list(CFGS))
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    if a.mode == "packs":  # only their compute_edits, for settings the checks stage did not build
        model = S.load_model()[0]
        for n in sorted({c for _, c in CURVE}):
            build_pack(model, n)
    elif a.mode == "checks":
        checks()
    else:
        run(a.cfgs, a.shard, CURVE if a.plan == "curve" else None)


if __name__ == "__main__":
    main()
