"""Checks for the add / add_re arms; run before generation; exits non-zero on any failure.

    cd efficient_reasoning && python reencode/checks.py [--alpha 1.2]      -> reencode_results/checks_s2e.json

 1 hook: with the hook active, every o_proj / down_proj output equals Steer2Edit's LinearInterventionHook
   (output + direction.type_as(output) * alpha, computed independently) bitwise; inactive, logits equal the
   unhooked model bitwise.
 2 add_re schedule: on a real generation (GSM8K probe question 0, closes </think>), restores happen exactly at each fed
   '\\n\\n' token and at </think>, the restored range ends at the last boundary, and the held opener is the only
   pending token after a '\\n\\n'.
 3 rebuilt cache: the restored range equals, bitwise, an independent rebuild (steered prefill, then the generated
   tokens replayed unsteered in the same pieces); a held opener's entry equals its steered entry bitwise; the restored
   range is within tolerance (per-layer relative Frobenius error < 0.05) of per-token unsteered processing
   after the same steered prompt, and closer to it than the add arm's cache is to its own unsteered reference.
 4 determinism: rerunning add_re on the same question / seed gives identical tokens.
"""
import argparse
import json

import torch
from transformers import DynamicCache

import s2e_reencode as S
from s2e_reencode import Noise


def rel_err(a, b, lo, hi):
    worst = 0.0
    for l in range(len(b.key_cache)):
        for X, R in ((a.key_cache[l], b.key_cache[l]), (a.value_cache[l], b.value_cache[l])):
            x, r = X[..., lo:hi, :].float(), R[..., lo:hi, :].float()
            worst = max(worst, float((x - r).norm() / r.norm()))
    return worst


def bit_eq(a, b, lo, hi):
    return all(torch.equal(a.key_cache[l][..., lo:hi, :], b.key_cache[l][..., lo:hi, :]) and
               torch.equal(a.value_cache[l][..., lo:hi, :], b.value_cache[l][..., lo:hi, :]) for l in range(len(a.key_cache)))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=max(S.ALPHAS))
    alpha = ap.parse_args().alpha
    model, tok, gcfg = S.load_model()
    dev = model.device
    report, ok = [], True

    def chk(name, cond, **info):
        nonlocal ok
        ok &= bool(cond)
        report.append({"check": name, "pass": bool(cond), **info})
        print(("PASS " if cond else "FAIL ") + name, json.dumps(info, default=str)[:300], flush=True)

    q = json.loads(open(S.PROBE).readline())
    ids = S.prompt_ids(tok, q["question"])
    x = torch.tensor([ids], device=dev)

    # 1 hook
    base = model(input_ids=x).logits
    steer = S.Steer(model, alpha)
    d = torch.load(S.DIRS, map_location="cpu")["post"]["direction"]
    cap = {}
    mods = [(f"model.layers[{i}].self_attn.o_proj", L.self_attn.o_proj) for i, L in enumerate(model.model.layers)] + \
           [(f"model.layers[{i}].mlp.down_proj", L.mlp.down_proj) for i, L in enumerate(model.model.layers)]
    # capture each module's input and final (post-hook) output; recompute the unhooked output from the input
    hs = [m.register_forward_hook((lambda n: lambda mod, i, o: cap.__setitem__(n, (i[0].detach(), o.detach())))(n))
          for n, m in mods]
    steer.on = True
    model(input_ids=x)
    steer.on = False
    worst_bit = True
    for n, m in mods:
        inp, out = cap[n]
        raw = torch.nn.functional.linear(inp, m.weight, m.bias)
        ref = raw + d[n].to(dev).type_as(raw) * alpha
        worst_bit &= torch.equal(out, ref)
    for h in hs:
        h.remove()
    off = model(input_ids=x).logits
    chk(f"hook == Steer2Edit LinearInterventionHook (alpha {alpha}) on all {len(mods)} components, bitwise; "
        "inactive hook leaves logits bitwise unchanged", worst_bit and torch.equal(off, base), n_components=len(mods))

    # 2-3 on a real generation
    nn_set, te = S.split_set(tok), tok.convert_tokens_to_ids("</think>")
    noise = lambda: Noise(f"probe|0", S.SEED, model.config.vocab_size, dev)
    g = S.generate(model, steer, ids, "add_re", gcfg, nn_set, te, noise(), 4096, return_cache=True)
    gen = g["gen_ids"]
    fed = gen[:-1]
    chk("add_re generation closed </think> within 4096 tokens", te in fed, n_gen=g["n_gen"])
    if te in fed:
        P, t_end = len(ids), fed.index(te)
        exp, pend, start = [], 0, P
        for i, t in enumerate(fed):
            pend += 1
            if t in nn_set or i == t_end:
                hold = 1 if t in nn_set else 0
                if pend - hold:
                    exp.append((start, start + pend - hold))
                    start += pend - hold
                pend = hold
        chk("restores exactly at each fed '\\n\\n' (opener held) and at </think>", g["restores"] == exp,
            n_restores=len(g["restores"]), expected=len(exp), restored_upto=g["restored_upto"], expected_upto=start)
        ct = g["cache"]
        # independent rebuild: steered prefill, then every restore span replayed unsteered in REPLAY_PIECE pieces,
        # held openers fed steered in between exactly as generation did
        reb = DynamicCache()
        steer.on = True
        model(input_ids=x, past_key_values=reb, use_cache=True)
        steer.on = False
        for lo, hi in exp:
            reb.crop(lo)
            span = fed[lo - P:hi - P]
            for j in range(0, len(span), S.REPLAY_PIECE):
                model(input_ids=torch.tensor([span[j:j + S.REPLAY_PIECE]], device=dev), past_key_values=reb, use_cache=True)
        hi = exp[-1][1]
        chk("restored range == independent rebuild (steered prompt + unsteered replay in the same pieces), bitwise",
            bit_eq(ct, reb, P, hi) and bit_eq(ct, reb, 0, P), restored=(P, hi))
        # per-token unsteered reference after the same steered prompt
        ref = DynamicCache()
        steer.on = True
        model(input_ids=x, past_key_values=ref, use_cache=True)
        steer.on = False
        for t in fed[:hi - P]:
            model(input_ids=torch.tensor([[t]], device=dev), past_key_values=ref, use_cache=True)
        e_re = rel_err(ct, ref, P, hi)
        g0 = S.generate(model, steer, ids, "add", gcfg, nn_set, te, noise(), len(fed[:hi - P]) + 1, return_cache=True)
        f0 = g0["gen_ids"][:-1]
        ref0 = DynamicCache()
        steer.on = True
        model(input_ids=x, past_key_values=ref0, use_cache=True)
        steer.on = False
        for t in f0:
            model(input_ids=torch.tensor([[t]], device=dev), past_key_values=ref0, use_cache=True)
        e_add = rel_err(g0["cache"], ref0, P, P + len(f0))
        chk("restored range ~= per-token unsteered processing (rel Frobenius < 0.05) and closer than the add arm",
            e_re < 0.05 and e_re < e_add, add_re_err=e_re, add_err=e_add, n_tokens=hi - P)
        # mid-step: stop 3 tokens after a '\n\n' that has >= 4 more tokens before the next boundary
        sp = [i for i, t in enumerate(fed[:t_end]) if t in nn_set]
        j = next(a for a, b in zip(sp, sp[1:] + [t_end]) if b - a > 4 and a >= 4)
        g2 = S.generate(model, steer, ids, "add_re", gcfg, nn_set, te, noise(), j + 5, return_cache=True)
        p2 = g2["gen_ids"][:-1]
        same = p2 == fed[:len(p2)]
        held_steered = rel_err(g2["cache"], ref, P + j, P + j + 1) > 3 * rel_err(g2["cache"], ref, P + j - 4, P + j)
        chk("mid-step: same tokens; restored range ends at the held opener; its entry stays steered (> 3x the error "
            "of the restored tokens before it)", same and g2["restored_upto"] == P + j and held_steered,
            split_pos=j, restored_upto=g2["restored_upto"], expected=P + j)
        # 4 determinism
        g3 = S.generate(model, steer, ids, "add_re", gcfg, nn_set, te, noise(), j + 5)
        chk("rerun gives identical tokens", g3["gen_ids"] == g2["gen_ids"])
    steer.remove()
    S.OUT.mkdir(exist_ok=True)
    (S.OUT / "checks_s2e.json").write_text(json.dumps({"all_pass": ok, "alpha": alpha, "report": report},
                                                            indent=1, default=str))
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
