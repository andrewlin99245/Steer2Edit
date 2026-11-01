"""Grade the weight-edit runs (edit vs edit_re) with their extract_answer / verify_answer, unmodified.

    cd efficient_reasoning && python reencode/evaluate_edit_reencode.py      (needs math_verify only; no GPU)
Vanilla is s2e_reencode.py's (same questions, seed, noise). Writes reencode_results/edit/eval/math100.jsonl and prints, per edit
setting: accuracy / mean reasoning length / cap hits for edit and edit_re, edit vs vanilla (their shortening), and
paired edit_re vs edit (one-arm-only counts with a sign test; mean length difference and geometric-mean length ratio
with question-level sign-flip permutation p); restores per edit_re run.
"""
import json
import math

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_reencode import OUT as S2E_OUT, flip_p, grade, sign_p  # noqa: E402

OUT = S2E_OUT / "edit"


def load(d, keep):
    recs = {}
    for fn in sorted(d.glob("math100.*.jsonl")):
        for l in open(fn):
            r = json.loads(l)
            if keep(r):
                recs[r["uid"]] = r
    return recs


def main():
    recs = load(S2E_OUT, lambda r: r["arm"] == "vanilla") | load(OUT, lambda r: True)
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    G = {}
    with open(OUT / "eval" / "math100.jsonl", "w") as f:
        for r in recs.values():
            pred, ok = grade(r)
            cfg = r.get("cfg", "-")
            G[(r["arm"], cfg, r["idx"])] = (ok, r["reasoning_length"], r["hit_eos"], r["n_restores"])
            f.write(json.dumps({k: r[k] for k in ("uid", "idx", "arm", "gold", "reasoning_length", "n_gen", "hit_eos",
                                                  "n_restores")} | {"cfg": cfg, "prediction": pred, "correct": ok}) + "\n")
    van = {i: v for (arm, _, i), v in G.items() if arm == "vanilla"}
    print(f"vanilla n={len(van)} acc {sum(v[0] for v in van.values())} mean len "
          f"{sum(v[1] for v in van.values()) / max(1, len(van)):.0f} caps {sum(not v[2] for v in van.values())}")
    for cfg in sorted({c for arm, c, _ in G if arm != "vanilla"}):
        print(f"\n== {cfg}")
        for arm in ("edit", "edit_re"):
            v = {i: x for (a, c, i), x in G.items() if a == arm and c == cfg}
            if not v:
                continue
            q = [i for i in v if i in van]
            print(f"  {arm:7s} n={len(v):3d} acc {sum(x[0] for x in v.values()):3d} mean len "
                  f"{sum(x[1] for x in v.values()) / len(v):5.0f} caps {sum(not x[2] for x in v.values())} | vs vanilla "
                  f"(n {len(q)}): len ratio {sum(v[i][1] for i in q) / max(1, sum(van[i][1] for i in q)):.3f} acc "
                  f"{sum(v[i][0] for i in q)} vs {sum(van[i][0] for i in q)}"
                  + (f" | restores/run {sum(x[3] for x in v.values()) / len(v):.0f}" if arm == "edit_re" else ""))
        Q = sorted(i for (a, c, i) in G if a == "edit_re" and c == cfg and ("edit", cfg, i) in G)
        if Q:
            A, B = (lambda i: G[("edit_re", cfg, i)]), (lambda i: G[("edit", cfg, i)])
            w = sum(A(i)[0] and not B(i)[0] for i in Q)
            l = sum(B(i)[0] and not A(i)[0] for i in Q)
            d = [A(i)[1] - B(i)[1] for i in Q]
            lr = [math.log(max(A(i)[1], 1) / max(B(i)[1], 1)) for i in Q]
            print(f"  paired edit_re vs edit n={len(Q)}: acc {sum(A(i)[0] for i in Q)} vs {sum(B(i)[0] for i in Q)}, "
                  f"only {w}/{l} sign p {sign_p(w, l):.3f} | mean len diff {sum(d) / len(d):+.0f} p {flip_p(d):.3f} | "
                  f"geo ratio {math.exp(sum(lr) / len(lr)):.3f} p {flip_p(lr):.3f}")


if __name__ == "__main__":
    main()
